import argparse
import os
import shutil
import json

# --- IMPORTANT ---
# Set this *before* importing JAX
# This tells the script to only see the GPU with ID '0' (the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import optax

from typing import NamedTuple, Optional
from flax.training.train_state import TrainState
from pathlib import Path
from datetime import datetime
from flax import serialization
from envs import ENV_REGISTRY
from envs import Pendulum, DroneV0, DroneV1
from gymnax.environments import environment
from envs.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    ClipAction,
    SensitivityOffWrapper,
    MujocoSensitivityOffWrapper,
    ResetEnvWrapper,
    MinMaxObservationWrapper,
)
from utils import (
    load_actor_params_from_file,
    merge_actor_params_with_fresh_critic,
    update_latest_symlink,
)
from torch.utils.tensorboard import SummaryWriter

from networks import ActorCritic, Actor, SoftQNetwork, get_sac_action_dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent with preset config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_sac.json"),
        help="Path to JSON config file (defaults to configs/default_sac.json)",
    )
    return parser.parse_args()


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


from buffers import (
    ReplayTransition,
    create_empty_buffer,
    add_batch_to_buffer,
    sample_batch_from_buffer,
)


def make_train(config, writer: SummaryWriter):
    # Ensure integer loop counts for scans even if TOTAL_TIMESTEPS is a float (e.g., 5e7)
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"])

    env = ENV_REGISTRY[config["ENV_NAME"]]()
    env_params = env.default_params
    action_space = env.action_space(env_params)
    # Correct action rescaling to map tanh outputs in [-1, 1] to [low, high]
    # a = tanh(x) * action_scale + action_bias, where
    # action_scale = (high - low)/2 and action_bias = (high + low)/2
    scale_action = (action_space.high - action_space.low) / 2.0
    bias_action = (action_space.high + action_space.low) / 2.0
    # Use JAX arrays to avoid implicit host-device transfers inside jit regions
    scale_action = jnp.asarray(scale_action, dtype=jnp.float32)
    bias_action = jnp.asarray(bias_action, dtype=jnp.float32)
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    # env = MinMaxObservationWrapper(env, env_params)
    if config["SENSITIVITY"]:
        if env.type == "mujoco":
            env = MujocoSensitivityOffWrapper(
                env,
                regularization_weight=config[
                    "SENSITIVITY_REGULARIZATION_WEIGHT"
                ],
            )
        else:
            env = SensitivityOffWrapper(
                env,
                regularization_weight=config[
                    "SENSITIVITY_REGULARIZATION_WEIGHT"
                ],
            )
    env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = 1.0 - (count // config["UPDATE_EPOCHS"]) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor = Actor(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        )
        qf1 = SoftQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        qf2 = SoftQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        qf1_target = SoftQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        qf2_target = SoftQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)

        # Load actor parameters from file if specified, otherwise initialize randomly
        if config.get("LOAD_ACTOR_PARAMS_PATH"):
            print(
                f"Loading actor parameters from: {config['LOAD_ACTOR_PARAMS_PATH']}"
            )
            print("Critic parameters will be randomly initialized")

            try:
                # Load only actor parameters
                actor_params = load_actor_params_from_file(
                    config["LOAD_ACTOR_PARAMS_PATH"]
                )

                print(
                    "Successfully loaded actor parameters (critic randomly initialized)"
                )

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: {e}")
                print(
                    "Falling back to random initialization for all parameters"
                )
                actor_params = actor.init(_rng, init_x)
            qf1_params = qf1.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf2_params = qf2.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf1_target_params = qf1_params
            qf2_target_params = qf2_params
        else:
            actor_params = actor.init(_rng, init_x)
            qf1_params = qf1.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf2_params = qf2.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf1_target_params = qf1_params
            qf2_target_params = qf2_params

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_q1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_q2 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["POLICY_LR"], eps=1e-5),
            )
            tx_q1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["Q_LR"], eps=1e-5),
            )
            tx_q2 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["Q_LR"], eps=1e-5),
            )

        train_state_actor = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=tx,
        )
        train_state_qf1 = TrainState.create(
            apply_fn=qf1.apply,
            params=qf1_params,
            tx=tx_q1,
        )
        train_state_qf2 = TrainState.create(
            apply_fn=qf2.apply,
            params=qf2_params,
            tx=tx_q2,
        )

        target_entropy = -jnp.prod(
            jnp.array(env.action_space(env_params).shape)
        )
        if config.get("ENT_COEF", 0.01) == "auto":
            log_ent_coef = jnp.array(0.0)
            ent_coef = jnp.exp(log_ent_coef)
        else:
            ent_coef = config.get("ENT_COEF", 0.01)
            log_ent_coef = jnp.log(jnp.array(ent_coef))

        ent_coef_optimizer = optax.chain(
            optax.clip_by_global_norm(0.01),
            optax.adam(config["ALPHA_LR"], eps=1e-5),
        )
        ent_coef_opt_state = ent_coef_optimizer.init(log_ent_coef)

        if config["SENSITIVITY"]:
            env.set_actor(
                actor,
                actor_params,
                get_sac_action_dist,
                scale_action,
                bias_action,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, _, _, info = env.step(
            rng_step,
            env_state,
            jnp.zeros((config["NUM_ENVS"],) + action_space.shape),
            env_params,
        )
        total_steps = 0

        rb = create_empty_buffer(
            buffer_size=int(config["BUFFER_SIZE"]),
            example_transition=ReplayTransition(
                done=jnp.zeros((1,)),
                action=jnp.zeros((1,) + env.action_space(env_params).shape),
                reward=jnp.zeros((1,)),
                obs=jnp.zeros((1,) + env.observation_space(env_params).shape),
                next_obs=jnp.zeros(
                    (1,) + env.observation_space(env_params).shape
                ),
            ),
        )

        # Sensitivity evaluation function
        def evaluate_sensitivity(actor_params, rng):
            """Run NUM_EVAL_STEPS environment steps to collect sensitivity metrics."""
            num_eval_steps = config.get("NUM_STEPS", 200)

            # Reset evaluation environment
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            eval_obs, eval_env_state = env.reset(reset_rng, env_params)

            if config["SENSITIVITY"]:
                env.update_network_params(actor_params)

            def eval_step(carry, _):
                eval_env_state, eval_obs, rng = carry

                # Select action using current policy
                rng, _rng = jax.random.split(rng)
                pi = actor.apply(actor_params, eval_obs)
                action, _, _ = get_sac_action_dist(
                    pi, scale_action, bias_action, _rng
                )

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                next_obs, next_env_state, _, _, info = env.step(
                    rng_step, eval_env_state, action, env_params
                )

                carry = (next_env_state, next_obs, rng)
                return carry, info["sensitivity_state"]

            # Run evaluation rollout
            _, infos = jax.lax.scan(
                eval_step,
                (eval_env_state, eval_obs, rng),
                None,
                length=num_eval_steps,
            )

            sensitivity_mean = infos.mean()
            sensitivity_max = infos.max(axis=(0, 1)).mean()

            return sensitivity_mean, sensitivity_max

        @jax.jit
        def env_step(runner_state, unused):
            (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                ent_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                log_ent_coef,
                env_state,
                last_obs,
                last_info,
                rb,
                rng,
                total_steps,
            ) = runner_state

            if config["SENSITIVITY"]:
                env.update_network_params(train_state_actor.params)

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi = actor.apply(train_state_actor.params, last_obs)
            action_pi, _, _ = get_sac_action_dist(
                pi, scale_action, bias_action, _rng
            )
            action_rand = jax.random.uniform(
                _rng,
                minval=action_space.low,
                maxval=action_space.high,
                shape=(config["NUM_ENVS"],) + action_space.shape,
            )
            action = jax.lax.select(
                total_steps < config["NUM_ENVS"] * config["LEARNING_STARTS"],
                action_rand,
                action_pi,
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            obsv = jax.lax.stop_gradient(obsv)
            env_state = jax.lax.stop_gradient(env_state)
            transition = ReplayTransition(done, action, reward, last_obs, obsv)
            rb = add_batch_to_buffer(rb, transition)

            total_steps = total_steps + config["NUM_ENVS"]

            runner_state = (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                ent_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                log_ent_coef,
                env_state,
                obsv,
                info,
                rb,
                rng,
                total_steps,
            )
            return runner_state, None

        @jax.jit
        def _update_step(runner_state, unused):

            # COLLECT ONE TRANSITION
            runner_state, _ = env_step(runner_state, None)

            (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                ent_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                log_ent_coef,
                env_state,
                obsv,
                info,
                rb,
                rng,
                total_steps,
            ) = runner_state

            # --- This logic replaces the scan over _update_epoch ---

            # Sample batch from replay buffer
            rng, _rng_batch = jax.random.split(rng)
            batch = sample_batch_from_buffer(
                rb, config["BATCH_SIZE"], _rng_batch
            )

            ent_coef = jnp.exp(log_ent_coef)

            # Compute target Q-values
            rng, _rng_target = jax.random.split(rng)
            next_state_actions = actor.apply(
                train_state_actor.params, batch.next_obs
            )
            next_state_actions_sample, next_state_log_pi, _ = (
                get_sac_action_dist(
                    next_state_actions,
                    scale_action,
                    bias_action,
                    _rng_target,
                )
            )

            next_state_log_pi = jax.lax.stop_gradient(next_state_log_pi)
            qf1_next_target = qf1_target.apply(
                qf1_target_params,
                batch.next_obs,
                next_state_actions_sample,
            )
            qf2_next_target = qf2_target.apply(
                qf2_target_params,
                batch.next_obs,
                next_state_actions_sample,
            )
            min_qf_next_target = (
                jnp.minimum(qf1_next_target, qf2_next_target).reshape(-1, 1)
                - ent_coef * next_state_log_pi
            )

            rewards_col = batch.reward[..., None]
            dones_col = batch.done[..., None]

            next_q_value = (
                rewards_col
                + (1 - dones_col) * config["GAMMA"] * min_qf_next_target
            )
            next_q_value = jax.lax.stop_gradient(next_q_value)

            # --- Q-function update (always happens) ---
            def q1_loss_fn(qf1_params):
                qf1_a_values = qf1.apply(qf1_params, batch.obs, batch.action)
                qf1_loss = jnp.mean(
                    (qf1_a_values.reshape(-1, 1) - next_q_value) ** 2
                )
                return qf1_loss

            def q2_loss_fn(qf2_params):
                qf2_a_values = qf2.apply(qf2_params, batch.obs, batch.action)
                qf2_loss = jnp.mean(
                    (qf2_a_values.reshape(-1, 1) - next_q_value) ** 2
                )
                return qf2_loss

            (q1_loss, q1_grads) = jax.value_and_grad(q1_loss_fn)(
                train_state_qf1.params
            )
            train_state_qf1 = train_state_qf1.apply_gradients(grads=q1_grads)

            (q2_loss, q2_grads) = jax.value_and_grad(q2_loss_fn)(
                train_state_qf2.params
            )
            train_state_qf2 = train_state_qf2.apply_gradients(grads=q2_grads)

            # --- Conditional Actor and Alpha update ---
            # This function defines the single update step for actor and alpha
            def actor_alpha_update_body(carry, _):
                (train_state_actor, log_ent_coef, ent_coef_opt_state, rng) = (
                    carry
                )
                rng, _rng_actor = jax.random.split(rng)

                # Use batch, ent_coef, train_state_qf1/2 from outer scope
                # Note: This uses the *just updated* Q-network parameters

                def actor_loss_fn(actor_params):
                    pi = actor.apply(actor_params, batch.obs)
                    actions, log_pi, _ = get_sac_action_dist(
                        pi, scale_action, bias_action, _rng_actor
                    )
                    qf1_pi = qf1.apply(
                        train_state_qf1.params, batch.obs, actions
                    )
                    qf2_pi = qf2.apply(
                        train_state_qf2.params, batch.obs, actions
                    )
                    min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
                    actor_loss = jnp.mean(
                        ent_coef * log_pi - min_qf_pi.reshape(-1, 1)
                    )
                    return actor_loss, log_pi

                (actor_loss, log_pi), actor_grads = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(train_state_actor.params)
                train_state_actor = train_state_actor.apply_gradients(
                    grads=actor_grads
                )

                # Alpha update
                rng, _rng_alpha = jax.random.split(rng)
                # Use new actor params for alpha loss
                pi_for_alpha = actor.apply(train_state_actor.params, batch.obs)
                _, log_pi_for_alpha, _ = get_sac_action_dist(
                    pi_for_alpha, scale_action, bias_action, _rng_alpha
                )
                log_pi_for_alpha = jax.lax.stop_gradient(log_pi_for_alpha)

                def ent_coef_loss_fn(alpha_params):
                    alpha_loss = (
                        -jnp.exp(alpha_params)
                        * (log_pi_for_alpha + target_entropy).mean()
                    )
                    return alpha_loss, None

                (alpha_loss, _), alpha_grads = jax.value_and_grad(
                    ent_coef_loss_fn, has_aux=True
                )(log_ent_coef)
                updates, ent_coef_opt_state = ent_coef_optimizer.update(
                    alpha_grads, ent_coef_opt_state
                )
                log_ent_coef = jax.lax.select(
                    config["ENT_COEF"] == "auto",
                    optax.apply_updates(log_ent_coef, updates),
                    log_ent_coef,
                )

                carry = (
                    train_state_actor,
                    log_ent_coef,
                    ent_coef_opt_state,
                    rng,
                )
                losses = (actor_loss, alpha_loss, log_pi.mean())
                return carry, losses

            # This function just passes the state through, doing nothing
            def no_actor_alpha_update_body(carry, _):
                losses = (0.0, 0.0, 0.0)  # actor_loss, alpha_loss, entropy
                return carry, losses

            # Get policy frequency from config, default to 1 (update every step)
            policy_frequency = config.get("POLICY_FREQUENCY", 1)
            initial_carry = (
                train_state_actor,
                log_ent_coef,
                ent_coef_opt_state,
                rng,
            )

            # --- Branch 1: Run the actual updates ---
            def do_policy_update(carry):

                # Scan the *actual* update body
                (
                    (new_ts_actor, new_log_ent, new_ent_opt_state, new_rng),
                    losses,
                ) = jax.lax.scan(
                    actor_alpha_update_body,
                    carry,
                    None,
                    length=policy_frequency,
                )
                new_carry = (
                    new_ts_actor,
                    new_log_ent,
                    new_ent_opt_state,
                    new_rng,
                )
                return new_carry, losses

            # --- Branch 2: Run a "do-nothing" scan ---
            def do_no_policy_update(carry):

                # Scan the "do nothing" body
                (
                    (new_ts_actor, new_log_ent, new_ent_opt_state, new_rng),
                    losses,
                ) = jax.lax.scan(
                    no_actor_alpha_update_body,
                    carry,
                    None,
                    length=policy_frequency,
                )
                new_carry = (
                    new_ts_actor,
                    new_log_ent,
                    new_ent_opt_state,
                    new_rng,
                )
                return new_carry, losses

            # Use lax.cond to run *either* the update scan *or* the do-nothing scan
            # Both branches take the 'initial_carry' and return
            # (final_carry, losses_tree) with the *same structure*.
            (
                (train_state_actor, log_ent_coef, ent_coef_opt_state, rng),
                (actor_losses, alpha_losses, entropies),
            ) = jax.lax.cond(
                (total_steps % policy_frequency) == 0,
                do_policy_update,  # Function to call if True
                do_no_policy_update,  # Function to call if False
                initial_carry,  # The argument to pass to *which_ever* func is called
            )

            # --- Conditional Target Network Update ---
            target_network_frequency = config.get("TARGET_NETWORK_FREQUENCY", 1)

            def update_targets(params):
                (qf1_target_params, qf2_target_params) = params
                tau = config.get("POLYAK_TAU", 0.005)
                qf1_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    qf1_target_params,
                    train_state_qf1.params,
                )
                qf2_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    qf2_target_params,
                    train_state_qf2.params,
                )
                return (qf1_target_params, qf2_target_params)

            def no_update_targets(params):
                return params

            (qf1_target_params, qf2_target_params) = jax.lax.cond(
                (total_steps % target_network_frequency) == 0,
                update_targets,
                no_update_targets,
                (qf1_target_params, qf2_target_params),
            )

            # --- Sensitivity Evaluation (every 100 iterations) ---
            eval_frequency = config.get("EVAL_SENSITIVITY_FREQUENCY", 100)

            def run_sensitivity_eval(carry):
                rng, actor_params = carry
                rng, _rng = jax.random.split(rng)
                sens_mean, sens_max = evaluate_sensitivity(actor_params, _rng)
                return (rng, sens_mean, sens_max)

            def no_sensitivity_eval(carry):
                rng, _ = carry
                return (rng, jnp.array(0.0), jnp.array(0.0))

            should_eval = jnp.logical_and(
                (total_steps % eval_frequency) == 0,
                config.get("SENSITIVITY", False),
            )

            rng, sensitivity_mean_eval, sensitivity_max_eval = jax.lax.cond(
                should_eval,
                run_sensitivity_eval,
                no_sensitivity_eval,
                (rng, train_state_actor.params),
            )

            # --- Logging ---
            metric = info  # Use the 'info' from env_step
            mean_actor_loss = jnp.mean(actor_losses)
            mean_alpha_loss = jnp.mean(
                alpha_losses
            )  # Not logged in CleanRL, but good to have
            mean_entropy = jnp.mean(entropies)
            mean_total_loss = (
                q1_loss + q2_loss + mean_actor_loss
            )  # alpha_loss is separate

            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]]
                        * config["NUM_ENVS"]
                    )
                    if timesteps.size == 0:
                        return
                    print(
                        f"global step={timesteps[-1]}, episodic return={return_values.mean()}"
                    )

                jax.debug.callback(callback, metric)

            # Log metrics to tensorboard
            def log_callback(
                mean_total,
                mean_qf1,
                mean_qf2,
                mean_actor,
                mean_ent,
                metric,
                timestep,
                sens_mean,
                sens_max,
                should_log_sens,
            ):
                if metric["timestep"][metric["returned_episode"]].size != 0:
                    episodic_return = metric["returned_episode_returns"][
                        metric["returned_episode"]
                    ].mean()
                    episodic_return_true = metric[
                        "returned_episode_returns_true"
                    ][metric["returned_episode"]].mean()
                    episodic_length = metric["returned_episode_lengths"][
                        metric["returned_episode"]
                    ].mean()
                    writer.add_scalar(
                        "returns/episodic_return",
                        float(episodic_return),
                        int(timestep),
                    )
                    writer.add_scalar(
                        "returns/episodic_return_true",
                        float(episodic_return_true),
                        int(timestep),
                    )
                    writer.add_scalar(
                        "returns/episodic_length",
                        float(episodic_length),
                        int(timestep),
                    )

                writer.add_scalar(
                    "train/total_loss", float(mean_total), int(timestep)
                )
                writer.add_scalar(
                    "train/qf1_loss", float(mean_qf1), int(timestep)
                )
                writer.add_scalar(
                    "train/qf2_loss", float(mean_qf2), int(timestep)
                )
                writer.add_scalar(
                    "train/policy_loss", float(mean_actor), int(timestep)
                )
                writer.add_scalar(
                    "train/entropy", float(mean_ent), int(timestep)
                )

                # Log sensitivity metrics when evaluation was performed
                if should_log_sens:
                    writer.add_scalar(
                        "returns/sensitivity_state_mean",
                        float(sens_mean),
                        int(timestep),
                    )
                    writer.add_scalar(
                        "returns/sensitivity_state_max",
                        float(sens_max),
                        int(timestep),
                    )

            jax.debug.callback(
                log_callback,
                mean_total_loss,
                q1_loss,  # q1_loss is scalar, not from scan
                q2_loss,  # q2_loss is scalar, not from scan
                mean_actor_loss,
                mean_entropy,
                metric,
                total_steps,
                sensitivity_mean_eval,
                sensitivity_max_eval,
                should_eval,
            )
            # --- End Logging ---

            runner_state = (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                ent_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                log_ent_coef,
                env_state,
                obsv,
                metric,  # Pass the metric (info) forward
                rb,
                rng,
                total_steps,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_actor,
            train_state_qf1,
            train_state_qf2,
            ent_coef_opt_state,
            qf1_target_params,
            qf2_target_params,
            log_ent_coef,
            env_state,
            obsv,
            info,
            rb,
            _rng,
            total_steps,
        )
        runner_state, _ = jax.lax.scan(
            env_step,
            runner_state,
            None,
            config["LEARNING_STARTS"] // config["NUM_ENVS"],
        )
        runner_state, _ = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["NUM_UPDATES"]
            - config["LEARNING_STARTS"] // config["NUM_ENVS"],
        )
        return {"runner_state": runner_state, "metrics": None}

    return train


LOG_STD_MAX = 2
LOG_STD_MIN = -5


if __name__ == "__main__":
    args = parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    rng = jax.random.PRNGKey(config.get("SEED", 30))

    # Create/rotate run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / config["ENV_NAME"] / f"SAC_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving latest results to: {run_dir}")

    update_latest_symlink(run_dir, runs_root)

    # tensorboard writer
    writer = SummaryWriter(f"{run_dir}")

    train_jit = jax.jit(make_train(config, writer))
    # train_jit = make_train(config, writer)
    out = train_jit(rng)

    # Save params
    (
        train_state_actor,
        train_state_qf1,
        train_state_qf2,
        _,
        qf1_target_params,
        qf2_target_params,
        log_ent_coef,
        env_state,
        obsv,
        _,
        _,
        _rng,
        _,
    ) = out["runner_state"]
    params_actor = train_state_actor.params
    (run_dir / "params_actor.msgpack").write_bytes(
        serialization.to_bytes(params_actor)
    )

    params_qf1 = train_state_qf1.params
    (run_dir / "params_qf1.msgpack").write_bytes(
        serialization.to_bytes(params_qf1)
    )

    params_qf2 = train_state_qf2.params
    (run_dir / "params_qf2.msgpack").write_bytes(
        serialization.to_bytes(params_qf2)
    )

    shutil.copy(config_path, run_dir / "config.json")
    # Prepare metrics for hparams logging
    hparam_dict = {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, bool, str))
    }
    metric_dict = {}

    """
    # Save raw metrics arrays for post-hoc analysis/plotting
    metrics = out["metrics"]

    # Minimal console summary
    ret = np.array(metrics["returned_episode_returns"]).reshape(-1)
    done = np.array(metrics["returned_episode"]).reshape(-1).astype(bool)
    ep_returns = ret[done]

        if ep_returns.size > 0:
        metric_dict = {
            "hparam/final_avg_return": float(ep_returns.mean()),
            "hparam/final_max_return": float(ep_returns.max()),
            "hparam/final_episodes": float(ep_returns.size),
        }
        print(
            f"Training finished. Episodes: {ep_returns.size}, Avg Return: {ep_returns.mean():.2f}"
        )
    """
    # Log hyperparameters with final metrics
    writer.add_hparams(hparam_dict, metric_dict, run_name=".")

    # close tensorboard writer
    writer.close()

    # Run rollouts for evaluation
    eval_config = {
        "NUM_ROLLOUTS": 10,
        "MAX_STEPS": 200,
        "DETERMINISTIC": True,
        "SEED": 42,
    }

    import subprocess

    # Call with positional and named args
    subprocess.run(
        [
            "python3",
            os.path.dirname(__file__) + "/rollouts.py",
            "--run_dirs",
            f"{run_dir}",
            "--max_step",
            f"{eval_config['MAX_STEPS']}",
            "--num_rollouts",
            f"{eval_config['NUM_ROLLOUTS']}",
        ]
    )

    subprocess.run(
        [
            "python3",
            (
                os.path.dirname(__file__) + "/plots/plot_drone_rollouts.py"
                if config["ENV_NAME"] == "drone_v0"
                or config["ENV_NAME"] == "drone_v1"
                else os.path.dirname(__file__)
                + "/plots/plot_pendulum_rollouts.py"
            ),
        ]
    )
