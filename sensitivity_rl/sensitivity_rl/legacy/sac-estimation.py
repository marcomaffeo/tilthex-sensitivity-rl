import argparse
import os
import shutil
import json

import sys
import os

# 1. Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)

# --- IMPORTANT ---
# Set this *before* importing JAX
# This tells the script to only see the GPU with ID '0' (the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

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
from envs import Pendulum, DroneV0, DroneV1, TilthexV0, TilthexV1
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

from networks import (
    ActorCritic,
    Actor,
    SoftQNetwork,
    SensQNetwork,
    get_sac_action_dist,
)


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

def get_inner_field(state, field_name):
    """Walk the wrapper state chain to find a field by name."""
    while not hasattr(state, field_name) or field_name == "env_state":
        if not hasattr(state, "env_state"):
            raise AttributeError(
                f"Field '{field_name}' not found in wrapper state chain"
            )
        state = state.env_state
    return getattr(state, field_name)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


from buffers import (
    ReplaySensTransition,
    ReplaySensOnlyTransition,
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

    update_sensitivity_batch = jax.vmap(
        env.update_sensitivity, in_axes=(0, None, 0, 0, 0, 0, 0)
    )

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
        qf1_sens = SensQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        qf2_sens = SensQNetwork(
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
        qf1_sens_target = SensQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        qf2_sens_target = SensQNetwork(
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
            qf1_sens_params = qf1_sens.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape), jnp.zeros((env.param_dim*env.state_dim))
            )
            qf2_sens_params = qf2_sens.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape), jnp.zeros((env.param_dim*env.state_dim))
            )
            qf1_target_params = qf1_params
            qf2_target_params = qf2_params
            qf1_sens_target_params = qf1_sens_params
            qf2_sens_target_params = qf2_sens_params
            target_action_params = actor_params
        else:
            actor_params = actor.init(_rng, init_x)
            qf1_params = qf1.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf2_params = qf2.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf1_sens_params = qf1_sens.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape), jnp.zeros((env.param_dim*env.state_dim))
            )
            qf2_sens_params = qf2_sens.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape), jnp.zeros((env.param_dim*env.state_dim))
            )
            qf1_target_params = qf1_params
            qf2_target_params = qf2_params
            qf1_sens_target_params = qf1_sens_params
            qf2_sens_target_params = qf2_sens_params
            target_action_params = actor_params

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
            tx_q1_sens = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_q2_sens = optax.chain(
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
            tx_q1_sens = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["Q_LR"], eps=1e-5),
            )
            tx_q2_sens = optax.chain(
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
        train_state_qf1_sens = TrainState.create(
            apply_fn=qf1_sens.apply,
            params=qf1_sens_params,
            tx=tx_q1_sens,
        )
        train_state_qf2_sens = TrainState.create(
            apply_fn=qf2_sens.apply,
            params=qf2_sens_params,
            tx=tx_q2_sens,
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

        # Initialize sensitivity regularization coefficient
        target_sensitivity = jnp.array(config.get("TARGET_SENSITIVITY", 150.0))
        if config.get("SENSITIVITY_REGULARIZATION_WEIGHT", 4.0) == "auto":
            log_sens_coef = jnp.array(4.0)
            sens_coef = jnp.exp(log_sens_coef)
        else:
            sens_coef = config.get("SENSITIVITY_REGULARIZATION_WEIGHT", 1.0)
            log_sens_coef = jnp.log(jnp.array(sens_coef))

        sens_coef_optimizer = optax.chain(
            optax.clip_by_global_norm(0.01),
            optax.adam(config.get("SENS_LR", 1e-5), eps=1e-5),
        )
        sens_coef_opt_state = sens_coef_optimizer.init(log_sens_coef)

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
            example_transition=ReplaySensTransition(
                done=jnp.zeros((1,)),
                action=jnp.zeros((1,) + env.action_space(env_params).shape),
                reward=jnp.zeros((1,)),
                obs=jnp.zeros((1,) + env.observation_space(env_params).shape),
                next_obs=jnp.zeros(
                    (1,) + env.observation_space(env_params).shape
                ),
            ),
        )

        # Create second buffer specifically for sensitivity data with potentially smaller size
        sens_buffer_size = int(config.get("SENSITIVITY_BUFFER_SIZE", config["BUFFER_SIZE"]))
        rb_sens = create_empty_buffer(
            buffer_size=sens_buffer_size,
            example_transition=ReplaySensOnlyTransition(
                done=jnp.zeros((1,)),
                action=jnp.zeros((1,) + env.action_space(env_params).shape),
                obs=jnp.zeros((1,) + env.observation_space(env_params).shape),
                next_obs=jnp.zeros(
                    (1,) + env.observation_space(env_params).shape
                ),
                sens=jnp.zeros((1,) + (int(env.state_dim), int(env.param_dim))),
                sens_next=jnp.zeros((1,) + (int(env.state_dim), int(env.param_dim))),
                F_sens_x=jnp.zeros(
                    (1,) + (int(env.state_dim), int(env.param_dim))
                ),
                F_sens_x_next=jnp.zeros(
                    (1,) + (int(env.state_dim), int(env.param_dim))
                ),
                df_du=jnp.zeros(
                    (1,) + (int(env.state_dim), int(env.action_dim))
                ),
                df_du_next=jnp.zeros(
                    (1,) + (int(env.state_dim), int(env.action_dim))
                ),
                dobs_dx=jnp.zeros(
                    (1,) + (int(env.obs_dim), int(env.param_dim))
                ),
                dobs_dx_next=jnp.zeros(
                    (1,) + (int(env.obs_dim), int(env.param_dim))
                ),
                log_sens_norm=jnp.zeros((1,)),
                log_sens_next_norm=jnp.zeros((1,)),
            ),
        )

        # Initialize statistics for normalization with exponential moving average
        # Statistics: [log_pi_min, log_pi_max, q_min, q_max, sens_min, sens_max]
        # Initialize with reasonable defaults that will be updated during training
        normalization_stats = jnp.array([-10.0, 0.0, -500.0, 100.0, -50.0, 50.0])
        stats_alpha = config.get("STATS_ALPHA", 0.01)  # Forgetting factor for EMA

        # Sensitivity evaluation function
        def evaluate_sensitivity(actor_params, rng):
            """Run NUM_EVAL_STEPS environment steps to collect sensitivity metrics."""
            num_eval_steps = config.get("NUM_STEPS", 200)

            # Reset evaluation environment
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, 20)
            eval_obs, eval_env_state = env.reset(reset_rng, env_params)

            if config["SENSITIVITY"]:
                env.update_network_params(actor_params)

            def eval_step(carry, _):
                eval_env_state, eval_obs, rng = carry

                # Select action using current policy
                rng, _rng = jax.random.split(rng)
                pi = actor.apply(actor_params, eval_obs)
                action, _, _, _ = get_sac_action_dist(
                    pi, scale_action, bias_action, _rng, jnp.zeros((eval_obs.shape[0],) + env.action_space(env.default_params).shape), True
                )

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, 20)
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
                train_state_qf1_sens,
                train_state_qf2_sens,
                ent_coef_opt_state,
                sens_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                qf1_sens_target_params,
                qf2_sens_target_params,
                target_action_params,
                log_ent_coef,
                log_sens_coef,
                env_state,
                last_obs,
                last_info,
                rb,
                rb_sens,
                normalization_stats,
                rng,
                total_steps,
            ) = runner_state

            # Update target action params conditionally
            def sens_param_update(operand):
                params, old_target_params = operand
                return params
            
            def not_sens_param_update(operand):
                params, old_target_params = operand
                return old_target_params
            
            target_action_params = jax.lax.cond(
                total_steps % config["NUM_STEPS"] == 0,
                sens_param_update,
                not_sens_param_update,
                operand=(train_state_actor.params, target_action_params)
            )
            
            # Update env network params with the selected target_action_params
            # This side effect happens outside the cond
            env.update_network_params(target_action_params)

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi = actor.apply(train_state_actor.params, last_obs)
            action_pi, _, _, _ = get_sac_action_dist(
                pi, scale_action, bias_action, _rng, jnp.zeros((last_obs.shape[0],) + env.action_space(env.default_params).shape), True
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
            prev_sensitivity = get_inner_field(env_state, "sensitivity")
            log_norm = prev_sensitivity.log_norm_old
            F_sens_x = prev_sensitivity.F_sens_x
            df_du = prev_sensitivity.df_du
            dobs_dx = prev_sensitivity.dobs_dx
            sens = prev_sensitivity.state
            obsv, env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )

            obsv = jax.lax.stop_gradient(obsv)
            env_state = jax.lax.stop_gradient(env_state)
            
            transition = ReplaySensTransition(
                done,
                action,
                reward,
                last_obs,
                obsv,
            )
            rb = add_batch_to_buffer(rb, transition)
            
            # Add sensitivity data to the dedicated sensitivity buffer
            next_sensitivity = get_inner_field(env_state, "sensitivity")
            sens_transition = ReplaySensOnlyTransition(
                done,
                action,
                last_obs,
                obsv,
                sens,
                next_sensitivity.state,
                F_sens_x,
                next_sensitivity.F_sens_x,
                df_du,
                next_sensitivity.df_du,
                dobs_dx,
                next_sensitivity.dobs_dx,
                next_sensitivity.log_norm_old,
                next_sensitivity.log_norm,
            )
            rb_sens = add_batch_to_buffer(rb_sens, sens_transition)

            total_steps = total_steps + config["NUM_ENVS"]

            runner_state = (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                train_state_qf1_sens,
                train_state_qf2_sens,
                ent_coef_opt_state,
                sens_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                qf1_sens_target_params,
                qf2_sens_target_params,
                target_action_params,
                log_ent_coef,
                log_sens_coef,
                env_state,
                obsv,
                info,
                rb,
                rb_sens,
                normalization_stats,
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
                train_state_qf1_sens,
                train_state_qf2_sens,
                ent_coef_opt_state,
                sens_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                qf1_sens_target_params,
                qf2_sens_target_params,
                target_action_params,
                log_ent_coef,
                log_sens_coef,
                env_state,
                obsv,
                info,
                rb,
                rb_sens,
                normalization_stats,
                rng,
                total_steps,
            ) = runner_state

            # Sample batch from replay buffer
            rng, _rng_batch = jax.random.split(rng)
            batch = sample_batch_from_buffer(
                rb, config["BATCH_SIZE"], _rng_batch
            )
            
            # Sample batch from sensitivity buffer for sensitivity Q-network updates
            rng, _rng_sens_batch = jax.random.split(rng)
            batch_sens = sample_batch_from_buffer(
                rb_sens, config["BATCH_SIZE"], _rng_sens_batch
            )

            ent_coef = jnp.exp(log_ent_coef)

            rng, _rng_sens = jax.random.split(rng)
            rng_sens = jax.random.split(_rng_sens, config["BATCH_SIZE"])

            """
            _, updated_sens_reward = update_sensitivity_batch(
                rng_sens,
                train_state_actor.params,
                batch.df_du_next,
                batch.dobs_dx_next,
                batch.next_obs,
                batch.log_sens_next_norm,
                batch.F_sens_x_next,
            )
            updated_sens_reward = -(
                updated_sens_reward - batch.log_sens_next_norm
            )
            

            # Normalize updated_sens_reward using buffer statistics
            #sens_mean = rb.mean[1]
            #sens_std = jnp.sqrt(rb.var[1]) + 1e-8
            #updated_sens_reward = (updated_sens_reward) / sens_std
            updated_sens_reward = jax.lax.stop_gradient(updated_sens_reward)
            """
            # Compute target Q-values
            rng, _rng_target = jax.random.split(rng)
            next_state_actions = actor.apply(
                train_state_actor.params, batch.next_obs
            )
            next_state_actions_sample, next_state_log_pi, _, _ = (
                get_sac_action_dist(
                    next_state_actions,
                    scale_action,
                    bias_action,
                    _rng_target,
                    jnp.zeros((batch.next_obs.shape[0],) +env.action_space(env.default_params).shape),
                    True,
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
            min_qf_next_target_norm = jnp.minimum(qf1_next_target, qf2_next_target).reshape(-1, 1)   
            
            # Combine normalized Q with entropy and sensitivity
            min_qf_next_target = (
                min_qf_next_target_norm
                - ent_coef * next_state_log_pi
                #+ sens_coef
                #* updated_sens_reward[..., None]
            )

            rewards_col = batch.reward[..., None]
            dones_col = batch.done[..., None]

            next_q_value = (
                rewards_col
                + (1 - dones_col) * config["GAMMA"] * min_qf_next_target
            )
            next_q_value = jax.lax.stop_gradient(next_q_value)

            # Update min/max statistics with EMA
            # log_pi stats
            log_pi_min_batch = jnp.min(next_state_log_pi)
            log_pi_max_batch = jnp.max(next_state_log_pi)
            normalization_stats = normalization_stats.at[0].set(
                jnp.minimum(normalization_stats[0], (1 - stats_alpha) * normalization_stats[0] + stats_alpha * log_pi_min_batch)
            )
            normalization_stats = normalization_stats.at[1].set(
                jnp.maximum(normalization_stats[1], (1 - stats_alpha) * normalization_stats[1] + stats_alpha * log_pi_max_batch)
            )
            
            # Q-value stats
            q_min_batch = jnp.min(min_qf_next_target)
            q_max_batch = jnp.max(min_qf_next_target)
            normalization_stats = normalization_stats.at[2].set(
                jnp.minimum(normalization_stats[2], (1 - stats_alpha) * normalization_stats[2] + stats_alpha * q_min_batch)
            )
            normalization_stats = normalization_stats.at[3].set(
                jnp.maximum(normalization_stats[3], (1 - stats_alpha) * normalization_stats[3] + stats_alpha * q_max_batch)
            )

            # --- Q-function update ---
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

            # --- Q_sens-function update (using sensitivity buffer) ---
            # Compute sensitivity Q-value targets using batch_sens
            rng, _rng_sens_target = jax.random.split(rng)
            next_state_actions_sens = actor.apply(
                train_state_actor.params, batch_sens.next_obs
            )
            next_state_actions_sens_sample, _, _, _ = (
                get_sac_action_dist(
                    next_state_actions_sens,
                    scale_action,
                    bias_action,
                    _rng_sens_target,
                    jnp.zeros((batch_sens.next_obs.shape[0],) + env.action_space(env.default_params).shape),
                    True,
                )
            )

            qf1_sens_next_target_sens = qf1_sens_target.apply(
                qf1_sens_target_params,
                batch_sens.next_obs,
                next_state_actions_sens_sample,
                batch_sens.sens_next.reshape(batch_sens.next_obs.shape[0], -1)
            )
            qf2_sens_next_target_sens = qf2_sens_target.apply(
                qf2_sens_target_params,
                batch_sens.next_obs,
                next_state_actions_sens_sample,
                batch_sens.sens_next.reshape(batch_sens.next_obs.shape[0], -1)
            )
            min_qf_sens_next_target_sens = (
                jnp.minimum(qf1_sens_next_target_sens, qf2_sens_next_target_sens).reshape(-1, 1)
            )

            dones_col_sens = batch_sens.done[..., None]
            next_q_sens_value = (
               batch_sens.log_sens_next_norm[..., None]-batch_sens.log_sens_norm[..., None]  + (1 - dones_col_sens) * config["GAMMA"] * (min_qf_sens_next_target_sens)
            )
            next_q_sens_value = jax.lax.stop_gradient(next_q_sens_value)
            
            # Sensitivity component stats (from batch_sens)
            sens_component = min_qf_sens_next_target_sens
            sens_min_batch = jnp.min(sens_component)
            sens_max_batch = jnp.max(sens_component)
            normalization_stats = normalization_stats.at[4].set(
                jnp.minimum(normalization_stats[4], (1 - stats_alpha) * normalization_stats[4] + stats_alpha * sens_min_batch)
            )
            normalization_stats = normalization_stats.at[5].set(
                jnp.maximum(normalization_stats[5], (1 - stats_alpha) * normalization_stats[5] + stats_alpha * sens_max_batch)
            )
            normalization_stats = jax.lax.stop_gradient(normalization_stats)
            
            def q1_sens_loss_fn(qf1_sens_params):
                qf1_sens_a_values = qf1_sens.apply(qf1_sens_params, batch_sens.obs, batch_sens.action, batch_sens.sens.reshape(batch_sens.obs.shape[0], -1))
                qf1_sens_loss = jnp.mean(
                    (qf1_sens_a_values.reshape(-1, 1) - next_q_sens_value) ** 2
                )
                return qf1_sens_loss

            def q2_sens_loss_fn(qf2_sens_params):
                qf2_sens_a_values = qf2_sens.apply(qf2_sens_params, batch_sens.obs, batch_sens.action, batch_sens.sens.reshape(batch_sens.obs.shape[0], -1))
                qf2_sens_loss = jnp.mean(
                    (qf2_sens_a_values.reshape(-1, 1) - next_q_sens_value) ** 2
                )
                return qf2_sens_loss

            (q1_sens_loss, q1_sens_grads) = jax.value_and_grad(q1_sens_loss_fn)(
                train_state_qf1_sens.params
            )
            train_state_qf1_sens = train_state_qf1_sens.apply_gradients(grads=q1_sens_grads)

            (q2_sens_loss, q2_sens_grads) = jax.value_and_grad(q2_sens_loss_fn)(
                train_state_qf2_sens.params
            )
            train_state_qf2_sens = train_state_qf2_sens.apply_gradients(grads=q2_sens_grads)

            # --- Conditional Actor and Alpha update ---
            # This function defines the single update step for actor and alpha
            def actor_alpha_update_body(carry, _):
                (train_state_actor, log_ent_coef, log_sens_coef, ent_coef_opt_state, sens_coef_opt_state, rng) = (
                    carry
                )
                rng, _rng_actor = jax.random.split(rng)
                rng, _rng_sens = jax.random.split(rng)
                rng_sens = jax.random.split(_rng_sens, config["BATCH_SIZE"])

                # Use batch, ent_coef, sens_coef, train_state_qf1/2 from outer scope
                # Note: This uses the *just updated* Q-network parameters
                sens_coef_local = jnp.exp(log_sens_coef)

                def actor_loss_fn(actor_params):
                    
                    sens_state, updated_sens_reward = update_sensitivity_batch(
                        rng_sens,
                        actor_params,
                        batch_sens.df_du,
                        batch_sens.dobs_dx,
                        batch_sens.obs,
                        batch_sens.log_sens_norm,
                        batch_sens.F_sens_x,
                    )

                    updated_sens_reward_r =  (
                        updated_sens_reward - batch_sens.log_sens_norm
                    )
                    
                    # Normalize updated_sens_reward using buffer statistics
                    #sens_std = jnp.sqrt(normalization_stats_local[3]) + 1e-8
                    #updated_sens_reward_norm = (updated_sens_reward ) / sens_std
                    
                    # Stop gradient to prevent nested differentiation (jacfwd inside grad)
                    # updated_sens_reward = jax.lax.stop_gradient(updated_sens_reward)
                    
                    pi = actor.apply(actor_params, batch.obs)
                    actions, log_pi, _, action_pre_tanh = get_sac_action_dist(
                        pi, scale_action, bias_action, _rng_actor, jnp.zeros((batch.obs.shape[0],) + env.action_space(env.default_params).shape), True
                    )
                    #sens log pi
                    pi_sens = actor.apply(target_action_params, batch.obs)
                    _, log_pi_sens, _, _ = get_sac_action_dist(
                        pi_sens, scale_action, bias_action, _rng_actor, action_pre_tanh, False
                    )

                    qf1_pi = qf1.apply(
                        train_state_qf1.params, batch.obs, actions
                    )
                    qf2_pi = qf2.apply(
                        train_state_qf2.params, batch.obs, actions
                    )
                    min_qf_pi_raw = jnp.minimum(qf1_pi, qf2_pi)

                    qf1_sens_pi = qf1_sens.apply(
                        train_state_qf1_sens.params, batch_sens.obs, actions, batch_sens.sens.reshape(batch_sens.obs.shape[0], -1)
                    )
                    qf2_sens_pi = qf2_sens.apply(
                        train_state_qf2_sens.params, batch_sens.obs, actions, batch_sens.sens.reshape(batch_sens.obs.shape[0], -1)
                    )
                    min_qf_sens_pi_raw = jnp.minimum(qf1_sens_pi, qf2_sens_pi)

                    # Compute sensitivity component
                    sens_component = min_qf_sens_pi_raw

                    # Normalize components to [-1, 1] using min-max scaling
                    # For log_pi (entropy term)
                    log_pi_range = normalization_stats[1] - normalization_stats[0] + 1e-6
                    log_pi_normalized = (log_pi - normalization_stats[0]) / log_pi_range
                    
                    # For Q-values
                    q_range = normalization_stats[3] - normalization_stats[2] + 1e-6
                    q_normalized = (min_qf_pi_raw - normalization_stats[2]) / q_range
                    
                    # For sensitivity component
                    sens_range = normalization_stats[5] - normalization_stats[4] + 1e-6
                    sens_normalized = (sens_component - normalization_stats[4]) / sens_range 
                    
                    # Clip the log difference to prevent inf when taking exp
                    log_diff = jnp.clip(log_pi - log_pi_sens, -50.0, 50.0)
                    ratio = jnp.exp(log_diff)

                    # PPO-style ratio clipping
                    ratio_clipped = jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0,
                    )

                    Q_sens_loss = (sens_normalized.reshape(-1,1)) * ratio
                    Q_sens_loss_clipped = (sens_normalized.reshape(-1,1))*ratio_clipped
                    
                    # Take minimum (most pessimistic) of clipped and unclipped
                    Q_sens_loss_final = jnp.minimum(Q_sens_loss, Q_sens_loss_clipped)
                    
                    q_1 = q_normalized.reshape(-1, 1)* ratio
                    q_2 = q_normalized.reshape(-1, 1)* ratio_clipped
                    q_final = jnp.minimum(q_1, q_2)
                    # Actor loss
                    
                    actor_loss = jnp.mean(
                        ent_coef * log_pi.reshape(-1, 1)
                        - q_normalized.reshape(-1, 1) 
                        + sens_coef_local *(Q_sens_loss_final.reshape(-1, 1) )
                    )
                    return actor_loss, (log_pi, updated_sens_reward)

                (actor_loss, (log_pi, updated_sens_reward_for_loss)), actor_grads = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(train_state_actor.params)

                train_state_actor = train_state_actor.apply_gradients(
                    grads=actor_grads
                )

                # Alpha update
                rng, _rng_alpha = jax.random.split(rng)
                # Use new actor params for alpha loss
                pi_for_alpha = actor.apply(train_state_actor.params, batch.obs)
                _, log_pi_for_alpha, _, _ = get_sac_action_dist(
                    pi_for_alpha, scale_action, bias_action, _rng_alpha, jnp.zeros((batch.obs.shape[0],) + env.action_space(env.default_params).shape), True
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

                # Sensitivity coefficient update
                # Use stop_gradient on updated_sens_reward to avoid nested differentiation
                sens_norm = jax.lax.stop_gradient(updated_sens_reward_for_loss)
                sens_norm = jnp.mean(sens_norm)
                
                def sens_coef_loss_fn(sens_params):
                    # Loss increases sens_coef when sens_norm < target, decreases when > target
                    sens_loss = (
                        -jnp.exp(sens_params)
                        * (sens_norm - target_sensitivity)
                    )
                    return sens_loss, None

                (sens_loss, _), sens_grads = jax.value_and_grad(
                    sens_coef_loss_fn, has_aux=True
                )(log_sens_coef)
                sens_updates, sens_coef_opt_state = sens_coef_optimizer.update(
                    sens_grads, sens_coef_opt_state
                )
                log_sens_coef = jax.lax.select(
                    config.get("SENSITIVITY_REGULARIZATION_WEIGHT", 1.0) == "auto",
                    optax.apply_updates(log_sens_coef, sens_updates),
                    log_sens_coef,
                )
                
                log_sens_coef = jnp.minimum(log_sens_coef, jnp.array(jnp.log(5.0)))  # Cap sens_coef to avoid overflow

                carry = (
                    train_state_actor,
                    log_ent_coef,
                    log_sens_coef,
                    ent_coef_opt_state,
                    sens_coef_opt_state,
                    rng,
                )
                losses = (actor_loss, alpha_loss, sens_loss, log_pi.mean(), sens_norm)
                return carry, losses

            # This function just passes the state through, doing nothing
            def no_actor_alpha_update_body(carry, _):
                losses = (0.0, 0.0, 0.0, 0.0, 0.0)  # actor_loss, alpha_loss, sens_loss, entropy, sens_norm
                return carry, losses

            # Get policy frequency from config, default to 1 (update every step)
            policy_frequency = config.get("POLICY_FREQUENCY", 1)
            initial_carry = (
                train_state_actor,
                log_ent_coef,
                log_sens_coef,
                ent_coef_opt_state,
                sens_coef_opt_state,
                rng,
            )

            # --- Branch 1: Run the actual updates ---
            def do_policy_update(carry):

                # Scan the *actual* update body
                (
                    (new_ts_actor, new_log_ent, new_log_sens, new_ent_opt_state, new_sens_opt_state, new_rng),
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
                    new_log_sens,
                    new_ent_opt_state,
                    new_sens_opt_state,
                    new_rng,
                )
                return new_carry, losses

            # --- Branch 2: Run a "do-nothing" scan ---
            def do_no_policy_update(carry):

                # Scan the "do nothing" body
                (
                    (new_ts_actor, new_log_ent, new_log_sens, new_ent_opt_state, new_sens_opt_state, new_rng),
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
                    new_log_sens,
                    new_ent_opt_state,
                    new_sens_opt_state,
                    new_rng,
                )
                return new_carry, losses

            # Use lax.cond to run *either* the update scan *or* the do-nothing scan
            # Both branches take the 'initial_carry' and return
            # (final_carry, losses_tree) with the *same structure*.
            (
                (train_state_actor, log_ent_coef, log_sens_coef, ent_coef_opt_state, sens_coef_opt_state, rng),
                (actor_losses, alpha_losses, sens_losses, entropies, sens_norms),
            ) = jax.lax.cond(
                (total_steps % policy_frequency) == 0,
                do_policy_update,  # Function to call if True
                do_no_policy_update,  # Function to call if False
                initial_carry,  # The argument to pass to *which_ever* func is called
            )

            # --- Conditional Target Network Update ---
            target_network_frequency = config.get("TARGET_NETWORK_FREQUENCY", 1)

            def update_targets(params):
                (
                    qf1_target_params,
                    qf2_target_params,
                    qf1_sens_target_params,
                    qf2_sens_target_params,
                ) = params
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
                qf1_sens_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    qf1_sens_target_params,
                    train_state_qf1_sens.params,
                )
                qf2_sens_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    qf2_sens_target_params,
                    train_state_qf2_sens.params,
                )
                return (
                    qf1_target_params,
                    qf2_target_params,
                    qf1_sens_target_params,
                    qf2_sens_target_params,
                )

            def no_update_targets(params):
                return params

            (
                qf1_target_params,
                qf2_target_params,
                qf1_sens_target_params,
                qf2_sens_target_params,
            ) = jax.lax.cond(
                (total_steps % target_network_frequency) == 0,
                update_targets,
                no_update_targets,
                (
                    qf1_target_params,
                    qf2_target_params,
                    qf1_sens_target_params,
                    qf2_sens_target_params,
                ),
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
            mean_sens_loss = jnp.mean(sens_losses)
            mean_entropy = jnp.mean(entropies)
            mean_total_loss = (
                q1_loss + q2_loss + mean_actor_loss
            )  # alpha_loss and sens_loss are separate

            if config.get("DEBUG"):

                def callback(info, sens_norm):
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
                    
                jax.debug.callback(callback, metric, sens_norms)
            # Log metrics to tensorboard
            def log_callback(
                mean_total,
                mean_qf1,
                mean_qf2,
                mean_qf1_sens,
                mean_qf2_sens,
                mean_actor,
                mean_ent,
                mean_sens_loss_val,
                sens_coef_val,
                metric,
                timestep,
                sens_mean,
                sens_max,
                should_log_sens,
                policy_frequency=policy_frequency,
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

                if (timestep % policy_frequency) == 0:
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
                        "train/qf1_sens_loss", float(mean_qf1_sens), int(timestep)
                    )
                    writer.add_scalar(
                        "train/qf2_sens_loss", float(mean_qf2_sens), int(timestep)
                    )
                    writer.add_scalar(
                        "train/policy_loss", float(mean_actor), int(timestep)
                    )
                    writer.add_scalar(
                        "train/entropy", float(mean_ent), int(timestep)
                    )
                    writer.add_scalar(
                        "train/sens_coef_loss", float(mean_sens_loss_val), int(timestep)
                    )
                    writer.add_scalar(
                        "train/sens_coef", float(sens_coef_val), int(timestep)
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
                q1_sens_loss,  # q1_sens_loss is scalar
                q2_sens_loss,  # q2_sens_loss is scalar
                mean_actor_loss,
                mean_entropy,
                mean_sens_loss,
                jnp.exp(log_sens_coef),
                metric,
                total_steps,
                sensitivity_mean_eval,
                sensitivity_max_eval,
                should_eval,
                config.get("POLICY_FREQUENCY", 1),
            )
            # --- End Logging ---

            runner_state = (
                train_state_actor,
                train_state_qf1,
                train_state_qf2,
                train_state_qf1_sens,
                train_state_qf2_sens,
                ent_coef_opt_state,
                sens_coef_opt_state,
                qf1_target_params,
                qf2_target_params,
                qf1_sens_target_params,
                qf2_sens_target_params,
                target_action_params,
                log_ent_coef,
                log_sens_coef,
                env_state,
                obsv,
                metric,  # Pass the metric (info) forward
                rb,
                rb_sens,
                normalization_stats,
                rng,
                total_steps,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_actor,
            train_state_qf1,
            train_state_qf2,
            train_state_qf1_sens,
            train_state_qf2_sens,
            ent_coef_opt_state,
            sens_coef_opt_state,
            qf1_target_params,
            qf2_target_params,
            qf1_sens_target_params,
            qf2_sens_target_params,
            target_action_params,
            log_ent_coef,
            log_sens_coef,
            env_state,
            obsv,
            info,
            rb,
            rb_sens,
            normalization_stats,
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


def run_experiment(config: dict, run_dir: Path, rng_key: jax.random.PRNGKey):
    """Run a single experiment with the given config.
    
    Args:
        config: Configuration dictionary
        run_dir: Directory to save results
        rng_key: JAX random key
    """
    # tensorboard writer
    writer = SummaryWriter(f"{run_dir}")

    train_jit = jax.jit(make_train(config, writer))
    # train_jit = make_train(config, writer)
    out = train_jit(rng_key)

    train_jit = jax.jit(make_train(config, writer))
    # train_jit = make_train(config, writer)
    out = train_jit(rng_key)

    # Save params
    (
        train_state_actor,
        train_state_qf1,
        train_state_qf2,
        train_state_qf1_sens,
        train_state_qf2_sens,
        _,
        _,
        qf1_target_params,
        qf2_target_params,
        qf1_sens_target_params,
        qf2_sens_target_params,
        target_action_params,
        log_ent_coef,
        log_sens_coef,
        env_state,
        obsv,
        _,
        _,
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

    params_qf1_sens = train_state_qf1_sens.params
    (run_dir / "params_qf1_sens.msgpack").write_bytes(
        serialization.to_bytes(params_qf1_sens)
    )

    params_qf2_sens = train_state_qf2_sens.params
    (run_dir / "params_qf2_sens.msgpack").write_bytes(
        serialization.to_bytes(params_qf2_sens)
    )

    # Save config as JSON
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Prepare metrics for hparams logging
    hparam_dict = {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, bool, str))
    }
    metric_dict = {}

    # Log hyperparameters with final metrics
    writer.add_hparams(hparam_dict, metric_dict, run_name=".")

    # close tensorboard writer
    writer.close()
    
    return run_dir


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
    run_dir = runs_root / config["ENV_NAME"] / f"SAC_QS_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving latest results to: {run_dir}")

    update_latest_symlink(run_dir, runs_root)
    
    # Run the experiment
    run_experiment(config, run_dir, rng)
