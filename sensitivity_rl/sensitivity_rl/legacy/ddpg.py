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
from envs import Pendulum, DroneV0, DroneV1, TilthexV0
from gymnax.environments import environment
from envs.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    ClipAction,
    SensitivityOffWrapper,
    ResetEnvWrapper,
    MinMaxObservationWrapper,
)
from utils import (
    load_actor_params_from_file,
    merge_actor_params_with_fresh_critic,
    update_latest_symlink,
)
from torch.utils.tensorboard import SummaryWriter

from networks import ActorCritic, Actor, SoftQNetwork, get_ddpg_action_dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent with preset config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_ddpg.json"),
        help="Path to JSON config file (defaults to configs/default_ddpg.json)",
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
        env = SensitivityOffWrapper(
            env,
            regularization_weight=config["SENSITIVITY_REGULARIZATION_WEIGHT"],
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
        qf1_target = SoftQNetwork(
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        )
        actor_target = Actor(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
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
            qf1_target_params = qf1_params
            actor_target_params = actor_params
        else:
            actor_params = actor.init(_rng, init_x)
            qf1_params = qf1.init(
                _rng, init_x, jnp.zeros(env.action_space(env_params).shape)
            )
            qf1_target_params = qf1_params
            actor_target_params = actor_params

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_q1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_q1 = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
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
        if config["SENSITIVITY"]:
            env.set_actor(
                actor,
                actor_params,
                get_ddpg_action_dist,
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

        def env_step(runner_state, unused):
            (
                train_state_actor,
                train_state_qf1,
                qf1_target_params,
                actor_target_params,
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
            action_pi, _, _ = get_ddpg_action_dist(
                pi, scale_action, bias_action, _rng
            )
            action = action_pi + config["EXP_NOISE"] * jax.random.normal(
                _rng, shape=action_pi.shape
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
                qf1_target_params,
                actor_target_params,
                env_state,
                obsv,
                info,
                rb,
                rng,
                total_steps,
            )
            return runner_state, None

        def _update_step(runner_state, unused):

            (
                train_state_actor,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = runner_state

            # COLLECT TRAJECTORIES
            if config["SENSITIVITY"]:
                env.update_network_params(train_state_actor.params)

            runner_state, _ = env_step(runner_state, None)

            (
                train_state_actor,
                train_state_qf1,
                qf1_target_params,
                actor_target_params,
                env_state,
                obsv,
                info,
                rb,
                rng,
                total_steps,
            ) = runner_state

            def _update_epoch(update_state, unused):
                (
                    train_state_actor,
                    train_state_qf1,
                    qf1_target_params,
                    actor_target_params,
                    rb,
                    rng,
                ) = update_state

                # Sample batch from replay buffer
                rng, _rng_batch = jax.random.split(rng)
                batch = sample_batch_from_buffer(
                    rb, config["BATCH_SIZE"], _rng_batch
                )

                # Compute target Q-values FIRST (before defining loss functions)
                rng, _rng_target = jax.random.split(rng)
                next_state_actions = actor.apply(
                    actor_target_params, batch.next_obs
                )
                next_state_actions_sample, _, _ = get_ddpg_action_dist(
                    next_state_actions,
                    scale_action,
                    bias_action,
                    _rng_target,
                )

                qf1_next_target = qf1_target.apply(
                    qf1_target_params,
                    batch.next_obs,
                    next_state_actions_sample,
                )
                # Flatten rewards and dones to match Q-value shape
                next_q_value = (
                    batch.reward.flatten()
                    + (1 - batch.done.flatten())
                    * config["GAMMA"]
                    * qf1_next_target
                )
                next_q_value = jax.lax.stop_gradient(next_q_value)

                # Q-function update (NOW next_q_value is defined)
                def q1_loss_fn(qf1_params):
                    # Current Q-values
                    qf1_a_values = qf1.apply(
                        qf1_params, batch.obs, batch.action
                    )
                    qf1_loss = jnp.mean((qf1_a_values - next_q_value) ** 2)
                    return qf1_loss

                # Update Q-functions
                grad_q1_fn = jax.value_and_grad(
                    q1_loss_fn,
                    has_aux=False,
                )
                q1_loss, q1_grads = grad_q1_fn(train_state_qf1.params)

                train_state_qf1 = train_state_qf1.apply_gradients(
                    grads=q1_grads
                )

                def actor_update_step(update_state, unused):

                    (
                        train_state_actor,
                        train_state_qf1,
                        rng,
                    ) = update_state
                    # Actor update (with fresh RNG key)
                    rng, _rng_actor = jax.random.split(rng)

                    def actor_loss_fn(actor_params):
                        pi = actor.apply(actor_params, batch.obs)
                        actions, _, _ = get_ddpg_action_dist(
                            pi, scale_action, bias_action, _rng_actor
                        )

                        qf1_pi = qf1.apply(
                            train_state_qf1.params, batch.obs, actions
                        )
                        actor_loss = jnp.mean(-qf1_pi)
                        return actor_loss

                    actor_loss, actor_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=False
                    )(train_state_actor.params)

                    train_state_actor = train_state_actor.apply_gradients(
                        grads=actor_grads
                    )

                    loss_info = (actor_loss,)
                    update_state = (
                        train_state_actor,
                        train_state_qf1,
                        rng,
                    )
                    return update_state, loss_info

                us, loss_actor = jax.lax.scan(
                    actor_update_step,
                    (
                        train_state_actor,
                        train_state_qf1,
                        rng,
                    ),
                    None,
                    config.get("ACTOR_EPOCHS", 1),
                )
                (
                    train_state_actor,
                    _,
                    rng,
                ) = us
                entropy = jnp.array(0.0)

                # Update target networks (soft update)
                tau = config.get("POLYAK_TAU", 0.005)
                qf1_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    qf1_target_params,
                    train_state_qf1.params,
                )
                actor_target_params = jax.tree.map(
                    lambda target, online: tau * online + (1 - tau) * target,
                    actor_target_params,
                    train_state_actor.params,
                )

                update_state = (
                    train_state_actor,
                    train_state_qf1,
                    qf1_target_params,
                    actor_target_params,
                    rb,
                    rng,
                )

                total_loss = q1_loss + loss_actor[0]
                return update_state, (
                    total_loss,
                    (q1_loss, loss_actor[0], entropy),
                )

            update_state = (
                train_state_actor,
                train_state_qf1,
                qf1_target_params,
                actor_target_params,
                rb,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state_actor = update_state[0]
            train_state_qf1 = update_state[1]
            qf1_target_params = update_state[2]
            actor_target_params = update_state[3]
            metric = info
            rng = update_state[-1]

            # Extract and compute mean losses from loss_info
            # loss_info shape: [UPDATE_EPOCHS, NUM_MINIBATCHES] where each element is (total_loss, (value_loss, actor_loss, entropy))
            # loss_info is a tuple of (total_losses_array, aux_losses_tuple)
            # where aux_losses_tuple = (value_losses_array, actor_losses_array, entropies_array)
            total_losses, (
                qf1_losses,
                actor_losses,
                entropies,
            ) = loss_info

            # Compute means across all scan iterations (epochs and minibatches)
            mean_total_loss = total_losses.mean()
            mean_qf1_loss = qf1_losses.mean()
            mean_actor_loss = actor_losses.mean()
            mean_entropy = entropies.mean()
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

            # Log metrics to tensorboard using the computed mean losses
            def log_callback(
                mean_total,
                mean_qf1,
                mean_actor,
                mean_ent,
                metric,
                timestep,
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
                    "train/policy_loss", float(mean_actor), int(timestep)
                )
                writer.add_scalar(
                    "train/entropy", float(mean_ent), int(timestep)
                )

                if "sensitivity_state" in metric:
                    sensitivity_state_max = (
                        metric["sensitivity_state"].max(axis=0).mean()
                    )
                    sensitivity_state_mean = metric["sensitivity_state"].mean()
                    writer.add_scalar(
                        "returns/sensitivity_state_max",
                        float(sensitivity_state_max),
                        int(timestep),
                    )
                    writer.add_scalar(
                        "returns/sensitivity_state_mean",
                        float(sensitivity_state_mean),
                        int(timestep),
                    )

            jax.debug.callback(
                log_callback,
                mean_total_loss,
                mean_qf1_loss,
                mean_actor_loss,
                mean_entropy,
                metric,
                total_steps,
            )

            runner_state = (
                train_state_actor,
                train_state_qf1,
                qf1_target_params,
                actor_target_params,
                env_state,
                obsv,
                info,
                rb,
                rng,
                total_steps,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_actor,
            train_state_qf1,
            qf1_target_params,
            actor_target_params,
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
        runner_state, metric = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["NUM_UPDATES"]
            - config["LEARNING_STARTS"] // config["NUM_ENVS"],
        )
        return {"runner_state": runner_state, "metrics": metric}

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
    run_dir = runs_root / config["ENV_NAME"] / f"DDPG_{ts}"
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
        qf1_target_params,
        actor_target_params,
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

    shutil.copy(config_path, run_dir / "config.json")

    # Save raw metrics arrays for post-hoc analysis/plotting
    metrics = out["metrics"]

    # Minimal console summary
    ret = np.array(metrics["returned_episode_returns"]).reshape(-1)
    done = np.array(metrics["returned_episode"]).reshape(-1).astype(bool)
    ep_returns = ret[done]

    # Prepare metrics for hparams logging
    hparam_dict = {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, bool, str))
    }
    metric_dict = {}
    if ep_returns.size > 0:
        metric_dict = {
            "hparam/final_avg_return": float(ep_returns.mean()),
            "hparam/final_max_return": float(ep_returns.max()),
            "hparam/final_episodes": float(ep_returns.size),
        }
        print(
            f"Training finished. Episodes: {ep_returns.size}, Avg Return: {ep_returns.mean():.2f}"
        )

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

    subprocess.run(
        [
            "python3",
            (
                os.path.dirname(__file__) + "/plots/plot_tilthex_rollouts.py"
                if config["ENV_NAME"] == "tilthex_v0"
                else os.path.dirname(__file__)
                + "/plots/plot_pendulum_rollouts.py"
            ),
        ]
    )
