import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Optional
from flax.training.train_state import TrainState
import distrax
from pathlib import Path
from datetime import datetime
from flax import serialization
from envs import Pendulum, DroneV0
from gymnax.environments import environment
from envs.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    ClipAction,
    SensitivityWrapper,
    SensitivityExtWrapper,
    ResetEnvWrapper,
    MinMaxObservationWrapper,
)
from utils import (
    load_actor_params_from_file,
    merge_actor_params_with_fresh_critic,
    update_latest_symlink,
)
from torch.utils.tensorboard import SummaryWriter


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


envs: Dict[str, environment.Environment] = {
    "pendulum": Pendulum,
    "drone_v0": DroneV0,
}


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Actor network with explicit names
        actor_mean = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense1",
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2",
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(actor_mean)
        actor_logtstd = self.param(
            "actor_log_std", nn.initializers.zeros, (self.action_dim,)
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Critic network with explicit names
        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1",
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2",
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, writer: SummaryWriter):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = envs[config["ENV_NAME"]]()
    env_params = env.default_params
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    # env = MinMaxObservationWrapper(env, env_params)
    if config["SENSITIVITY"]:
        env = SensitivityExtWrapper(
            env,
            regularization_weight=config["SENSITIVITY_REGULARIZATION_WEIGHT"],
        )
    env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
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
                # Always initialize fresh parameters first
                fresh_params = network.init(_rng, init_x)

                # Load only actor parameters
                actor_params = load_actor_params_from_file(
                    config["LOAD_ACTOR_PARAMS_PATH"]
                )

                # Merge actor params with fresh critic params
                network_params = merge_actor_params_with_fresh_critic(
                    actor_params, fresh_params
                )
                print(
                    "Successfully loaded actor parameters (critic randomly initialized)"
                )

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: {e}")
                print(
                    "Falling back to random initialization for all parameters"
                )
                network_params = network.init(_rng, init_x)
        else:
            network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        if config["SENSITIVITY"]:
            env.set_actor(network, network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            if config["SENSITIVITY"]:
                train_state, env_state, last_obs, rng = runner_state
                env.update_network_params(train_state.params)
                runner_state = (train_state, env_state, last_obs, rng)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                obsv = jax.lax.stop_gradient(obsv)
                env_state = jax.lax.stop_gradient(env_state)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        log_ratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_ratio)

                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clipfracs = jnp.mean(
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        )

                        # get sensitivity
                        sensitivity_state = traj_batch.info[
                            "sensitivity_state_rw"
                        ]
                        ss_normalized = (sensitivity_state) / (
                            jnp.max(sensitivity_state)
                            - jnp.min(sensitivity_state)
                            + 1e-8
                        )
                        # ss_normalized = jnp.clip(ss_normalized, -10.0, 10.0)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        gae = (
                            gae
                            + config["SENSITIVITY_REGULARIZATION_WEIGHT"]
                            * ss_normalized
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            old_approx_kl,
                            approx_kl,
                            clipfracs,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state

                rng, _rng = jax.random.split(rng)
                batch_size = (
                    config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                )
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Extract and compute mean losses from loss_info
            # loss_info shape: [UPDATE_EPOCHS, NUM_MINIBATCHES] where each element is (total_loss, (value_loss, actor_loss, entropy))
            # loss_info is a tuple of (total_losses_array, aux_losses_tuple)
            # where aux_losses_tuple = (value_losses_array, actor_losses_array, entropies_array)
            total_losses, (
                value_losses,
                actor_losses,
                entropies,
                old_approx_kl,
                approx_kl,
                clipfracs,
            ) = loss_info

            # Compute means across all scan iterations (epochs and minibatches)
            mean_total_loss = total_losses.mean()
            mean_value_loss = value_losses.mean()
            mean_actor_loss = actor_losses.mean()
            mean_entropy = entropies.mean()
            mean_old_approx_kl = old_approx_kl.mean()
            mean_approx_kl = approx_kl.mean()
            mean_clipfrac = jnp.concatenate(clipfracs).mean()
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]]
                        * config["NUM_ENVS"]
                    )
                    print(
                        f"global step={timesteps[-1]}, episodic return={return_values.mean()}"
                    )

                jax.debug.callback(callback, metric)

            # logging variables
            traj_batch.value
            targets_var = jnp.var(targets)
            explained_var = 1 - jnp.var(targets - traj_batch.value) / (
                targets_var + 1e-8
            )
            # explained_var = explained_var.mean()

            # Log metrics to tensorboard using the computed mean losses
            def log_callback(
                mean_total,
                mean_value,
                mean_actor,
                mean_ent,
                metric,
                explained_var,
                mean_old_approx_kl,
                mean_approx_kl,
                mean_clipfrac,
                timestep,
            ):

                episodic_return = metric["returned_episode_returns"][
                    metric["returned_episode"]
                ].mean()
                episodic_return_true = metric["returned_episode_returns_true"][
                    metric["returned_episode"]
                ].mean()
                episodic_length = metric["returned_episode_lengths"][
                    metric["returned_episode"]
                ].mean()
                sensitivity_state_max = (
                    metric["sensitivity_state"].max(axis=0).mean()
                )
                sensitivity_state_mean = metric["sensitivity_state"].mean()

                writer.add_scalar(
                    "train/total_loss", float(mean_total), int(timestep)
                )
                writer.add_scalar(
                    "train/value_loss", float(mean_value), int(timestep)
                )
                writer.add_scalar(
                    "train/policy_loss", float(mean_actor), int(timestep)
                )
                writer.add_scalar(
                    "train/entropy", float(mean_ent), int(timestep)
                )
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
                    "returns/sensitivity_state_max",
                    float(sensitivity_state_max),
                    int(timestep),
                )
                writer.add_scalar(
                    "returns/sensitivity_state_mean",
                    float(sensitivity_state_mean),
                    int(timestep),
                )

                writer.add_scalar(
                    "train/old_approx_kl",
                    float(mean_old_approx_kl),
                    int(timestep),
                )
                writer.add_scalar(
                    "train/approx_kl", float(mean_approx_kl), int(timestep)
                )
                writer.add_scalar(
                    "train/clipfrac", float(mean_clipfrac), int(timestep)
                )
                writer.add_scalar(
                    "train/explained_variance",
                    float(explained_var),
                    int(timestep),
                )

            # Get current timestep from metric
            current_timestep = metric["timestep"].mean() * config["NUM_ENVS"]
            jax.debug.callback(
                log_callback,
                mean_total_loss,
                mean_value_loss,
                mean_actor_loss,
                mean_entropy,
                metric,
                explained_var,
                mean_old_approx_kl,
                mean_approx_kl,
                mean_clipfrac,
                current_timestep,
            )

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 1e-3,
        "NUM_ENVS": 400,
        "NUM_STEPS": 200,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 64,
        "GAMMA": 0.95,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
        "SENSITIVITY": True,
        "SENSITIVITY_REGULARIZATION_WEIGHT": 0.1,
        "ENV_NAME": "drone_v0",
        # "LOAD_ACTOR_PARAMS_PATH": "runs/latest/params.msgpack",  # Path to load actor params from
    }
    rng = jax.random.PRNGKey(30)

    # Create/rotate run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / config["ENV_NAME"] / f"SPPO_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving latest results to: {run_dir}")

    update_latest_symlink(run_dir, config["ENV_NAME"], runs_root)

    # tensorboard writer
    writer = SummaryWriter(f"{run_dir}")

    train_jit = jax.jit(make_train(config, writer))
    # train_jit = make_train(config, writer)
    out = train_jit(rng)

    # Save params
    train_state, env_state, obsv, _rng = out["runner_state"]
    params = train_state.params
    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))

    # Save config
    import json

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save raw metrics arrays for post-hoc analysis/plotting
    metrics = out["metrics"]
    np.savez_compressed(
        run_dir / "metrics_raw.npz",
        returned_episode_returns=np.array(metrics["returned_episode_returns"]),
        returned_episode_lengths=np.array(metrics["returned_episode_lengths"]),
        returned_episode=np.array(metrics["returned_episode"]),
        timestep=np.array(metrics["timestep"]),
    )

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
            "--deterministic",
        ]
    )

    subprocess.run(
        [
            "python3",
            (
                os.path.dirname(__file__) + "/plots/plot_drone_rollouts.py"
                if config["ENV_NAME"] == "drone_v0"
                else os.path.dirname(__file__)
                + "/plots/plot_pendulum_rollouts.py"
            ),
            f"{run_dir}/rollouts/",
        ]
    )
