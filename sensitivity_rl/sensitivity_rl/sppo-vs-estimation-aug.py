import argparse
import os
import shutil
import json
from omegaconf import DictConfig, OmegaConf
import hydra

# --- IMPORTANT ---
# Set this *before* importing JAX
# This tells the script to only see the GPU with ID '0' (the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
# Set JAX platforms to use (cuda for GPU, cpu for fallback)
os.environ["JAX_PLATFORMS"] = "cuda,cpu"


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
    NormalizeVecRewardSensitivity,
    ClipAction,
    SensitivityWrapper,
    SensitivityExtWrapper,
    MujocoSensitivityExtWrapper,
    ResetEnvWrapper,
    MinMaxObservationWrapper,
)
from utils import (
    load_actor_params_from_file,
    merge_actor_params_with_fresh_critic,
    update_latest_symlink,
)
from torch.utils.tensorboard import SummaryWriter
from buffers import _calculate_gae_s, RolloutSensAugTransition
from envs import ENV_REGISTRY
from networks import ActorCriticSensAug, ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent with preset config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="Path to JSON config file (defaults to configs/default.json)",
    )
    return parser.parse_args()


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
    out = train_jit(rng_key)

    # Save params
    train_state, _, env_state, obsv, _, _rng = out["runner_state"]
    params = train_state.params
    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))

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

    writer.close()
    
    return run_dir


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


def make_train(config, writer: SummaryWriter):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = ENV_REGISTRY[config["ENV_NAME"]]()
    env_params = env.default_params
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    # env = MinMaxObservationWrapper(env, env_params)
    if config["SENSITIVITY"]:
        if env.type == "mujoco":
            env = MujocoSensitivityExtWrapper(
                env,
                regularization_weight=config[
                    "SENSITIVITY_REGULARIZATION_WEIGHT"
                ],
            )
        else:
            env = SensitivityExtWrapper(
                env,
                regularization_weight=config[
                    "SENSITIVITY_REGULARIZATION_WEIGHT"
                ],
            )
    env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        if config["SENSITIVITY"]:
            env = NormalizeVecRewardSensitivity(env, config["GAMMA"])
        else:
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
            actor_layer_sizes=config.get("ACTOR_LAYER_SIZES", [256, 256]),
            critic_layer_sizes=config.get("CRITIC_LAYER_SIZES", [256, 256]),
        )
        network_vs = ActorCriticSensAug(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            actor_layer_sizes=config.get("ACTOR_LAYER_SIZES", [256, 256]),
            critic_layer_sizes=config.get("CRITIC_LAYER_SIZES", [256, 256]),
            gru_hidden_size=config.get("GRU_HIDDEN_SIZE", 64),
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        init_gru_hidden = jnp.zeros((1, config.get("GRU_HIDDEN_SIZE", 64)))

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

        network_vs_params = network_vs.init(_rng, init_x, init_gru_hidden)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tvx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tvx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        train_state_vs = TrainState.create(
            apply_fn=network_vs.apply,
            params=network_vs_params,
            tx=tvx,
        )
        if config["SENSITIVITY"]:
            env.set_actor(network, network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # Initialize GRU hidden state
        gru_hidden = jnp.zeros(
            (config["NUM_ENVS"], config.get("GRU_HIDDEN_SIZE", 64))
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            if config["SENSITIVITY"]:
                (
                    train_state,
                    train_state_vs,
                    env_state,
                    last_obs,
                    gru_hidden,
                    rng,
                ) = runner_state
                env.update_network_params(train_state.params)
                runner_state = (
                    train_state,
                    train_state_vs,
                    env_state,
                    last_obs,
                    gru_hidden,
                    rng,
                )

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    train_state_vs,
                    env_state,
                    last_obs,
                    gru_hidden,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                value_s, new_gru_hidden = network_vs.apply(
                    train_state_vs.params,
                    last_obs,
                    gru_hidden,
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                reward_s = info.get(
                    "sensitivity_state_rw", jnp.zeros_like(reward)
                )
                new_gru_hidden = (
                    1 - done.reshape(-1, 1)
                ) * new_gru_hidden + done.reshape(-1, 1) * jnp.zeros_like(
                    new_gru_hidden
                )
                obsv = jax.lax.stop_gradient(obsv)
                env_state = jax.lax.stop_gradient(env_state)
                new_gru_hidden = jax.lax.stop_gradient(new_gru_hidden)

                transition = RolloutSensAugTransition(
                    done,
                    action,
                    value,
                    value_s,
                    reward,
                    reward_s,
                    log_prob,
                    last_obs,
                    gru_hidden,  # Store GRU hidden state
                    info,
                )
                runner_state = (
                    train_state,
                    train_state_vs,
                    env_state,
                    obsv,
                    new_gru_hidden,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # Normalize sensitivity rewards
            # traj_batch = traj_batch._replace(
            #     reward_s=(traj_batch.reward_s)
            #     / (
            #         jnp.max(traj_batch.reward_s)
            #         - jnp.min(traj_batch.reward_s)
            #         + 1e-8
            #     )
            # )

            # CALCULATE ADVANTAGE
            (
                train_state,
                train_state_vs,
                env_state,
                last_obs,
                gru_hidden,
                rng,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            last_val_s, _ = network_vs.apply(
                train_state_vs.params,
                last_obs,
                gru_hidden,
            )

            advantages, advantages_s, targets, targets_s = _calculate_gae_s(
                config, traj_batch, last_val, last_val_s
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(ts, batch_info):
                    (train_state, train_state_vs) = ts

                    traj_batch, advantages, advantages_s, targets, targets_s = (
                        batch_info
                    )

                    def _loss_fn_vs(
                        params_vs,
                        traj_batch,
                        targets_s,
                    ):
                        # Process batch through GRU
                        # traj_batch.gru_hidden shape: [num_steps, num_envs, gru_hidden_size]
                        # traj_batch.obs shape: [num_steps, num_envs, obs_dim]
                        # We need to reshape for batch processing
                        batch_size = traj_batch.obs.shape[0]
                        value_s, _ = network_vs.apply(
                            params_vs,
                            traj_batch.obs,
                            traj_batch.gru_hidden,
                        )

                        # CALCULATE VALUE SENSITIVITY LOSS
                        value_s_pred_clipped = traj_batch.value_s + (
                            value_s - traj_batch.value_s
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_s_losses = jnp.square(value_s - targets_s)
                        value_s_losses_clipped = jnp.square(
                            value_s_pred_clipped - targets_s
                        )
                        value_loss_s = config["VF_COEF"] * (
                            0.5
                            * jnp.maximum(
                                value_s_losses, value_s_losses_clipped
                            ).mean()
                        )
                        return value_loss_s

                    def _loss_fn(
                        params,
                        traj_batch,
                        gae,
                        gae_s,
                        targets,
                    ):
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

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        gae_s = (gae_s - gae_s.mean()) / (gae_s.std() + 1e-8)
                        gae = (
                            gae
                            + config["SENSITIVITY_REGULARIZATION_WEIGHT"]
                            * gae_s
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
                        train_state.params,
                        traj_batch,
                        advantages,
                        advantages_s,
                        targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    grad_fn_vs = jax.value_and_grad(_loss_fn_vs, has_aux=False)
                    total_loss_vs, grads_vs = grad_fn_vs(
                        train_state_vs.params,
                        traj_batch,
                        targets_s,
                    )
                    train_state_vs = train_state_vs.apply_gradients(
                        grads=grads_vs
                    )
                    return (
                        (train_state, train_state_vs),
                        total_loss + (total_loss_vs,),
                    )

                (
                    train_state,
                    train_state_vs,
                    traj_batch,
                    advantages,
                    advantages_s,
                    targets,
                    targets_s,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                batch_size = (
                    config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                )
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (
                    traj_batch,
                    advantages,
                    advantages_s,
                    targets,
                    targets_s,
                )
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
                (train_state, train_state_vs), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, train_state_vs), minibatches
                )
                update_state = (
                    train_state,
                    train_state_vs,
                    traj_batch,
                    advantages,
                    advantages_s,
                    targets,
                    targets_s,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                train_state_vs,
                traj_batch,
                advantages,
                advantages_s,
                targets,
                targets_s,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            train_state_vs = update_state[1]
            metric = traj_batch.info
            rng = update_state[-1]

            # Extract and compute mean losses from loss_info
            # loss_info shape: [UPDATE_EPOCHS, NUM_MINIBATCHES] where each element is (total_loss, (value_loss, actor_loss, entropy))
            # loss_info is a tuple of (total_losses_array, aux_losses_tuple)
            # where aux_losses_tuple = (value_losses_array, actor_losses_array, entropies_array)
            (
                total_losses,
                (
                    value_losses,
                    actor_losses,
                    entropies,
                    old_approx_kl,
                    approx_kl,
                    clipfracs,
                ),
                value_losses_s,
            ) = loss_info

            # Compute means across all scan iterations (epochs and minibatches)
            mean_total_loss = total_losses.mean()
            mean_value_s_loss = value_losses_s.mean()
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
                mean_values_s,
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
                    "train/value_loss_s",
                    float(mean_values_s),
                    int(timestep),
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
                mean_value_s_loss,
                mean_actor_loss,
                mean_entropy,
                metric,
                explained_var,
                mean_old_approx_kl,
                mean_approx_kl,
                mean_clipfrac,
                current_timestep,
            )

            runner_state = (
                train_state,
                train_state_vs,
                env_state,
                last_obs,
                gru_hidden,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            train_state_vs,
            env_state,
            obsv,
            gru_hidden,
            _rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": None}

    return train

@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig):
    """Main function decorated with Hydra.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert OmegaConf to regular dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize random key
    rng = jax.random.PRNGKey(config.get("SEED", 30))
    
    # Create/rotate run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / config["ENV_NAME"] / f"SPPOVSA_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving latest results to: {run_dir}")

    update_latest_symlink(run_dir, runs_root)
    
    # Run the experiment
    run_experiment(config, run_dir, rng)

    # Run rollouts for evaluation
    eval_config = {
        "NUM_ROLLOUTS": 10,
        "MAX_STEPS": 200,
        "DETERMINISTIC": True,
        "SEED": 42,
        "PARAM_SAMPLES": 1,
    }

    import subprocess

    print("Running evaluation rollouts (nominal)...")
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
            "--param_samples",
            f"{eval_config['PARAM_SAMPLES']}",
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
            "--output",
            "nominal.png",
        ]
    )

    print("Running evaluation rollouts (perturbed)...")
    eval_config["PARAM_SAMPLES"] = 10
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
            "--param_samples",
            f"{eval_config['PARAM_SAMPLES']}",
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
            "--output",
            "perturbed.png",
        ]
    )

if __name__ == "__main__":
    main()
    