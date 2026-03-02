#!/usr/bin/env python3
"""Unified training entry point for PPO-family algorithms.

Usage:
    # Default config (ppo)
    python train.py

    # Select algorithm via Hydra override
    python train.py ++ALGORITHM=sppo_aug

    # Override environment and parameters
    python train.py ++ALGORITHM=ppo env=drone_v0 SEED=42
    python train.py ++ALGORITHM=sppo_aug env=drone_v1 LR=0.0005
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Environment setup before JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=8 --xla_cpu_enable_fast_math=true --xla_gpu_enable_fast_min_max=true "
)
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp
import optax
import hydra
from flax import nnx, serialization
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

from algorithms import ALGORITHM_REGISTRY, AlgorithmSpec
from buffers import UpdateState, RunnerState
from envs import ENV_REGISTRY
from utils import (
    update_latest_symlink,
    shuffle_batch,
)


# JAX compilation cache setup
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


# Module-level mutable state for wall-clock timing inside jax.debug.callback.
# jax.debug.callback returns None, so timing cannot be threaded through JAX
# scans.  Instead we track it purely on the Python/host side.
_timing_state: dict = {"last_time": None}


def log_metrics(
    writer, losses, metric, targets, traj_batch_value, timestep,
    num_steps, num_envs, extra_losses=None,
):
    """Create logging callback for training metrics."""
    targets_var = jnp.var(targets)
    explained_var = 1 - jnp.var(targets - traj_batch_value) / (targets_var + 1e-8)

    def log_callback(losses, metric, explained_var, timestep, extra_losses):
        episodic_return = metric["returned_episode_returns"][
            metric["returned_episode"]
        ].mean()
        episodic_return_true = metric["returned_episode_returns_true"][
            metric["returned_episode"]
        ].mean()
        episodic_length = metric["returned_episode_lengths"][
            metric["returned_episode"]
        ].mean()

        writer.add_scalar("train/total_loss", float(losses["total"]), int(timestep))
        writer.add_scalar("train/value_loss", float(losses["value"]), int(timestep))
        writer.add_scalar("train/policy_loss", float(losses["actor"]), int(timestep))
        writer.add_scalar("train/entropy", float(losses["entropy"]), int(timestep))
        writer.add_scalar(
            "returns/episodic_return", float(episodic_return), int(timestep)
        )
        writer.add_scalar(
            "returns/episodic_return_true", float(episodic_return_true), int(timestep)
        )
        writer.add_scalar(
            "returns/episodic_length", float(episodic_length), int(timestep)
        )

        # Sensitivity metrics (only present when sensitivity wrapper is active)
        if "sensitivity_state" in metric:
            sensitivity_state_max = metric["sensitivity_state"].max(axis=0).mean()
            sensitivity_state_mean = metric["sensitivity_state"].mean()
            # print(f"  sensitivity: max={float(sensitivity_state_max):.4f}, mean={float(sensitivity_state_mean):.4f}")
            writer.add_scalar(
                "returns/sensitivity_state_max", float(sensitivity_state_max), int(timestep)
            )
            writer.add_scalar(
                "returns/sensitivity_state_mean",
                float(sensitivity_state_mean),
                int(timestep),
            )

        writer.add_scalar(
            "train/old_approx_kl", float(losses["old_approx_kl"]), int(timestep)
        )
        writer.add_scalar("train/approx_kl", float(losses["approx_kl"]), int(timestep))
        writer.add_scalar("train/clipfrac", float(losses["clipfrac"]), int(timestep))
        writer.add_scalar(
            "train/explained_variance", float(explained_var), int(timestep)
        )

        # Wall-clock timing via Python-side mutable state
        now = time.time()
        if _timing_state["last_time"] is not None:
            elapsed = now - _timing_state["last_time"]
            steps_per_second = (num_steps * num_envs) / elapsed
            print(
                f"step={int(timestep)}  {elapsed:.2f}s  ({steps_per_second:.0f} steps/s)"
            )
            writer.add_scalar("perf/steps_per_second", steps_per_second, int(timestep))
        else:
            print(f"step={int(timestep)}  (first update, timing starts now)")
        _timing_state["last_time"] = now

        if extra_losses:
            for name, val in extra_losses.items():
                writer.add_scalar(f"train/{name}", float(val), int(timestep))

    jax.debug.callback(
        log_callback, losses, metric, explained_var, timestep, extra_losses
    )


def debug_callback(config, metric):
    """Debug callback for printing training progress."""

    def callback(info):
        return_values = info["returned_episode_returns"][info["returned_episode"]]
        timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
        print(
            f"global step={timesteps[-1]}, mean episodic return={return_values.mean()}"
        )

    jax.debug.callback(callback, metric)


def make_train(config, spec: AlgorithmSpec, writer: SummaryWriter, init_rng):
    """Create unified training function for PPO-family algorithms.

    Args:
        config: Training configuration dict
        spec: Algorithm specification
        writer: TensorBoard writer
        init_rng: PRNG key used for network initialisation (concrete, not traced)

    Returns:
        JIT-compilable train function
    """

    # Setup derived config values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = ENV_REGISTRY[config["ENV_NAME"]]()
    env_params = env.default_params
    env = spec.wrap_env(env, config)

    # INIT NETWORKS outside JIT (NNX needs concrete rngs)
    networks = spec.init_networks(
        config,
        env,
        env_params,
        init_rng,
    )
    graphdef = networks["graphdef"]

    if config.get("SENSITIVITY", False):
        env.set_actor(graphdef, networks["train_state"].params)

    # Create algorithm-specific functions (closures over graphdef)
    loss_fn = spec.make_loss_fn(config, graphdef)
    collect_fn = spec.make_collect_fn(config, env, env_params, networks)
    update_fn = spec.make_update_fn(config, loss_fn, shuffle_batch, networks)

    
    total_steps = config["TOTAL_TIMESTEPS"]

    def train(rng):
        # INIT RUNNER STATE (algorithm handles state structure and RNG splitting)
        runner_state = spec.init_runner_state(config, env, env_params, networks, rng)

        # TRAIN LOOP
        def _update_step(runner_state, _unused):
            if config.get("SENSITIVITY", False):
                env.update_network_params(runner_state.train_state.params)

            # COLLECT TRAJECTORIES
            collect_state, traj_batch = jax.lax.scan(
                collect_fn, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE GAE
            gae_results, collect_state = spec.calculate_gae(
                config,
                traj_batch,
                networks,
                collect_state,
            )

            # UPDATE NETWORK
            gae_extras = {
                k: v
                for k, v in gae_results.items()
                if k not in ("advantages", "targets")
            }
            update_state = UpdateState(
                train_state=collect_state.train_state,
                traj_batch=traj_batch,
                advantages=gae_results["advantages"],
                targets=gae_results["targets"],
                rng=collect_state.rng,
                extras={**collect_state.extras, **gae_extras},
            )
            update_state, loss_info = jax.lax.scan(
                update_fn, update_state, None, config["UPDATE_EPOCHS"]
            )

            # Extract losses
            losses, extra_losses = spec.extract_losses(loss_info)

            metric = traj_batch.info
            if config.get("DEBUG", False):
                debug_callback(config, metric)

            # Logging (timing tracked via Python-side _timing_state)
            current_timestep = metric["timestep"].mean() * config["NUM_ENVS"]
            log_metrics(
                writer,
                losses,
                metric,
                gae_results["targets"],
                traj_batch.value,
                current_timestep,
                num_steps=config["NUM_STEPS"],
                num_envs=config["NUM_ENVS"],
                extra_losses=extra_losses,
            )

            # Rebuild runner state with updated train_state(s)
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=collect_state.env_state,
                last_obs=collect_state.last_obs,
                rng=update_state.rng,
                extras={**collect_state.extras, **update_state.extras},
            )
            return runner_state, None

        # Run training loop
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return runner_state

    return train


def run_evaluation(config, run_dir):
    """Run evaluation rollouts and plotting."""

    script_dir = os.path.dirname(__file__)

    print("Running evaluation rollouts (nominal)...")
    subprocess.run(
        [
            "python3",
            f"{script_dir}/rollouts.py",
            "--run_dirs",
            f"{run_dir}",
            "--plot",
        ]
    )

    print("Running evaluation rollouts (perturbed)...")
    subprocess.run(
        [
            "python3",
            f"{script_dir}/rollouts.py",
            "--run_dirs",
            f"{run_dir}",
            "--param_samples",
            "10",
            "--plot",
        ]
    )


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig):
    """Main function decorated with Hydra.

    Args:
        cfg: Hydra configuration object
    """
    # Convert OmegaConf to regular dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)

    # Get algorithm from config (default to ppo)
    algorithm = config.get("ALGORITHM", "ppo").lower()

    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    spec = ALGORITHM_REGISTRY[algorithm]

    # Ensure SENSITIVITY is set for algorithms that need it
    if spec.uses_sensitivity and "SENSITIVITY" not in config:
        config["SENSITIVITY"] = True

    rng = jax.random.PRNGKey(config.get("SEED", 30))

    # Create run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = spec.run_dir(config, runs_root, ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {run_dir}")

    update_latest_symlink(run_dir, runs_root)

    # Setup tensorboard
    writer = SummaryWriter(f"{run_dir}")

    # Create unified training function
    rng, init_rng = jax.random.split(rng)
    train_fn = make_train(config, spec, writer, init_rng)

    # JIT compile and train
    train_jit = jax.jit(train_fn)
    print("Compiling and training...")
    start_time = time.time()
    final_state = train_jit(rng)
    jax.block_until_ready(final_state)
    end_time = time.time()

    # Log training speed (total time includes compilation)
    total_time = end_time - start_time
    total_steps = config["TOTAL_TIMESTEPS"]
    
    writer.add_scalar("perf/total_time_seconds", total_time, total_steps)

    # Save params
    params = final_state.train_state.params
    pure_params = params.to_pure_dict()
    (run_dir / "policy_params.msgpack").write_bytes(serialization.to_bytes(pure_params))

    # Add environment-specific attributes to config
    env = ENV_REGISTRY[config["ENV_NAME"]]()
    env_params = env.default_params
    config["env"] = {}

    # List of attributes to extract from env and env_params
    env_attrs = [
        ("action_dim", "env", "action_dim"),
        ("obs_dim", "env", "obs_dim"),
        ("param_dim", "env", "param_dim"),
        ("MIN_ACTION", "env_params", "min_action"),
        ("MAX_ACTION", "env_params", "max_action"),
        ("MIN_INPUT", "env_params", "min_input"),
        ("MAX_INPUT", "env_params", "max_input"),
    ]

    for config_key, source, attr_name in env_attrs:
        try:
            source_obj = env if source == "env" else env_params
            attr_value = getattr(source_obj, attr_name)
            # Convert to int if it's a scalar, to list if it's an array
            if isinstance(attr_value, (int, float, bool)):
                config["env"][config_key] = (
                    int(attr_value)
                    if isinstance(attr_value, (int, bool))
                    else attr_value
                )
            else:
                config["env"][config_key] = attr_value.tolist()
        except AttributeError:
            print(
                f"Warning: Environment attribute '{attr_name}' not found in {source}. Skipping."
            )

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Log hparams
    hparam_dict = {
        k: v for k, v in config.items() if isinstance(v, (int, float, bool, str))
    }
    writer.add_hparams(hparam_dict, {}, run_name=".")
    writer.close()

    # Evaluation
    run_evaluation(config, run_dir)

    print(f"Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
