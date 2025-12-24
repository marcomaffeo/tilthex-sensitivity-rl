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
from pathlib import Path
from datetime import datetime

# Environment setup before JAX import
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import jax
import jax.numpy as jnp
import optax
import hydra
from flax import serialization
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

from algorithms import ALGORITHM_REGISTRY, AlgorithmSpec
from envs import ENV_REGISTRY
from utils import (
    load_actor_params_from_file,
    merge_actor_params_with_fresh_critic,
    update_latest_symlink,
    init_network_params,
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



def log_metrics(writer, losses, metric, targets, traj_batch_value, timestep, extra_losses=None):
    """Create logging callback for training metrics."""
    targets_var = jnp.var(targets)
    explained_var = 1 - jnp.var(targets - traj_batch_value) / (targets_var + 1e-8)

    def log_callback(losses, metric, explained_var, timestep, extra_losses):
        episodic_return = metric["returned_episode_returns"][metric["returned_episode"]].mean()
        episodic_return_true = metric["returned_episode_returns_true"][metric["returned_episode"]].mean()
        episodic_length = metric["returned_episode_lengths"][metric["returned_episode"]].mean()
        sensitivity_state_max = metric["sensitivity_state"].max(axis=0).mean()
        sensitivity_state_mean = metric["sensitivity_state"].mean()

        writer.add_scalar("train/total_loss", float(losses["total"]), int(timestep))
        writer.add_scalar("train/value_loss", float(losses["value"]), int(timestep))
        writer.add_scalar("train/policy_loss", float(losses["actor"]), int(timestep))
        writer.add_scalar("train/entropy", float(losses["entropy"]), int(timestep))
        writer.add_scalar("returns/episodic_return", float(episodic_return), int(timestep))
        writer.add_scalar("returns/episodic_return_true", float(episodic_return_true), int(timestep))
        writer.add_scalar("returns/episodic_length", float(episodic_length), int(timestep))
        writer.add_scalar("returns/sensitivity_state_max", float(sensitivity_state_max), int(timestep))
        writer.add_scalar("returns/sensitivity_state_mean", float(sensitivity_state_mean), int(timestep))
        writer.add_scalar("train/old_approx_kl", float(losses["old_approx_kl"]), int(timestep))
        writer.add_scalar("train/approx_kl", float(losses["approx_kl"]), int(timestep))
        writer.add_scalar("train/clipfrac", float(losses["clipfrac"]), int(timestep))
        writer.add_scalar("train/explained_variance", float(explained_var), int(timestep))
        
        if extra_losses:
            for name, val in extra_losses.items():
                writer.add_scalar(f"train/{name}", float(val), int(timestep))

    jax.debug.callback(log_callback, losses, metric, explained_var, timestep, extra_losses)


def debug_callback(config, metric):
    """Debug callback for printing training progress."""
    def callback(info):
        return_values = info["returned_episode_returns"][info["returned_episode"]]
        timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
        print(f"global step={timesteps[-1]}, mean episodic return={return_values.mean()}")
    jax.debug.callback(callback, metric)


def make_train(config, spec: AlgorithmSpec, writer: SummaryWriter):
    """Create unified training function for PPO-family algorithms.
    
    Args:
        config: Training configuration dict
        spec: Algorithm specification
        writer: TensorBoard writer
        
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
    
    def train(rng):
        # INIT NETWORKS (algorithm handles all network creation)
        rng, init_rng = jax.random.split(rng)
        networks = spec.init_networks(
            config, env, env_params, init_rng,
            load_params_fn=init_network_params,
        )
        network = networks["network"]
        
        if config.get("SENSITIVITY", False):
            env.set_actor(network, networks["train_state"].params)

        # Create algorithm-specific functions (closures over network)
        loss_fn = spec.make_loss_fn(config, network)
        collect_fn = spec.make_collect_fn(config, env, env_params, networks)
        update_fn = spec.make_update_fn(config, loss_fn, shuffle_batch, networks)

        # INIT RUNNER STATE (algorithm handles state structure and RNG splitting)
        runner_state = spec.init_runner_state(config, env, env_params, networks, rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            if config.get("SENSITIVITY", False):
                env.update_network_params(runner_state.train_state.params)

            # COLLECT TRAJECTORIES
            collect_state, traj_batch = jax.lax.scan(
                collect_fn, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE GAE
            gae_results = spec.calculate_gae(
                config, traj_batch, networks, collect_state,
            )

            # UPDATE NETWORK
            update_state = spec.pack_update_state(collect_state, traj_batch, gae_results)
            update_state, loss_info = jax.lax.scan(
                update_fn, update_state, None, config["UPDATE_EPOCHS"]
            )

            # Extract losses
            losses, extra_losses = spec.extract_losses(loss_info)

            metric = traj_batch.info
            if config.get("DEBUG", False):
                debug_callback(config, metric)

            # Logging
            current_timestep = metric["timestep"].mean() * config["NUM_ENVS"]
            log_metrics(writer, losses, metric, gae_results["targets"], traj_batch.value, current_timestep, extra_losses)

            # Rebuild runner state with updated train_state(s)
            runner_state = spec.rebuild_runner_state(update_state, collect_state)
            return runner_state, metric

        # Run training loop
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return runner_state

    return train


def run_evaluation(config, run_dir):
    """Run evaluation rollouts and plotting."""
    eval_config = {
        "NUM_ROLLOUTS": 10,
        "MAX_STEPS": 200,
        "DETERMINISTIC": True,
        "SEED": 42,
        "PARAM_SAMPLES": 1,
    }

    script_dir = os.path.dirname(__file__)

    print("Running evaluation rollouts (nominal)...")
    subprocess.run([
        "python3", f"{script_dir}/rollouts.py",
        "--run_dirs", f"{run_dir}",
        "--max_step", f"{eval_config['MAX_STEPS']}",
        "--num_rollouts", f"{eval_config['NUM_ROLLOUTS']}",
        "--param_samples", f"{eval_config['PARAM_SAMPLES']}",
        "--plot",
    ])

    print("Running evaluation rollouts (perturbed)...")
    eval_config["PARAM_SAMPLES"] = 10
    subprocess.run([
        "python3", f"{script_dir}/rollouts.py",
        "--run_dirs", f"{run_dir}",
        "--max_step", f"{eval_config['MAX_STEPS']}",
        "--num_rollouts", f"{eval_config['NUM_ROLLOUTS']}",
        "--param_samples", f"{eval_config['PARAM_SAMPLES']}",
        "--plot",
    ])


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
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(ALGORITHM_REGISTRY.keys())}")
    
    spec = ALGORITHM_REGISTRY[algorithm]
    
    # Ensure SENSITIVITY is set for algorithms that need it
    if spec.uses_sensitivity and "SENSITIVITY" not in config:
        config["SENSITIVITY"] = True
    
    rng = jax.random.PRNGKey(config.get("SEED", 30))
    
    # Create run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_name = algorithm.upper()
    run_dir = runs_root / config["ENV_NAME"] / f"{algo_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {run_dir}")
    
    update_latest_symlink(run_dir, runs_root)
    
    # Setup tensorboard
    writer = SummaryWriter(f"{run_dir}")
    
    # Create unified training function
    train_fn = make_train(config, spec, writer)
    
    # Train
    train_jit = jax.jit(train_fn)
    final_state = train_jit(rng)

    # Save params    
    params = final_state.train_state.params
    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    
    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Log hparams
    hparam_dict = {k: v for k, v in config.items() if isinstance(v, (int, float, bool, str))}
    writer.add_hparams(hparam_dict, {}, run_name=".")
    writer.close()
    
    # Evaluation
    run_evaluation(config, run_dir)
    
    print(f"Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
