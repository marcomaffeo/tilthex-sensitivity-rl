#!/usr/bin/env python3
"""Training entry point for PPO-family algorithms.

Usage:
    # Defaults from configs/hydra/config.yaml
    python sensitivity_rl/train.py

    # Select the algorithm and the environment
    python sensitivity_rl/train.py ALGORITHM=sppo_aug_grad_param_dual env=tilthex_v1 SEED=14
    python sensitivity_rl/train.py ALGORITHM=ppo env=pendulum
"""

import os
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Restrict to only the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Environment setup before JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# XLA determinism mode:
#
# REPRODUCIBLE (slow): autotune_level=0, no kernel benchmarking.
#   Use for experiments where exact reproducibility across runs is required.
#
# FAST+REPRODUCIBLE (used here): autotune_level=4 with a per-fusion autotune cache.
# The first run benchmarks the kernels and writes the cache; later runs
# load the cached selections, so the same kernels and the same floating-point
# op order are used every time, with the speed of full autotuning.
#   Cache is stored at XLA_AUTOTUNE_CACHE_DIR (default /tmp/xla_autotune_cache).
#   To reset the cache (e.g. after a GPU/driver change), delete that directory.
#
# --xla_cpu_enable_fast_math and --xla_gpu_enable_fast_min_max allow FP
#   reassociation and can introduce tiny numerical differences on the CPU path
# and min/max ops respectively; turn them off for bit-exact results.
#
_XLA_AUTOTUNE_CACHE = os.environ.get("XLA_AUTOTUNE_CACHE_DIR", "/tmp/xla_autotune_cache")

# Fast + reproducible: full autotuning with persistent kernel-selection cache.
# The first run populates the cache; later runs reuse it.
os.environ["XLA_FLAGS"] = (
    f"--xla_gpu_autotune_level=4 "
    f"--xla_gpu_per_fusion_autotune_cache_dir={_XLA_AUTOTUNE_CACHE} "
    "--xla_gpu_force_compilation_parallelism=8 "
    "--xla_cpu_enable_fast_math=true "
    "--xla_gpu_enable_fast_min_max=true "
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_triton_gemm_any=true "
)

os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_PLATFORMS"] = "cuda,cpu"

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
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches",
#     "xla_gpu_per_fusion_autotune_cache_dir"
# )
# jax.clear_caches()

# Module-level mutable state for wall-clock timing inside jax.debug.callback.
# jax.debug.callback returns None, so timing cannot be threaded through JAX
# scans.  Instead we track it purely on the Python/host side.
_timing_state: dict = {"last_time": None}


def log_metrics(
    writer,
    losses,
    metric,
    targets,
    traj_batch_value,
    timestep,
    num_steps,
    num_envs,
    extra_losses=None,
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
                "returns/sensitivity_state_max",
                float(sensitivity_state_max),
                int(timestep),
            )
            writer.add_scalar(
                "returns/sensitivity_state_mean",
                float(sensitivity_state_mean),
                int(timestep),
            )

        # Constraint metrics (only present when constraint wrapper is active)
        if "constraint_violated" in metric:
            violation_rate = metric["constraint_violated"].mean()
            mean_violation = metric["constraint_violation"].mean()
            writer.add_scalar(
                "constraints/violation_rate", float(violation_rate), int(timestep)
            )
            writer.add_scalar(
                "constraints/mean_violation", float(mean_violation), int(timestep)
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
        returned = info["returned_episode"]
        if not returned.any():
            return
        return_values = info["returned_episode_returns"][returned]
        timesteps = info["timestep"][returned] * config["NUM_ENVS"]
        print(
            f"global step={timesteps[-1]}, mean episodic return={return_values.mean()}"
        )

    jax.debug.callback(callback, metric)


def _periodic_eval_log_callback(
    writer,
    run_dir,
    update_idx,
    timestep,
    should_eval,
    episodic_return,
    sensitivity_mean,
    sensitivity_max,
):
    """Host callback for scalar evaluation logging only."""
    if not bool(should_eval):
        return

    step = int(timestep)
    update_num = int(update_idx) + 1
    episodic_return_f = float(episodic_return)
    sensitivity_mean_f = float(sensitivity_mean)
    sensitivity_max_f = float(sensitivity_max)
    has_sensitivity = not math.isnan(sensitivity_mean_f)

    writer.add_scalar("eval/episodic_return", episodic_return_f, step)
    if has_sensitivity:
        writer.add_scalar("eval/sensitivity_state_mean", sensitivity_mean_f, step)
        writer.add_scalar("eval/sensitivity_state_max", sensitivity_max_f, step)

    record = {
        "trigger": f"during_training_update_{update_num}",
        "step": step,
        "episodic_return": episodic_return_f,
    }
    if has_sensitivity:
        record["sensitivity_state_mean"] = sensitivity_mean_f
        record["sensitivity_state_max"] = sensitivity_max_f

    eval_summary_path = run_dir / "evaluation_summary.json"
    all_records = []
    if eval_summary_path.exists():
        try:
            previous_records = json.loads(eval_summary_path.read_text())
            if isinstance(previous_records, list):
                all_records.extend(previous_records)
        except json.JSONDecodeError:
            pass
    all_records.append(record)
    eval_summary_path.write_text(json.dumps(all_records, indent=2))


def make_train(
    config,
    spec: AlgorithmSpec,
    writer: SummaryWriter,
    init_rng,
    run_dir: Path,
):
    """Create unified training function for PPO-family algorithms.

    Args:
        config: Training configuration dict
        spec: Algorithm specification
        writer: TensorBoard writer
        init_rng: Concrete PRNG key used for network initialisation

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

    override_kwargs = {}
    if "GAIN_STAB" in config:
        override_kwargs["gain_stab"] = float(config["GAIN_STAB"])
    if "GAIN_ATT" in config:
        override_kwargs["gain_att"] = float(config["GAIN_ATT"])
    if "GAIN_REF" in config:
        override_kwargs["gain_ref"] = float(config["GAIN_REF"])
    if "GAIN_SM" in config:
        override_kwargs["gain_sm"] = float(config["GAIN_SM"])
    if "REWARD_POS_TOLERANCE" in config:
        override_kwargs["reward_pos_tolerance"] = float(config["REWARD_POS_TOLERANCE"])

    if override_kwargs:
        env_params = env_params.replace(**override_kwargs)

    
    # Use the environment's true max_steps length for training (where it learns).
    # This decouples the episode length from NUM_STEPS (which only controls buffer unrolls).
    train_max_steps = config.get("TRAINING_MAX_STEPS_IN_EPISODE", env.default_params.max_steps_in_episode)
    
    env_params = env_params.replace(max_steps_in_episode=train_max_steps)
    env = spec.wrap_env(env, config)

    eval_config = config.copy()
    eval_config["DOMAIN_RANDOMIZATION"] = False
    eval_env = ENV_REGISTRY[eval_config["ENV_NAME"]]()
    eval_env_params = eval_env.default_params
    
    # Testing can use a larger horizon if EVAL_MAX_STEPS_IN_EPISODE is provided
    eval_max_steps = config.get("EVAL_MAX_STEPS_IN_EPISODE", train_max_steps)
    eval_env_params = eval_env_params.replace(max_steps_in_episode=eval_max_steps)
    eval_env = spec.wrap_env(eval_env, eval_config)

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
        eval_env.set_actor(graphdef, networks["train_state"].params)

    # Create algorithm-specific functions (closures over graphdef)
    loss_fn = spec.make_loss_fn(config, graphdef)
    collect_fn = spec.make_collect_fn(config, env, env_params, networks)
    update_fn = spec.make_update_fn(config, loss_fn, shuffle_batch, networks)
    eval_step_fn = spec.make_eval_step_fn(
        eval_config, graphdef, eval_env, eval_env_params
    )

    eval_during_training_enabled = bool(
        config.get("EVAL_DURING_TRAINING_ENABLED", False)
    )
    eval_every_updates = int(config.get("EVAL_EVERY_UPDATES", 0))

    total_steps = config["TOTAL_TIMESTEPS"]

    def train(rng):
        # INIT RUNNER STATE (algorithm handles state structure and RNG splitting)
        runner_state = spec.init_runner_state(config, env, env_params, networks, rng)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
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

            update_extras = {**collect_state.extras, **gae_extras}

            leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(
                collect_state.train_state.params
            )
            update_extras["old_actor_params"] = tuple(
                leaf
                for path, leaf in leaves_with_paths
                if any("actor" in str(path_entry) for path_entry in path)
            )

            update_state = UpdateState(
                train_state=collect_state.train_state,
                traj_batch=traj_batch,
                advantages=gae_results["advantages"],
                targets=gae_results["targets"],
                rng=collect_state.rng,
                extras=update_extras,
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

            if eval_during_training_enabled and eval_every_updates > 0:
                should_eval = ((update_idx + 1) % eval_every_updates) == 0

                # Create a fresh runner state for eval_env to circumvent domain randomization PyTree mismatch
                fresh_eval_rng, _ = jax.random.split(collect_state.rng)
                fresh_eval_runner_state = spec.init_runner_state(
                    eval_config, eval_env, eval_env_params, networks, fresh_eval_rng
                )
                eval_start_state = spec.build_eval_state(
                    update_state, fresh_eval_runner_state
                )

                episodic_return, sensitivity_mean, sensitivity_max = jax.lax.cond(
                    should_eval,
                    lambda s: _run_deterministic_eval_rollout(
                        eval_config, s, eval_step_fn
                    ),
                    lambda _s: (
                        jnp.array(jnp.nan, dtype=jnp.float32),
                        jnp.array(jnp.nan, dtype=jnp.float32),
                        jnp.array(jnp.nan, dtype=jnp.float32),
                    ),
                    eval_start_state,
                )

                jax.debug.callback(
                    lambda update_idx, timestep, should_eval, episodic_return, sensitivity_mean, sensitivity_max: _periodic_eval_log_callback(
                        writer,
                        run_dir,
                        update_idx,
                        timestep,
                        should_eval,
                        episodic_return,
                        sensitivity_mean,
                        sensitivity_max,
                    ),
                    update_idx,
                    current_timestep,
                    should_eval,
                    episodic_return,
                    sensitivity_mean,
                    sensitivity_max,
                )

            # Rebuild runner state with updated train_state(s)
            runner_extras = {
                k: v for k, v in update_state.extras.items() if k != "old_actor_params"
            }
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=collect_state.env_state,
                last_obs=collect_state.last_obs,
                rng=update_state.rng,
                extras={**collect_state.extras, **runner_extras},
            )
            return runner_state, current_timestep

        update_indices = jnp.arange(config["NUM_UPDATES"])
        runner_state, _ = jax.lax.scan(_update_step, runner_state, update_indices)
        return runner_state

    return train


def _run_deterministic_eval_rollout(config, eval_start_state, eval_step_fn):
    """Run deterministic rollout for NUM_STEPS and aggregate evaluation metrics."""
    _final_eval_state, info_batch = jax.lax.scan(
        eval_step_fn,
        eval_start_state,
        None,
        config["NUM_STEPS"],
    )

    returned_mask = info_batch["returned_episode"].astype(jnp.float32)
    returned_returns = info_batch["returned_episode_returns"]
    num_returned = returned_mask.sum()
    episodic_return = jnp.where(
        num_returned > 0,
        (returned_returns * returned_mask).sum() / num_returned,
        jnp.nan,
    )

    sensitivity_mean = jnp.nan
    sensitivity_max = jnp.nan
    if "sensitivity_state" in info_batch:
        sensitivity_state = info_batch["sensitivity_state"]
        sensitivity_mean = jnp.mean(sensitivity_state)
        sensitivity_max = jnp.mean(jnp.max(sensitivity_state, axis=0))

    return episodic_return, sensitivity_mean, sensitivity_max


def run_evaluation_loop(
    config,
    eval_step_fn,
    eval_start_state,
    run_dir=None,
    writer=None,
    step=None,
    trigger="post_training",
):
    """Run deterministic in-process evaluation loop and log return/sensitivity."""
    if not config.get("EVAL_ENABLED", True):
        return []

    episodic_return, sensitivity_mean, sensitivity_max = (
        _run_deterministic_eval_rollout(
            config,
            eval_start_state,
            eval_step_fn,
        )
    )

    episodic_return_f = float(episodic_return)
    sensitivity_mean_f = float(sensitivity_mean)
    sensitivity_max_f = float(sensitivity_max)
    log_step = int(step) if step is not None else 0
    has_sensitivity = not math.isnan(sensitivity_mean_f)

    if writer is not None:
        writer.add_scalar("eval/episodic_return", episodic_return_f, log_step)
        if has_sensitivity:
            writer.add_scalar(
                "eval/sensitivity_state_mean", sensitivity_mean_f, log_step
            )
            writer.add_scalar("eval/sensitivity_state_max", sensitivity_max_f, log_step)

    record = {
        "trigger": trigger,
        "step": log_step,
        "episodic_return": episodic_return_f,
    }
    if has_sensitivity:
        record["sensitivity_state_mean"] = sensitivity_mean_f
        record["sensitivity_state_max"] = sensitivity_max_f

    if run_dir is not None:
        eval_summary_path = run_dir / "evaluation_summary.json"
        all_records = []
        if eval_summary_path.exists():
            try:
                previous_records = json.loads(eval_summary_path.read_text())
                if isinstance(previous_records, list):
                    all_records.extend(previous_records)
            except json.JSONDecodeError:
                pass

        all_records.append(record)
        eval_summary_path.write_text(json.dumps(all_records, indent=2))

    print(
        "[Eval] "
        f"trigger={trigger} step={log_step} "
        f"return={episodic_return_f:.4f} "
        f"sensitivity_mean={sensitivity_mean_f:.4f} "
        f"sensitivity_max={sensitivity_max_f:.4f}"
    )
    return [record]


def run_evaluation(config, run_dir):
    """Run evaluation rollouts and plotting."""

    script_dir = os.path.dirname(__file__)

    print("Running evaluation rollouts (nominal)...")
    subprocess.run(
        [
            sys.executable,
            f"{script_dir}/rollouts.py",
            "--run_dirs",
            f"{run_dir}",
            "--plot",
        ]
    )

    print("Running evaluation rollouts (perturbed)...")
    subprocess.run(
        [
            sys.executable,
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
    train_fn = make_train(config, spec, writer, rng, run_dir)

    print("Compiling and training...")
    start_time = time.time()
    final_state = jax.jit(train_fn)(rng)
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

    # Save config (convert any remaining JAX arrays to Python types)
    config.pop("SENS_W", None)

    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return obj

    with open(run_dir / "config.json", "w") as f:
        json.dump(_to_serializable(config), f, indent=2)

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
