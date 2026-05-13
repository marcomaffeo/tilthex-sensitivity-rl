#!/usr/bin/env python3
"""Sequential rollout script for trained PPO policies.

- Reconstruct the evaluation environment from the saved config.
- Load the trained ActorCritic parameters.
- Run rollouts.
- Save the environment state trajectory and the action sequence for
    each episode under the respective run directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization
from envs import ENV_REGISTRY
from envs import Pendulum, DroneV0, DroneV1, TilthexV0, TilthexV1
from envs.wrappers import GymnaxWrapper, ClipAction
from jax.flatten_util import ravel_pytree

from networks import ActorCritic, ActorCriticDual, ActorCriticGRU, ActorDCritic
from utils import sample_env_params

import numpy as np
from pathlib import Path

@dataclass
class RolloutConfig:
    num_rollouts: int
    max_steps: int
    deterministic: bool
    seed: int
    param_samples: int
    plot: bool


def load_run_artifacts(run_dir: Path) -> Tuple[Dict[str, Any], bytes]:
    cfg_path = run_dir / "config.json"
    params_path = run_dir / "policy_params.msgpack"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing policy_params.msgpack in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    params_bytes = params_path.read_bytes()
    return config, params_bytes


def build_env_from_config(
    config: Dict[str, Any],
) -> Tuple[GymnaxWrapper, Any, str, type]:
    env_name = config.get("ENV_NAME", "drone_v0").lower()
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {sorted(ENV_REGISTRY)}"
        )

    env_cls = ENV_REGISTRY[env_name]
    env = env_cls()
    env_params = env.default_params
    # Apply EVAL_MAX_STEPS_IN_EPISODE from config if explicitly set (allows using a
    # longer episode for testing/rollouts than the NUM_STEPS used during training).
    if "EVAL_MAX_STEPS_IN_EPISODE" in config:
        env_params = env_params.replace(
            max_steps_in_episode=int(config["EVAL_MAX_STEPS_IN_EPISODE"])
        )
    env_params = env_params.replace(is_testing=True)

    wrapped_env = GymnaxWrapper(env)
    wrapped_env = ClipAction(wrapped_env)

    return wrapped_env, env_params, env_name, env_cls


def _flatten_value(x):
    """Flatten a pytree or array/scalar into a 1D JAX array for logging."""
    try:
        flat, _ = ravel_pytree(x)
    except Exception:
        flat = jnp.ravel(jnp.asarray(x))
    return flat


# NOTE: Treating env and graphdef as static keeps the function JIT-able while
# allowing vmap to supply different rng/env_params externally.
@partial(
    jax.jit,
    static_argnames=("env", "graphdef", "max_steps", "deterministic"),
)
def rollout_episode(
    env,
    env_params,
    graphdef,
    params,
    rng_key: jax.Array,
    max_steps: int,
    deterministic: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    rng_reset, rng = jax.random.split(rng_key)
    obs, env_state = env.reset(rng_reset, env_params)

    # Determine horizon
    horizon = (
        max_steps
        if max_steps > 0
        else int(getattr(env_params, "max_steps_in_episode", 200))
    )
    # Preallocate action buffer
    act_shape = env.action_space(env_params).shape
    actions_buf = jnp.zeros((horizon,) + act_shape, dtype=jnp.float32)

    # Preallocate log buffers for requested keys and write initial state at index 0
    logs_buf: Dict[str, jnp.ndarray] = {}
    for key in env_state.__dataclass_fields__.keys():
        val0 = getattr(env_state, key)
        flat0 = _flatten_value(val0)
        logs_buf[key] = jnp.zeros((horizon + 1, flat0.shape[0]), dtype=flat0.dtype)
        logs_buf[key] = logs_buf[key].at[0].set(flat0)

    # Preallocate eval_metrics buffers if the environment supports it
    eval_metrics_buf: Dict[str, jnp.ndarray] = {}
    if hasattr(env, "eval_metrics"):
        initial_metrics = env.eval_metrics(env_state, env_params)
        for key, val0 in initial_metrics.items():
            flat0 = _flatten_value(val0)
            eval_metrics_buf[key] = jnp.zeros(
                (horizon + 1, flat0.shape[0]), dtype=flat0.dtype
            )
            eval_metrics_buf[key] = eval_metrics_buf[key].at[0].set(flat0)

    def body_fun(i, carry):
        rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf = carry

        model = nnx.merge(graphdef, params)
        pi = model.actor(obs)
        if deterministic:
            action = pi.mean()
        else:
            rng, act_key = jax.random.split(rng)
            action = pi.sample(seed=act_key)

        rng, step_key = jax.random.split(rng)
        obs, env_state, reward, _, _ = env.step(step_key, env_state, action, env_params)

        # Record action after environment step (matches previous semantics)
        actions_buf = actions_buf.at[i].set(getattr(env_state, "last_commanded_action", getattr(env_state, "last_u", action)))

        # Record selected state fields at index i+1
        for key in env_state.__dataclass_fields__.keys():
            val = getattr(env_state, key)
            flat = _flatten_value(val)
            logs_buf[key] = logs_buf[key].at[i + 1].set(flat)

        # Record eval_metrics at index i+1
        if hasattr(env, "eval_metrics"):
            metrics = env.eval_metrics(env_state, env_params)
            for key, val in metrics.items():
                flat = _flatten_value(val)
                eval_metrics_buf[key] = eval_metrics_buf[key].at[i + 1].set(flat)

        return (rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf)

    rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf = jax.lax.fori_loop(
        0,
        horizon,
        body_fun,
        (rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf),
    )

    return actions_buf, logs_buf, eval_metrics_buf


@partial(
    jax.jit,
    static_argnames=("env", "graphdef", "max_steps", "deterministic"),
)
def rollout_episode_from_state(
    env,
    env_params,
    graphdef,
    params,
    initial_obs: jax.Array,
    initial_state,
    rng_key: jax.Array,
    max_steps: int,
    deterministic: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Like rollout_episode but starts from an explicitly provided initial state,
    so multiple env_params samples can share the same starting position."""

    rng = rng_key
    obs = initial_obs
    env_state = initial_state

    # Determine horizon
    horizon = (
        max_steps
        if max_steps > 0
        else int(getattr(env_params, "max_steps_in_episode", 200))
    )
    # Preallocate action buffer
    act_shape = env.action_space(env_params).shape
    actions_buf = jnp.zeros((horizon,) + act_shape, dtype=jnp.float32)

    # Preallocate log buffers for requested keys and write initial state at index 0
    logs_buf: Dict[str, jnp.ndarray] = {}
    for key in env_state.__dataclass_fields__.keys():
        val0 = getattr(env_state, key)
        flat0 = _flatten_value(val0)
        logs_buf[key] = jnp.zeros((horizon + 1, flat0.shape[0]), dtype=flat0.dtype)
        logs_buf[key] = logs_buf[key].at[0].set(flat0)

    # Preallocate eval_metrics buffers if the environment supports it
    eval_metrics_buf: Dict[str, jnp.ndarray] = {}
    if hasattr(env, "eval_metrics"):
        initial_metrics = env.eval_metrics(env_state, env_params)
        for key, val0 in initial_metrics.items():
            flat0 = _flatten_value(val0)
            eval_metrics_buf[key] = jnp.zeros(
                (horizon + 1, flat0.shape[0]), dtype=flat0.dtype
            )
            eval_metrics_buf[key] = eval_metrics_buf[key].at[0].set(flat0)

    def body_fun(i, carry):
        rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf = carry

        model = nnx.merge(graphdef, params)
        pi = model.actor(obs)
        if deterministic:
            action = pi.mean()
        else:
            rng, act_key = jax.random.split(rng)
            action = pi.sample(seed=act_key)

        rng, step_key = jax.random.split(rng)
        obs, env_state, reward, _, _ = env.step(step_key, env_state, action, env_params)

        actions_buf = actions_buf.at[i].set(getattr(env_state, "last_commanded_action", getattr(env_state, "last_u", action)))

        for key in env_state.__dataclass_fields__.keys():
            val = getattr(env_state, key)
            flat = _flatten_value(val)
            logs_buf[key] = logs_buf[key].at[i + 1].set(flat)

        if hasattr(env, "eval_metrics"):
            metrics = env.eval_metrics(env_state, env_params)
            for key, val in metrics.items():
                flat = _flatten_value(val)
                eval_metrics_buf[key] = eval_metrics_buf[key].at[i + 1].set(flat)

        return (rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf)

    rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf = jax.lax.fori_loop(
        0,
        horizon,
        body_fun,
        (rng, obs, env_state, actions_buf, logs_buf, eval_metrics_buf),
    )

    return actions_buf, logs_buf, eval_metrics_buf


def save_trajectory(
    path: Path,
    actions: np.ndarray,
    state_filtered: Dict[str, np.ndarray],
    eval_metrics: Dict[str, np.ndarray] = None,
    env_params=None,
) -> None:
    to_save = {"actions": actions}
    to_save.update(state_filtered)
    if eval_metrics:
        to_save["eval_metrics"] = eval_metrics

    path.write_bytes(serialization.msgpack_serialize(to_save))


def extract_actor_params(pure_dict: Dict) -> Dict:
    """Extract actor parameters from a full model's NNX pure dict.

    Kept for reference / debug but not used in the default NNX flow
    which loads the full model directly.
    """
    actor_dict = {}
    for key, value in pure_dict.items():
        if key.startswith("actor_"):
            actor_dict[key[len("actor_") :]] = value
    return actor_dict


def init_model_and_params(config: Dict[str, Any], env, env_params, params_bytes: bytes):
    actor_layer_sizes = tuple(config["ACTOR_LAYER_SIZES"])
    critic_layer_sizes = tuple(
        config.get("CRITIC_LAYER_SIZES", config["ACTOR_LAYER_SIZES"])
    )
    action_dim = env.action_space(env_params).shape[0]
    obs_dim = env.observation_space(env_params).shape[0]
    activation = config.get("ACTIVATION", "tanh")
    rngs = nnx.Rngs(0)  # dummy rngs; weights are overwritten below

    algorithm = config.get("ALGORITHM", "ppo").lower()
    if algorithm in ("rappo",):
        model = ActorDCritic(
            obs_dim,
            action_dim,
            activation=activation,
            actor_layer_sizes=actor_layer_sizes,
            rngs=rngs,
        )
    elif algorithm in ("sppo", "sppo_grad"):
        gru_hidden_size = config.get("HIDDEN_SIZE", 64)
        model = ActorCriticGRU(
            obs_dim,
            action_dim,
            activation=activation,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            gru_hidden_size=gru_hidden_size,
            rngs=rngs,
        )
    elif algorithm in ("sppo_aug",):
        model = ActorCritic(
            obs_dim,
            action_dim,
            activation=activation,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            rngs=rngs,
        )
    else:
        model = ActorCritic(
            obs_dim,
            action_dim,
            activation=activation,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            rngs=rngs,
        )

    # Extract the pure dict from the dummy model to use as a strict typing template
    _, template_state = nnx.split(model)
    template_pure_dict = template_state.to_pure_dict()

    # Let Flax parse the bytes against the template, coercing string keys back to integers
    restored_pure_dict = serialization.from_bytes(template_pure_dict, params_bytes)

    # Safely update the model with the strictly typed dictionary
    nnx.update(model, restored_pure_dict)

    # Split the updated model to get the final restored state for JAX
    graphdef, restored_state = nnx.split(model)

    return graphdef, restored_state


def run_rollouts_for_run(run_dir: Path, rcfg: RolloutConfig) -> None:

    config, params_bytes = load_run_artifacts(run_dir)
    env, env_params, env_name, env_cls = build_env_from_config(config)

    # If an explicit --max_steps was given on the CLI, it overrides both the
    # rollout horizon AND the env terminal condition so the episode runs for the
    # full requested length without auto-resetting early.
    if rcfg.max_steps > 0:
        env_params = env_params.replace(max_steps_in_episode=rcfg.max_steps)

    graphdef, params = init_model_and_params(config, env, env_params, params_bytes)

    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(rcfg.seed)
    param_sample_count = max(1, rcfg.param_samples)

    # When using multiple param samples, all perturbed rollouts share the same
    # initial state as the nominal rollout (reset once with nominal params).
    # This isolates the effect of parameter perturbations from initial-condition
    # variability. For a single param set we keep the original per-episode reset.
    use_shared_reset = param_sample_count > 1

    if use_shared_reset:
        batched_rollout = jax.vmap(
            rollout_episode_from_state,
            # env, env_params, graphdef, params, initial_obs, initial_state, rng_key, max_steps, deterministic
            in_axes=(None, 0, None, None, None, None, None, None, None),
            out_axes=(0, 0, 0),
        )
    else:
        batched_rollout = jax.vmap(
            rollout_episode,
            in_axes=(None, 0, None, None, 0, None, None),
            out_axes=(0, 0, 0),
        )

    print(
        f"[rollouts] Using {param_sample_count} environment parameter set(s)"
        f" for {run_dir.name}"
    )

    all_series_nominal = []
    all_series_perturbed = []

    for idx in range(rcfg.num_rollouts):
        rng, episode_key = jax.random.split(rng)

        # Resample perturbed parameters independently for each episode so that
        # parameter variability is not confounded with initial-state variability.
        if param_sample_count > 1:
            rng, sample_key = jax.random.split(rng)
            lows, highs = env.params_min, env.params_max
            sampled_params = sample_env_params(
                env=env,
                base_params=env_params,
                lows=lows,
                highs=highs,
                num_samples=param_sample_count - 1,
                rng_key=sample_key,
            )
            env_param_sets = [env_params] + sampled_params
        else:
            env_param_sets = [env_params]

        env_params_batched = jax.tree_util.tree_map(
            lambda *vals: jnp.stack([jnp.asarray(v) for v in vals], axis=0),
            *env_param_sets,
        )
        nominal_env_params = env_param_sets[0]

        if use_shared_reset:
            # Reset with nominal params to get the shared initial state; all
            # perturbed param samples will start from the same position.
            rng_reset, rng_step = jax.random.split(episode_key)
            nominal_obs, nominal_state = env.reset(rng_reset, nominal_env_params)
            actions_batch, state_batch, eval_metrics_batch = batched_rollout(
                env,
                env_params_batched,
                graphdef,
                params,
                nominal_obs,
                nominal_state,
                rng_step,
                rcfg.max_steps,
                rcfg.deterministic,
            )
        else:
            episode_keys = jax.random.split(episode_key, param_sample_count)
            actions_batch, state_batch, eval_metrics_batch = batched_rollout(
                env,
                env_params_batched,
                graphdef,
                params,
                episode_keys,
                rcfg.max_steps,
                rcfg.deterministic,
            )

        actions_batch = np.asarray(actions_batch)
        state_batch_np = {key: np.asarray(val) for key, val in state_batch.items()}
        eval_metrics_batch_np = {
            key: np.asarray(val) for key, val in eval_metrics_batch.items()
        }

        env_params_samples = [
            jax.tree_util.tree_map(lambda x: np.asarray(x[i]), env_params_batched)
            for i in range(param_sample_count)
        ]

        traj_id = idx + 1

        for sample_idx in range(param_sample_count):
            if param_sample_count == 1:
                traj_file = rollouts_dir / f"traj_{traj_id:05d}.msgpack"
            else:
                traj_file = (
                    rollouts_dir / f"traj_{traj_id:05d}_sample_{sample_idx:02d}.msgpack"
                )

            actions = actions_batch[sample_idx]
            state_flat = {
                key: value[sample_idx] for key, value in state_batch_np.items()
            }
            eval_metrics_flat = {
                key: value[sample_idx] for key, value in eval_metrics_batch_np.items()
            }
            save_trajectory(
                traj_file,
                actions,
                state_flat,
                eval_metrics_flat,
                env_params_samples[sample_idx],
            )

            sample_params = env_params_samples[sample_idx]
            default_dt = getattr(env.default_params, "dt", 1.0)
            if isinstance(sample_params, dict):
                dt_val = sample_params.get("dt", default_dt)
            else:
                dt_val = getattr(sample_params, "dt", default_dt)
            try:
                dt_val = float(np.asarray(dt_val))
            except Exception:
                dt_val = float(default_dt)
            
            series_entry = {"actions": actions}
            series_entry.update(state_flat)
            series_entry["eval_metrics"] = eval_metrics_flat
            series_entry["dt"] = dt_val
            
            # Separa i nominali dai perturbati
            if sample_idx == 0:
                all_series_nominal.append(series_entry)
            else:
                all_series_perturbed.append(series_entry)

            suffix = " (nominal)" if sample_idx == 0 else f" (sample {sample_idx})"
            print(
                f"[rollouts] Saved trajectory {traj_id}/{rcfg.num_rollouts}{suffix}"
                f" -> {traj_file}"
            )

    print(f"[rollouts] Completed {rcfg.num_rollouts} trajectories for {run_dir}")

    def save_series_to_npz(series_list, filename):
        if not series_list: return
        savez_path = run_dir / filename
        
        save_dict = {
            "state": np.stack([s["state"] for s in series_list]),
            "actions": np.stack([s["actions"] for s in series_list])
        }
        if "target_pos" in series_list[0]:
            save_dict["target_pos"] = np.stack([s["target_pos"] for s in series_list])
        if "target_attitude" in series_list[0]:
            save_dict["target_attitude"] = np.stack([s["target_attitude"] for s in series_list])
            
        np.savez(savez_path, **save_dict)
        print(f"[rollouts] Dati salvati in -> {savez_path}")

    save_series_to_npz(all_series_nominal, "rollout_data_nominal.npz")
    
    if all_series_perturbed:
        save_series_to_npz(all_series_nominal + all_series_perturbed, "rollout_data_perturbed.npz")

    if rcfg.plot:
        if all_series_nominal:
            print(f"[rollouts] Generating plot -> {rollouts_dir.parent / 'rollouts_nominal.png'}")
            try:
                env_cls.plot_rollouts(all_series_nominal, rollouts_dir.parent / "rollouts_nominal.png")
            except Exception as exc:
                print(f"[rollouts] Plotting nominal failed: {exc}")
                
        if all_series_perturbed:
            print(f"[rollouts] Generating plot -> {rollouts_dir.parent / 'rollouts_perturbed.png'}")
            try:
                # Plotta tutto insieme (nominal + perturbato) per far vedere la dispersione
                env_cls.plot_rollouts(all_series_nominal + all_series_perturbed, rollouts_dir.parent / "rollouts_perturbed.png")
            except Exception as exc:
                print(f"[rollouts] Plotting perturbed failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequential rollouts from learned policies."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=["runs/latest"],
        help="One or more run directories containing policy_params.msgpack and config.json.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollouts to generate per run directory.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum steps per episode (0 uses the environment default).",
    )
    parser.add_argument(
        "--deterministic",
        type=bool,
        default=True,
        help="Use policy mean action instead of sampling from the distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base PRNG seed for rollouts.",
    )
    parser.add_argument(
        "--param_samples",
        type=int,
        default=1,
        help=(
            "Number of environment parameter samples per rollout (>=1). "
            "Value 1 uses only the nominal parameters."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots of the rollouts after saving trajectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rcfg = RolloutConfig(
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
        param_samples=max(1, args.param_samples),
        plot=args.plot,
    )

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        if not run_dir.exists():
            print(f"[rollouts] Skipping missing run directory: {run_dir}")
            continue
        try:
            # Clear JAX caches to avoid pytree metadata conflicts between runs
            jax.clear_caches()
            run_rollouts_for_run(run_dir, rcfg)
        except Exception as exc:  # pragma: no cover - guardrail for CLI use
            print(f"[rollouts] Error processing {run_dir}: {exc}")


if __name__ == "__main__":
    main()