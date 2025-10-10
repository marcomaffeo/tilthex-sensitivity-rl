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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from envs import Pendulum, DroneV0
from envs.wrappers import GymnaxWrapper, ClipAction, SensitivityWrapper
from jax.flatten_util import ravel_pytree

ENV_REGISTRY = {
    "pendulum": Pendulum,
    "drone_v0": DroneV0,
}


@dataclass
class RolloutConfig:
    num_rollouts: int
    max_steps: int
    deterministic: bool
    seed: int


class ActorCriticEval(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor = nn.Dense(
            self.actor_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense1",
        )(x)
        actor = activation_fn(actor)
        actor = nn.Dense(
            self.actor_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2",
        )(actor)
        actor = activation_fn(actor)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_output",
        )(actor)
        actor_log_std = self.param(
            "actor_log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

        critic = nn.Dense(
            self.critic_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1",
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            self.critic_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2",
        )(critic)
        critic = activation_fn(critic)
        critic_value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_output",
        )(critic)

        return pi, jnp.squeeze(critic_value, axis=-1)

def _load_run_artifacts(run_dir: Path) -> Tuple[Dict[str, Any], bytes]:
    cfg_path = run_dir / "config.json"
    params_path = run_dir / "params.msgpack"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.msgpack in {run_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    params_bytes = params_path.read_bytes()
    return config, params_bytes


def _build_env_from_config(
    config: Dict[str, Any],
) -> Tuple[GymnaxWrapper, Any, str]:
    env_name = config.get("ENV_NAME", "pendulum").lower()
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {sorted(ENV_REGISTRY)}"
        )

    env_cls = ENV_REGISTRY[env_name]
    env = env_cls()
    env_params = env.default_params
    env_params.replace(is_testing=True)

    wrapped_env = GymnaxWrapper(env)
    wrapped_env = ClipAction(wrapped_env)

    return wrapped_env, env_params, env_name


def _unwrap_base_env_state(env_state: Any) -> Any:
    while hasattr(env_state, "env_state"):
        env_state = env_state.env_state
    return env_state


def _flatten_value(x):
    """Flatten a pytree or array/scalar into a 1D JAX array for logging."""
    try:
        flat, _ = ravel_pytree(x)
    except Exception:
        flat = jnp.ravel(jnp.asarray(x))
    return flat

@partial(jax.jit, static_argnames=("env", "actor_critic", "max_steps", "deterministic"))
def rollout_episode(
    env,
    env_params,
    actor_critic,
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

    def body_fun(i, carry):
        rng, obs, env_state, actions_buf, logs_buf = carry

        pi, _ = actor_critic.apply(params, obs)
        if deterministic:
            action = pi.mean()
        else:
            rng, act_key = jax.random.split(rng)
            action = pi.sample(seed=act_key)

        rng, step_key = jax.random.split(rng)
        obs, env_state, reward, _, _ = env.step(step_key, env_state, action, env_params)

        # Record action after environment step (matches previous semantics)
        actions_buf = actions_buf.at[i].set(env_state.last_u)

        # Record selected state fields at index i+1
        for key in env_state.__dataclass_fields__.keys():
            val = getattr(env_state, key)
            flat = _flatten_value(val)
            logs_buf[key] = logs_buf[key].at[i + 1].set(flat)

        return (rng, obs, env_state, actions_buf, logs_buf)

    rng, obs, env_state, actions_buf, logs_buf = jax.lax.fori_loop(
        0, horizon, body_fun, (rng, obs, env_state, actions_buf, logs_buf)
    )

    return actions_buf, logs_buf



def save_trajectory(
    path: Path, actions: np.ndarray, state_filtered: Dict[str, np.ndarray]
) -> None:
    to_save = {"actions": actions}
    to_save.update(state_filtered)
    path.write_bytes(serialization.msgpack_serialize(to_save))


def _init_model_and_params(
    config: Dict[str, Any], env, env_params, params_bytes: bytes
):
    params = serialization.msgpack_restore(params_bytes)

    actor_hidden = 256
    critic_hidden = 256
    try:
        actor_hidden = int(params["params"]["actor_dense1"]["kernel"].shape[1])
    except (KeyError, AttributeError, TypeError, IndexError):
        actor_hidden = 256
    try:
        critic_hidden = int(
            params["params"]["critic_dense1"]["kernel"].shape[1]
        )
    except (KeyError, AttributeError, TypeError, IndexError):
        critic_hidden = 256

    action_dim = env.action_space(env_params).shape[0]
    model = ActorCriticEval(
        action_dim=action_dim,
        activation=config.get("ACTIVATION", "tanh"),
        actor_hidden_dim=actor_hidden,
        critic_hidden_dim=critic_hidden,
    )
    return model, params


def run_rollouts_for_run(run_dir: Path, rcfg: RolloutConfig) -> None:
    config, params_bytes = _load_run_artifacts(run_dir)
    env, env_params, env_name = _build_env_from_config(config)
    actor_critic, params = _init_model_and_params(
        config, env, env_params, params_bytes
    )


    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(rcfg.seed)

    for idx in range(rcfg.num_rollouts):
        rng, episode_key = jax.random.split(rng)
        actions, state_flat = rollout_episode(
            env=env,
            env_params=env_params,
            actor_critic=actor_critic,
            params=params,
            rng_key=episode_key,
            max_steps=rcfg.max_steps,
            deterministic=rcfg.deterministic,
        )

        traj_id = idx + 1
        traj_file = rollouts_dir / f"traj_{traj_id:05d}.msgpack"

        save_trajectory(traj_file, actions, state_flat)

        print(
            f"[rollouts] Saved trajectory {traj_id}/{rcfg.num_rollouts} -> {traj_file}"
        )

    print(
        f"[rollouts] Completed {rcfg.num_rollouts} trajectories for {run_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequential rollouts from learned policies."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=["runs/latest"],
        help="One or more run directories containing params.msgpack and config.json.",
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
        default=0,
        help="Base PRNG seed for rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rcfg = RolloutConfig(
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        if not run_dir.exists():
            print(f"[rollouts] Skipping missing run directory: {run_dir}")
            continue
        try:
            run_rollouts_for_run(run_dir, rcfg)
        except Exception as exc:  # pragma: no cover - guardrail for CLI use
            print(f"[rollouts] Error processing {run_dir}: {exc}")


if __name__ == "__main__":
    main()
