"""Base classes and shared utilities for PPO-family algorithms."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from buffers import RunnerState


# ---------------------------------------------------------------------------
# Default implementations for common algorithm patterns.
# Algorithms only override the ones where they differ.
# ---------------------------------------------------------------------------


def default_extract_losses(loss_info):
    """Turn the loss information of a PPO-style update into scalar metrics.

    Args:
        loss_info (tuple): ``(total_loss, (value_loss, actor_loss, entropy, old_approx_kl, approx_kl, clipfrac))``.

    Returns:
        tuple[dict, None]: Mean of each metric, and no algorithm-specific metrics.
    """
    total_losses, (
        value_losses,
        actor_losses,
        entropies,
        old_approx_kl,
        approx_kl,
        clipfracs,
    ) = loss_info
    losses = {
        "total": total_losses.mean(),
        "value": value_losses.mean(),
        "actor": actor_losses.mean(),
        "entropy": entropies.mean(),
        "old_approx_kl": old_approx_kl.mean(),
        "approx_kl": approx_kl.mean(),
        "clipfrac": jnp.concatenate(clipfracs).mean(),
    }
    return losses, None


def default_init_runner_state(config, env, env_params, networks, rng):
    """Reset the environments and build the initial runner state, without extras.

    Args:
        config (dict): Training configuration (``NUM_ENVS``).
        env (GymnaxWrapper): Wrapped environment.
        env_params (EnvParams): Environment parameters.
        networks (dict): Output of ``init_networks``.
        rng (jax.Array): PRNG key.

    Returns:
        RunnerState: Initial runner state.
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)
    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,
    )


def default_run_dir(algo_name, config, runs_root, ts):
    """Run directory ``<runs_root>/<ENV_NAME>/<ALGO>_[DR_]<timestamp>``.

    Args:
        algo_name (str): Short algorithm label, e.g. ``"PPO"``.
        config (dict): Training configuration (``ENV_NAME``, ``DOMAIN_RANDOMIZATION``).
        runs_root (pathlib.Path): Root folder of the runs.
        ts (str): Timestamp.

    Returns:
        pathlib.Path: Run directory.
    """
    suffix = (
        f"{algo_name}_DR_{ts}"
        if config.get("DOMAIN_RANDOMIZATION", False)
        else f"{algo_name}_{ts}"
    )
    return runs_root / config["ENV_NAME"] / suffix


def default_build_eval_state(update_state, collect_state):
    """Build the carry of the deterministic evaluation loop for non-recurrent policies.

    Args:
        update_state (UpdateState): Current update state.
        collect_state (RunnerState): Current runner state.

    Returns:
        tuple: ``(params, env_state, last_obs, rng)``.
    """
    return (
        update_state.train_state.params,
        collect_state.env_state,
        collect_state.last_obs,
        update_state.rng,
    )


# ---------------------------------------------------------------------------
# AlgorithmSpec
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmSpec:
    """Specification for a PPO-family algorithm variant.

    Only ``algo_name`` and the six "required" callables must be provided.
    Everything else has sensible defaults that work for the standard PPO
    training loop.

    Required:
        algo_name:       Short uppercase label (e.g. ``"PPO"``, ``"SPPO_AUG"``).
        wrap_env:        Function to apply algorithm-specific wrappers.
        make_loss_fn:    Factory that returns the loss function.
        make_collect_fn: Factory for trajectory collection step.
        make_update_fn:  Factory for update epoch step.
        calculate_gae:   GAE computation returning a results dict.
        init_networks:   Network / TrainState initialisation.

    Optional (defaults provided):
        extract_losses:       Extract logging metrics from loss_info.
        init_runner_state:    Create the initial RunnerState.
        run_dir:              Build the run directory path.
        uses_sensitivity:     Whether the algorithm needs sensitivity wrappers.
        build_eval_state:     Build minimal eval carry from update/collect state.
        make_eval_step_fn:    Build algorithm-specific deterministic eval step.
    """

    # --- required ---
    algo_name: str
    wrap_env: Callable
    make_loss_fn: Callable
    make_collect_fn: Callable
    make_update_fn: Callable
    calculate_gae: Callable
    init_networks: Callable

    # --- optional (have defaults) ---
    uses_sensitivity: bool = False
    extract_losses: Optional[Callable] = None
    init_runner_state: Optional[Callable] = None
    run_dir: Optional[Callable] = None
    build_eval_state: Optional[Callable] = None
    make_eval_step_fn: Optional[Callable] = None

    def __post_init__(self):
        """Replace the optional fields left to ``None`` with the default implementations."""
        if self.extract_losses is None:
            self.extract_losses = default_extract_losses
        if self.init_runner_state is None:
            self.init_runner_state = default_init_runner_state
        if self.run_dir is None:
            name = self.algo_name
            self.run_dir = lambda config, runs_root, ts: default_run_dir(
                name, config, runs_root, ts
            )
        if self.build_eval_state is None:
            self.build_eval_state = default_build_eval_state
