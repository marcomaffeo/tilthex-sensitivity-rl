"""Helpers of the training loop: normalisation, loading of saved parameters, sampling of environment parameters, batch shuffling."""

import jax
import jax.numpy as jnp
import os
from pathlib import Path
from typing import Any, Mapping, Tuple
from flax import nnx, serialization



def angle_normalize(x: jax.Array) -> jax.Array:
    """Wrap an angle to ``[-pi, pi)``.

    Args:
        x (jax.Array): Angle [rad].

    Returns:
        jax.Array: Wrapped angle [rad].
    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def normalize(a, a_min, a_max):
    """Map a value from ``[a_min, a_max]`` to ``[-1, 1]``.

    Args:
        a (jax.Array): Value to map.
        a_min (jax.Array): Lower bound.
        a_max (jax.Array): Upper bound.

    Returns:
        jax.Array: Mapped value.
    """
    return 2 * (a - a_min) / (a_max - a_min) - 1

def load_actor_params_from_file(params_path: str):
    """Load the parameters saved by the training loop in a msgpack file.

    The file holds the pure-dict representation of the full NNX state
    (``state.to_pure_dict()``). Relative paths are also searched from the
    directory where Hydra was launched.

    Args:
        params_path (str): Path to ``policy_params.msgpack``.

    Returns:
        dict: Restored pure dict of the whole model; the caller extracts the actor keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    params_file = Path(params_path)
    if not params_file.is_absolute():
        params_file = Path.cwd() / params_file
    if not params_file.exists():
        hydra_orig_cwd = os.environ.get("HYDRA_ORIG_CWD")
        if hydra_orig_cwd:
            alt = Path(hydra_orig_cwd) / params_path
            if alt.exists():
                params_file = alt

    if not params_file.exists():
        raise FileNotFoundError(
            f"Parameters file not found: {params_path} (resolved: {params_file})"
        )

    try:
        params_bytes = params_file.read_bytes()
        return serialization.msgpack_restore(params_bytes)
    except Exception as e:
        raise ValueError(f"Failed to load parameters from {params_file}: {e}")


def merge_actor_params_with_fresh_critic(loaded_pure_dict, fresh_state):
    """Merge loaded actor parameters into a freshly initialised NNX state.

    Args:
        loaded_pure_dict (dict): Pure dict restored from a previous run (whole model).
        fresh_state (nnx.State): State of a freshly initialised model with the same architecture.

    Returns:
        nnx.State: State with the actor leaves from ``loaded_pure_dict`` and the critic leaves from ``fresh_state``.
    """
    treedef = jax.tree.structure(fresh_state)
    fresh_pure = fresh_state.to_pure_dict()

    # Overlay loaded actor values onto fresh dict
    merged = dict(fresh_pure)
    for key, value in loaded_pure_dict.items():
        if key.startswith("actor_"):
            merged[key] = value

    flat_leaves = jax.tree.leaves(merged)
    return jax.tree.unflatten(treedef, flat_leaves)


def sample_env_params(
    env,
    base_params,
    lows,
    highs,
    num_samples: int,
    rng_key: jax.Array,
):
    """Sample environment parameter pytrees uniformly within given bounds.

    Args:
        env (environment.Environment): Environment, used for context only.
        base_params (EnvParams): Reference parameters, usually ``env.default_params``.
        lows (EnvParams | list): Lower bounds, in the leaf order of ``base_params``.
        highs (EnvParams | list): Upper bounds, in the same order.
        num_samples (int): Number of parameter sets.
        rng_key (jax.Array): PRNG key.

    Returns:
        list[EnvParams]: Sampled parameter sets.

    Raises:
        ValueError: If the bounds do not match the leaves of ``base_params``.
    """

    leaves, treedef = jax.tree_util.tree_flatten(base_params)
    low_leaves, _ = jax.tree_util.tree_flatten(lows)
    high_leaves, _ = jax.tree_util.tree_flatten(highs)

    if len(low_leaves) != len(leaves) or len(high_leaves) != len(leaves):
        raise ValueError("Bounds must match the number of leaves in base_params")

    samples_per_leaf = []
    key = rng_key
    for leaf, low_bound, high_bound in zip(leaves, low_leaves, high_leaves):
        leaf = jnp.asarray(leaf)
        leaf_shape = leaf.shape
        low = jnp.asarray(low_bound, dtype=leaf.dtype)
        high = jnp.asarray(high_bound, dtype=leaf.dtype)
        low = jnp.broadcast_to(low, leaf_shape)
        high = jnp.broadcast_to(high, leaf_shape)

        key, subkey = jax.random.split(key)
        sample_shape = (num_samples,) + leaf_shape
        uniform = jax.random.uniform(subkey, sample_shape)
        sample = uniform * (high - low)[None, ...] + low[None, ...]
        samples_per_leaf.append(sample)

    sampled_params = []
    for idx in range(num_samples):
        leaves_idx = [leaf_samples[idx] for leaf_samples in samples_per_leaf]
        sampled_params.append(jax.tree_util.tree_unflatten(treedef, leaves_idx))

    return sampled_params


def update_latest_symlink(run_dir: Path, env_name: str, runs_root: Path | None = None) -> Path:
    """Point ``<runs_root>/latest`` to the most recent run directory.

    Args:
        run_dir (pathlib.Path): Directory of the most recent run.
        env_name (str): Environment name.
        runs_root (pathlib.Path, optional): Root folder of the runs, ``runs`` by default.

    Returns:
        pathlib.Path: Path of the ``latest`` link.
    """

    runs_root = runs_root or Path("runs")

    latest_dir = runs_root / "latest"
    target = run_dir.resolve()

    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)

    latest_dir.symlink_to(target, target_is_directory=True)
    return latest_dir


def init_network_params(model, config):
    """Load actor weights from a previous run into a model, when requested.

    If ``config["LOAD_ACTOR_PARAMS_PATH"]`` is set, the actor leaves are
    replaced by the saved ones and the critic keeps its fresh initialisation.

    Args:
        model (nnx.Module): Initialised NNX model.
        config (dict): Training configuration.

    Returns:
        nnx.Module: The same model, with the actor weights replaced when a path is given.
    """
    load_path = config.get("LOAD_ACTOR_PARAMS_PATH")
    if not load_path:
        return model

    print(f"Loading actor parameters from: {load_path}")
    try:
        loaded_dict = load_actor_params_from_file(load_path)
        graphdef, fresh_state = nnx.split(model)
        merged_state = merge_actor_params_with_fresh_critic(loaded_dict, fresh_state)
        model = nnx.merge(graphdef, merged_state)
        print("Successfully loaded actor parameters")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: {e}. Using random initialisation.")
    return model


def shuffle_batch(rng, batch, config):
    """Shuffle a rollout batch and split it into minibatches.

    Args:
        rng (jax.Array): PRNG key.
        batch (Any): Pytree of arrays with leading shape ``(NUM_STEPS, NUM_ENVS)``.
        config (dict): Training configuration (``MINIBATCH_SIZE``, ``NUM_MINIBATCHES``, ``NUM_STEPS``, ``NUM_ENVS``).

    Returns:
        Any: Pytree of arrays with leading shape ``(NUM_MINIBATCHES, MINIBATCH_SIZE)``.
    """
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (
        batch_size == config["NUM_STEPS"] * config["NUM_ENVS"] 
        ), "batch size must be equal to number of steps * number of envs"
                
    permutation = jax.random.permutation(rng, batch_size)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    return jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
        shuffled_batch,
    )

    