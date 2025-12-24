import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Any, Mapping, Tuple
from flax import serialization



def angle_normalize(x: jax.Array) -> jax.Array:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def normalize(a, a_min, a_max):
    """
    Maps input a from [a_min, a_max] to [-1, 1]
    """
    return 2 * (a - a_min) / (a_max - a_min) - 1

def load_actor_params_from_file(params_path: str):
    """Load actor parameters from a msgpack file and merge with fresh critic params.
    
    Args:
        params_path: Path to the params.msgpack file
        
    Returns:
        Loaded parameters dictionary containing only actor parameters
        
    Raises:
        FileNotFoundError: If the params file doesn't exist
        ValueError: If the params file is corrupted or invalid
    """
    params_file = Path(params_path)
    if not params_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    try:
        params_bytes = params_file.read_bytes()
        full_params = serialization.from_bytes(None, params_bytes)
        
        if not (isinstance(full_params, dict) and 'params' in full_params):
            raise ValueError("Invalid parameter structure: expected dict with 'params' key")
        
        # Extract only actor parameters
        all_params = full_params['params']
        actor_params = {}
        
        actor_layer_names = ['actor_dense1', 'actor_dense2', 'actor_output', 'actor_log_std']
        
        for layer_name in actor_layer_names:
            if layer_name in all_params:
                actor_params[layer_name] = all_params[layer_name]
                print(f"Loaded actor layer: {layer_name}")
            else:
                print(f"Warning: Actor layer {layer_name} not found in saved parameters")
        
        return {'params': actor_params}
        
    except Exception as e:
        raise ValueError(f"Failed to load parameters from {params_path}: {e}")


def merge_actor_params_with_fresh_critic(actor_params, fresh_params):
    """Merge loaded actor parameters with freshly initialized critic parameters.
    
    Args:
        actor_params: Dictionary containing only actor parameters
        fresh_params: Freshly initialized full network parameters
        
    Returns:
        Merged parameters dictionary
    """
    merged_params = fresh_params.copy()
    
    if 'params' in actor_params and 'params' in fresh_params:
        # Start with fresh parameters
        merged_layer_params = fresh_params['params'].copy()
        
        # Override with loaded actor parameters
        for layer_name, layer_params in actor_params['params'].items():
            if layer_name.startswith('actor_'):
                merged_layer_params[layer_name] = layer_params
        
        merged_params['params'] = merged_layer_params
    
    return merged_params


def sample_env_params(
    env,
    base_params,
    lows,
    highs,
    num_samples: int,
    rng_key: jax.Array,
):
    """Sample environment parameter pytrees within user-defined ranges.

    Parameters
    ----------
    env: environment.Environment
        Environment instance used for context; not mutated.
    base_params: pytree
        Reference parameters (usually ``env.default_params``).
    lows, highs:
        Lower and upper bounds supplied in the same flattened order as the
        leaves returned by ``jax.tree_util.tree_flatten(base_params)``.
        Each entry must broadcast with the corresponding leaf.
    num_samples: int
        Number of parameter copies to generate.
    rng_key: jax.Array
        PRNG key used for sampling.

    Returns
    -------
    list
        List of parameter pytrees sampled within provided bounds.
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
    """Create or refresh the latest symlink for a given environment.

    Parameters
    ----------
    run_dir:
        The directory containing the results of the most recent PPO run.
    runs_root:
        Base directory that contains the per-environment run folders. Defaults to
        the "runs" folder relative to the current working directory.

    Returns
    -------
    Path
        Absolute path to the refreshed latest symlink.
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


def init_network_params(rng, network, init_x, config):
    """Initialize or load network parameters."""
    if config.get("LOAD_ACTOR_PARAMS_PATH"):
        print(f"Loading actor parameters from: {config['LOAD_ACTOR_PARAMS_PATH']}")
        try:
            fresh_params = network.init(rng, init_x)
            actor_params = load_actor_params_from_file(config["LOAD_ACTOR_PARAMS_PATH"])
            network_params = merge_actor_params_with_fresh_critic(actor_params, fresh_params)
            print("Successfully loaded actor parameters")
            return network_params
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e}. Using random initialization.")
    return network.init(rng, init_x)


def shuffle_batch(rng, batch, config):
    """Shuffle and reshape batch into minibatches."""
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

    