import jax
import jax.numpy as jnp
from pathlib import Path
from flax import serialization
from envs.base import Sensitivity


def default_sensitivity(env) -> Sensitivity:
        """Default sensitivity parameters."""
        return Sensitivity(
            state=jnp.zeros((env.state_dim, env.param_dim)), # ds/dp
            action=jnp.zeros((env.action_dim, env.param_dim )), # du/dp
            policy_gains=jnp.zeros((env.action_dim, env.obs_dim)), # dmu/dobs
        )

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



def update_latest_symlink(run_dir: Path, env_name: str, runs_root: Path | None = None) -> Path:
    """Create or refresh the latest symlink for a given environment.

    Parameters
    ----------
    run_dir:
        The directory containing the results of the most recent PPO run.
    env_name:
        Environment key (e.g. "drone_v0") used to build the runs/<env> hierarchy.
    runs_root:
        Base directory that contains the per-environment run folders. Defaults to
        the "runs" folder relative to the current working directory.

    Returns
    -------
    Path
        Absolute path to the refreshed latest symlink.
    """

    runs_root = runs_root or Path("runs")
    env_root = runs_root / env_name
    env_root.mkdir(parents=True, exist_ok=True)

    latest_dir = env_root / "latest"
    target = run_dir.resolve()

    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)

    latest_dir.symlink_to(target, target_is_directory=True)
    return latest_dir