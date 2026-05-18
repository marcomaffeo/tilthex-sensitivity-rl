"""
Tuner script for running multiple experiments with Hydra configuration.

This script allows you to:
1. Run multiple experiments with different configurations in parallel
2. Sweep over hyperparameters defined in SWEEP_PARAMS
3. Use method-specific sweeps via METHOD_SPECIFIC_SWEEPS
4. Organize results in a single sweep directory

Edit SWEEP_PARAMS or METHOD_SPECIFIC_SWEEPS below to define your parameter sweeps.

USAGE:
    python tuner.py

CONFIGURATION:
    Edit the following variables in this file:
    - SWEEP_PARAMS: Dictionary of parameters to sweep (shared across all methods)
    - METHOD_SPECIFIC_SWEEPS: Dictionary of method-specific parameter sweeps
    - NUM_PARALLEL_JOBS: Number of experiments to run in parallel
    - RUN_EVAL_ROLLOUTS: Whether to run evaluation rollouts
    - EVAL_CONFIG: Evaluation settings

EXAMPLE 1 - Shared sweeps (all methods use same parameters):
    SWEEP_PARAMS = {
        "SEED": [30, 40, 50],
        "SENSITIVITY_REGULARIZATION_WEIGHT": [0.0, 0.5, 0.9],
        "LR": [0.001, 0.0005],
    }
    METHOD_SPECIFIC_SWEEPS = {}

This will run 3 × 3 × 2 = 18 experiments per method.

EXAMPLE 2 - Method-specific sweeps (different parameters per method):
    METHOD_SPECIFIC_SWEEPS = {
        "sppo_aug": {
            "SEED": [0, 1, 2],
            "SENSITIVITY_REGULARIZATION_WEIGHT": [0.5, 0.9],
        },
        "ppo": {
            "SEED": [0, 1, 2],
            "DOMAIN_RANDOMIZATION": [True, False],
        },
    }

This runs 6 experiments for sppo_aug and 6 for ppo, with different parameters.
"""

import os
import sys
import copy
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import itertools
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set JAX platforms before importing JAX (cuda for GPU, cpu for fallback)
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

# Add parent directory to path to import sensitivity_rl modules
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
import jax
from flax import serialization
from torch.utils.tensorboard import SummaryWriter

from algorithms import ALGORITHM_REGISTRY
from train import make_train, run_evaluation

# Import plotting function
try:
    from plots.plot_sweep_results import plot_sweep_results

    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: plot_sweep_results module not found. Plotting will be skipped.")
    PLOTTING_AVAILABLE = False


# =============================================================================
# SWEEP PARAMETERS - Edit this section to define your parameter sweeps
# =============================================================================

# List of methods to sweep over. Each method uses its corresponding script.
# Available: "sppo-vs-estimation-aug", "sppo-vs-estimation", "ppo", "sac"
METHODS = ["ppo","sppo_aug_grad_param_dual"]  # Add or remove methods as needed

# List of environments to sweep over. Each needs a corresponding config file.
# Available: "halfcheetah_v5", "drone_v0", "drone_v1", "pendulum_v0", etc.
ENVIRONMENTS = ["tilthex_v1"]

# Parameter sweeps - can be method-specific or shared across all methods
# Option 1: Use SWEEP_PARAMS for shared parameters across all methods (backward compatible)
# Option 2: Use METHOD_SPECIFIC_SWEEPS for method-specific parameters
# If METHOD_SPECIFIC_SWEEPS is defined and not empty, it takes precedence

# Shared parameter sweeps - applied to all methods if METHOD_SPECIFIC_SWEEPS is not used
SWEEP_PARAMS = {
    # Example sweeps - uncomment and modify as needed
    "SEED": [12, 54, 99, 123, 80, 77, 14, 66, 39, 87, 224, 44],  # Random seeds for reproducibility
    "GAIN_STAB": [0.2],          # Penalità velocità angolare
    "GAIN_ATT": [2.0],                 # Reward per allineamento
    "GAIN_SM": [1],        # Penalità variazione comandi
    "GAIN_REF": [1],       # Penalità distanza target
    "REWARD_POS_TOLERANCE": [0.04], # Tolleranza per reward di posizione

    #"SENSITIVITY_REGULARIZATION_WEIGHT": [0.01, 0.1, 0.05],
    #"LR": [0.001, 0.0005, 0.0001],
    #"NUM_ENVS": [2048, 4096],
    #"GAMMA": [0.98],
    #"GAE_LAMBDA": [0.9, 0.95],
    "CLIP_EPS": [0.1, 0.2],
    #"UPDATE_EPOCHS": [5],
    #"ENT_COEF": [0.001],
    #"DOMAIN_RANDOMIZATION": [True, False],
    "TOTAL_TIMESTEPS": [50_000_000],
}

# Method-specific parameter sweeps (optional)
# If specified, each method uses its own sweep parameters
# Methods not listed here will use SWEEP_PARAMS (if USE_SHARED_PARAMS_AS_FALLBACK is True)
METHOD_SPECIFIC_SWEEPS = {
    "ppo": {
        "DOMAIN_RANDOMIZATION": [True, False],
    },
    "rappo": {
        "DOMAIN_RANDOMIZATION": [False],
    },
    "cppo": {
        "DOMAIN_RANDOMIZATION": [False],
    },
    "sppo_grad": {
        "SENSITIVITY_REGULARIZATION_WEIGHT": [0.3, 0.2],
        "DOMAIN_RANDOMIZATION": [False],
    },
    "sppo_grad_param": {
        "SENSITIVITY_REGULARIZATION_WEIGHT": [0.3, 0.2],
        "XI": [0.08, 0.04, 0.1],
        "RHO": [0.5, 0.1, 0.3],
        "TAU": [1e-3, 2e-3, 1e-4, 5e-4],
        "DOMAIN_RANDOMIZATION": [True, False],
    },
    "sppo_aug_grad_param_dual": {
        "DOMAIN_RANDOMIZATION": [True, False],
        "SENSITIVITY_REGULARIZATION_WEIGHT": [0.01, 0.02, 0.03, 0.04, 0.05],
    },
    "sac": {
        "SEED": [0, 1],
        "LR": [0.001, 0.0005],
    },
}

# If True, methods not in METHOD_SPECIFIC_SWEEPS will use SWEEP_PARAMS as fallback
# If False, methods not in METHOD_SPECIFIC_SWEEPS will run with no parameter sweep (1 experiment per env)
USE_SHARED_PARAMS_AS_FALLBACK = False

# Method name mapping for config selection
# Maps script names to config group names
METHOD_CONFIG_NAMES = {
    "ppo": "ppo",
    "sppo_grad": "ppo",
    "sppo_aug_grad_param": "ppo",
    "rappo": "rappo",
    "cppo": "cppo",
    "sac-estimation": "sac",
    "sac": "sac",
}

# Number of experiments to run in parallel (set based on your CPU/GPU resources)
# Set to 1 for sequential execution, or higher for parallel execution
# Note: Each parallel job will run in a separate process
NUM_PARALLEL_JOBS = 4

# Set to True to run evaluation rollouts after each experiment
RUN_EVAL_ROLLOUTS = True

# Set to True to generate plots after sweep completes
GENERATE_PLOTS = True

# Evaluation settings
EVAL_CONFIG = {
    "NUM_ROLLOUTS": 10,
    "PARAM_SAMPLES": 10,
}

# =============================================================================


def generate_param_combinations(
    sweep_params: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from the sweep configuration.

    Args:
        sweep_params: Dictionary mapping parameter names to lists of values

    Returns:
        List of dictionaries, each containing one parameter combination
    """
    if not sweep_params:
        return [{}]

    keys = list(sweep_params.keys())
    values = list(sweep_params.values())

    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


# Fields that should be coerced to integers (Hydra may parse as float)
INT_FIELDS = {
    "SEED",
    "NUM_ENVS",
    "NUM_STEPS",
    "TOTAL_TIMESTEPS",
    "NUM_MINIBATCHES",
    "NUM_ROLLOUTS",
    "MAX_STEPS",
    "PARAM_SAMPLES",
}


def normalize_config(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Ensure config types are JSON/JAX friendly and annotate algorithm."""
    cfg = copy.deepcopy(config)
    cfg["ALGORITHM"] = method

    for key in INT_FIELDS:
        if key in cfg:
            try:
                cfg[key] = int(cfg[key])
            except (TypeError, ValueError):
                pass
    return cfg


def run_experiment_job(
    method: str, config_dict: Dict[str, Any], run_dir: Path, rng: jax.random.PRNGKey
):
    """Run a single experiment using the unified AlgorithmSpec pipeline."""
    if method not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    # Clear JAX caches to avoid pytree metadata conflicts between experiments
    jax.clear_caches()

    spec = ALGORITHM_REGISTRY[method]
    config = normalize_config(config_dict, method)

    writer = SummaryWriter(str(run_dir))
    train_fn = make_train(config, spec, writer, rng, run_dir)
    train_jit = jax.jit(train_fn)
    final_state = train_jit(rng)

    params = final_state.train_state.params
    pure_params = params.to_pure_dict()
    (run_dir / "policy_params.msgpack").write_bytes(serialization.to_bytes(pure_params))

    keys_to_remove = ["hydra", "defaults", "_self_"]
    clean_config = {
        k: v for k, v in config.items() 
        if k not in keys_to_remove and not str(k).startswith("override")
    }

    # Estrazione dinamica esatta come in train.py
    try:
        from envs import ENV_REGISTRY
        env = ENV_REGISTRY[clean_config["ENV_NAME"]]()
        env_params = env.default_params
        
        clean_config["env"] = {}
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
            source_obj = env if source == "env" else env_params
            attr_value = getattr(source_obj, attr_name)
            if isinstance(attr_value, (int, float, bool)):
                clean_config["env"][config_key] = (
                    int(attr_value) if isinstance(attr_value, (int, bool)) else attr_value
                )
            else:
                clean_config["env"][config_key] = attr_value.tolist()
                
    except Exception as e:
        print(f"Warning: Failed to inject physical env mapping. {e}")

    
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        if hasattr(obj, "tolist"):  # Intercetta sia JAX (ArrayImpl) che Numpy
            return obj.tolist()
        return obj

    clean_config.pop("SENS_W", None)

    with open(run_dir / "config.json", "w") as f:
        json.dump(_to_serializable(clean_config), f, indent=2)

    hparam_dict = {
        k: v for k, v in config.items() if isinstance(v, (int, float, bool, str))
    }
    writer.add_hparams(hparam_dict, {}, run_name=".")
    writer.close()

    if RUN_EVAL_ROLLOUTS:
        run_evaluation(config, run_dir)

    return run_dir


def _run_experiment_subprocess(args_tuple):
    """
    Wrapper function for running experiment in a subprocess.
    This function will be called by ProcessPoolExecutor.

    Args:
        args_tuple: Tuple of (method, config_dict, experiment_name, run_dir, idx, total)

    Returns:
        Tuple of (success, run_dir, error_message, experiment_name)
    """
    method, config_dict, experiment_name, run_dir, idx, total = args_tuple

    # Set GPU memory limits
    import os
    import shutil

    # Prevent pre-allocation of all GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Set memory fraction per process
    memory_fraction = 1.0 / NUM_PARALLEL_JOBS
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    # Optional: Set allocator to avoid fragmentation
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Now import JAX and other dependencies
    import jax
    import traceback

    try:
        rng = jax.random.PRNGKey(config_dict.get("SEED", 30))

        print(f"\n[{idx}/{total}] Starting experiment: {experiment_name}")
        print(f"[{idx}/{total}] Method: {method}, Env: {config_dict.get('ENV_NAME')}")
        print(f"[{idx}/{total}] GPU Memory Fraction: {memory_fraction:.2%}")
        print(f"[{idx}/{total}] Run directory: {run_dir}\n")

        run_experiment_job(method, config_dict, run_dir, rng)

        print(f"\n[{idx}/{total}] Completed experiment: {experiment_name}")
        print(f"[{idx}/{total}] Results saved to: {run_dir}\n")

        return (True, str(run_dir), None, experiment_name)

    except Exception as e:
        error_msg = f"Experiment {idx} ({method}) failed with error: {e}"
        print(f"\nERROR: {error_msg}")
        traceback.print_exc()
        return (False, str(run_dir), str(e), experiment_name)


def run_single_experiment(method, config_dict, experiment_name, run_dir, idx, total):
    """
    Run a single experiment in the current process (for sequential execution).

    Args:
        method: Method script name to import
        config_dict: Configuration dictionary
        experiment_name: Name of the experiment
        run_dir: Directory to save results
        idx: Experiment index
        total: Total number of experiments

    Returns:
        Tuple of (success, run_dir, error_message)
    """
    try:
        rng = jax.random.PRNGKey(config_dict.get("SEED", 30))

        print(f"\n[{idx}/{total}] Starting experiment: {experiment_name}")
        print(f"[{idx}/{total}] Method: {method}, Env: {config_dict.get('ENV_NAME')}")
        print(f"[{idx}/{total}] Run directory: {run_dir}\n")

        run_experiment_job(method, config_dict, run_dir, rng)

        print(f"\n[{idx}/{total}] Completed experiment: {experiment_name}")
        print(f"[{idx}/{total}] Results saved to: {run_dir}\n")

        return (True, run_dir, None)

    except Exception as e:
        error_msg = f"Experiment {idx} ({method}) failed with error: {e}"
        print(f"\nERROR: {error_msg}")
        import traceback

        traceback.print_exc()
        return (False, run_dir, str(e))


def run_sweep_experiments():
    """
    Run experiments for all combinations of methods, environments, and parameters.

    This function loads configs and runs experiments in parallel for each
    combination. All experiments are saved in a single sweep directory.
    Supports method-specific parameter sweeps via METHOD_SPECIFIC_SWEEPS.
    """
    # Determine which sweep configuration to use
    use_method_specific = bool(METHOD_SPECIFIC_SWEEPS)

    # Build method-specific parameter combinations
    method_param_combinations = {}
    for method in METHODS:
        if use_method_specific and method in METHOD_SPECIFIC_SWEEPS:
            for key in SWEEP_PARAMS.keys():
                if key not in METHOD_SPECIFIC_SWEEPS[method]:
                    METHOD_SPECIFIC_SWEEPS[method][key] = SWEEP_PARAMS[key]
        method_param_combinations[method] = generate_param_combinations(
            METHOD_SPECIFIC_SWEEPS[method]
        )

    # Create a single sweep directory with timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = Path("runs") / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total number of experiments
    total_experiments = sum(
        len(ENVIRONMENTS) * len(method_param_combinations[method]) for method in METHODS
    )

    # Save sweep configuration
    sweep_config = {
        "methods": METHODS,
        "environments": ENVIRONMENTS,
        "method_config_names": METHOD_CONFIG_NAMES,
        "sweep_params": SWEEP_PARAMS if not use_method_specific else None,
        "method_specific_sweeps": (
            METHOD_SPECIFIC_SWEEPS if use_method_specific else None
        ),
        "use_method_specific": use_method_specific,
        "use_shared_params_as_fallback": USE_SHARED_PARAMS_AS_FALLBACK,
        "num_experiments": total_experiments,
        "num_parallel_jobs": NUM_PARALLEL_JOBS,
        "timestamp": ts,
    }
    with open(sweep_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"PARAMETER SWEEP CONFIGURATION")
    print(f"{'='*80}")
    print(f"Methods: {METHODS}")
    print(f"Environments: {ENVIRONMENTS}")
    print(f"Sweep directory: {sweep_dir}")
    print(f"Total experiments: {total_experiments}")
    print(f"Running sequentially in one process")
    if use_method_specific:
        print(f"Using method-specific sweeps:")
        for method in METHODS:
            params = METHOD_SPECIFIC_SWEEPS.get(
                method, SWEEP_PARAMS if USE_SHARED_PARAMS_AS_FALLBACK else {}
            )
            print(f"  {method}: {list(params.keys())}")
    else:
        print(f"Sweep parameters (shared): {list(SWEEP_PARAMS.keys())}")
    print(f"{'='*80}\n")

    # Update latest symlink to point to this sweep
    latest_link = Path("runs") / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(
        sweep_dir.relative_to(Path("runs")), target_is_directory=True
    )

    # Prepare all experiment configurations
    experiment_args = []
    exp_counter = 0

    for method in METHODS:
        if method not in ALGORITHM_REGISTRY:
            print(
                f"WARNING: Method '{method}' is not registered. Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
            continue

        for env_name in ENVIRONMENTS:
            # Get the method config name (maps method script to config group)
            method_config = METHOD_CONFIG_NAMES.get(method, "ppo")

            # Load base config with defaults for this method and environment
            config_path = (
                Path(__file__).parent.parent / "configs" / "hydra" / "config.yaml"
            )

            if not config_path.exists():
                config_path = (
                    Path(__file__).parent.parent / "configs" / "hydra" / "base_config.yaml"
                )
                if not config_path.exists():
                    print(f"ERROR: Base config not found: {config_path}")
                    continue

            # Load base config and compose with method and env
            base_cfg = OmegaConf.load(config_path)

            # Load method config
            method_path = (
                Path(__file__).parent.parent
                / "configs"
                / "hydra"
                / "method"
                / f"{method_config}.yaml"
            )
            if not method_path.exists():
                print(f"WARNING: Method config not found: {method_path}")
                print(f"Skipping {method} experiments")
                continue
            method_cfg = OmegaConf.load(method_path)

            # Load environment config
            env_path = (
                Path(__file__).parent.parent
                / "configs"
                / "hydra"
                / "env"
                / f"{env_name}.yaml"
            )
            if not env_path.exists():
                print(f"WARNING: Environment config not found: {env_path}")
                print(f"Skipping {env_name} experiments")
                continue
            env_cfg = OmegaConf.load(env_path)

            # Merge configs: base <- method <- env
            merged_cfg = OmegaConf.merge(base_cfg, method_cfg, env_cfg)
            base_config = OmegaConf.to_container(merged_cfg, resolve=True)

            # Ensure ENV_NAME is set correctly
            base_config["ENV_NAME"] = env_name

            # Now iterate over parameter combinations for this method
            param_combinations = method_param_combinations[method]
            for param_override in param_combinations:
                exp_counter += 1

                # Create config for this experiment
                config = base_config.copy()
                config.update(param_override)

                # Generate experiment name
                method_short = method.replace("sppo-vs-estimation-aug", "sppovsa")
                method_short = method_short.replace("sppo-vs-estimation", "sppovs")
                method_short = method_short.replace("-", "")

                experiment_name = f"exp_{exp_counter:03d}_{method_short}_{env_name}"
                for key, value in param_override.items():
                    # Shorten common parameter names
                    key_short = key.replace("SENSITIVITY_REGULARIZATION_WEIGHT", "srw")
                    key_short = key_short.replace("LEARNING_RATE", "lr")
                    key_short = key_short.lower()
                    experiment_name += f"_{key_short}{value}"

                # Create run directory inside the sweep directory
                run_dir = sweep_dir / experiment_name
                run_dir.mkdir(parents=True, exist_ok=True)

                experiment_args.append(
                    (
                        method,
                        config,
                        experiment_name,
                        run_dir,
                        exp_counter,
                        total_experiments,
                    )
                )

    # Run experiments in parallel or sequentially based on NUM_PARALLEL_JOBS
    successful_runs = []
    failed_runs = []

    if NUM_PARALLEL_JOBS <= 1:
        # Sequential execution
        print(f"Starting sequential execution...\n")
        print(f"Total experiments to run: {len(experiment_args)}\n")

        for args in experiment_args:
            method, config, experiment_name, run_dir, idx, total = args
            success, run_dir, error_msg = run_single_experiment(
                method, config, experiment_name, run_dir, idx, total
            )
            if success:
                successful_runs.append(run_dir)
            else:
                failed_runs.append((run_dir, error_msg))
    else:
        # Parallel execution in batches
        print(f"Starting parallel execution with {NUM_PARALLEL_JOBS} workers...\n")
        print(f"Total experiments to run: {len(experiment_args)}\n")
        print(f"Processing in batches of {NUM_PARALLEL_JOBS} experiments\n")

        # Process experiments in batches of NUM_PARALLEL_JOBS
        for batch_start in range(0, len(experiment_args), NUM_PARALLEL_JOBS):
            batch_end = min(batch_start + NUM_PARALLEL_JOBS, len(experiment_args))
            batch = experiment_args[batch_start:batch_end]

            print(f"\n{'='*80}")
            print(
                f"Processing batch {batch_start//NUM_PARALLEL_JOBS + 1} of {(len(experiment_args) + NUM_PARALLEL_JOBS - 1)//NUM_PARALLEL_JOBS}"
            )
            print(
                f"Experiments {batch_start + 1} to {batch_end} of {len(experiment_args)}"
            )
            print(f"{'='*80}\n")

            with ProcessPoolExecutor(max_workers=NUM_PARALLEL_JOBS) as executor:
                # Submit batch of experiments
                future_to_exp = {
                    executor.submit(_run_experiment_subprocess, args): args
                    for args in batch
                }

                # Wait for all experiments in this batch to complete
                completed_in_batch = 0
                for future in as_completed(future_to_exp):
                    completed_in_batch += 1
                    args = future_to_exp[future]
                    _, _, experiment_name, _, idx, _ = args

                    try:
                        success, run_dir_str, error_msg, exp_name = future.result()
                        run_dir = Path(run_dir_str)

                        if success:
                            successful_runs.append(run_dir)
                            print(
                                f"[Batch {batch_start//NUM_PARALLEL_JOBS + 1}] [{completed_in_batch}/{len(batch)}] ✓ Completed: {exp_name}"
                            )
                        else:
                            failed_runs.append((run_dir, error_msg))
                            print(
                                f"[Batch {batch_start//NUM_PARALLEL_JOBS + 1}] [{completed_in_batch}/{len(batch)}] ✗ Failed: {exp_name}"
                            )

                    except Exception as e:
                        _, _, experiment_name, run_dir, _, _ = args
                        error_msg = f"Subprocess execution failed: {e}"
                        failed_runs.append((run_dir, error_msg))
                        print(
                            f"[Batch {batch_start//NUM_PARALLEL_JOBS + 1}] [{completed_in_batch}/{len(batch)}] ✗ Failed: {experiment_name}"
                        )

            # All processes in this batch have finished before starting next batch
            print(
                f"\nBatch {batch_start//NUM_PARALLEL_JOBS + 1} completed. Waiting before starting next batch...\n"
            )
            import time

            time.sleep(2)  # Brief pause to ensure cleanup

    # Print summary
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiment_args)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")
    if failed_runs:
        print(f"\nFailed experiments:")
        for run_dir, error in failed_runs:
            print(f"  - {run_dir.name}: {error[:100]}")
    print(f"\nResults saved in: {sweep_dir}")
    print(f"{'='*80}\n")

    # Generate plots if enabled
    if GENERATE_PLOTS and PLOTTING_AVAILABLE and len(successful_runs) > 0:
        print(f"\n{'='*80}")
        print(f"GENERATING PLOTS")
        print(f"{'='*80}\n")
        try:
            plot_sweep_results(sweep_dir, output_subdir="plots")
        except Exception as e:
            print(f"Warning: Plotting failed with error: {e}")
            traceback.print_exc()
    elif GENERATE_PLOTS and not PLOTTING_AVAILABLE:
        print("\nSkipping plot generation (plot_sweep_results not available)")
    elif len(successful_runs) == 0:
        print("\nSkipping plot generation (no successful runs)")

    return successful_runs


if __name__ == "__main__":
    run_sweep_experiments()
