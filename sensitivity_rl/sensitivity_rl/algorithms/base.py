"""Base classes and shared utilities for PPO-family algorithms."""

from dataclasses import dataclass
from typing import Callable, Type, NamedTuple




@dataclass
class AlgorithmSpec:
    """Specification for a PPO-family algorithm variant.
    
    Attributes:
        wrap_env: Function to apply algorithm-specific wrappers
        make_loss_fn: Factory that returns the loss function
        make_collect_fn: Factory for trajectory collection function
        make_update_fn: Factory for update epoch function
        calculate_gae: Function to calculate GAE and return dict of results
        pack_update_state: Function to pack state for update loop
        rebuild_runner_state: Function to rebuild runner state from update state
        extract_losses: Function to extract loss metrics from loss_info
        init_networks: Function to initialize all networks and train states
        init_runner_state: Function to create the initial runner state
        uses_sensitivity: Whether this algorithm uses sensitivity wrappers
    """
    wrap_env: Callable
    make_loss_fn: Callable
    make_collect_fn: Callable
    make_update_fn: Callable
    calculate_gae: Callable
    pack_update_state: Callable
    rebuild_runner_state: Callable
    extract_losses: Callable
    init_networks: Callable
    init_runner_state: Callable
    run_dir: Callable
    uses_sensitivity: bool = False
