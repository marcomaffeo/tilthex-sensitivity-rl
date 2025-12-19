"""Base classes and shared utilities for PPO-family algorithms."""

from dataclasses import dataclass
from typing import Callable, Type, NamedTuple




@dataclass
class AlgorithmSpec:
    """Specification for a PPO-family algorithm variant.
    
    Attributes:
        name: Short identifier for the algorithm
        transition_cls: NamedTuple class for storing rollout transitions
        make_loss_fn: Factory that returns the loss function
        wrap_env: Function to apply algorithm-specific wrappers
        network_cls: Network class (ActorCritic or variant)
        uses_sensitivity: Whether this algorithm uses sensitivity wrappers
        gae_fn: GAE calculation function
        make_collect_fn: Factory for trajectory collection function
        make_update_fn: Factory for update epoch function
        calculate_gae: Function to calculate GAE and return dict of results
        pack_update_state: Function to pack state for update loop
        unpack_update_state: Function to unpack state after update loop
        rebuild_runner_state: Function to rebuild runner state from update state
        extract_losses: Function to extract loss metrics from loss_info
        init_networks: Function to initialize all networks and train states
        init_runner_state: Function to create the initial runner state
        unpack_runner_state: Function to unpack runner state for saving
    """
    name: str
    transition_cls: Type[NamedTuple]
    make_loss_fn: Callable
    wrap_env: Callable
    network_cls: Type
    uses_sensitivity: bool = False
    gae_fn: Callable = None
    make_collect_fn: Callable = None
    make_update_fn: Callable = None
    calculate_gae: Callable = None
    pack_update_state: Callable = None
    rebuild_runner_state: Callable = None
    extract_losses: Callable = None
    init_networks: Callable = None
    init_runner_state: Callable = None
