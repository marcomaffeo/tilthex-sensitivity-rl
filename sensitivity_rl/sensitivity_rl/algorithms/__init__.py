"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import sppo_aug

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "sppo_aug": sppo_aug.SPEC,
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
