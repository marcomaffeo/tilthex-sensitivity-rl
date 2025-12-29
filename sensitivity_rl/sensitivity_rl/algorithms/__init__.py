"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import sppo
from . import sppo_aug
from . import cppo

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "sppo": sppo.SPEC,
    "sppo_aug": sppo_aug.SPEC,
    "cppo" : cppo.SPEC
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
