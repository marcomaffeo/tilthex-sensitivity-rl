"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import sppo
from . import sppo_aug
from . import cppo
from . import rappo
from . import sppo_grad

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "sppo": sppo.SPEC,
    "sppo_aug": sppo_aug.SPEC,
    "sppo_grad": sppo_grad.SPEC,
    "cppo" : cppo.SPEC,
    "rappo" : rappo.SPEC,
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
