"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import sppo
from . import sppo_aug
from . import cppo
from . import rappo
from . import sppo_grad
from . import sppo_grad_param
from . import sppo_aug_grad_param
from . import sppo_aug_grad_param_dual

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "sppo": sppo.SPEC,
    "sppo_aug": sppo_aug.SPEC,
    "sppo_grad": sppo_grad.SPEC,
    "sppo_grad_param": sppo_grad_param.SPEC,
    "cppo": cppo.SPEC,
    "rappo": rappo.SPEC,
    "sppo_aug_grad_param": sppo_aug_grad_param.SPEC,
    "sppo_aug_grad_param_dual": sppo_aug_grad_param_dual.SPEC,
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
