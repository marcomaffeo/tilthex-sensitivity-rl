from .pendulum import Pendulum
from .base import Sensitivity
from .drone_v3 import DroneV3
from .tilthex_v1 import TilthexV1

ENV_REGISTRY = {
    "pendulum": Pendulum,
    "drone_v3": DroneV3,
    "tilthex_v1": TilthexV1,
}

__all__ = [
    "Pendulum",
    "Sensitivity",
    "DroneV3",
    "TilthexV1",
]
