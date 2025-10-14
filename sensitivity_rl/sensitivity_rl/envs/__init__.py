from .pendulum import Pendulum
from .base import Sensitivity
from .drone_v0 import DroneV0
from .drone_v1 import DroneV1

ENV_REGISTRY = {
    "pendulum": Pendulum,
    "drone_v0": DroneV0,
    "drone_v1": DroneV1,
}

__all__ = [
    "Pendulum",
    "Sensitivity",
    "DroneV0",
    "DroneV1",
]
