from .pendulum import Pendulum
from .base import Sensitivity
from .drone_v0 import DroneV0
from .drone_v1 import DroneV1
from .drone_v2 import DroneV2
from .drone_v3 import DroneV3
from .drone_obstacle import DroneObstacle
from .drone_flip import DroneFlipV0
from .halfcheetah import HalfCheetahV5
from .reacher import ReacherV5
from .tilthex_v0 import TilthexV0

ENV_REGISTRY = {
    "pendulum": Pendulum,
    "drone_v0": DroneV0,
    "drone_v1": DroneV1,
    "drone_v2": DroneV2,
    "drone_v3": DroneV3,
    "drone_obstacle": DroneObstacle,
    "drone_flip_v0": DroneFlipV0,
    "halfcheetah_v5": HalfCheetahV5,
    "reacher_v5": ReacherV5,
    "tilthex_v0": TilthexV0,
}

__all__ = [
    "Pendulum",
    "Sensitivity",
    "DroneV0",
    "DroneV1",
    "DroneV2",
    "DroneFlipV0",
    "DroneV3",
    "DroneObstacle",
    "HalfCheetahV5",
    "ReacherV5",
    "TilthexV0"
]
