"""Remote vision-language-action inference for the real xArm system."""

from .config import InferenceConfig
from .contracts import (
    build_observation,
    interpolate_action_chunk,
    model_to_ros_commands,
    validate_action_chunk,
)
from .websocket_policy import PolicyClient

__all__ = [
    "InferenceConfig",
    "PolicyClient",
    "build_observation",
    "interpolate_action_chunk",
    "model_to_ros_commands",
    "validate_action_chunk",
]
