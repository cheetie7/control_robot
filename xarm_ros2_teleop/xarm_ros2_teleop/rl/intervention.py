"""Device-independent ownership switching and sparse human rewards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from xarm_ros2_teleop.rl.episode_services import RL_EPISODE_SERVICES

# Stable Trigger endpoints used by keyboard-equivalent Web operator commands.
RL_SERVICE_COMMANDS = {
    **RL_EPISODE_SERVICES,
    "switch": "/rl/switch_control",
    "success": "/rl/success",
    "failure": "/rl/failure",
    "stop": "/rl/stop",
}


class InterventionState(str, Enum):
    """Exclusive policy/human ownership; STOPPING is an automatic transition."""

    POLICY = "POLICY"
    STOPPING = "STOPPING"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class RewardEvent:
    """One timestamped sparse reward supplied by the human operator.

    Attributes:
        timestamp: Monotonic event time in seconds.
        reward: Scalar reward assigned to the latest executed transition.
        event: Semantic label such as ``task_success``.
        terminated: Whether the task reached a terminal outcome.
        truncated: Whether the operator aborted without a task terminal state.
        source: Reward provenance; currently always ``human``.
    """

    timestamp: float
    reward: float
    event: str
    terminated: bool
    truncated: bool
    source: str = "human"

    def to_json(self) -> str:
        """Return a stable JSON object for ROS String transport and logging."""

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class InterventionController:
    """Manage a no-jump one-press takeover in canonical 8D action space.

    Robot and human-input arrays have shape ``(8,)``. Robot values are seven
    absolute radians plus normalized gripper ``[0,1]``. Human values are
    continuous seven-axis command coordinates plus a scalar gripper coordinate;
    only their displacement after automatic stop confirmation is used.
    """

    def __init__(self, *, gripper_delta_scale: float = -1.0) -> None:
        """Initialize policy ownership and the input-to-gripper delta scale."""

        self.state = InterventionState.POLICY
        self.hold_action: np.ndarray | None = None
        self.robot_origin: np.ndarray | None = None
        self.human_origin: np.ndarray | None = None
        self.gripper_delta_scale = float(gripper_delta_scale)

    def begin_takeover(self, robot_state: np.ndarray) -> np.ndarray:
        """Latch measured robot state and enter the stopping phase.

        Args:
            robot_state: Current physical state, finite float array shape ``(8,)``.

        Returns:
            Copied hold target shape ``(8,)`` that must replace policy actions.
        """

        if self.state not in {InterventionState.POLICY, InterventionState.PAUSED}:
            raise RuntimeError(f"cannot begin takeover from {self.state}")
        self.hold_action = _state(robot_state, "robot_state")
        self.state = InterventionState.STOPPING
        return self.hold_action.copy()

    def engage_human(self, robot_state: np.ndarray, human_state: np.ndarray) -> np.ndarray:
        """Capture both origins after the automatic stop gate and enter relative control.

        The caller must first confirm fresh, healthy measured stop feedback.

        Args:
            robot_state: Latest physical robot state, float shape ``(8,)``.
            human_state: Latest human-device coordinates, float shape ``(8,)``.

        Returns:
            First human target, exactly equal to the measured robot state.
        """

        if self.state != InterventionState.STOPPING:
            raise RuntimeError(f"cannot engage human from {self.state}")
        self.robot_origin = _state(robot_state, "robot_state")
        self.human_origin = _state(human_state, "human_state")
        self.hold_action = self.robot_origin.copy()
        self.state = InterventionState.HUMAN_CONTROL
        return self.human_action(self.human_origin)

    def human_action(self, human_state: np.ndarray) -> np.ndarray:
        """Map human-device displacement after engagement onto the robot origin.

        Args:
            human_state: Current continuous leader coordinates, shape ``(8,)``.

        Returns:
            Absolute robot target shape ``(8,)``. Arm displacement is applied
            one-to-one after device-specific sign mapping; gripper is clipped to ``[0,1]``.
        """

        if self.state != InterventionState.HUMAN_CONTROL:
            raise RuntimeError(f"human action unavailable in {self.state}")
        assert self.robot_origin is not None and self.human_origin is not None
        human = _state(human_state, "human_state")
        target = self.robot_origin.copy()
        target[:7] += human[:7] - self.human_origin[:7]
        target[7] += self.gripper_delta_scale * (human[7] - self.human_origin[7])
        target[7] = np.clip(target[7], 0.0, 1.0)
        return target

    def pause(self, robot_state: np.ndarray) -> np.ndarray:
        """Hold the latest measured state and require explicit policy resume."""

        self.hold_action = _state(robot_state, "robot_state")
        self.state = InterventionState.PAUSED
        return self.hold_action.copy()

    def resume_policy(self) -> None:
        """Return ownership to a freshly observed policy rollout."""

        if self.state != InterventionState.PAUSED:
            raise RuntimeError(f"cannot resume policy from {self.state}")
        self.state = InterventionState.POLICY

def _state(value: np.ndarray, name: str) -> np.ndarray:
    """Return a copied finite float32 canonical state with shape ``(8,)``."""

    state = np.asarray(value, dtype=np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"{name} must be finite shape (8,)")
    return state.copy()
