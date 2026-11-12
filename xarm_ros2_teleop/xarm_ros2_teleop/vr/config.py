"""Validate the small, explicit VR teleoperation profile."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VRConfig:
    """Metre/radian control limits and fixed quest_floor-to-base axis mapping."""

    hand: str
    rotation: np.ndarray
    position_scale: float
    timeout: float
    stable_sec: float
    linear_speed: float
    linear_acceleration: float
    angular_speed: float
    lower: np.ndarray | None
    upper: np.ndarray | None

    @classmethod
    def from_mapping(cls, data):
        """Parse teleop.vr mapping; empty workspace permits preview only."""
        hand = data.get("hand", "right")
        if hand not in {"left", "right"}:
            raise ValueError("teleop.vr.hand must be left or right")
        rotation = np.asarray(data.get("base_from_floor_rotation", np.eye(3)), dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("base_from_floor_rotation must be a proper rotation [3,3]")
        values = [
            float(data.get(key, default))
            for key, default in (
                ("position_scale", 1.0),
                ("input_timeout_sec", 0.25),
                ("ready_stable_sec", 0.3),
                ("max_linear_speed_m_s", 0.1),
                ("max_linear_acceleration_m_s2", 0.2),
                ("max_angular_speed_rad_s", 0.5),
            )
        ]
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("VR scales, timeouts and speeds must be positive and finite")
        lower, upper = data.get("workspace_lower_m"), data.get("workspace_upper_m")
        if (lower is None) != (upper is None):
            raise ValueError("VR workspace requires both lower and upper bounds")
        if lower is not None:
            lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.all(np.isfinite([lower, upper]))
                or np.any(lower >= upper)
            ):
                raise ValueError("VR workspace bounds must be finite ordered xyz [3]")
        return cls(hand, rotation, *values, lower, upper)

    def require_workspace(self) -> None:
        """Reject physical motion without an explicitly configured metre workspace."""
        if self.lower is None:
            raise ValueError("VR motion requires workspace_lower_m and workspace_upper_m")

    def check_workspace(self, position) -> None:
        """Reject, rather than clip, xyz [3] outside the configured base-frame box."""
        if self.lower is not None and (
            np.any(position < self.lower) or np.any(position > self.upper)
        ):
            raise ValueError("TCP target outside configured workspace; release and re-grip")
