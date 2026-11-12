"""Device-independent freshness gate for a continuously running human controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HumanInput:
    """Cache validated leader coordinates, not physical robot targets.

    Arm is float32 [7] in sign-mapped radians; gripper is one continuous device
    coordinate scaled by InterventionController. Adapters supply monotonic arrival
    times and connection state. FACTR is the current adapter, not a dependency.
    """

    arm: np.ndarray | None = None
    gripper: float | None = None
    arm_received_at: float = 0.0
    gripper_received_at: float = 0.0
    connected: bool = False
    ready_since: float | None = None

    def sample(self, now: float, timeout_sec: float) -> np.ndarray | None:
        """Return fresh combined float32 [8] coordinates, or None when unavailable."""

        if self.arm is None or self.gripper is None or not self.connected:
            return None
        if now - min(self.arm_received_at, self.gripper_received_at) > timeout_sec:
            return None
        return np.concatenate([self.arm, np.asarray([self.gripper], dtype=np.float32)])

    def ready(self, now: float, timeout_sec: float, stable_sec: float) -> bool:
        """Require fresh input continuously for stable_sec; hand motion is allowed.

        Times are monotonic seconds. Lost connection or stale input resets the
        window; reconnect never silently restores prior control ownership.
        """

        if self.sample(now, timeout_sec) is None:
            self.ready_since = None
            return False
        if self.ready_since is None:
            self.ready_since = now
        return now - self.ready_since >= stable_sec
