"""One inference phase/reason contract shared by the recorder, UI and monitor."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceStatus:
    """Machine-readable phase and operator-facing reason; neither owns robot safety."""

    phase: str
    reason: str = ""

    @classmethod
    def from_wire(cls, value: str) -> "InferenceStatus":
        """Decode JSON status, or a pre-upgrade ``PHASE: reason`` ROS message.

        Args:
            value: String message data. Older publishers remain readable during
                a rolling restart; new publishers always emit schema version 1.
        Returns:
            Stable phase and separate display text, without prefix matching.
        """

        if value.lstrip().startswith("{"):
            data = json.loads(value)
            if data["schema_version"] != 1:
                raise ValueError("unsupported inference status schema")
            return cls(str(data["phase"]), str(data["reason"]))
        phase, _, reason = value.partition(":")
        return cls(phase.strip(), reason.strip())

    def to_json(self) -> str:
        """Return the versioned ROS String JSON payload, with separate phase/reason."""

        return json.dumps(
            {"schema_version": 1, "phase": self.phase, "reason": self.reason},
            ensure_ascii=False,
        )

    @property
    def text(self) -> str:
        """Return readable text for logs, not a machine control predicate."""

        return f"{self.phase}: {self.reason}" if self.reason else self.phase

    @property
    def failed(self) -> bool:
        """Whether a request/connection fault requires mode-specific handling."""

        return self.phase in {"EPISODE_FAILED", "STOPPED"}

    @property
    def reusable(self) -> bool:
        """Whether a connected policy is idle or intentionally paused for a new episode."""

        return self.phase in {"WAITING_FOR_EPISODE", "PAUSED_BY_HUMAN"}

    @property
    def can_stop(self) -> bool:
        """Whether an inference stop request is meaningful in this phase."""

        return self.phase not in {"DISABLED", "UNKNOWN", "STOPPED"}
