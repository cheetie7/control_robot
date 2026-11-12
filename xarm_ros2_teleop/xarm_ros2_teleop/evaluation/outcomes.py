"""Durable sidecars for human-labeled policy evaluation outcomes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class EvaluationOutcomeWriter:
    """Append auditable success/failure records beside one LeRobot dataset.

    Args:
        dataset_root: LeRobot directory receiving ``evaluation/outcomes.jsonl``.

    Each JSON line identifies one committed LeRobot episode. Discarded attempts
    are intentionally absent because they have no durable episode index.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        """Resolve the sidecar path without creating files before first save."""

        self.path = Path(dataset_root).expanduser().resolve() / "evaluation" / "outcomes.jsonl"

    def append(self, outcome: Mapping[str, Any]) -> None:
        """Append and fsync one JSON outcome mapping.

        Args:
            outcome: JSON-compatible scalar provenance and terminal fields for
                one committed episode. ``episode_index`` is a zero-based int.

        Returns:
            ``None``. One complete newline-delimited record is durably written.
        """

        episode_index = outcome.get("episode_index")
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            raise ValueError("evaluation outcome requires an integer episode_index")
        payload = json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(file_descriptor, payload.encode("utf-8"))
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)

    def counts(self) -> tuple[int, int]:
        """Return persisted ``(successes, failures)`` from valid JSON lines."""

        if not self.path.is_file():
            return 0, 0
        successes = 0
        failures = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("outcome") == "success":
                successes += 1
            elif record.get("outcome") == "failure":
                failures += 1
        return successes, failures
