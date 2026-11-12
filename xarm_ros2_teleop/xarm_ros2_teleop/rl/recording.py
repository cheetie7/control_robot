"""RL-only frame provenance and durable episode labels; no learning dependency."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np

from xarm_ros2_teleop.recording.control_contract import JOINT_ACTION, TCP_ACTION
from xarm_ros2_teleop.recording.synchronizer import SynchronizationError

ACTION_SOURCES = {"policy": 0, "human": 1, "hold": 2}


class RLFrameHistory:
    """Match driver-accepted actions to timestamped arbiter selections.

    Attributes:
        samples: Bounded history of JSON command events on the ROS clock.
        duration_sec: Retention window in seconds, shared with command history.
    """

    def __init__(self, duration_sec: float, action_format: str = JOINT_ACTION) -> None:
        """Allocate a recent history covering ``duration_sec`` seconds."""

        self.duration_sec = float(duration_sec)
        self.samples: deque[dict] = deque()
        self.action_format = action_format
        self.acceptances: deque[dict] = deque()

    def clear(self) -> None:
        """Remove selections from previous episodes or reset motion."""

        self.samples.clear()
        self.acceptances.clear()

    def add(self, event: dict) -> None:
        """Validate and retain one selected command, action arrays shaped ``[8]``.

        Args:
            event: ROS-clock timestamp, source, selected action, optional policy
                action and its validity flag. Invalid events raise ValueError.
        """

        timestamp = float(event["timestamp"])
        action = np.asarray(event["selected_action"], dtype=np.float32)
        policy = np.asarray(event["policy_action"], dtype=np.float32)
        if (
            not np.isfinite(timestamp)
            or action.shape != (8,)
            or policy.shape != (8,)
            or not np.all(np.isfinite(action))
            or not np.all(np.isfinite(policy))
            or event["source"] not in ACTION_SOURCES
            or event.get("action_format", JOINT_ACTION) != self.action_format
        ):
            raise ValueError("RL command event requires finite [8] actions and a known source")
        sample = {**event, "selected_action": action, "policy_action": policy}
        self.samples.append(sample)
        while self.samples and self.samples[0]["timestamp"] < timestamp - self.duration_sec:
            self.samples.popleft()

    def add_acceptance(self, event: dict) -> None:
        """Retain a successful SDK TCP ACK with actual pose [7] and selection time.

        Both timestamps use ROS seconds. A null selection denotes driver hold,
        not a policy/human command. No ACK is produced for SDK failures.
        """
        pose = np.asarray(event["accepted_pose"], dtype=np.float32)
        stamp = float(event["timestamp"])
        selected = event["selection_timestamp"]
        if pose.shape != (7,) or not np.all(np.isfinite(pose)) or not np.isfinite(stamp):
            raise ValueError("invalid TCP SDK acceptance")
        if selected is not None and (not np.isfinite(selected) or selected > stamp):
            raise ValueError("TCP acceptance references a future/invalid selection")
        self.acceptances.append({**event, "accepted_pose": pose})
        while self.acceptances and self.acceptances[0]["timestamp"] < stamp - self.duration_sec:
            self.acceptances.popleft()

    def _tcp_selection(self, action: np.ndarray, arm_time: float) -> dict:
        """Join actual TCP [8] to SDK ACK and original selection without pose equality.

        SDK rotation limiting changes the accepted pose; quaternion sign may also
        change in recording. Verify ACK against actual pose up to q == -q, then
        join requested action by its ROS timestamp. Missing evidence rejects row.
        """
        acknowledgement = next(
            (
                event
                for event in reversed(self.acceptances)
                if abs(event["timestamp"] - arm_time) < 1e-6
            ),
            None,
        )
        if acknowledgement is None:
            raise SynchronizationError("No TCP SDK acknowledgement for accepted action")
        pose = acknowledgement["accepted_pose"]
        if not np.allclose(action[:3], pose[:3], atol=1e-5, rtol=0) or not np.isclose(
            abs(action[3:7] @ pose[3:7]), 1, atol=1e-5
        ):
            raise SynchronizationError("TCP SDK acknowledgement pose mismatch")
        selection = acknowledgement["selection_timestamp"]
        if selection is None:
            return {
                "selected_action": action,
                "policy_action": np.zeros(8, np.float32),
                "source": "hold",
                "policy_valid": False,
            }
        sample = next(
            (
                event
                for event in reversed(self.samples)
                if abs(event["timestamp"] - selection) < 1e-6
            ),
            None,
        )
        if sample is None:
            raise SynchronizationError("No causal RL TCP selection for SDK acknowledgement")
        return sample

    def annotate(self, frame: dict, timestamp: float, command_times: tuple[float, float]) -> dict:
        """Extend a synchronized row with the provenance of its accepted arm target.

        Args:
            frame: Standard LeRobot row, with driver-accepted ``action[8]``.
            timestamp: Original anchor image time in ROS seconds.
            command_times: Accepted arm and gripper command timestamps in seconds.

        Returns:
            Same mapping with float32 numeric RL fields. The arm target must
            match a causal selection within 1e-5 rad; otherwise the row is
            rejected. Gripper acceptance stays separately auditable because
            its driver may quantize or suppress repeated commands.
        """

        arm_time, gripper_time = command_times
        sample = (
            self._tcp_selection(frame["action"], arm_time)
            if self.action_format == TCP_ACTION
            else next(
                (
                    candidate
                    for candidate in reversed(self.samples)
                    if 0 <= arm_time - candidate["timestamp"] <= self.duration_sec
                    and np.allclose(
                        candidate["selected_action"][:7], frame["action"][:7], atol=1e-5, rtol=0
                    )
                ),
                None,
            )
        )
        if sample is None:
            raise SynchronizationError("No causal RL selection matches the accepted xArm command")
        frame.update(rl_schema_fields())
        frame["observation.timestamp_ros_s"] = np.asarray([timestamp], dtype=np.float64)
        frame["action.command_timestamp_ros_s"] = np.asarray(
            [arm_time, gripper_time], dtype=np.float64
        )
        frame["action.selected"] = sample["selected_action"].copy()
        frame["action.policy"] = sample["policy_action"].copy()
        frame["action.policy_valid"][0] = float(sample["policy_valid"])
        frame["action.source"][0] = ACTION_SOURCES[sample["source"]]
        frame["action.intervention"][0] = float(sample["source"] == "human")
        return frame


def rl_schema_fields() -> dict[str, np.ndarray]:
    """Return fixed-shape RL fields for schema initialization or a normal row.

    Returns:
        Zero float32 arrays (one scalar per flag), except absolute ROS seconds
        use float64 to preserve sub-frame timing at Unix-scale timestamps.
    """

    fields = {
        "action.selected": np.zeros(8, dtype=np.float32),
        "action.policy": np.zeros(8, dtype=np.float32),
        "action.policy_valid": np.zeros(1, dtype=np.float32),
        "action.source": np.zeros(1, dtype=np.float32),
        "action.intervention": np.zeros(1, dtype=np.float32),
        "reward": np.zeros(1, dtype=np.float32),
        "terminated": np.zeros(1, dtype=np.float32),
        "truncated": np.zeros(1, dtype=np.float32),
    }
    fields["observation.timestamp_ros_s"] = np.zeros(1, dtype=np.float64)
    fields["action.command_timestamp_ros_s"] = np.zeros(2, dtype=np.float64)
    return fields


def write_episode_label(dataset_root: Path, episode_index: int, label: dict) -> None:
    """Atomically save one committed episode's human reward and provenance.

    Args:
        dataset_root: Exact category LeRobot directory.
        episode_index: Zero-based episode index local to that category.
        label: JSON-compatible outcome, timestamps, source counts and policy
            metadata. The human end event is also assigned to the last row.

    Returns:
        ``None``. A durable ``rl/episode_NNNNNN.json`` file is published.
    """

    if label.get("outcome") not in {"success", "failure", "crash"}:
        raise ValueError("RL outcome must be success, failure or crash")
    if not np.isfinite(float(label["reward"])):
        raise ValueError("RL episode reward must be finite")
    if label["terminated"] and label["truncated"]:
        raise ValueError("episode cannot be both terminated and truncated")
    output_dir = dataset_root / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"episode_{episode_index:06d}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    **label,
                    "schema_version": 1,
                    "episode_index": episode_index,
                    "action_format": label.get("action_format", JOINT_ACTION),
                    "action_source_encoding": ACTION_SOURCES,
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
