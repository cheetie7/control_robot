"""Assemble one causal LeRobot frame around the configured camera timestamp."""

from dataclasses import dataclass

import numpy as np

from xarm_ros2_teleop.recording.config import RecordingConfig
from xarm_ros2_teleop.recording.timestamp_buffer import (
    TimestampBuffer,
    TimestampedSample,
)


class SynchronizationError(RuntimeError):
    """Report a missing or stale sample that prevents a fixed-schema frame."""


@dataclass(frozen=True)
class XarmState:
    """Measured xArm arrays with position, velocity and effort shape ``(7,)``."""

    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


class FrameSynchronizer:
    """Buffer sensor streams and align them to configured anchor timestamps."""

    def __init__(self, config: RecordingConfig) -> None:
        """Allocate bounded histories for timestamp-aligned observations.

        Args:
            config: Recording timing and stream configuration. Image buffers
                retain enough decoded ``(H,W,3)`` frames to choose the sample
                nearest each anchor; numeric state/action buffers are smaller.
        """

        self.config = config
        # Four frame periods cover normal cross-driver callback reordering while
        # keeping decoded image memory bounded.
        image_history_sec = 4.0 / config.fps
        state_history_sec = max(2.0 * config.max_state_skew_sec, 2.0 / config.fps)
        self.images = {
            stream.key: TimestampBuffer(image_history_sec) for stream in config.streams
        }
        self.xarm_state = TimestampBuffer(state_history_sec)
        self.robotiq_state = TimestampBuffer(state_history_sec)
        # Action history is numeric and tiny. Retaining it preserves the
        # causal label when a delayed image carries a timestamp slightly older
        # than the newest command already received by this process.
        self.xarm_action = TimestampBuffer(config.action_history_sec)
        self.robotiq_action = TimestampBuffer(config.action_history_sec)
        self.teleop_status = TimestampBuffer(state_history_sec)

    def clear(self) -> None:
        """Remove all pre-reset and pre-episode observations and actions."""

        for buffer in (
            *self.images.values(),
            self.xarm_state,
            self.robotiq_state,
            self.xarm_action,
            self.robotiq_action,
            self.teleop_status,
        ):
            buffer.clear()

    def assemble(self, anchor: TimestampedSample, task: str) -> dict:
        """Build one frame using timestamp-nearest observations and causal commands.

        Args:
            anchor: Decoded anchor-camera RGB image and its ROS timestamp.
            task: Natural-language task stored for the current episode.

        Returns:
            Complete LeRobot frame with videos, ``observation.state`` shape
            ``(8,)``, ``action`` shape ``(8,)`` and synchronization diagnostics.
        """

        timestamp = anchor.timestamp
        frame: dict[str, np.ndarray | str | float] = {"task": task}
        skews = []
        for stream in self.config.streams:
            sample = (
                anchor
                if stream.key == self.config.anchor_key
                else self.images[stream.key].nearest(timestamp)
            )
            if sample is None:
                raise SynchronizationError(f"No sample available for {stream.key}")
            frame[f"observation.images.{stream.key}"] = sample.value
            skews.append(float(sample.timestamp - timestamp))

        xarm_state = _require_near(
            self.xarm_state.nearest(timestamp),
            timestamp,
            self.config.max_state_skew_sec,
            "xarm_state",
        )
        robotiq_state = _require_near(
            self.robotiq_state.nearest(timestamp),
            timestamp,
            self.config.max_state_skew_sec,
            "robotiq_state",
        )
        xarm_action = _require_causal(
            self.xarm_action.latest_before(timestamp),
            timestamp,
            self.config.max_action_age_sec,
            "xarm_action",
        )
        robotiq_action = _require_causal(
            self.robotiq_action.latest_before(timestamp),
            timestamp,
            self.config.max_action_age_sec,
            "robotiq_action",
        )
        state_value: XarmState = xarm_state.value
        frame["observation.state"] = np.concatenate(
            (state_value.position, np.asarray(robotiq_state.value, dtype=np.float32))
        ).astype(np.float32)
        frame["action"] = np.concatenate(
            (
                np.asarray(xarm_action.value, dtype=np.float32),
                np.asarray(robotiq_action.value, dtype=np.float32),
            )
        ).astype(np.float32)
        frame["observation.xarm_velocity"] = state_value.velocity.astype(np.float32)
        frame["observation.xarm_effort"] = state_value.effort.astype(np.float32)
        if self.config.action_format == "absolute_tcp_pose_xyzw_plus_gripper":
            status = _require_near(self.teleop_status.nearest(timestamp), timestamp,
                                   self.config.max_state_skew_sec, "teleop_status")
            frame["observation.teleop_status"] = np.asarray(status.value, dtype=np.float32)
        skews.extend(
            (
                xarm_state.timestamp - timestamp,
                robotiq_state.timestamp - timestamp,
                xarm_action.timestamp - timestamp,
                robotiq_action.timestamp - timestamp,
            )
        )
        frame["observation.sync_skew_s"] = np.asarray(skews, dtype=np.float32)
        return frame

    def schema_frame(self, task: str) -> dict:
        """Build one non-recorded example from the latest image shapes.

        Args:
            task: Natural-language instruction stored as a scalar string.

        Returns:
            A complete LeRobot-shaped mapping. Robot values are zero arrays;
            the frame is used only to prepare dataset features during warm-up.

        Raises:
            SynchronizationError: Before every configured image has arrived.
        """

        frame: dict[str, np.ndarray | str] = {"task": task}
        for stream in self.config.streams:
            sample = self.images[stream.key].newest()
            if sample is None:
                raise SynchronizationError(f"No sample available for {stream.key}")
            frame[f"observation.images.{stream.key}"] = sample.value
        frame["observation.state"] = np.zeros(8, dtype=np.float32)
        frame["action"] = np.zeros(8, dtype=np.float32)
        if self.config.action_format == "absolute_tcp_pose_xyzw_plus_gripper":
            frame["action"][6] = 1.0
            frame["observation.teleop_status"] = np.zeros(2, dtype=np.float32)
        frame["observation.xarm_velocity"] = np.zeros(7, dtype=np.float32)
        frame["observation.xarm_effort"] = np.zeros(7, dtype=np.float32)
        frame["observation.sync_skew_s"] = np.zeros(
            len(self.config.streams) + 4,
            dtype=np.float32,
        )
        return frame


def _require_near(sample, timestamp, limit, name) -> TimestampedSample:
    """Return a sample within an absolute timestamp limit or raise."""

    if sample is None:
        raise SynchronizationError(f"No sample available for {name}")
    delta = sample.timestamp - timestamp
    if abs(delta) > limit:
        raise SynchronizationError(
            f"{name} timestamp delta {delta:+.4f}s exceeds ±{limit:.4f}s"
        )
    return sample


def _require_causal(sample, timestamp, max_age, name) -> TimestampedSample:
    """Return a command active at ``timestamp`` without selecting future data."""

    if sample is None:
        raise SynchronizationError(f"No causal command available for {name}")
    age = timestamp - sample.timestamp
    if age < 0 or age > max_age:
        raise SynchronizationError(f"{name} age {age:.4f}s exceeds {max_age:.4f}s")
    return sample
