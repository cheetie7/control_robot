"""Validate the LeRobot recorder section of the unified system YAML."""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RecordingStream:
    """Describe one ROS image topic stored as a LeRobot video feature.

    Args:
        key: Dataset suffix used under ``observation.images``, as a string.
        topic: Absolute ROS image topic, as a string.
        kind: Either ``rgb`` or ``depth``.
        depth_unit: Physical input unit, ``mm`` or ``m``, for depth images.
        depth_min_m: Lower depth visualization bound in metres.
        depth_max_m: Upper depth visualization bound in metres.
        source_size: Optional topic ``(width, height)`` that frames must match
            when an ``output_size`` training view is configured.
        output_size: Optional stored ``(width, height)``; the recorder takes the
            largest aspect-preserving center crop of the topic frame and
            rescales it to exactly this size.
    """

    key: str
    topic: str
    kind: str
    depth_unit: str
    depth_min_m: float
    depth_max_m: float
    source_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class RecordingConfig:
    """Validated fixed-schema configuration for one LeRobot dataset session.

    Vector observations have shape ``(8,)`` for seven xArm joints and one
    normalized Robotiq joint. Video shapes are inferred from the first coherent
    frame and remain fixed for the complete dataset.
    """

    enabled: bool
    root: Path
    dataset_name: str
    append_timestamp: bool
    repo_id: str
    robot_type: str
    fps: int
    instruction: str
    num_episodes: int
    reset_joint_positions: tuple[float, ...]
    reset_action_name: str
    keyboard_enabled: bool
    keyboard_device: str
    auto_start_first_episode: bool
    anchor_key: str
    streams: tuple[RecordingStream, ...]
    action_history_sec: float
    max_state_skew_sec: float
    max_action_age_sec: float
    vcodec: str
    image_writer_threads: int
    streaming_encoding: bool
    writer_queue_size: int
    action_format: str = "absolute_joint_rad_plus_gripper"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        default_reset_joint_positions: list[float] | None = None,
    ) -> "RecordingConfig":
        """Build a recording configuration from one YAML mapping.

        Args:
            value: Scalar YAML mapping under ``recording`` or ``None``.

        Returns:
            Validated immutable configuration. Invalid paths, topics, stream
            kinds, seven-axis reset arrays or synchronization limits raise.
        """

        data = _mapping(value, "recording")
        enabled = _boolean(data.get("enabled", False), "recording.enabled")
        root_value = data.get("root", "/tmp/control_robot_lerobot")
        dataset_name = data.get("dataset_name", "")
        repo_id = data.get("repo_id", "local/control_robot")
        robot_type = data.get("robot_type", "xarm7_factr")
        instruction = data.get("instruction", data.get("task", "teleoperation"))
        for name, item in (
            ("root", root_value),
            ("repo_id", repo_id),
            ("robot_type", robot_type),
            ("instruction", instruction),
        ):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"recording.{name} must be a non-empty string")
        if not isinstance(dataset_name, str):
            raise ValueError("recording.dataset_name must be a string")
        if "/" in dataset_name or dataset_name in {".", ".."}:
            raise ValueError("recording.dataset_name must be one path-safe folder name")
        append_timestamp = _boolean(
            data.get("append_timestamp", False), "recording.append_timestamp"
        )
        if "task" in data and "instruction" in data and data["task"] != data["instruction"]:
            raise ValueError("recording.task and recording.instruction must not disagree")

        fps = _positive_integer(data.get("fps", 30), "recording.fps")
        if fps != 30:
            raise ValueError("recording.fps must be 30")
        num_episodes = _positive_integer(data.get("num_episodes", 1), "recording.num_episodes")
        reset_positions = data.get(
            "reset_joint_positions",
            default_reset_joint_positions
            if default_reset_joint_positions is not None
            else [0.0] * 7,
        )
        if not isinstance(reset_positions, list) or len(reset_positions) != 7:
            raise ValueError("recording.reset_joint_positions must contain 7 values")
        if any(not _finite_number(item) for item in reset_positions):
            raise ValueError("recording.reset_joint_positions must contain finite numbers")
        reset_action_name = data.get("reset_action_name", "/teleop/reset")
        if not isinstance(reset_action_name, str) or not reset_action_name.startswith("/"):
            raise ValueError("recording.reset_action_name must be an absolute ROS name")

        keyboard = _mapping(data.get("keyboard"), "recording.keyboard")
        keyboard_enabled = _boolean(
            keyboard.get("enabled", True), "recording.keyboard.enabled"
        )
        keyboard_device = keyboard.get("device", "/dev/tty")
        if not isinstance(keyboard_device, str) or not keyboard_device.startswith("/"):
            raise ValueError("recording.keyboard.device must be an absolute path")
        auto_start_first_episode = _boolean(
            keyboard.get("auto_start_first_episode", True),
            "recording.keyboard.auto_start_first_episode",
        )

        anchor = _mapping(data.get("anchor"), "recording.anchor")
        anchor_key = anchor.get("key", "front_camera")
        if not isinstance(anchor_key, str) or not anchor_key:
            raise ValueError("recording.anchor.key must be a non-empty string")

        stream_data = _mapping(data.get("streams"), "recording.streams")
        streams = tuple(
            _stream_from_mapping(key, _mapping(stream, f"recording.streams.{key}"))
            for key, stream in stream_data.items()
            if _boolean(
                _mapping(stream, f"recording.streams.{key}").get("enabled", True),
                f"recording.streams.{key}.enabled",
            )
        )
        if enabled and not streams:
            raise ValueError("recording.enabled requires at least one video stream")
        stream_keys = [stream.key for stream in streams]
        topics = [stream.topic for stream in streams]
        if len(stream_keys) != len(set(stream_keys)) or len(topics) != len(set(topics)):
            raise ValueError("recording stream keys and topics must be unique")
        if enabled and anchor_key not in stream_keys:
            raise ValueError("recording.anchor.key must name an enabled stream")
        if enabled:
            anchor_stream = streams[stream_keys.index(anchor_key)]
            if anchor_stream.kind != "rgb":
                raise ValueError("recording anchor stream must be rgb")

        sync = _mapping(data.get("sync"), "recording.sync")
        action_history = _positive_number(
            sync.get("action_history_sec", 1.0), "action_history_sec"
        )
        max_state_skew = _positive_number(
            sync.get("max_state_skew_sec", 0.025), "max_state_skew_sec"
        )
        max_action_age = _positive_number(
            sync.get("max_action_age_sec", 0.05), "max_action_age_sec"
        )

        video = _mapping(data.get("video"), "recording.video")
        vcodec = video.get("vcodec", "h264")
        if vcodec not in {"h264", "hevc", "libsvtav1", "auto"}:
            raise ValueError("recording.video.vcodec is unsupported")
        image_writer_threads = _positive_integer(
            video.get("image_writer_threads", 4), "recording.video.image_writer_threads"
        )
        streaming_encoding = _boolean(
            video.get("streaming_encoding", False),
            "recording.video.streaming_encoding",
        )
        writer_queue_size = _positive_integer(
            video.get("writer_queue_size", 16),
            "recording.video.writer_queue_size",
        )
        return cls(
            enabled=enabled,
            root=Path(root_value).expanduser(),
            dataset_name=dataset_name.strip(),
            append_timestamp=append_timestamp,
            repo_id=repo_id.strip(),
            robot_type=robot_type.strip(),
            fps=fps,
            instruction=instruction.strip(),
            num_episodes=num_episodes,
            reset_joint_positions=tuple(float(item) for item in reset_positions),
            reset_action_name=reset_action_name,
            keyboard_enabled=keyboard_enabled,
            keyboard_device=keyboard_device,
            auto_start_first_episode=auto_start_first_episode,
            anchor_key=anchor_key,
            streams=streams,
            action_history_sec=action_history,
            max_state_skew_sec=max_state_skew,
            max_action_age_sec=max_action_age,
            vcodec=vcodec,
            image_writer_threads=image_writer_threads,
            streaming_encoding=streaming_encoding,
            writer_queue_size=writer_queue_size,
        )

    @property
    def task(self) -> str:
        """Return the VLA instruction under LeRobot's standard ``task`` name."""

        return self.instruction

    def dataset_root(self, started_at: datetime | None = None) -> Path:
        """Return the concrete directory for one recorder process.

        Args:
            started_at: Optional local session time used only when
                ``append_timestamp`` is true.

        Returns:
            Full dataset directory ``root/dataset_name[_YYYYmmdd_HHMMSS]``.
        """

        base = self.root
        name = self.dataset_name
        if self.append_timestamp:
            session_time = started_at or datetime.now()
            if name:
                name = f"{name}_{session_time:%Y%m%d_%H%M%S}"
            else:
                return base.parent / f"{base.name}_{session_time:%Y%m%d_%H%M%S}"
        return base / name if name else base


def _stream_from_mapping(key: str, data: Mapping[str, Any]) -> RecordingStream:
    """Return one validated image-stream description from scalar YAML values."""

    if not isinstance(key, str) or not key or "/" in key:
        raise ValueError("recording stream keys must be non-empty path-safe strings")
    topic = data.get("topic")
    kind = data.get("type", "rgb")
    if not isinstance(topic, str) or not topic.startswith("/"):
        raise ValueError(f"recording stream '{key}' topic must be absolute")
    if kind not in {"rgb", "depth"}:
        raise ValueError(f"recording stream '{key}' type must be rgb or depth")
    depth_unit = data.get("depth_unit", "mm")
    if depth_unit not in {"mm", "m"}:
        raise ValueError(f"recording stream '{key}' depth_unit must be mm or m")
    depth_min = _positive_number(data.get("depth_min_m", 0.05), "depth_min_m")
    depth_max = _positive_number(data.get("depth_max_m", 3.0), "depth_max_m")
    if depth_min >= depth_max:
        raise ValueError(f"recording stream '{key}' depth_min_m must be below depth_max_m")
    source_size = _optional_size(data.get("source_size"), f"recording stream '{key}' source_size")
    output_size = _optional_size(data.get("output_size"), f"recording stream '{key}' output_size")
    if (source_size is None) != (output_size is None):
        raise ValueError(
            f"recording stream '{key}' source_size and output_size must be configured together"
        )
    return RecordingStream(
        key=key,
        topic=topic,
        kind=kind,
        depth_unit=depth_unit,
        depth_min_m=depth_min,
        depth_max_m=depth_max,
        source_size=source_size,
        output_size=output_size,
    )


def _optional_size(value: Any, name: str) -> tuple[int, int] | None:
    """Return one optional positive-integer ``(width, height)`` pair from YAML."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a [width, height] pair")
    if any(not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive integers")
    return (int(value[0]), int(value[1]))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return a YAML mapping or raise a field-specific error."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _boolean(value: Any, name: str) -> bool:
    """Return a strict YAML boolean rather than accepting truthy strings."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_number(value: Any) -> bool:
    """Return whether a scalar is a finite non-boolean number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: Any, name: str) -> float:
    """Return one positive finite float or raise a field-specific error."""

    if not _finite_number(value) or float(value) <= 0:
        raise ValueError(f"recording.sync.{name} must be a positive finite number")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    """Return one positive non-boolean integer or raise a field-specific error."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
