"""Validated configuration for real-robot remote policy inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class InferenceImageConfig:
    """Describe one RGB observation sent to the policy server.

    Attributes:
        key: Model-side image key, identical to the LeRobot recording stream key.
        topic: Absolute ROS Image topic containing the physical camera stream.
        source_size: Optional configured source ``(width, height)`` in pixels.
        output_size: Optional center-cropped output ``(width, height)`` in pixels.
    """

    key: str
    topic: str
    source_size: tuple[int, int] | None
    output_size: tuple[int, int] | None


@dataclass(frozen=True)
class InferenceConfig:
    """Store the network, timing, observation and motion-safety contract.

    Policy state and action arrays both have shape ``(8,)`` and are ordered as
    seven absolute xArm joint angles in radians followed by normalized Robotiq
    position, where zero is open and one is closed.
    """

    enabled: bool
    dry_run: bool
    host: str
    port: int
    prompt: str
    policy_frequency_hz: float
    inference_timeout_sec: float
    observation_timeout_sec: float
    max_camera_skew_sec: float
    max_runtime_sec: float
    max_chunks: int
    max_joint_step_rad: float
    joint_lower_rad: tuple[float, ...]
    joint_upper_rad: tuple[float, ...]
    image_streams: tuple[str, ...]
    images: tuple[InferenceImageConfig, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "InferenceConfig":
        """Parse one YAML mapping and reject unsafe or ambiguous values.

        Args:
            value: Inference YAML mapping or ``None`` when inference is disabled.

        Returns:
            Immutable validated configuration. Image topics and transforms are
            resolved later from the shared recording/sensor configuration.
        """

        data = {} if value is None else value
        if not isinstance(data, Mapping):
            raise ValueError("inference must be a mapping")
        enabled = _boolean(data.get("enabled", False), "inference.enabled")
        dry_run = _boolean(data.get("dry_run", True), "inference.dry_run")
        host = _nonempty_string(data.get("host", "127.0.0.1"), "inference.host")
        prompt = _nonempty_string(data.get("prompt", "complete the task"), "inference.prompt")
        port = data.get("port", 8000)
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("inference.port must be an integer in [1, 65535]")

        positive_values = {
            "policy_frequency_hz": float(data.get("policy_frequency_hz", 25.0)),
            "inference_timeout_sec": float(data.get("inference_timeout_sec", 30.0)),
            "observation_timeout_sec": float(data.get("observation_timeout_sec", 0.25)),
            "max_camera_skew_sec": float(data.get("max_camera_skew_sec", 0.08)),
            "max_runtime_sec": float(data.get("max_runtime_sec", 120.0)),
            "max_joint_step_rad": float(data.get("max_joint_step_rad", 0.5)),
        }
        for name, scalar in positive_values.items():
            if not math.isfinite(scalar) or scalar <= 0:
                raise ValueError(f"inference.{name} must be a positive finite number")
        max_chunks = data.get("max_chunks", 100)
        if not isinstance(max_chunks, int) or isinstance(max_chunks, bool) or max_chunks < 1:
            raise ValueError("inference.max_chunks must be a positive integer")

        lower = _joint_vector(
            data.get(
                "joint_lower_rad", [-6.2832, -2.059, -6.2832, -0.19198, -6.2832, -1.69297, -6.2832]
            ),
            "inference.joint_lower_rad",
        )
        upper = _joint_vector(
            data.get("joint_upper_rad", [6.2832, 2.0944, 6.2832, 3.927, 6.2832, 3.1416, 6.2832]),
            "inference.joint_upper_rad",
        )
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("inference joint lower limits must be below upper limits")

        image_streams_value = data.get("image_streams", [])
        if not isinstance(image_streams_value, list) or any(
            not isinstance(item, str) or not item.strip() for item in image_streams_value
        ):
            raise ValueError("inference.image_streams must be a list of non-empty strings")
        image_streams = tuple(item.strip() for item in image_streams_value)
        if len(image_streams) != len(set(image_streams)):
            raise ValueError("inference.image_streams must not contain duplicates")
        return cls(
            enabled=enabled,
            dry_run=dry_run,
            host=host,
            port=port,
            prompt=prompt,
            max_chunks=max_chunks,
            joint_lower_rad=lower,
            joint_upper_rad=upper,
            image_streams=image_streams,
            images=(),
            **positive_values,
        )

    def with_images(self, images: tuple[InferenceImageConfig, ...]) -> "InferenceConfig":
        """Return this policy configuration with shared YAML streams resolved.

        Args:
            images: RGB stream descriptors derived from enabled physical
                sensors and ``recording.streams``.

        Returns:
            A copy whose observation keys, topics and crop transforms match
            collection. Enabled inference requires at least one RGB stream.
        """

        if self.enabled and not images:
            raise ValueError("inference requires at least one enabled RGB sensor stream")
        return replace(self, images=images)


def _joint_vector(value: Any, name: str) -> tuple[float, ...]:
    """Return seven finite joint-limit values in radians."""

    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{name} must contain seven values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _boolean(value: Any, name: str) -> bool:
    """Return a strict YAML boolean."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    """Return one stripped non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
