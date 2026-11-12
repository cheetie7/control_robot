"""Parse and validate the unified robot, camera and tactile configuration."""

import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from xarm_ros2_teleop.dashboard.config import DashboardConfig
from xarm_ros2_teleop.inference.config import InferenceConfig, InferenceImageConfig
from xarm_ros2_teleop.recording.config import RecordingConfig
from xarm_ros2_teleop.system.session_config import SpeechConfig, WarmupConfig
from xarm_ros2_teleop.vr.config import VRConfig

_ROS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TOPIC_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_/]*$")
_SUPPORTED_CAMERAS = {
    ("realsense", "d435"),
    ("orbbec", "femto_bolt"),
}
_SUPPORTED_TACTILE_OUTPUTS = (
    "rectify",
    "difference",
    "depth",
    "marker_2d",
    "force",
    "force_norm",
    "force_resultant",
)


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for one image stream.

    Args:
        enabled: Whether the physical stream is published, as ``bool``.
        width: Optional image width in pixels, as ``int``.
        height: Optional image height in pixels, as ``int``.
        fps: Optional frame rate in frames per second, as ``int``.

    The stream has image shape ``(height, width, channels)`` when dimensions are
    specified. ``None`` dimensions select the vendor profile defaults.
    """

    enabled: bool
    width: int | None = None
    height: int | None = None
    fps: int | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None, *, default_enabled: bool
    ) -> "StreamConfig":
        """Build a stream config from a YAML mapping.

        Args:
            value: YAML mapping with scalar stream parameters, or ``None``.
            default_enabled: Default physical stream state, as ``bool``.

        Returns:
            A validated ``StreamConfig``. Optional image dimensions are scalar
            integers and therefore have no array shape.
        """

        data = dict(value or {})
        dimensions = tuple(data.get(key) for key in ("width", "height", "fps"))
        if any(item is not None for item in dimensions) and not all(
            item is not None for item in dimensions
        ):
            raise ValueError("width, height and fps must be configured together")
        if any(
            item is not None and (not isinstance(item, int) or item <= 0) for item in dimensions
        ):
            raise ValueError("width, height and fps must be positive integers")
        return cls(
            enabled=_as_bool(data.get("enabled", default_enabled), "stream.enabled"),
            width=dimensions[0],
            height=dimensions[1],
            fps=dimensions[2],
        )


@dataclass(frozen=True)
class CameraConfig:
    """Validated configuration for one physical RGB-D camera.

    Args:
        enabled: Whether launch opens this configured physical camera.
        name: Unique ROS-safe logical camera name.
        vendor: Driver family, ``realsense`` or ``orbbec``.
        model: Supported hardware model, ``d435`` or ``femto_bolt``.
        serial_number: Manufacturer serial number used for exclusive binding.
        topic_prefix: Relative ROS namespace used by canonical output topics.
        required: Whether missing or stale output is a system error instead of
            a degraded warning.
        startup_delay_sec: Absolute delay from launch start before opening the
            USB device, in seconds.
        color: Color image stream configuration.
        depth: Aligned depth image stream configuration.
        infrared: Stereo infrared pair (infra1/infra2) stream configuration.
        point_cloud: Colored point-cloud stream configuration.

    Image outputs use their configured ``(height, width)`` shape. Point clouds
    are unorganized or organized ``sensor_msgs/PointCloud2`` data depending on
    the vendor driver.
    """

    enabled: bool
    name: str
    vendor: str
    model: str
    serial_number: str
    topic_prefix: str
    required: bool
    startup_delay_sec: float
    color: StreamConfig
    depth: StreamConfig
    infrared: StreamConfig
    point_cloud: StreamConfig

    @property
    def infrared_enabled(self) -> bool:
        """Return the explicit stereo-infrared generation switch."""

        return self.infrared.enabled

    @property
    def color_enabled(self) -> bool:
        """Return the explicit color-stream generation switch."""

        return self.color.enabled

    @property
    def depth_enabled(self) -> bool:
        """Return the explicit depth-stream generation switch."""

        return self.depth.enabled

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraConfig":
        """Build one camera config from its YAML mapping.

        Args:
            value: YAML mapping describing one physical camera.

        Returns:
            A validated scalar camera configuration. Stream image shapes are
            carried by the nested ``StreamConfig`` values.
        """

        data = dict(value)
        name = _required_string(data, "name", "camera")
        vendor = _required_string(data, "vendor", "camera").lower()
        model = _required_string(data, "model", "camera").lower()
        serial_number = _required_string(data, "serial_number", "camera")
        topic_prefix = _normalize_topic_prefix(_required_string(data, "topic_prefix", "camera"))
        if not _ROS_NAME.fullmatch(name):
            raise ValueError(f"Invalid camera name '{name}'")
        if (vendor, model) not in _SUPPORTED_CAMERAS:
            raise ValueError(f"Unsupported camera '{vendor}/{model}'")
        delay = float(data.get("startup_delay_sec", 0.0))
        if delay < 0:
            raise ValueError("startup_delay_sec must be non-negative")
        color = StreamConfig.from_mapping(data.get("color"), default_enabled=True)
        depth = StreamConfig.from_mapping(data.get("depth"), default_enabled=False)
        infrared = StreamConfig.from_mapping(data.get("infrared"), default_enabled=False)
        point_cloud = StreamConfig.from_mapping(data.get("point_cloud"), default_enabled=False)
        if depth.enabled and not color.enabled:
            raise ValueError("Aligned depth requires color.enabled=true")
        if point_cloud.enabled and not (color.enabled and depth.enabled):
            raise ValueError(
                "Colored point cloud requires color.enabled=true and depth.enabled=true"
            )
        return cls(
            enabled=_as_bool(data.get("enabled", True), "camera.enabled"),
            name=name,
            vendor=vendor,
            model=model,
            serial_number=serial_number,
            topic_prefix=topic_prefix,
            required=_as_bool(data.get("required", True), "camera.required"),
            startup_delay_sec=delay,
            color=color,
            depth=depth,
            infrared=infrared,
            point_cloud=point_cloud,
        )


@dataclass(frozen=True)
class TactileConfig:
    """Validated configuration for one Xense tactile-vision sensor.

    Args:
        enabled: Whether launch opens this configured tactile sensor.
        name: Unique ROS-safe logical sensor name.
        serial_number: Xense hardware serial used for exact device binding.
        topic_prefix: Relative prefix for all published ROS topics.
        required: Whether missing or stale output is a system error instead of
            a degraded warning.
        startup_delay_sec: Absolute delay from launch start before opening the
            USB device, in seconds.
        publish_rate_hz: Acquisition and publication frequency in hertz.
        config_path: Optional Xense calibration file or directory path.
        outputs: Ordered tuple of enabled Xense output names.

    Image outputs have their native SDK shape. Tensor outputs are flattened in
    ROS messages while preserving their original dimensions in the layout.
    """

    enabled: bool
    name: str
    serial_number: str
    topic_prefix: str
    required: bool
    startup_delay_sec: float
    publish_rate_hz: float
    config_path: str
    outputs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TactileConfig":
        """Build one Xense sensor configuration from a YAML mapping.

        Args:
            value: Scalar YAML mapping for one physical tactile sensor.

        Returns:
            Validated ``TactileConfig`` with a one-dimensional output-name tuple.
        """

        data = dict(value)
        name = _required_string(data, "name", "tactile")
        serial_number = _required_string(data, "serial_number", "tactile")
        topic_prefix = _normalize_topic_prefix(_required_string(data, "topic_prefix", "tactile"))
        if not _ROS_NAME.fullmatch(name):
            raise ValueError(f"Invalid tactile name '{name}'")

        delay = float(data.get("startup_delay_sec", 0.0))
        publish_rate_hz = float(data.get("publish_rate_hz", 30.0))
        if delay < 0:
            raise ValueError("startup_delay_sec must be non-negative")
        if publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be positive")

        output_data = _mapping(data.get("outputs"), "tactile.outputs")
        unknown_outputs = set(output_data) - set(_SUPPORTED_TACTILE_OUTPUTS)
        if unknown_outputs:
            raise ValueError(f"Unsupported tactile outputs: {sorted(unknown_outputs)}")
        outputs = tuple(
            output_name
            for output_name in _SUPPORTED_TACTILE_OUTPUTS
            if _as_bool(
                output_data.get(output_name, output_name == "rectify"),
                f"tactile.outputs.{output_name}",
            )
        )
        if not outputs:
            raise ValueError("At least one tactile output must be enabled")

        config_path = data.get("config_path", "")
        if not isinstance(config_path, str):
            raise ValueError("tactile.config_path must be a string")
        return cls(
            enabled=_as_bool(data.get("enabled", True), "tactile.device.enabled"),
            name=name,
            serial_number=serial_number,
            topic_prefix=topic_prefix,
            required=_as_bool(data.get("required", True), "tactile.required"),
            startup_delay_sec=delay,
            publish_rate_hz=publish_rate_hz,
            config_path=config_path.strip(),
            outputs=outputs,
        )


@dataclass(frozen=True)
class SystemConfig:
    """Top-level teleoperation system configuration.

    Args:
        name: Human-readable system name.
        teleop: Scalar leader-node parameter mapping.
        xarm: Scalar follower-node parameter mapping.
        robotiq: Scalar direct Modbus RTU gripper parameter mapping.
        reset: Scalar multi-device reset parameter mapping.
        inference: Validated remote-policy network, timing and safety configuration.
        evaluation: Policy-evaluation episode, labeling and timeout settings.
        rl: Human intervention, sparse reward and online rollout parameters.
        cameras_enabled: Global camera subsystem switch.
        cameras: Tuple of enabled physical camera definitions.
        tactile_enabled: Global tactile subsystem switch.
        tactile_devices: Tuple of enabled Xense sensor definitions.
        recording: Validated fixed-schema LeRobot recorder configuration.
        dashboard: Validated local dashboard and Rerun configuration.
        warmup: Pre-recording sensor and actuator readiness configuration.
        speech: Local queued text-to-speech configuration.

    Returns no data directly; it describes the set of ROS processes to launch.
    """

    name: str
    teleop: Mapping[str, Any]
    xarm: Mapping[str, Any]
    robotiq: Mapping[str, Any]
    reset: Mapping[str, Any]
    inference: InferenceConfig
    evaluation: Mapping[str, Any]
    rl: Mapping[str, Any]
    cameras_enabled: bool
    cameras: tuple[CameraConfig, ...]
    tactile_enabled: bool
    tactile_devices: tuple[TactileConfig, ...]
    recording: RecordingConfig
    dashboard: DashboardConfig
    warmup: WarmupConfig
    speech: SpeechConfig


def load_system_config(path: str | Path) -> SystemConfig:
    """Load and validate a unified system YAML file.

    Args:
        path: Filesystem path to a UTF-8 YAML mapping, as ``str`` or ``Path``.

    Returns:
        ``SystemConfig`` containing scalar robot parameters and a one-dimensional
        tuples of enabled camera and tactile configurations.
    """

    raw = load_system_mapping(path)
    if not isinstance(raw, Mapping):
        raise ValueError("System configuration must be a YAML mapping")

    system = _mapping(raw.get("system"), "system")
    teleop = _mapping(raw.get("teleop"), "teleop")
    xarm = _mapping(raw.get("xarm"), "xarm")
    robotiq = _mapping(raw.get("robotiq"), "robotiq")
    reset = _mapping(raw.get("reset"), "reset")
    inference = InferenceConfig.from_mapping(raw.get("inference"))
    evaluation = _mapping(raw.get("evaluation"), "evaluation")
    rl = _mapping(raw.get("rl"), "rl")
    cameras_section = _mapping(raw.get("cameras"), "cameras")
    tactile_section = _mapping(raw.get("tactile"), "tactile")
    recording = RecordingConfig.from_mapping(
        raw.get("recording"),
        default_reset_joint_positions=xarm.get("initial_joint_positions"),
    )
    dashboard = DashboardConfig.from_mapping(raw.get("dashboard"))
    warmup = WarmupConfig.from_mapping(raw.get("warmup"))
    speech = SpeechConfig.from_mapping(raw.get("speech"))
    _as_bool(teleop.get("enabled", True), "teleop.enabled")
    source = teleop.get("source", "factr")
    if source not in {"factr", "vr"}:
        raise ValueError("teleop.source must be factr or vr")
    if source == "vr":
        VRConfig.from_mapping(_mapping(teleop.get("vr"), "teleop.vr"))
        recording = replace(recording, action_format="absolute_tcp_pose_xyzw_plus_gripper")
        if recording.robot_type == "xarm7_factr":
            recording = replace(recording, robot_type="xarm7_vr_tcp")
    _as_bool(xarm.get("enabled", True), "xarm.enabled")
    _as_bool(robotiq.get("enabled", False), "robotiq.enabled")
    _validate_robot_configs(xarm, robotiq)
    _validate_teleop_calibration(teleop)
    _validate_reset_config(teleop, xarm, robotiq, reset)
    _validate_evaluation_config(evaluation)
    _validate_rl_config(rl)
    cameras_enabled = _as_bool(cameras_section.get("enabled", False), "cameras.enabled")
    camera_devices = cameras_section.get("devices", [])
    tactile_enabled = _as_bool(tactile_section.get("enabled", False), "tactile.enabled")
    tactile_devices_raw = tactile_section.get("devices", [])
    if not isinstance(camera_devices, list):
        raise ValueError("cameras.devices must be a list")
    if not isinstance(tactile_devices_raw, list):
        raise ValueError("tactile.devices must be a list")

    cameras = tuple(
        CameraConfig.from_mapping(_mapping(device, "camera")) for device in camera_devices
    )
    tactile_devices = tuple(
        TactileConfig.from_mapping(_mapping(device, "tactile device"))
        for device in tactile_devices_raw
    )
    _validate_unique_cameras(cameras)
    _validate_unique_tactile_devices(tactile_devices)
    _validate_unique_sensor_names_and_topics(cameras, tactile_devices)
    enabled_cameras = tuple(camera for camera in cameras if cameras_enabled and camera.enabled)
    enabled_tactile = tuple(
        device for device in tactile_devices if tactile_enabled and device.enabled
    )
    inference = _resolve_inference_images(
        inference,
        recording,
        enabled_cameras,
        enabled_tactile,
    )
    recording = _validate_recording_topics(
        recording,
        enabled_cameras,
        enabled_tactile,
    )
    if recording.enabled and not all(
        (
            _as_bool(xarm.get("enabled", True), "xarm.enabled"),
            _as_bool(robotiq.get("enabled", False), "robotiq.enabled"),
        )
    ):
        raise ValueError("recording.enabled requires xarm and robotiq")
    if recording.enabled and not _as_bool(reset.get("enabled", False), "reset.enabled"):
        raise ValueError("episode recording reset requires reset.enabled")
    if dashboard.enabled and dashboard.rerun.enabled and not recording.streams:
        raise ValueError("dashboard.rerun.enabled requires at least one recording stream")
    if warmup.enabled and not _as_bool(reset.get("enabled", False), "reset.enabled"):
        raise ValueError("warmup.enabled requires reset.enabled")
    return SystemConfig(
        name=str(system.get("name", "control_robot")),
        teleop=teleop,
        xarm=xarm,
        robotiq=robotiq,
        reset=reset,
        inference=inference,
        evaluation=evaluation,
        rl=rl,
        cameras_enabled=cameras_enabled,
        cameras=cameras,
        tactile_enabled=tactile_enabled,
        tactile_devices=tactile_devices,
        recording=recording,
        dashboard=dashboard,
        warmup=warmup,
        speech=speech,
    )


def load_system_mapping(path, parents=()) -> dict:
    """Load a YAML profile with relative base_config, merging nested mappings.

    Lists replace, rather than concatenate, the base value. Cycle checks keep
    a malformed profile from recursing; returned data is the resolved config.
    """
    path = Path(path).expanduser().resolve()
    if path in parents:
        raise ValueError(f"Cyclic base_config: {path}")
    with path.open(encoding="utf-8") as source:
        data = yaml.safe_load(source)
    if not isinstance(data, dict):
        raise ValueError("System configuration must be a YAML mapping")
    base = data.pop("base_config", None)
    if base is None:
        return data
    result = load_system_mapping(path.parent / base, (*parents, path))
    _merge_profile(result, data)
    return result


def _merge_profile(base, overlay) -> None:
    """Update a resolved YAML dictionary in place; recurse only for mappings."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_profile(base[key], value)
        else:
            base[key] = value


def _validate_unique_cameras(cameras: tuple[CameraConfig, ...]) -> None:
    """Reject camera identifiers that would bind or publish ambiguously."""

    for label, values in (
        ("name", [camera.name for camera in cameras]),
        ("topic_prefix", [camera.topic_prefix for camera in cameras]),
        ("vendor/serial", [(camera.vendor, camera.serial_number) for camera in cameras]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate camera {label}")


def _validate_rl_config(rl: Mapping[str, Any]) -> None:
    """Validate human takeover, keyboard and sparse-reward scalar settings."""

    _as_bool(rl.get("enabled", False), "rl.enabled")
    _as_bool(rl.get("keyboard_enabled", True), "rl.keyboard_enabled")
    for key, default in (
        ("policy_action_topic", "/control/policy_action"),
        ("human_arm_topic", "/control/factr_arm"),
        ("human_gripper_topic", "/control/factr_gripper"),
        ("human_connected_topic", "/factr/connected"),
    ):
        value = rl.get(key, default)
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"rl.{key} must be an absolute ROS topic")
    device = rl.get("keyboard_device", "/dev/tty")
    if not isinstance(device, str) or not device.startswith("/"):
        raise ValueError("rl.keyboard_device must be an absolute path")
    for key, default in (
        ("stop_velocity_rad_s", 0.03),
        ("stop_stable_sec", 0.3),
        ("human_ready_stable_sec", 0.3),
        ("human_input_timeout_sec", 0.25),
        ("policy_timeout_sec", 0.25),
        ("max_human_joint_step_rad", 0.05),
    ):
        value = rl.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"rl.{key} must be a positive finite number")
    for key, default in (
        ("gripper_delta_scale", -1.0),
        ("success_reward", 1.0),
        ("failure_reward", 0.0),
        ("crash_reward", 0.0),
    ):
        value = rl.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"rl.{key} must be a finite number")


def _validate_evaluation_config(evaluation: Mapping[str, Any]) -> None:
    """Validate policy-evaluation episode limits and sparse outcome values."""

    _as_bool(evaluation.get("enabled", False), "evaluation.enabled")
    dataset_root = evaluation.get("dataset_root")
    if dataset_root is not None:
        if not isinstance(dataset_root, str) or not dataset_root.strip():
            raise ValueError("evaluation.dataset_root must be a non-empty string")
        if not Path(dataset_root).expanduser().is_absolute():
            raise ValueError("evaluation.dataset_root must be an absolute path")
    for key, default in (("max_episode_sec", 30.0),):
        value = evaluation.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"evaluation.{key} must be a positive finite number")
    max_episodes = evaluation.get("max_episodes", 100)
    if (
        not isinstance(max_episodes, int)
        or isinstance(max_episodes, bool)
        or max_episodes < 1
    ):
        raise ValueError("evaluation.max_episodes must be a positive integer")
    for key, default in (("success_reward", 1.0), ("failure_reward", 0.0)):
        value = evaluation.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"evaluation.{key} must be a finite number")


def _validate_robot_configs(xarm: Mapping[str, Any], robotiq: Mapping[str, Any]) -> None:
    """Validate physical robot limits before any launch process is created."""

    for key, default in (
        ("control_period", 0.01),
        ("state_read_rate_hz", 60.0),
        ("initial_joint_speed_rad_s", 0.5),
        ("max_joint_speed_rad_s", 1.0),
        ("max_joint_acceleration_rad_s2", 0.8),
        ("reset_lift_speed_mm_s", 50.0),
        ("reset_lift_skip_above_z_mm", 400.0),
    ):
        value = xarm.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"xarm.{key} must be a positive number")

    reset_lift_mm = xarm.get("reset_lift_mm", 100.0)
    if (
        not isinstance(reset_lift_mm, (int, float))
        or isinstance(reset_lift_mm, bool)
        or not math.isfinite(float(reset_lift_mm))
        or reset_lift_mm < 0
    ):
        raise ValueError("xarm.reset_lift_mm must be a non-negative finite number")

    if not _as_bool(robotiq.get("enabled", False), "robotiq.enabled"):
        return
    for key in ("com_port", "command_topic"):
        value = robotiq.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"robotiq.{key} must be a non-empty string")
    _as_bool(robotiq.get("debug", False), "robotiq.debug")
    _as_bool(
        robotiq.get("activate_on_startup", True),
        "robotiq.activate_on_startup",
    )
    device_id = robotiq.get("device_id", 9)
    if not isinstance(device_id, int) or isinstance(device_id, bool) or not 1 <= device_id <= 247:
        raise ValueError("robotiq.device_id must be an integer in [1, 247]")
    control_period = robotiq.get("control_period", 0.05)
    if (
        not isinstance(control_period, (int, float))
        or isinstance(control_period, bool)
        or control_period <= 0
    ):
        raise ValueError("robotiq.control_period must be a positive number")
    activation_timeout = robotiq.get("activation_timeout_sec", 15.0)
    if (
        not isinstance(activation_timeout, (int, float))
        or isinstance(activation_timeout, bool)
        or activation_timeout <= 0
    ):
        raise ValueError("robotiq.activation_timeout_sec must be a positive number")
    for key, default in (("speed", 255), ("force", 100), ("minimal_motion", 2)):
        value = robotiq.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
            raise ValueError(f"robotiq.{key} must be an integer in [0, 255]")


def _validate_teleop_calibration(teleop: Mapping[str, Any]) -> None:
    """Validate FACTR calibration paths and the cross-arm joint directions."""

    if (not _as_bool(teleop.get("enabled", True), "teleop.enabled")
            or teleop.get("source", "factr") == "vr"):
        return
    for key in ("offset_file", "config_file"):
        value = teleop.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"teleop.{key} must be a non-empty string")
    mapping_signs = teleop.get("xarm_to_factr_signs")
    if not isinstance(mapping_signs, list) or len(mapping_signs) != 7:
        raise ValueError("teleop.xarm_to_factr_signs must contain 7 values")
    if any(value not in {-1, 1} for value in mapping_signs):
        raise ValueError("teleop.xarm_to_factr_signs values must be -1 or 1")


def _validate_reset_config(
    teleop: Mapping[str, Any],
    xarm: Mapping[str, Any],
    robotiq: Mapping[str, Any],
    reset: Mapping[str, Any],
) -> None:
    """Validate multi-device reset settings before opening any hardware.

    Args:
        teleop: Leader component YAML mapping.
        xarm: Seven-axis follower YAML mapping.
        robotiq: Direct gripper YAML mapping.
        reset: Reset timing, tolerance and FACTR current-servo mapping.

    Returns:
        Nothing. Invalid component combinations, arrays or scalar limits raise
        ``ValueError`` before launch actions are created.
    """

    enabled = _as_bool(reset.get("enabled", False), "reset.enabled")
    xarm_enabled = _as_bool(xarm.get("enabled", True), "xarm.enabled")
    if not xarm_enabled and not enabled:
        return
    initial_positions = xarm.get("initial_joint_positions")
    if not isinstance(initial_positions, list) or len(initial_positions) != 7:
        raise ValueError("xarm.initial_joint_positions must contain 7 values")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in initial_positions
    ):
        raise ValueError("xarm.initial_joint_positions must contain finite numbers")
    if not enabled:
        return
    if not all(
        (
            _as_bool(teleop.get("enabled", True), "teleop.enabled"),
            _as_bool(xarm.get("enabled", True), "xarm.enabled"),
            _as_bool(robotiq.get("enabled", False), "robotiq.enabled"),
        )
    ):
        raise ValueError("reset.enabled requires teleop, xarm and robotiq")
    action_name = reset.get("action_name", "/teleop/reset")
    if not isinstance(action_name, str) or not action_name.startswith("/"):
        raise ValueError("reset.action_name must be an absolute ROS name")
    for key, default in (
        ("timeout_sec", 15.0),
        ("state_timeout_sec", 0.5),
        ("stable_time_sec", 0.3),
        ("xarm_position_tolerance_rad", 0.03),
        ("xarm_velocity_tolerance_rad_s", 0.05),
        ("factr_state_publish_rate_hz", 60.0),
        ("factr_max_current_ma", 150.0),
    ):
        value = reset.get(key, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"reset.{key} must be a positive finite number")
    if float(reset.get("factr_max_current_ma", 150.0)) > 900:
        raise ValueError("reset.factr_max_current_ma must not exceed 900")
    threshold = reset.get("robotiq_open_threshold_bits", 5)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or not 0 <= threshold <= 255:
        raise ValueError("reset.robotiq_open_threshold_bits must be an integer in [0, 255]")


def _validate_unique_tactile_devices(devices: tuple[TactileConfig, ...]) -> None:
    """Reject duplicate Xense serial numbers within the tactile subsystem."""

    serial_numbers = [device.serial_number for device in devices]
    if len(serial_numbers) != len(set(serial_numbers)):
        raise ValueError("Duplicate tactile serial_number")


def _validate_recording_topics(
    recording: RecordingConfig,
    cameras: tuple[CameraConfig, ...],
    tactile_devices: tuple[TactileConfig, ...],
) -> RecordingConfig:
    """Drop recording streams no enabled sensor produces, then check the anchor.

    Returns:
        ``RecordingConfig`` with unsupported streams pruned. Streams whose
        topics are not published by any enabled camera or tactile device are
        automatically disabled with a stderr warning, so a sensor can be
        switched off with its single ``enabled`` YAML flag. The anchor stream
        is required and raises when its topic is unavailable.
    """

    if not recording.enabled:
        return recording
    available_topics = set()
    camera_color_topics = {}
    for camera in cameras:
        prefix = f"/{camera.topic_prefix}"
        if camera.color_enabled:
            color_topic = f"{prefix}/color"
            available_topics.add(color_topic)
            camera_color_topics[color_topic] = camera.color.fps
        if camera.depth_enabled:
            available_topics.add(f"{prefix}/depth")
        if camera.infrared_enabled:
            available_topics.add(f"{prefix}/infra1")
            available_topics.add(f"{prefix}/infra2")
    for device in tactile_devices:
        available_topics.update(
            f"/{device.topic_prefix}/{output}"
            for output in device.outputs
            if output in {"rectify", "difference", "depth"}
        )
    kept = [stream for stream in recording.streams if stream.topic in available_topics]
    dropped = [stream.key for stream in recording.streams if stream.topic not in available_topics]
    if dropped:
        # Auto-pruning keeps sensor enable/disable a one-line YAML change:
        # recorders, visualizers and the dataset schema all derive from this
        # stream list.
        print(
            f"警告: 录制流 {', '.join(dropped)} 的传感器未启用，已自动禁用。",
            file=sys.stderr,
        )
    anchor = next((stream for stream in kept if stream.key == recording.anchor_key), None)
    if anchor is None or anchor.topic not in camera_color_topics:
        raise ValueError(
            f"recording anchor '{recording.anchor_key}' must use an enabled "
            "camera color topic"
        )
    anchor_fps = camera_color_topics[anchor.topic]
    if anchor_fps is not None and anchor_fps != recording.fps:
        raise ValueError(
            f"recording anchor '{recording.anchor_key}' source must run at "
            f"{recording.fps} FPS, got {anchor_fps}"
        )
    if len(kept) != len(recording.streams):
        return replace(recording, streams=tuple(kept))
    return recording


def _resolve_inference_images(
    inference: InferenceConfig,
    recording: RecordingConfig,
    cameras: tuple[CameraConfig, ...],
    tactile_devices: tuple[TactileConfig, ...],
) -> InferenceConfig:
    """Derive policy RGB observations from the same streams used for collection.

    Args:
        inference: Policy configuration containing an optional stream-key subset.
        recording: LeRobot stream schema defining keys, topics and crop sizes.
        cameras: Physical cameras enabled by the unified YAML.
        tactile_devices: Physical tactile sensors enabled by the unified YAML.

    Returns:
        Inference configuration whose images contain only RGB recording streams
        produced by enabled sensors. An empty ``image_streams`` selection means
        all such streams; an explicit selection must be available in full.
    """

    available_topics = set()
    for camera in cameras:
        prefix = f"/{camera.topic_prefix}"
        if camera.color_enabled:
            available_topics.add(f"{prefix}/color")
    for device in tactile_devices:
        available_topics.update(
            f"/{device.topic_prefix}/{output}"
            for output in device.outputs
            if output in {"rectify", "difference"}
        )
    rgb_streams = {
        stream.key: stream
        for stream in recording.streams
        if stream.kind == "rgb" and stream.topic in available_topics
    }
    selected_keys = inference.image_streams or tuple(rgb_streams)
    unavailable = [key for key in selected_keys if key not in rgb_streams]
    if unavailable:
        raise ValueError(
            "inference.image_streams are not enabled RGB sensor streams: " + ", ".join(unavailable)
        )
    images = tuple(
        InferenceImageConfig(
            key=rgb_streams[key].key,
            topic=rgb_streams[key].topic,
            source_size=rgb_streams[key].source_size,
            output_size=rgb_streams[key].output_size,
        )
        for key in selected_keys
    )
    return inference.with_images(images)


def _validate_unique_sensor_names_and_topics(
    cameras: tuple[CameraConfig, ...], tactile_devices: tuple[TactileConfig, ...]
) -> None:
    """Prevent camera and tactile nodes from sharing logical names or topics."""

    devices = (*cameras, *tactile_devices)
    for label, values in (
        ("name", [device.name for device in devices]),
        ("topic_prefix", [device.topic_prefix for device in devices]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate sensor {label}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a field-specific error."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_string(data: Mapping[str, Any], key: str, section: str) -> str:
    """Read a required non-empty string from one named YAML section."""

    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _normalize_topic_prefix(value: str) -> str:
    """Normalize a relative topic prefix and reject ambiguous ROS names."""

    prefix = value.strip("/")
    if not prefix or "//" in prefix or not _TOPIC_PREFIX.fullmatch(prefix):
        raise ValueError(f"Invalid topic_prefix '{value}'")
    return prefix


def _as_bool(value: Any, name: str) -> bool:
    """Return a strict YAML boolean rather than accepting truthy strings."""

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
