"""Aggregate robot and sensor health into standard ROS diagnostics."""

import json
import time
from functools import partial

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray, String

from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.monitor.health import ERROR, WARN, ComponentHealth
from xarm_ros2_teleop.system.config import load_system_config


def runtime_component_flags(
    system, operation_mode: str, motion_enabled: bool, inference_enabled: bool
) -> dict[str, bool]:
    """Return mode-aware enable flags for launch-managed runtime components.

    Args:
        system: Validated SystemConfig carrying scalar subsystem settings.
        operation_mode: One of ``teleop``, ``inference`` or ``rl``.
        motion_enabled: Whether launch explicitly armed physical actuators.
        inference_enabled: Whether the remote-policy node is part of the launch.

    Returns:
        Mapping from component identifier to whether that process is expected
        to publish in the selected runtime mode.
    """

    teleop_mode = operation_mode == "teleop"
    vr = system.teleop.get("source", "factr") == "vr"
    live_teleop = teleop_mode and (not vr or motion_enabled)
    rl_mode = operation_mode == "rl"
    evaluation_mode = bool(
        operation_mode == "inference"
        and motion_enabled
        and system.evaluation.get("enabled", False)
    )
    warmup_enabled = system.warmup.enabled and live_teleop
    return {
        "factr": not vr and bool(system.teleop.get("enabled", True)) and (teleop_mode or rl_mode),
        "vr": vr,
        "xarm": bool(system.xarm.get("enabled", True)) and (live_teleop or motion_enabled),
        "robotiq": bool(system.robotiq.get("enabled", False))
        and (live_teleop or motion_enabled),
        "recording": system.recording.enabled
        and (live_teleop or evaluation_mode or rl_mode and motion_enabled),
        "reset": bool(system.reset.get("enabled", False))
        and (live_teleop or evaluation_mode or rl_mode and motion_enabled),
        "rerun": system.dashboard.enabled and system.dashboard.rerun.enabled,
        "session": warmup_enabled,
        "speech": (teleop_mode or evaluation_mode)
        and (system.recording.enabled or system.warmup.enabled)
        and system.speech.enabled,
        "inference": inference_enabled,
        "rl": rl_mode,
    }


class SystemMonitor(Node):
    """Publish freshness and controller health for every configured component."""

    def __init__(self) -> None:
        """Load unified YAML, create dynamic subscriptions and start diagnostics."""

        super().__init__("system_monitor")
        self.declare_parameter("config_path", "")
        self.declare_parameter("operation_mode", "teleop")
        self.declare_parameter("motion_enabled", True)
        self.declare_parameter("inference_enabled", False)
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided by teleop.launch.py")
        self.system = load_system_config(config_path)
        self.operation_mode = str(self.get_parameter("operation_mode").value)
        if self.operation_mode not in {"teleop", "inference", "rl"}:
            raise ValueError("operation_mode must be teleop, inference or rl")
        self.motion_enabled = bool(self.get_parameter("motion_enabled").value)
        self.inference_enabled = bool(self.get_parameter("inference_enabled").value)
        self.started_at = time.monotonic()
        self.trackers: dict[str, ComponentHealth] = {}
        self.topic_trackers: dict[str, list[ComponentHealth]] = {}
        self.details: dict[str, dict[str, str]] = {}
        self.xarm_control_ok = None
        self.xarm_error = 0
        self.xarm_warn = 0
        self.recording_level = 0
        self._build_trackers()

        self.publisher = self.create_publisher(
            DiagnosticArray, "/system/diagnostics", 10
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool, "/xarm/control_ok", self.control_ok_callback, 10
        )
        self.create_subscription(
            Int32MultiArray, "/xarm/error_warn", self.error_callback, 10
        )
        self.create_subscription(
            DiagnosticArray,
            "/recording/metrics",
            self.recording_metrics_callback,
            latched_qos,
        )
        self.create_subscription(
            String, "/teleop/reset/status", self.reset_status_callback, latched_qos
        )
        self.create_subscription(
            String, "/visualization/rerun/status", self.rerun_status_callback, 10
        )
        self.create_subscription(
            String, "/system/session/status", self.session_status_callback, latched_qos
        )
        self.create_subscription(
            String, "/system/speech/status", self.speech_status_callback, latched_qos
        )
        self.create_subscription(
            String, "/inference/status", self.inference_status_callback, latched_qos
        )
        self.create_subscription(String, "/rl/status", self.rl_status_callback, latched_qos)
        self.create_timer(1.0 / self.system.dashboard.status_rate_hz, self.publish_diagnostics)

    def _build_trackers(self) -> None:
        """Create component definitions and subscriptions from validated YAML."""

        runtime_enabled = runtime_component_flags(
            self.system,
            self.operation_mode,
            self.motion_enabled,
            self.inference_enabled,
        )
        self._add_tracker("vr", "robot", runtime_enabled["vr"], True,
                          (("/vr/teleop_status", String),), 0.5)
        if runtime_enabled["vr"]:
            self.create_subscription(String, "/vr/teleop_status", self.vr_status_callback, 10)
        self._add_tracker(
            "factr",
            "robot",
            runtime_enabled["factr"],
            True,
            (("/factr/arm_joint_state", JointState),),
            0.5,
        )
        self._add_tracker(
            "xarm",
            "robot",
            runtime_enabled["xarm"],
            True,
            (("/xarm/arm_joint_state", JointState),),
            0.5,
        )
        self._add_tracker(
            "robotiq",
            "gripper",
            runtime_enabled["robotiq"],
            True,
            (("/robotiq/joint_state", JointState),),
            max(0.5, 5.0 * float(self.system.robotiq.get("control_period", 0.05))),
        )
        for camera in self.system.cameras:
            component_enabled = self.system.cameras_enabled and camera.enabled
            prefix = f"/{camera.topic_prefix}"
            topics = []
            if camera.color_enabled:
                topics.append((f"{prefix}/color", Image))
            if camera.depth_enabled:
                topics.append((f"{prefix}/depth", Image))
            if camera.infrared_enabled:
                topics.append((f"{prefix}/infra1", Image))
                topics.append((f"{prefix}/infra2", Image))
            if camera.point_cloud.enabled:
                topics.append((f"{prefix}/point_cloud", PointCloud2))
            self._add_tracker(
                f"camera/{camera.name}",
                "camera",
                component_enabled,
                camera.required,
                tuple(topics),
                1.0,
                camera.startup_delay_sec + 10.0,
                hardware_id=camera.serial_number,
            )
        for tactile in self.system.tactile_devices:
            component_enabled = self.system.tactile_enabled and tactile.enabled
            topics = tuple(
                (
                    f"/{tactile.topic_prefix}/{output}",
                    Image
                    if output in {"rectify", "difference", "depth"}
                    else Float32MultiArray,
                )
                for output in tactile.outputs
            )
            self._add_tracker(
                f"tactile/{tactile.name}",
                "tactile",
                component_enabled,
                tactile.required,
                topics,
                max(1.0, 3.0 / tactile.publish_rate_hz),
                tactile.startup_delay_sec + 10.0,
                hardware_id=tactile.serial_number,
            )
        self._add_tracker(
            "recording",
            "software",
            runtime_enabled["recording"],
            False,
            (("/recording/metrics", DiagnosticArray),),
            2.0,
        )
        self._add_tracker(
            "reset",
            "software",
            runtime_enabled["reset"],
            False,
            (("/teleop/reset/status", String),),
            None,
        )
        self._add_tracker(
            "rerun",
            "software",
            runtime_enabled["rerun"],
            False,
            (("/visualization/rerun/status", String),),
            2.0,
        )
        self._add_tracker(
            "session",
            "software",
            runtime_enabled["session"],
            True,
            (("/system/session/status", String),),
            None,
        )
        self._add_tracker(
            "speech",
            "software",
            runtime_enabled["speech"],
            False,
            (("/system/speech/status", String),),
            None,
        )
        self._add_tracker(
            "inference",
            "software",
            runtime_enabled["inference"],
            runtime_enabled["inference"],
            (("/inference/status", String),),
            None,
        )
        self._add_tracker(
            "rl",
            "software",
            runtime_enabled["rl"],
            runtime_enabled["rl"],
            (("/rl/status", String),),
            None,
        )

    def _add_tracker(
        self,
        component_id: str,
        kind: str,
        enabled: bool,
        required: bool,
        topics: tuple[tuple[str, type], ...],
        timeout_sec: float | None,
        startup_grace_sec: float = 10.0,
        *,
        hardware_id: str = "",
    ) -> None:
        """Register one component and subscribe to each physical output topic."""

        tracker = ComponentHealth(
            component_id=component_id,
            kind=kind,
            enabled=bool(enabled),
            required=bool(required),
            topics=tuple(topic for topic, _ in topics),
            timeout_sec=timeout_sec,
            startup_grace_sec=float(startup_grace_sec),
            started_at=self.started_at,
        )
        self.trackers[component_id] = tracker
        self.details[component_id] = {"hardware_id": hardware_id}
        for topic, message_type in topics:
            self.topic_trackers.setdefault(topic, []).append(tracker)
            if not enabled or component_id in {
                "recording",
                "reset",
                "rerun",
                "session",
                "speech",
                "inference",
                "rl",
            }:
                continue
            self.create_subscription(
                message_type,
                topic,
                partial(self.observe_topic, topic),
                qos_profile_sensor_data,
            )

    def observe_topic(self, topic: str, _message) -> None:
        """Mark one hardware output as received at the current monotonic time."""

        now = time.monotonic()
        for tracker in self.topic_trackers.get(topic, ()):
            tracker.observe(topic, now)

    def control_ok_callback(self, message: Bool) -> None:
        """Store the xArm controller's explicit command/read health flag."""

        self.xarm_control_ok = bool(message.data)

    def error_callback(self, message: Int32MultiArray) -> None:
        """Store scalar xArm controller error and warning codes."""

        if len(message.data) >= 2:
            self.xarm_error, self.xarm_warn = map(int, message.data[:2])

    def recording_metrics_callback(self, message: DiagnosticArray) -> None:
        """Store recorder metrics and refresh its software heartbeat."""

        self.observe_topic("/recording/metrics", message)
        if message.status:
            status = message.status[0]
            self.recording_level = _diagnostic_level(status.level)
            self.details["recording"] = {
                "hardware_id": status.hardware_id,
                "message": status.message,
                **{item.key: item.value for item in status.values},
            }

    def reset_status_callback(self, message: String) -> None:
        """Store the latest latched reset phase."""

        self.observe_topic("/teleop/reset/status", message)
        self.details["reset"]["state"] = message.data

    def rerun_status_callback(self, message: String) -> None:
        """Store the Rerun visualizer heartbeat payload."""

        self.observe_topic("/visualization/rerun/status", message)
        state, _, server_uri = message.data.partition("|")
        self.details["rerun"].update(state=state, server_uri=server_uri)

    def session_status_callback(self, message: String) -> None:
        """Store warm-up phase and refresh its latched software tracker."""

        self.observe_topic("/system/session/status", message)
        self.details["session"]["state"] = message.data

    def speech_status_callback(self, message: String) -> None:
        """Store local TTS health and refresh its latched software tracker."""

        self.observe_topic("/system/speech/status", message)
        state, _, detail = message.data.partition("|")
        self.details["speech"].update(state=state, detail=detail)

    def inference_status_callback(self, message: String) -> None:
        """Store the latest latched remote-policy rollout phase."""

        self.observe_topic("/inference/status", message)
        status = InferenceStatus.from_wire(message.data)
        self.details["inference"].update(state=status.phase, detail=status.reason)

    def rl_status_callback(self, message: String) -> None:
        """Store the stable RL state token and its operator-facing detail."""

        self.observe_topic("/rl/status", message)
        state, _, detail = message.data.partition("|")
        self.details["rl"].update(state=state.strip(), detail=detail.strip())

    def vr_status_callback(self, message: String) -> None:
        """Attach selected-controller readiness and reason to hardware diagnostics."""
        self.details["vr"].update(json.loads(message.data))

    def publish_diagnostics(self) -> None:
        """Evaluate all components and publish one complete diagnostic snapshot."""

        now = time.monotonic()
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        statuses = []
        for component_id, tracker in self.trackers.items():
            result = tracker.evaluate(now)
            level = result.level
            text = result.message
            details = dict(self.details[component_id])
            if component_id == "xarm" and tracker.enabled:
                details.update(
                    control_ok=str(self.xarm_control_ok),
                    error_code=str(self.xarm_error),
                    warn_code=str(self.xarm_warn),
                )
                if self.xarm_error:
                    level, text = ERROR, f"xArm error {self.xarm_error}"
                elif self.xarm_control_ok is False:
                    level, text = ERROR, "xArm control is not healthy"
            if component_id == "recording" and self.recording_level == ERROR:
                level, text = ERROR, details.get("last_error", "Recording failed")
            if component_id == "session" and details.get("state") == "FAILED":
                level, text = ERROR, "Warm-up failed"
            if component_id == "reset" and details.get("state") == "FAILED":
                level, text = ERROR, "Reset failed"
            if component_id == "speech" and details.get("state") == "ERROR":
                level, text = WARN, details.get("detail", "Speech failed")
            if component_id in {"inference", "rl"} and details.get("state") == "STOPPED":
                level, text = WARN, details.get("detail", "Rollout stopped")
            if component_id == "vr" and tracker.enabled:
                ready = details.get("ready", False)
                if not ready:
                    level, text = max(level, WARN), details.get("reason", "waiting for VR input")
                    details["state"] = "WAITING"
            status = DiagnosticStatus()
            status.name = component_id
            status.hardware_id = details.pop("hardware_id", "")
            # Humble represents ``diagnostic_msgs/DiagnosticStatus.level`` as
            # one byte rather than a Python integer.
            status.level = bytes([level])
            status.message = text
            values = {
                "state": result.state,
                "kind": tracker.kind,
                "enabled": tracker.enabled,
                "required": tracker.required,
                "age_sec": "" if result.age_sec is None else f"{result.age_sec:.3f}",
                "missing_topics": ",".join(result.missing_topics),
                **details,
            }
            status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
            statuses.append(status)
        message.status = statuses
        self.publisher.publish(message)


def _diagnostic_level(value: int | bytes) -> int:
    """Convert Humble's one-byte diagnostic level into an integer."""

    if isinstance(value, bytes):
        return value[0] if value else 0
    return int(value)


def main(args=None) -> None:
    """Run the health monitor with ROS arguments from the unified launch."""

    rclpy.init(args=args)
    node = SystemMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
