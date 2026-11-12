"""Gate recording until sensors and all reset-controlled actuators are ready."""

import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray, KeyValue
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectoryPoint

from xarm_ros2_teleop.system.config import load_system_config


class SessionCoordinator(Node):
    """Run warm-up and publish the authoritative recorder-ready gate.

    Hardware readiness comes from ``/system/diagnostics``. Once FACTR, xArm,
    Robotiq and every selected sensor are stable, a reset Action moves xArm to
    the shared seven-radian initial pose, aligns FACTR and opens Robotiq. The
    ready gate is true only after the settling delay and final announcement.
    """

    def __init__(self) -> None:
        """Load YAML, create ROS interfaces and enter ``WAITING_HARDWARE``."""

        super().__init__("session_coordinator")
        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided by teleop.launch.py")
        self.system = load_system_config(config_path)
        self.config = self.system.warmup
        self.speech = self.system.speech
        self.initial_joint_positions = tuple(
            float(value) for value in self.system.xarm["initial_joint_positions"]
        )
        self.started_at = time.monotonic()
        self.hardware_stable_since = None
        self.settle_until = None
        self.diagnostics_received = False
        self.component_states: dict[str, tuple[str, bool, bool]] = {}
        self.pending_announcement = None
        self.announcement_deadline = None
        self.reset_goal_pending = False
        self.reset_goal_handle = None
        self.reset_deadline = None
        self.reset_timeout_sec = float(self.system.reset.get("timeout_sec", 15.0)) + 2.0
        self.phase = "WAITING_HARDWARE"

        qos = QoSProfile(
            depth=20,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.ready_pub = self.create_publisher(Bool, "/system/session/ready", qos)
        self.status_pub = self.create_publisher(String, "/system/session/status", qos)
        self.announcement_pub = self.create_publisher(
            KeyValue, "/system/announcement/request", qos
        )
        self.create_subscription(
            DiagnosticArray,
            "/system/diagnostics",
            self.diagnostics_callback,
            10,
        )
        self.create_subscription(
            String,
            "/system/announcement/finished",
            self.announcement_finished_callback,
            qos,
        )
        self.reset_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.system.reset.get("action_name", "/teleop/reset")),
        )
        self.create_timer(0.1, self.control_loop)
        self._publish_ready(False)
        self._set_status("WAITING_HARDWARE")
        self._announce("warmup_started", self.speech.phrase("warmup_started"))

    def diagnostics_callback(self, message: DiagnosticArray) -> None:
        """Store actuator and enabled-sensor health from one snapshot."""

        component_states = {}
        for status in message.status:
            is_sensor = status.name.startswith(("camera/", "tactile/"))
            if not is_sensor and status.name not in {"factr", "vr", "xarm", "robotiq"}:
                continue
            values = {item.key: item.value for item in status.values}
            component_states[status.name] = (
                values.get("state", "UNKNOWN"),
                values.get("enabled", "False").lower() == "true",
                True if not is_sensor else values.get("required", "False").lower() == "true",
            )
        self.component_states = component_states
        self.diagnostics_received = True

    def announcement_finished_callback(self, message: String) -> None:
        """Unlock recording after the final warm-up phrase has completed."""

        if message.data != self.pending_announcement:
            return
        self.pending_announcement = None
        self.announcement_deadline = None
        if self.phase == "ANNOUNCING_READY":
            self._finish_warmup()

    def control_loop(self) -> None:
        """Advance hardware wait, reset, settling and announcement phases."""

        now = time.monotonic()
        if (
            self.phase == "RESETTING"
            and self.reset_deadline is not None
            and now >= self.reset_deadline
        ):
            if self.reset_goal_handle is not None:
                self.reset_goal_handle.cancel_goal_async()
            self.get_logger().error("warm-up reset 响应超时")
            self.reset_deadline = None
            self._set_status("FAILED")
            return
        if self.phase == "WAITING_HARDWARE":
            if now - self.started_at > self.config.hardware_timeout_sec:
                missing = ", ".join(self._missing_components()) or "diagnostics"
                self.get_logger().error(f"warm-up 等待硬件超时: {missing}")
                self._set_status("FAILED")
                return
            if not self._hardware_ready():
                self.hardware_stable_since = None
                return
            if self.hardware_stable_since is None:
                self.hardware_stable_since = now
                return
            if now - self.hardware_stable_since >= self.config.sensor_ready_stable_time_sec:
                self._start_initial_reset()
            return
        if self.phase == "SETTLING" and now >= self.settle_until:
            event_id = self._announce(
                "warmup_finished", self.speech.phrase("warmup_finished")
            )
            if event_id is None:
                self._finish_warmup()
            else:
                self.pending_announcement = event_id
                self.announcement_deadline = now + self.speech.timeout_sec + 1.0
                self._set_status("ANNOUNCING_READY")
            return
        if (
            self.phase == "ANNOUNCING_READY"
            and self.announcement_deadline is not None
            and now >= self.announcement_deadline
        ):
            self.get_logger().warn("warm-up finished 语音未确认，继续解锁录制")
            self._finish_warmup()

    def _hardware_ready(self) -> bool:
        """Return true when actuators and every selected sensor report ``OK``."""

        if not self.diagnostics_received:
            return False
        selected = [
            state
            for state, enabled, required in self.component_states.values()
            if enabled and (self.config.require_all_enabled_devices or required)
        ]
        return all(state == "OK" for state in selected)

    def _missing_components(self) -> list[str]:
        """Return selected hardware names that have not reached ``OK``."""

        return [
            name
            for name, (state, enabled, required) in self.component_states.items()
            if enabled
            and (self.config.require_all_enabled_devices or required)
            and state != "OK"
        ]

    def _start_initial_reset(self) -> None:
        """Send the shared xArm initial pose to the reset coordinator once."""

        if self.reset_goal_pending:
            return
        if not self.reset_client.wait_for_server(timeout_sec=0.0):
            return
        self.reset_goal_pending = True
        self.reset_deadline = time.monotonic() + self.reset_timeout_sec
        self._set_status("RESETTING")
        self._announce("warmup_reset", self.speech.phrase("reset_started"))
        future = self.reset_client.send_goal_async(self._reset_goal())
        future.add_done_callback(self._reset_goal_response)

    def _reset_goal(self) -> FollowJointTrajectory.Goal:
        """Build one absolute position-only reset goal with seven radians."""

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f"joint{index + 1}" for index in range(7)]
        point = JointTrajectoryPoint()
        point.positions = list(self.initial_joint_positions)
        goal.trajectory.points = [point]
        return goal

    def _reset_goal_response(self, future) -> None:
        """Attach the result callback or fail warm-up when the goal is rejected."""

        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"warm-up reset 请求失败: {exc}")
            self._set_status("FAILED")
            return
        if self.phase != "RESETTING":
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.get_logger().error("warm-up reset goal 被拒绝")
            self._set_status("FAILED")
            return
        self.reset_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._reset_result)

    def _reset_result(self, future) -> None:
        """Start the two-second stable hold after all reset devices succeed."""

        if self.phase != "RESETTING":
            return
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f"warm-up reset 结果失败: {exc}")
            self._set_status("FAILED")
            return
        self.reset_deadline = None
        self.reset_goal_handle = None
        successful = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        )
        if not successful:
            self.get_logger().error(
                f"warm-up reset 失败: {wrapped.result.error_string}"
            )
            self._set_status("FAILED")
            return
        self._announce("warmup_reset_finished", self.speech.phrase("reset_finished"))
        self.settle_until = time.monotonic() + self.config.finished_delay_sec
        self._set_status("SETTLING")

    def _announce(self, event: str, phrase: str) -> str | None:
        """Publish one uniquely identified phrase, or return ``None`` if disabled."""

        if not self.speech.enabled:
            return None
        event_id = f"session:{event}:{time.monotonic_ns()}"
        message = KeyValue(key=event_id, value=phrase)
        self.announcement_pub.publish(message)
        return event_id

    def _finish_warmup(self) -> None:
        """Publish READY only after reset, settling and final announcement."""

        self.pending_announcement = None
        self.announcement_deadline = None
        self._set_status("READY")
        self._publish_ready(True)
        self.get_logger().info("warm-up 完成，录制控制已激活")

    def _publish_ready(self, ready: bool) -> None:
        """Publish the authoritative latched recorder and actuator gate."""

        message = Bool()
        message.data = bool(ready)
        self.ready_pub.publish(message)

    def _set_status(self, status: str) -> None:
        """Publish one scalar session lifecycle state."""

        self.phase = status
        message = String()
        message.data = status
        self.status_pub.publish(message)


def main(args=None) -> None:
    """Run warm-up coordination with ROS futures and health callbacks."""

    rclpy.init(args=args)
    node = SessionCoordinator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
