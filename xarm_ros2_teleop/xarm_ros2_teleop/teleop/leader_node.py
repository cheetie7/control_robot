import time

import numpy as np
import rclpy
from factr_teleop.dynamixel.driver import DynamixelCommunicationError
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from serial import SerialException
from std_msgs.msg import Bool, Float64

from xarm_ros2_teleop.teleop.controller import FACTRTeleop


def create_array_msg(data):
    """Convert scalar or one-dimensional joint data to a ROS ``JointState``.

    Args:
        data: Joint positions in radians, as a scalar or array-like ``(N,)`` value.

    Returns:
        ``JointState`` whose ``position`` field has shape ``(N,)``.
    """

    msg = JointState()
    msg.position = list(map(float, np.atleast_1d(data)))
    return msg


def should_hold_factr(
    reset_enabled: bool,
    reset_active: bool,
    session_gate_enabled: bool,
    session_ready: bool,
) -> bool:
    """Return whether low-current position control must own FACTR.

    Args:
        reset_enabled: Whether coordinated reset is configured.
        reset_active: Whether a reset transaction currently owns the actuators.
        session_gate_enabled: Whether warm-up gates normal teleoperation.
        session_ready: Whether warm-up has completed successfully.

    Returns:
        ``True`` during every reset phase and throughout pre-ready warm-up.
    """

    return bool(
        reset_enabled and (reset_active or (session_gate_enabled and not session_ready))
    )


class Factr_teleop_Ros2(FACTRTeleop):
    """Expose the working FACTR current controller through fixed xArm topics."""

    # Keep the rebased zero command alive long enough for the follower to
    # observe the independently delivered episode-active transition.
    ORIGIN_REBASE_ZERO_FRAMES = 20
    CONNECTION_HEARTBEAT_PERIOD_SEC = 1.0

    def __init__(self):
        """Initialize FACTR control and xArm command/feedback ROS interfaces."""

        try:
            super().__init__()
        except Exception:
            if hasattr(self, "driver"):
                self.shut_down()
            raise
        self.gripper_feedback_gain = self.config["controller"]["gripper_feedback"]["gain"]
        self.gripper_torque_ema_beta = self.config["controller"]["gripper_feedback"]["ema_beta"]
        self.gripper_external_torque = 0.0
        self.external_arm_torque = None
        self._publish_factr_connected(True, force=True)

    def set_up_communication(self):
        """Create ROS publishers and optional force-feedback subscriptions."""

        self.declare_parameter("reset_enabled", False)
        self.declare_parameter("xarm_initial_joint_positions", [0.0] * 7)
        self.declare_parameter("reset_state_publish_rate_hz", 60.0)
        self.declare_parameter("reset_max_current_ma", 150.0)
        self.declare_parameter("session_gate_enabled", False)
        self.declare_parameter("arm_command_topic", "/xarm/arm_pos")
        self.declare_parameter("gripper_command_topic", "/xarm/gripper_pos")
        self.declare_parameter("reconnect_period_sec", 1.0)
        self.declare_parameter("recovery_successful_reads", 5)
        self.reset_enabled = bool(self.get_parameter("reset_enabled").value)
        self.session_gate_enabled = bool(
            self.get_parameter("session_gate_enabled").value
        )
        self.session_ready = not self.session_gate_enabled
        self.xarm_initial_joint_positions = np.asarray(
            self.get_parameter("xarm_initial_joint_positions").value, dtype=float
        )
        state_rate = float(self.get_parameter("reset_state_publish_rate_hz").value)
        self.reset_state_publish_period = 1.0 / state_rate
        self.reset_max_current_ma = float(
            self.get_parameter("reset_max_current_ma").value
        )
        if not 0 < self.reset_max_current_ma <= 900:
            raise ValueError("reset_max_current_ma must be in (0, 900]")
        self.reset_active = False
        self.reset_aligned = False
        self.reset_align_requested = False
        self.episode_active = False
        self.factr_connected = True
        self.factr_reconnect_verifying = False
        self.factr_recovery_reads = 0
        self.factr_next_reconnect_time = 0.0
        self.last_connection_publish_time = 0.0
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )
        self.recovery_successful_reads = int(
            self.get_parameter("recovery_successful_reads").value
        )
        if self.reconnect_period_sec <= 0 or self.recovery_successful_reads < 1:
            raise ValueError("FACTR reconnect period and recovery reads must be positive")
        # A reset completion or idle-to-active transition re-seeds the publish
        # origin from the measured pose, so relative deltas restart at zero.
        self.origin_rebase_pending = False
        self.origin_zero_frames_remaining = 0
        self.last_state_publish_time = 0.0
        self.last_alignment_log_time = 0.0

        arm_command_topic = str(self.get_parameter("arm_command_topic").value)
        gripper_command_topic = str(self.get_parameter("gripper_command_topic").value)
        self.arm_pos_publisher = self.create_publisher(JointState, arm_command_topic, 10)
        self.gripper_pos_publisher = self.create_publisher(
            JointState, gripper_command_topic, 10
        )
        self.state_publisher = self.create_publisher(
            JointState, "/factr/arm_joint_state", 10
        )
        self.reset_aligned_publisher = self.create_publisher(
            Bool, "/teleop/reset/factr_aligned", 10
        )
        self.reset_active_sub = self.create_subscription(
            Bool, "/teleop/reset/active", self.reset_active_callback, 10
        )
        self.reset_align_sub = self.create_subscription(
            Bool, "/teleop/reset/align_factr", self.reset_align_callback, 10
        )
        session_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.connected_publisher = self.create_publisher(
            Bool, "/factr/connected", session_qos
        )
        self.session_ready_sub = self.create_subscription(
            Bool,
            "/system/session/ready",
            self.session_ready_callback,
            session_qos,
        )
        self.episode_active_sub = self.create_subscription(
            Bool,
            "/teleop/episode_active",
            self.episode_active_callback,
            session_qos,
        )
        if self.enable_torque_feedback:
            self.torque_feedback_sub = self.create_subscription(
                JointState, f"{self.name}/xarm/joint_state", self.arm_external_torque_callback, 10
            )
        if self.enable_gripper_feedback:
            self.gripper_torque_feedback_sub = self.create_subscription(
                Float64, f"{self.name}/gripper_torque", self._gripper_external_torque_callback, 10
            )

    def reset_active_callback(self, msg):
        """Enter or leave coordinated reset and schedule an origin rebase.

        On the reset's falling edge the publish origin re-seeds from wherever
        the operator is holding the leader, so relative deltas restart at zero
        and the follower stays at its initial pose without any jump.
        """

        active = bool(msg.data) and self.reset_enabled
        if active != self.reset_active and not active:
            self.origin_rebase_pending = True
        if active:
            self.reset_align_requested = False
            self._publish_reset_aligned(False)
        self.reset_active = active

    def reset_align_callback(self, msg):
        """Acknowledge FACTR alignment immediately; the leader is free-held.

        The operator positions the gravity-compensated leader by hand, so no
        servo alignment has to converge before the reset may complete.
        """

        self.reset_align_requested = bool(msg.data)
        if not self.reset_align_requested:
            self._publish_reset_aligned(False)
        elif self.reset_active and self.factr_connected:
            self._publish_reset_aligned(True)

    def control_loop_callback(self) -> None:
        """Run FACTR control or advance bounded serial reconnection.

        Expected USB/serial failures are converted into a latched health event
        and an internal one-second retry loop. Unexpected programming errors
        still escape and retain the launch-wide critical-process behavior.
        """

        now = time.monotonic()
        if not self.factr_connected:
            self._attempt_factr_reconnect(now)
            return
        try:
            super().control_loop_callback()
        except (DynamixelCommunicationError, SerialException, OSError) as exc:
            self._mark_factr_disconnected(exc, now)
            return
        if now - self.last_connection_publish_time >= self.CONNECTION_HEARTBEAT_PERIOD_SEC:
            self._publish_factr_connected(True, force=True)

    def _mark_factr_disconnected(self, error: Exception, now: float) -> None:
        """Latch one communication fault and safely schedule reconnect attempts.

        Args:
            error: Serial/Dynamixel exception raised by the 200 Hz control loop.
            now: Monotonic fault time in seconds.

        Returns:
            ``None``. No leader commands are published until recovery verifies
            several consecutive complete bus reads.
        """

        if not self.factr_connected:
            return
        self.factr_connected = False
        self.factr_reconnect_verifying = False
        self.factr_recovery_reads = 0
        self.factr_next_reconnect_time = now + self.reconnect_period_sec
        self.origin_rebase_pending = True
        self._publish_reset_aligned(False)
        self._publish_factr_connected(False, force=True)
        self.get_logger().error(
            f"FACTR 通信断开，暂停当前 episode 并后台重连: {error}"
        )
        if not self.driver.emergency_disable(retries=1):
            self.get_logger().error(
                "FACTR 串口不可用，无法确认零电流；请检查主手是否仍保持力矩"
            )
        self._close_factr_port()

    def _attempt_factr_reconnect(self, now: float) -> None:
        """Reopen FACTR and require consecutive valid reads before recovery.

        Args:
            now: Current monotonic time in seconds.

        Returns:
            ``None``. Recovery publishes ``/factr/connected=true`` only after
            ``recovery_successful_reads`` complete position/velocity reads.
        """

        if not self.factr_reconnect_verifying and now < self.factr_next_reconnect_time:
            return
        try:
            if not self.factr_reconnect_verifying:
                self.driver.reconnect_current_control()
                self.factr_reconnect_verifying = True
                self.factr_recovery_reads = 0
            self.driver.get_positions_and_velocities(tries=0)
            self.factr_recovery_reads += 1
        except (DynamixelCommunicationError, SerialException, OSError) as exc:
            self.factr_reconnect_verifying = False
            self.factr_recovery_reads = 0
            self.factr_next_reconnect_time = now + self.reconnect_period_sec
            self._close_factr_port()
            self.get_logger().warn(f"FACTR 重连尚未成功，将继续重试: {exc}")
            return
        if self.factr_recovery_reads < self.recovery_successful_reads:
            return
        self.factr_connected = True
        self.factr_reconnect_verifying = False
        self.factr_recovery_reads = 0
        self.origin_rebase_pending = True
        self.origin_zero_frames_remaining = self.ORIGIN_REBASE_ZERO_FRAMES
        self._publish_factr_connected(True, force=True)
        if self.reset_active and self.reset_align_requested:
            self._publish_reset_aligned(True)
        self.get_logger().info("FACTR 串口已恢复并通过连续读校验，等待 reset 完成")

    def _close_factr_port(self) -> None:
        """Best-effort close of a serial port already known to be unhealthy."""

        try:
            self.driver.close()
        except (SerialException, OSError) as exc:
            self.get_logger().warn(f"FACTR 关闭失效串口时返回错误，将继续重连: {exc}")

    def _publish_factr_connected(self, connected: bool, *, force: bool = False) -> None:
        """Publish the latched FACTR bus health and refresh its heartbeat time.

        Args:
            connected: Whether normal read/write transactions are verified.
            force: Publish even when the boolean state did not change.

        Returns:
            ``None``. The message is a scalar ROS ``Bool``.
        """

        if not force and bool(connected) == self.factr_connected:
            return
        message = Bool()
        message.data = bool(connected)
        self.connected_publisher.publish(message)
        self.last_connection_publish_time = time.monotonic()

    def session_ready_callback(self, msg):
        """Release warm-up position hold only after session readiness is true."""

        self.session_ready = bool(msg.data) or not self.session_gate_enabled

    def episode_active_callback(self, msg):
        """Rebase FACTR exactly when idle control first becomes active.

        Args:
            msg: Latched recorder gate. A rising edge means either direct
                recording or entry into staging; repeated ``True`` while
                transitioning from staging to recording preserves the pose.

        Returns:
            Nothing. The next FACTR commands form a short zero-delta window,
            preventing motion accumulated while xArm held in ``IDLE``.
        """

        active = bool(msg.data)
        if active and not self.episode_active:
            self.origin_rebase_pending = True
        self.episode_active = active

    def _publish_reset_aligned(self, aligned):
        """Publish reset alignment state only when its boolean value changes."""

        aligned = bool(aligned)
        if aligned == self.reset_aligned:
            return
        self.reset_aligned = aligned
        msg = Bool()
        msg.data = aligned
        self.reset_aligned_publisher.publish(msg)

    def update_state_communication(
        self, leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
    ):
        """Publish FACTR absolute arm position and velocity at the configured rate."""

        now = time.monotonic()
        if now - self.last_state_publish_time < self.reset_state_publish_period:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint{index + 1}" for index in range(self.num_arm_joints)]
        msg.position = [float(value) for value in leader_arm_pos]
        msg.velocity = [float(value) for value in leader_arm_vel]
        self.state_publisher.publish(msg)
        self.last_state_publish_time = now

    def run_reset_control(
        self, leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
    ):
        """Own FACTR during warm-up/reset with a gravity-compensated free hold.

        The leader stays in Dynamixel current mode the entire time and is
        never servo-driven: reset only carries the arm weight so the operator
        can reposition it by hand, while ``reset_max_current_ma`` bounds the
        hold current. Alignment is acknowledged immediately in
        ``reset_align_callback`` once the coordinator asks for it.

        Args:
            leader_arm_pos: Logical arm position array shape ``(7,)`` in rad.
            leader_arm_vel: Logical arm velocity array shape ``(7,)`` in rad/s.
            leader_gripper_pos: Logical gripper position scalar in rad.
            leader_gripper_vel: Logical gripper velocity scalar in rad/s.

        Returns:
            ``True`` while warm-up/reset owns control; ``False`` during normal
            teleoperation, which resumes plain gravity-compensated torque.
        """

        hold_requested = should_hold_factr(
            self.reset_enabled,
            self.reset_active,
            self.session_gate_enabled,
            self.session_ready,
        )
        if not hold_requested:
            return False
        torque_arm = np.zeros(self.num_arm_joints)
        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
        self.set_leader_joint_torque(
            torque_arm, 0.0, max_current_ma=self.reset_max_current_ma
        )
        return True

    def get_leader_arm_external_joint_torque(self):
        """Return follower torque in N·m as an array of shape ``(7,)``."""

        if self.external_arm_torque is None:
            return np.zeros(self.num_arm_joints)
        return self.external_arm_torque

    def arm_external_torque_callback(self, msg):
        """Store ``JointState.effort`` as a float array of shape ``(7,)``."""

        self.external_arm_torque = np.array(msg.effort, dtype=float)

    def _gripper_external_torque_callback(self, msg):
        """Low-pass filter one scalar follower gripper feedback sample."""

        gripper_external_torque = msg.data
        self.gripper_external_torque = (
            self.gripper_torque_ema_beta * self.gripper_external_torque
            + (1 - self.gripper_torque_ema_beta) * gripper_external_torque
        )

    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        """Convert scalar follower feedback into leader gripper torque in N·m."""

        torque_gripper = -1.0 * gripper_feedback / self.gripper_feedback_gain
        return torque_gripper

    def get_leader_gripper_feedback(self):
        """Return the filtered scalar follower gripper feedback value."""

        return self.gripper_external_torque

    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        """Publish arm ``(7,)`` and scalar gripper position commands in radians.

        While a reset owns control the published delta is pinned to zero, so
        the follower cannot move on stale pre-reset deltas. After an origin
        rebase, a short zero-command window spans the independently delivered
        activation event before relative deltas resume.
        """

        if self.reset_active:
            self.arm_pos_publisher.publish(
                create_array_msg(np.zeros(self.num_arm_joints))
            )
            self.gripper_pos_publisher.publish(create_array_msg([leader_gripper_pos]))
            return
        if self.origin_rebase_pending:
            self.leader_arm_publish_origin = np.asarray(
                leader_arm_pos, dtype=float
            ).copy()
            self.origin_rebase_pending = False
            self.origin_zero_frames_remaining = self.ORIGIN_REBASE_ZERO_FRAMES
            self.get_logger().info("FACTR 原点已重定基，相对增量从当前位姿起算")
        if self.origin_zero_frames_remaining > 0:
            # Absorb any hand motion during the short activation guard so it
            # cannot reappear as one accumulated step when the guard expires.
            self.leader_arm_publish_origin = np.asarray(
                leader_arm_pos, dtype=float
            ).copy()
            leader_arm_delta = np.zeros(self.num_arm_joints)
            self.origin_zero_frames_remaining -= 1
        else:
            leader_arm_delta = self.xarm_to_factr_signs * (
                leader_arm_pos - self.leader_arm_publish_origin
            )
        self.arm_pos_publisher.publish(create_array_msg(leader_arm_delta))
        self.gripper_pos_publisher.publish(create_array_msg([leader_gripper_pos]))


def main(args=None):
    """Run the leader node; ``args`` is an optional sequence of ROS arguments."""

    rclpy.init(args=args)
    rosnode = None
    try:
        rosnode = Factr_teleop_Ros2()
        rclpy.spin(rosnode)
    except KeyboardInterrupt:
        if rosnode is not None:
            rosnode.get_logger().info("键盘控制停止")
    finally:
        if rosnode is not None:
            rosnode.shut_down()
            rosnode.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
