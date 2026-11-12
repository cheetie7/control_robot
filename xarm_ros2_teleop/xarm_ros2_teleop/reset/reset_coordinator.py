"""Coordinate xArm, FACTR and Robotiq episode resets through ROS 2."""

import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, Int32MultiArray, String
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES = tuple(f"joint{index + 1}" for index in range(7))
ACTIVE_PHASES = {
    "CLEARING_ERROR",
    "MOVING_XARM",
    "WAITING_FACTR",
    "ALIGNING_FACTR",
}


def normalize_reset_goal(request, default_tolerance, default_timeout):
    """Validate one reset Action goal and normalize it to canonical joint order.

    Args:
        request: ``FollowJointTrajectory.Goal`` containing exactly one point.
        default_tolerance: Positive default xArm tolerance in rad.
        default_timeout: Positive default reset timeout in seconds.

    Returns:
        Tuple ``(target, tolerances, timeout)`` where target and tolerances are
        finite float arrays with shape ``(7,)`` and timeout is a scalar second value.
    """

    trajectory = request.trajectory
    names = list(trajectory.joint_names)
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(JOINT_NAMES):
        raise ValueError("joint_names must contain joint1 through joint7 exactly once")
    if len(trajectory.points) != 1:
        raise ValueError("reset accepts exactly one trajectory point")
    point = trajectory.points[0]
    if point.velocities or point.accelerations or point.effort:
        raise ValueError("reset point must contain positions only")
    if len(point.positions) != 7:
        raise ValueError("reset point must contain seven positions")
    index_by_name = {name: index for index, name in enumerate(names)}
    target = np.asarray([point.positions[index_by_name[name]] for name in JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(target)):
        raise ValueError("reset positions must be finite")
    if request.path_tolerance or request.component_path_tolerance:
        raise ValueError("path tolerances are not supported by joint-space reset")
    if request.component_goal_tolerance:
        raise ValueError("component goal tolerances are not supported for revolute joints")

    tolerances = np.full(7, float(default_tolerance), dtype=float)
    for tolerance in request.goal_tolerance:
        if tolerance.name not in JOINT_NAMES or tolerance.position < 0:
            raise ValueError("goal tolerances must name joint1 through joint7")
        if tolerance.position > 0:
            tolerances[JOINT_NAMES.index(tolerance.name)] = tolerance.position
    timeout_extension = (
        float(request.goal_time_tolerance.sec) + float(request.goal_time_tolerance.nanosec) * 1e-9
    )
    timeout = float(default_timeout) + timeout_extension
    return target, tolerances, timeout


class ResetCoordinator(Node):
    """Coordinate one reset Action across the follower, leader and gripper.

    Main attributes hold the latest xArm state arrays with shape ``(7,)``, the
    configured initial pose in radians, and a scalar phase string. Hardware is
    controlled indirectly through ROS topics; this node never opens a device.
    """

    def __init__(self):
        """Read reset parameters, create ROS interfaces and start the state timer."""

        super().__init__("reset_coordinator")
        self.declare_parameter("action_name", "/teleop/reset")
        self.declare_parameter("timeout_sec", 15.0)
        self.declare_parameter("state_timeout_sec", 0.5)
        self.declare_parameter("stable_time_sec", 0.3)
        self.declare_parameter("xarm_position_tolerance_rad", 0.03)
        self.declare_parameter("xarm_velocity_tolerance_rad_s", 0.05)
        self.declare_parameter("robotiq_open_threshold_bits", 5)
        self.declare_parameter("initial_joint_positions", [0.0] * 7)
        self.declare_parameter("require_factr_alignment", True)
        self.action_name = str(self.get_parameter("action_name").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.state_timeout_sec = float(self.get_parameter("state_timeout_sec").value)
        self.stable_time_sec = float(self.get_parameter("stable_time_sec").value)
        self.position_tolerance = float(self.get_parameter("xarm_position_tolerance_rad").value)
        self.velocity_tolerance = float(self.get_parameter("xarm_velocity_tolerance_rad_s").value)
        self.robotiq_open_threshold = int(self.get_parameter("robotiq_open_threshold_bits").value)
        self.initial_joint_positions = np.asarray(
            self.get_parameter("initial_joint_positions").value, dtype=float
        )
        self.require_factr_alignment = bool(
            self.get_parameter("require_factr_alignment").value
        )

        self.lock = threading.RLock()
        self.phase = "IDLE"
        self.reset_active = False
        self.reset_target = None
        self.goal_tolerances = np.full(7, self.position_tolerance)
        self.reset_started_at = None
        self.reset_deadline = None
        self.xarm_stable_since = None
        self.gripper_stable_since = None
        self.latest_xarm_position = None
        self.latest_xarm_velocity = None
        self.latest_xarm_time = None
        self.latest_error_code = None
        self.control_ok = False
        self.xarm_target_valid = None
        self.factr_aligned = not self.require_factr_alignment
        self.factr_connected = self.require_factr_alignment
        self.factr_wait_started = None
        self.robotiq_position = None
        self.robotiq_state_time = None
        self.goal_reserved = False
        self.active_goal_handle = None
        self.goal_done = threading.Event()
        self.terminal_kind = None
        self.terminal_code = FollowJointTrajectory.Result.SUCCESSFUL
        self.terminal_message = ""
        self.cancel_requested = False
        self.error_during_goal = False
        self.xarm_motion_started = False

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.active_pub = self.create_publisher(Bool, "/teleop/reset/active", latched_qos)
        self.target_pub = self.create_publisher(
            JointState, "/teleop/reset/xarm_target", latched_qos
        )
        self.align_pub = self.create_publisher(Bool, "/teleop/reset/align_factr", latched_qos)
        self.status_pub = self.create_publisher(String, "/teleop/reset/status", latched_qos)
        self.create_subscription(JointState, "/xarm/arm_joint_state", self.xarm_state_callback, 10)
        self.create_subscription(Int32MultiArray, "/xarm/error_warn", self.error_callback, 10)
        self.create_subscription(Bool, "/xarm/control_ok", self.control_ok_callback, 10)
        self.create_subscription(
            Bool,
            "/teleop/reset/xarm_target_valid",
            self.target_valid_callback,
            10,
        )
        self.create_subscription(
            Bool, "/teleop/reset/factr_aligned", self.factr_aligned_callback, 10
        )
        self.create_subscription(
            Bool, "/factr/connected", self.factr_connected_callback, latched_qos
        )
        self.create_subscription(
            Bool, "/xarm/leader_stale", self.leader_stale_callback, latched_qos
        )
        self.create_subscription(
            Int32, "/robotiq/position_bits", self.robotiq_position_callback, 10
        )
        callback_group = ReentrantCallbackGroup()
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=callback_group,
        )
        self.timer = self.create_timer(0.02, self.control_loop)
        self._publish_active(False)
        self._publish_align(False)
        self._set_status("IDLE")

    def goal_callback(self, goal_request):
        """Accept one valid goal only when no reset transaction owns the system."""

        try:
            normalize_reset_goal(goal_request, self.position_tolerance, self.timeout_sec)
        except ValueError as exc:
            self.get_logger().error(f"拒绝 reset goal: {exc}")
            return GoalResponse.REJECT
        with self.lock:
            completing_goal = self.phase == "COMPLETE" and self.active_goal_handle is not None
            if self.goal_reserved or self.phase in ACTIVE_PHASES or completing_goal:
                return GoalResponse.REJECT
            self.goal_reserved = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept cancellation; the state machine first performs a safe handover."""

        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute one reset goal while timer callbacks drive non-blocking phases."""

        target, tolerances, timeout = normalize_reset_goal(
            goal_handle.request, self.position_tolerance, self.timeout_sec
        )
        with self.lock:
            self.active_goal_handle = goal_handle
            self.goal_reserved = False
            self.goal_done.clear()
            self.terminal_kind = None
            self.error_during_goal = False
            self.cancel_requested = False
            self._begin_reset(target, tolerances, timeout)

        while not self.goal_done.wait(0.05):
            if goal_handle.is_cancel_requested:
                with self.lock:
                    self._request_safe_cancel()
            self._publish_feedback(goal_handle)

        result = FollowJointTrajectory.Result()
        with self.lock:
            terminal_kind = self.terminal_kind
            result.error_code = int(self.terminal_code)
            result.error_string = self.terminal_message
            self.active_goal_handle = None
            if self.phase == "COMPLETE":
                self._set_status("IDLE")
        if terminal_kind == "success":
            goal_handle.succeed()
        elif terminal_kind == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _begin_reset(self, target, tolerances, timeout):
        """Initialize one reset transaction under ``self.lock``.

        Args:
            target: Absolute xArm joint positions, float array shape ``(7,)``
                in radians.
            tolerances: Per-joint position tolerances, float array shape
                ``(7,)`` in radians.
            timeout: Positive scalar reset deadline in seconds.
        Returns:
            ``None``. The non-blocking reset state machine is activated.
        """

        now = time.monotonic()
        self.reset_active = True
        self.reset_target = np.asarray(target, dtype=float).copy()
        self.goal_tolerances = np.asarray(tolerances, dtype=float).copy()
        self.reset_started_at = now
        self.reset_deadline = now + float(timeout)
        self.xarm_stable_since = None
        self.gripper_stable_since = None
        self.xarm_target_valid = None
        self.factr_aligned = not self.require_factr_alignment
        self.factr_wait_started = (
            None
            if not self.require_factr_alignment or self.factr_connected
            else now
        )
        self.cancel_requested = False
        self.xarm_motion_started = not bool(self.latest_error_code)
        self.phase = "CLEARING_ERROR" if self.latest_error_code else "MOVING_XARM"
        self._publish_active(True)
        self._publish_align(False)
        self._publish_target(self.reset_target)
        self._set_status(self.phase)

    def _request_safe_cancel(self):
        """Replace the active goal with a hold-at-current target before canceling."""

        if self.cancel_requested:
            return
        self.cancel_requested = True
        if self._xarm_state_fresh():
            self.reset_target = self.latest_xarm_position.copy()
            self.goal_tolerances = np.full(7, self.position_tolerance)
            self.xarm_target_valid = None
            self.xarm_stable_since = None
            self.phase = "MOVING_XARM"
            self._publish_align(False)
            self._publish_target(self.reset_target)
            self._set_status(self.phase)
            return
        self._finish_failure(
            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
            "Action canceled while xArm state was stale; teleoperation remains paused",
        )

    def control_loop(self):
        """Advance automatic recovery, target motion, alignment and completion states."""

        with self.lock:
            now = time.monotonic()
            if self.phase == "COMPLETE" and self.active_goal_handle is None:
                self._set_status("IDLE")
            if not self.reset_active:
                if self.latest_error_code:
                    self._begin_reset(
                        self.initial_joint_positions,
                        np.full(7, self.position_tolerance),
                        self.timeout_sec,
                    )
                return
            if self.phase == "FAILED":
                return
            if (
                (not self.require_factr_alignment or self.factr_connected)
                and self.reset_deadline is not None
                and now >= self.reset_deadline
            ):
                self._finish_failure(
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                    f"Reset timed out in phase {self.phase}",
                )
                return
            if self.latest_error_code:
                if (
                    self.active_goal_handle is not None
                    and self.xarm_motion_started
                    and not self.error_during_goal
                ):
                    self.error_during_goal = True
                    self.reset_target = self.initial_joint_positions.copy()
                    self.goal_tolerances = np.full(7, self.position_tolerance)
                    self.xarm_target_valid = None
                    self._publish_target(self.reset_target)
                self.phase = "CLEARING_ERROR"
                self._set_status(self.phase)
                return
            if self.phase == "CLEARING_ERROR":
                if not self.control_ok:
                    return
                self.phase = "MOVING_XARM"
                self.xarm_motion_started = True
                self._set_status(self.phase)
            if self.xarm_target_valid is False:
                self._finish_failure(
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    "xArm SDK rejected the reset target",
                )
                return
            if self.xarm_target_valid is None:
                if now - self.reset_started_at > self.state_timeout_sec:
                    self._finish_failure(
                        FollowJointTrajectory.Result.INVALID_GOAL,
                        "No xArm target validation response",
                    )
                return
            if self.phase == "MOVING_XARM":
                if not self._xarm_state_fresh():
                    if now - self.reset_started_at > self.state_timeout_sec:
                        self._finish_failure(
                            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                            "xArm state became stale during reset",
                        )
                    return
                position_ok = np.all(
                    np.abs(self.latest_xarm_position - self.reset_target) <= self.goal_tolerances
                )
                velocity_ok = np.max(np.abs(self.latest_xarm_velocity)) <= self.velocity_tolerance
                if position_ok and velocity_ok:
                    if self.xarm_stable_since is None:
                        self.xarm_stable_since = now
                    elif now - self.xarm_stable_since >= self.stable_time_sec:
                        if self.require_factr_alignment:
                            self.phase = (
                                "ALIGNING_FACTR"
                                if self.factr_connected
                                else "WAITING_FACTR"
                            )
                            self._publish_align(self.factr_connected)
                        else:
                            self.phase = "OPENING_GRIPPER"
                            self._publish_align(False)
                        self._set_status(self.phase)
                else:
                    self.xarm_stable_since = None
                return
            if self.phase in {"ALIGNING_FACTR", "OPENING_GRIPPER"}:
                if not self._robotiq_state_fresh():
                    self._finish_failure(
                        FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                        "Robotiq state became stale during reset",
                    )
                    return
                gripper_open = self.robotiq_position <= self.robotiq_open_threshold
                if gripper_open:
                    if self.gripper_stable_since is None:
                        self.gripper_stable_since = now
                else:
                    self.gripper_stable_since = None
                gripper_stable = (
                    self.gripper_stable_since is not None
                    and now - self.gripper_stable_since >= self.stable_time_sec
                )
                factr_ready = not self.require_factr_alignment or self.factr_aligned
                if factr_ready and gripper_stable:
                    self._finish_success()

    def factr_connected_callback(self, msg: Bool) -> None:
        """Pause reset completion while FACTR reconnects in the background.

        Args:
            msg: Latched scalar bus-health message from the leader node.

        Returns:
            ``None``. Offline duration is added back to the reset deadline so
            an unplugged leader cannot turn a recoverable reset into a timeout.
        """

        if not self.require_factr_alignment:
            return
        with self.lock:
            connected = bool(msg.data)
            if connected == self.factr_connected:
                return
            now = time.monotonic()
            self.factr_connected = connected
            self.factr_aligned = False
            if not connected:
                if self.reset_active and self.factr_wait_started is None:
                    self.factr_wait_started = now
                if self.phase == "ALIGNING_FACTR":
                    self.phase = "WAITING_FACTR"
                    self._publish_align(False)
                    self._set_status(self.phase)
                return
            if self.reset_active and self.factr_wait_started is not None:
                if self.reset_deadline is not None:
                    self.reset_deadline += now - self.factr_wait_started
                self.factr_wait_started = None
            if self.phase == "WAITING_FACTR":
                self.phase = "ALIGNING_FACTR"
                self._publish_align(True)
                self._set_status(self.phase)

    def leader_stale_callback(self, msg: Bool) -> None:
        """Treat an xArm command timeout as an offline FACTR dependency.

        Args:
            msg: Latched scalar xArm watchdog state. Only ``True`` changes
                reset state; recovery is authoritative only when the FACTR
                leader publishes ``/factr/connected=true`` again.

        Returns:
            ``None``. An active reset moves to or eventually enters
            ``WAITING_FACTR`` instead of timing out while alignment is absent.
        """

        if not self.require_factr_alignment or not bool(msg.data):
            return
        disconnected = Bool()
        disconnected.data = False
        self.factr_connected_callback(disconnected)

    def _finish_success(self):
        """Release reset ownership after all three devices reached stable targets."""

        self._publish_align(False)
        self._publish_active(False)
        self.reset_active = False
        self.phase = "COMPLETE"
        self._set_status("COMPLETE")
        if self.cancel_requested:
            self.terminal_kind = "canceled"
            self.terminal_code = FollowJointTrajectory.Result.SUCCESSFUL
            self.terminal_message = "Reset canceled after safe leader handover"
        elif self.error_during_goal:
            self.terminal_kind = "aborted"
            self.terminal_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            self.terminal_message = "Goal caused xArm error; recovered to configured initial pose"
        else:
            self.terminal_kind = "success"
            self.terminal_code = FollowJointTrajectory.Result.SUCCESSFUL
            devices = (
                "xArm and Robotiq"
                if not self.require_factr_alignment
                else "xArm, FACTR and Robotiq"
            )
            self.terminal_message = f"{devices} reset complete"
        if self.active_goal_handle is not None:
            self.goal_done.set()

    def _finish_failure(self, error_code, message):
        """Abort the current transaction while retaining safe reset ownership."""

        self._publish_align(False)
        if self._xarm_state_fresh():
            # Replace a timed-out or rejected goal with the measured pose so the
            # follower holds instead of continuing toward the failed target.
            self.reset_target = self.latest_xarm_position.copy()
            self.xarm_target_valid = None
            self._publish_target(self.reset_target)
        self.reset_active = True
        self.phase = "FAILED"
        self.terminal_kind = "aborted"
        self.terminal_code = int(error_code)
        self.terminal_message = str(message)
        self._set_status("FAILED")
        self.get_logger().error(message)
        if self.active_goal_handle is not None:
            self.goal_done.set()

    def _xarm_state_fresh(self):
        """Return true when finite seven-axis xArm feedback is recent enough."""

        return bool(
            self.latest_xarm_time is not None
            and time.monotonic() - self.latest_xarm_time <= self.state_timeout_sec
            and self.latest_xarm_position is not None
            and self.latest_xarm_velocity is not None
        )

    def _robotiq_state_fresh(self):
        """Return true when the scalar Robotiq bit position is recent enough."""

        return bool(
            self.robotiq_state_time is not None
            and time.monotonic() - self.robotiq_state_time <= self.state_timeout_sec
            and self.robotiq_position is not None
        )

    def _publish_feedback(self, goal_handle):
        """Publish desired, actual and error seven-axis Action feedback arrays."""

        with self.lock:
            if self.reset_target is None or self.latest_xarm_position is None:
                return
            desired = self.reset_target.copy()
            actual = self.latest_xarm_position.copy()
            velocity = (
                np.zeros(7)
                if self.latest_xarm_velocity is None
                else self.latest_xarm_velocity.copy()
            )
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(JOINT_NAMES)
        feedback.desired = JointTrajectoryPoint()
        feedback.actual = JointTrajectoryPoint()
        feedback.error = JointTrajectoryPoint()
        feedback.desired.positions = desired.tolist()
        feedback.actual.positions = actual.tolist()
        feedback.actual.velocities = velocity.tolist()
        feedback.error.positions = (desired - actual).tolist()
        goal_handle.publish_feedback(feedback)

    def xarm_state_callback(self, msg):
        """Store finite xArm position and velocity arrays with shape ``(7,)``."""

        if len(msg.position) < 7 or len(msg.velocity) < 7:
            return
        position = np.asarray(msg.position[:7], dtype=float)
        velocity = np.asarray(msg.velocity[:7], dtype=float)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            return
        with self.lock:
            self.latest_xarm_position = position
            self.latest_xarm_velocity = velocity
            self.latest_xarm_time = time.monotonic()

    def error_callback(self, msg):
        """Store the scalar xArm error code from ``[error, warning]`` data."""

        if msg.data:
            with self.lock:
                self.latest_error_code = int(msg.data[0])

    def control_ok_callback(self, msg):
        """Store whether the latest xArm read and command cycle succeeded."""

        with self.lock:
            self.control_ok = bool(msg.data)

    def target_valid_callback(self, msg):
        """Store the xArm SDK joint-limit validation result for the current target."""

        with self.lock:
            self.xarm_target_valid = bool(msg.data)

    def factr_aligned_callback(self, msg):
        """Store whether FACTR arm and gripper remained inside reset tolerances."""

        with self.lock:
            self.factr_aligned = bool(msg.data) and self.factr_connected

    def robotiq_position_callback(self, msg):
        """Store one Robotiq position sample in physical controller bits."""

        with self.lock:
            self.robotiq_position = int(msg.data)
            self.robotiq_state_time = time.monotonic()

    def _publish_active(self, active):
        """Publish reset ownership as a latched ROS boolean."""

        msg = Bool()
        msg.data = bool(active)
        self.active_pub.publish(msg)

    def _publish_align(self, align):
        """Publish FACTR active-alignment permission as a latched boolean."""

        msg = Bool()
        msg.data = bool(align)
        self.align_pub.publish(msg)

    def _publish_target(self, target):
        """Publish one absolute seven-axis xArm reset target in radians.

        Args:
            target: Finite float array shape ``(7,)`` in radians.

        Returns:
            ``None``. The target is published to the xArm controller.
        """

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(value) for value in target]
        self.target_pub.publish(msg)

    def _set_status(self, status):
        """Set and publish one documented reset phase string."""

        self.phase = str(status)
        msg = String()
        msg.data = self.phase
        self.status_pub.publish(msg)

    def destroy_node(self):
        """Destroy the Action server before releasing the ROS node."""

        self.action_server.destroy()
        super().destroy_node()


def main(args=None):
    """Run the reset coordinator with concurrent Action and state callbacks."""

    rclpy.init(args=args)
    node = ResetCoordinator()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
