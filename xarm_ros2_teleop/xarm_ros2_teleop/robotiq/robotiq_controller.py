"""Control a PC-connected Robotiq gripper through Modbus RTU."""

import numpy as np
import pyrobotiqgripper as rq
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32


class RobotiqNode(Node):
    """Follow a scalar leader command through direct Modbus RTU.

    The input is ``JointState.position[0]`` in leader-gripper radians. It is
    mapped to the Robotiq position range ``[0, 255]``, where 0 is open and 255
    is closed. Serial communication is independent from the xArm control loop.
    """

    def __init__(self):
        """Read ROS parameters, connect the gripper and start command updates."""

        super().__init__("robotiq_node")
        self.declare_parameter("com_port", "/dev/robotiq_rs485")
        self.declare_parameter("device_id", 9)
        self.declare_parameter("debug", False)
        self.declare_parameter("command_topic", "/xarm/gripper_pos")
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("speed", 255)
        self.declare_parameter("force", 100)
        self.declare_parameter("minimal_motion", 2)
        self.declare_parameter("activate_on_startup", True)
        self.declare_parameter("activation_timeout_sec", 15.0)
        self.declare_parameter("reset_enabled", False)
        self.declare_parameter("session_gate_enabled", False)
        self.declare_parameter("recording_gate_enabled", False)

        self.com_port = str(self.get_parameter("com_port").value)
        self.device_id = int(self.get_parameter("device_id").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.speed = int(self.get_parameter("speed").value)
        self.force = int(self.get_parameter("force").value)
        self.minimal_motion = int(self.get_parameter("minimal_motion").value)
        self.activate_on_startup = bool(self.get_parameter("activate_on_startup").value)
        self.activation_timeout_sec = float(
            self.get_parameter("activation_timeout_sec").value
        )
        self.reset_enabled = bool(self.get_parameter("reset_enabled").value)
        self.session_gate_enabled = bool(self.get_parameter("session_gate_enabled").value)
        self.recording_gate_enabled = bool(
            self.get_parameter("recording_gate_enabled").value
        )
        self.session_ready = not self.session_gate_enabled
        self.episode_active = not self.recording_gate_enabled
        self._validate_parameters()

        # The current pyRobotiqGripper API opens the configured RTU connection
        # when the object is constructed; no xArm tool-port transport is used.
        self.gripper = rq.RobotiqGripper(
            com_port=self.com_port,
            device_id=self.device_id,
            connection_type=rq.GRIPPER_MODE_RTU,
            debug=self.debug,
        )
        self._prepare_gripper()

        self.target_position = None
        self.last_commanded_position = None
        self.reset_active = False
        self.position_pub = self.create_publisher(Int32, "/robotiq/position_bits", 10)
        self.joint_state_pub = self.create_publisher(JointState, "/robotiq/joint_state", 10)
        self.commanded_joint_pub = self.create_publisher(
            JointState, "/robotiq/commanded_joint_state", 10
        )
        self.command_sub = self.create_subscription(
            JointState, self.command_topic, self.gripper_callback, 10
        )
        self.reset_active_sub = self.create_subscription(
            Bool, "/teleop/reset/active", self.reset_active_callback, 10
        )
        self.session_ready_sub = self.create_subscription(
            Bool, "/system/session/ready", self.session_ready_callback, 10
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.episode_active_sub = self.create_subscription(
            Bool,
            "/teleop/episode_active",
            self.episode_active_callback,
            latched_qos,
        )
        self.timer = self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            f"Robotiq 已通过 {self.com_port} 连接并启动, device_id={self.device_id}"
        )

    def _prepare_gripper(self) -> None:
        """Bring a cold Robotiq controller into its activated motion state.

        The method has no inputs or return value. A powered-on but inactive
        gripper is reset and calibrated once; an already activated gripper is
        not reset when this ROS node restarts. The configured timeout is in
        seconds and is applied to the third-party SDK's blocking activation.
        """

        if not self.activate_on_startup:
            return
        if not self.gripper.isActivated(refreshStatus=True):
            self.get_logger().info("Robotiq 未激活，执行冷启动 reset 和 activation")
            runtime_timeout = self.gripper.timeOut
            self.gripper.timeOut = self.activation_timeout_sec
            try:
                self.gripper.activate(reset=True, start=True, refreshStatus=False)
            finally:
                self.gripper.timeOut = runtime_timeout
        if not self.gripper.isStarted(refreshStatus=True):
            self.gripper.start(refreshStatus=False, readStatus=True)
        if not self.gripper.isActivated(refreshStatus=True):
            raise RuntimeError("Robotiq 激活流程结束，但控制器仍未进入 activated 状态")
        if not self.gripper.isStarted(refreshStatus=True):
            raise RuntimeError("Robotiq 已激活，但未能进入 started 状态")

    def _validate_parameters(self):
        """Validate scalar communication and motion parameters.

        Returns:
            Nothing. Invalid port, timing or Robotiq byte-range values raise
            ``ValueError`` before the serial device is opened.
        """

        if not self.com_port:
            raise ValueError("com_port must not be empty")
        if not 1 <= self.device_id <= 247:
            raise ValueError("device_id must be in [1, 247]")
        if self.control_period <= 0:
            raise ValueError("control_period must be positive")
        if self.activation_timeout_sec <= 0:
            raise ValueError("activation_timeout_sec must be positive")
        if not 0 <= self.speed <= 255:
            raise ValueError("speed must be in [0, 255]")
        if not 0 <= self.force <= 255:
            raise ValueError("force must be in [0, 255]")
        if not 0 <= self.minimal_motion <= 255:
            raise ValueError("minimal_motion must be in [0, 255]")

    def gripper_callback(self, msg):
        """Convert one leader-gripper radian command to Robotiq bits.

        Args:
            msg: ROS ``JointState`` with at least one scalar ``position`` value.
                The first value is the leader gripper position in radians.

        Returns:
            Nothing. ``target_position`` is updated to an integer in ``[0, 255]``.
        """

        if not msg.position:
            self.get_logger().warn("夹爪位置消息为空")
            return
        if self.reset_active or not self.session_ready or not self.episode_active:
            return
        self.target_position = int(np.clip(-float(msg.position[0]) * 255, 0, 255))

    def reset_active_callback(self, msg):
        """Open the gripper during reset and discard pre-reset commands.

        Args:
            msg: ROS ``Bool`` whose data enables the coordinated reset mode.

        Returns:
            Nothing. Both reset transitions clear cached normal commands.
        """

        active = bool(msg.data) and self.reset_enabled
        if active != self.reset_active:
            self.target_position = 0 if active else None
            self.last_commanded_position = None
        self.reset_active = active

    def session_ready_callback(self, msg):
        """Ignore normal gripper commands until warm-up has completed."""

        ready = bool(msg.data) or not self.session_gate_enabled
        if ready != self.session_ready:
            self.target_position = None
            self.last_commanded_position = None
        self.session_ready = ready

    def episode_active_callback(self, msg):
        """Discard leader gripper commands outside active episode capture.

        Args:
            msg: Latched ROS ``Bool`` from the recorder; false spans episode
                completion, physical reset and the following idle interval.

        Returns:
            ``None``. Physical feedback continues while cached commands are
            cleared at every gate transition.
        """

        active = bool(msg.data) or not self.recording_gate_enabled
        if active != self.episode_active:
            self.target_position = None
            self.last_commanded_position = None
        self.episode_active = active

    def control_loop(self):
        """Send changed gripper targets through Modbus RTU.

        Returns:
            Nothing. Commands whose accumulated change is smaller than
            ``minimal_motion`` bits are not written to the serial bus.
        """

        actual_position = int(self.gripper.position(refreshStatus=True))
        stamp = self.get_clock().now().to_msg()
        position_msg = Int32()
        position_msg.data = actual_position
        self.position_pub.publish(position_msg)
        self.joint_state_pub.publish(_position_message(actual_position, stamp))

        target_position = 0 if self.reset_active else self.target_position
        if not self.reset_active and (not self.session_ready or not self.episode_active):
            return
        if target_position is None:
            return
        if (
            self.last_commanded_position is not None
            and abs(target_position - self.last_commanded_position) < self.minimal_motion
        ):
            self.commanded_joint_pub.publish(_position_message(self.last_commanded_position, stamp))
            return
        self.gripper.move(
            target_position,
            speed=self.speed,
            force=self.force,
            wait=False,
            readStatus=False,
        )
        self.last_commanded_position = target_position
        self.commanded_joint_pub.publish(_position_message(target_position, stamp))

    def destroy_node(self):
        """Close the Modbus RTU connection before destroying the ROS node."""

        self.gripper.disconnect()
        super().destroy_node()


def main(args=None):
    """Run the direct Robotiq node with optional ROS argument strings."""

    rclpy.init(args=args)
    node = None
    try:
        node = RobotiqNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _position_message(position_bits, stamp):
    """Create a stamped normalized Robotiq joint position.

    Args:
        position_bits: Integer controller position in ``[0, 255]`` where zero
            is fully open and 255 is fully closed.
        stamp: ROS time message associated with the read or command operation.

    Returns:
        ``JointState`` with one normalized position in shape ``(1,)``.
    """

    msg = JointState()
    msg.header.stamp = stamp
    msg.name = ["robotiq_finger"]
    msg.position = [float(position_bits) / 255.0]
    return msg
