"""Direct Modbus RTU command tests for the Robotiq ROS node."""

import sys
from types import ModuleType

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class _FakeGripper:
    """Record direct gripper connection, activation and motion calls."""

    instance = None
    activated_on_connect = False
    started_on_connect = False

    def __init__(self, **kwargs):
        """Store scalar connection arguments and expose empty call histories."""

        self.connection = kwargs
        self.activation_count = 0
        self.start_count = 0
        self.activated = self.activated_on_connect
        self.started = self.started_on_connect
        self.timeOut = 5.0
        self.moves = []
        self.current_position = 0
        self.disconnected = False
        type(self).instance = self

    def isActivated(self, refreshStatus=True):
        """Return whether the fake gripper has completed activation."""

        return self.activated

    def isStarted(self, refreshStatus=True):
        """Return whether the fake gripper has received a start request."""

        return self.started

    def activate(self, **kwargs):
        """Record one activation request and its scalar options."""

        self.activation_count += 1
        self.activation_options = kwargs
        self.activation_timeout = self.timeOut
        self.activated = True
        self.started = bool(kwargs.get("start", True))

    def start(self, **kwargs):
        """Record one start request and its scalar options."""

        self.start_count += 1
        self.start_options = kwargs
        self.started = True

    def move(self, position, **kwargs):
        """Record one integer position and its scalar motion arguments."""

        self.moves.append((position, kwargs))
        self.current_position = position

    def position(self, refreshStatus=True):
        """Return the fake physical position in controller bits."""

        return self.current_position

    def disconnect(self):
        """Record that the serial connection was closed and return ``None``."""

        self.disconnected = True


_fake_api = ModuleType("pyrobotiqgripper")
_fake_api.GRIPPER_MODE_RTU = "RTU"
_fake_api.RobotiqGripper = _FakeGripper
sys.modules["pyrobotiqgripper"] = _fake_api


def test_node_uses_bound_rtu_port_and_filters_small_motion() -> None:
    """The node must use the configured RTU device and avoid duplicate writes."""

    from xarm_ros2_teleop.robotiq.robotiq_controller import RobotiqNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "com_port:=/dev/robotiq_rs485",
            "-p",
            "device_id:=9",
            "-p",
            "minimal_motion:=2",
        ]
    )
    node = RobotiqNode()
    gripper = _FakeGripper.instance
    try:
        assert gripper.connection == {
            "com_port": "/dev/robotiq_rs485",
            "device_id": 9,
            "connection_type": "RTU",
            "debug": False,
        }
        assert gripper.activation_count == 1
        assert gripper.activation_options == {
            "reset": True,
            "start": True,
            "refreshStatus": False,
        }
        assert gripper.activation_timeout == 15.0
        assert gripper.timeOut == 5.0
        assert gripper.start_count == 0

        message = JointState()
        message.position = [-0.5]
        node.gripper_callback(message)
        node.control_loop()
        node.control_loop()
        assert gripper.moves == [
            (
                127,
                {
                    "speed": 255,
                    "force": 100,
                    "wait": False,
                    "readStatus": False,
                },
            )
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()
    assert gripper.disconnected is True


def test_node_does_not_recalibrate_an_already_active_gripper() -> None:
    """A ROS node restart must preserve an activated Robotiq calibration."""

    from xarm_ros2_teleop.robotiq.robotiq_controller import RobotiqNode

    _FakeGripper.activated_on_connect = True
    _FakeGripper.started_on_connect = True
    rclpy.init()
    node = RobotiqNode()
    gripper = _FakeGripper.instance
    try:
        assert gripper.activation_count == 0
        assert gripper.start_count == 0
        assert gripper.timeOut == 5.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
        _FakeGripper.activated_on_connect = False
        _FakeGripper.started_on_connect = False


def test_reset_ignores_leader_command_and_opens_gripper() -> None:
    """Reset ownership must discard buffered commands and request bit position zero."""

    from xarm_ros2_teleop.robotiq.robotiq_controller import RobotiqNode

    rclpy.init(args=["--ros-args", "-p", "reset_enabled:=true"])
    node = RobotiqNode()
    gripper = _FakeGripper.instance
    try:
        leader = JointState()
        leader.position = [-0.5]
        node.gripper_callback(leader)
        active = Bool()
        active.data = True
        node.reset_active_callback(active)
        node.gripper_callback(leader)
        node.control_loop()
        assert gripper.moves[-1][0] == 0
        assert node.target_position == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_episode_gate_blocks_gripper_until_capture_starts() -> None:
    """Robotiq feedback stays live while inter-episode leader commands are ignored."""

    from xarm_ros2_teleop.robotiq.robotiq_controller import RobotiqNode

    rclpy.init(args=["--ros-args", "-p", "recording_gate_enabled:=true"])
    node = RobotiqNode()
    gripper = _FakeGripper.instance
    leader = JointState()
    leader.position = [-0.5]
    try:
        node.gripper_callback(leader)
        node.control_loop()
        assert gripper.moves == []

        node.episode_active_callback(Bool(data=True))
        node.gripper_callback(leader)
        node.control_loop()
        assert gripper.moves[-1][0] == 127

        node.episode_active_callback(Bool(data=False))
        leader.position = [-1.0]
        node.gripper_callback(leader)
        node.control_loop()
        assert gripper.moves[-1][0] == 127
    finally:
        node.destroy_node()
        rclpy.shutdown()
