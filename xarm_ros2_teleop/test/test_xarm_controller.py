"""Leader glitch-guard tests for the xArm follower node."""

import sys
from types import ModuleType

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class _FakeXArmAPI:
    """Record connection lifecycle and servo commands without hardware."""

    instance = None

    def __init__(self, ip, is_radian=True, do_not_open=False):
        """Store scalar constructor arguments and start with empty histories."""

        self.ip = ip
        self.connected = False
        self.servo_commands = []
        self.position_commands = []
        self.mode_commands = []
        self.state_commands = []
        self.controller_state = 0
        self.tcp_pose = [300.0, 0.0, 200.0, 3.14, 0.0, 0.0]
        type(self).instance = self

    def connect(self):
        """Mark the fake controller as connected and return success."""

        self.connected = True
        return 0

    def motion_enable(self, enable):
        """Accept one boolean enable request."""

        return 0

    def set_mode(self, mode):
        """Record one integer xArm control mode."""

        self.mode_commands.append(int(mode))
        return 0

    def set_state(self, state):
        """Record one integer xArm controller state."""

        self.state_commands.append(int(state))
        return 0

    def get_position(self, is_radian=True):
        """Return the fake TCP pose as ``(code, [x, y, z, roll, pitch, yaw])``."""

        return 0, list(self.tcp_pose)

    def get_state(self):
        """Return the scalar controller state used to supervise mode changes."""

        return 0, self.controller_state

    def set_position(
        self,
        x=None,
        y=None,
        z=None,
        roll=None,
        pitch=None,
        yaw=None,
        speed=None,
        mvacc=None,
        is_radian=True,
        wait=False,
    ):
        """Record one absolute TCP command, represented by its z value in millimetres."""

        self.position_commands.append(z)
        return 0

    def is_joint_limit(self, angles, is_radian=True):
        """Report every joint target as within limits."""

        return 0, False

    def set_servo_angle(self, angle=None, speed=None, mvacc=None, is_radian=True, wait=False):
        """Record one absolute joint target in radians with shape ``(7,)``."""

        self.servo_commands.append(tuple(angle))
        return 0

    def disconnect(self):
        """Mark the fake controller as disconnected."""

        self.connected = False


_fake_xarm = ModuleType("xarm")
_fake_xarm.__path__ = []
_fake_wrapper = ModuleType("xarm.wrapper")
_fake_wrapper.XArmAPI = _FakeXArmAPI
sys.modules["xarm"] = _fake_xarm
sys.modules["xarm.wrapper"] = _fake_wrapper


def _leader_message(delta) -> JointState:
    """Build one leader ``JointState`` from a seven-value delta in radians."""

    message = JointState()
    message.position = [float(value) for value in delta]
    return message


def test_jump_guard_holds_target_and_stops_on_sustained_rejects(monkeypatch) -> None:
    """Single glitches keep the previous target; persistent ones stop following."""

    import xarm_ros2_teleop.xarm.xarm_controller as controller_module
    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "max_leader_step_rad:=0.2",
            "-p",
            "max_leader_reject_sec:=0.5",
            "-p",
            "reset_lift_mm:=0.0",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        node.latest_error_code = 0
        init = np.asarray(node.init_qpos)

        # The first leader sample is accepted unconditionally and commanded.
        node.arm_callback(_leader_message(np.zeros(7)))
        node.control_loop()
        assert arm.servo_commands == [tuple(init)]

        # Normal small steps are accepted immediately.
        node.arm_callback(_leader_message(np.full(7, 0.05)))
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(init + 0.05)

        # One jumped sample is dropped and the previous target keeps being sent.
        node.arm_callback(_leader_message(np.full(7, 0.9)))
        assert np.allclose(node.target_arm_pos, init + 0.05)
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(init + 0.05)

        # Non-finite samples are dropped the same way; one good sample then
        # clears the rejection streak before the sustained-reject phase.
        node.arm_callback(_leader_message(np.full(7, np.nan)))
        assert np.allclose(node.target_arm_pos, init + 0.05)
        node.arm_callback(_leader_message(np.full(7, 0.05)))

        # Rejections sustained past max_leader_reject_sec latch the fault and
        # stop follower commands entirely.
        fake_now = {"value": 100.0}
        monkeypatch.setattr(controller_module.time, "monotonic", lambda: fake_now["value"])
        node.arm_callback(_leader_message(np.full(7, 0.9)))
        fake_now["value"] = 100.6
        node.arm_callback(_leader_message(np.full(7, 0.9)))
        assert node.leader_fault_active is True

        arm.servo_commands.clear()
        node.control_loop()
        assert arm.servo_commands == []

        # Consistent samples rebuild the streak; following resumes only after
        # LEADER_RESUME_STREAK consecutive accepts and not before.
        for _ in range(XarmNode.LEADER_RESUME_STREAK - 1):
            node.arm_callback(_leader_message(np.full(7, 0.05)))
        assert node.leader_fault_active is True
        node.arm_callback(_leader_message(np.full(7, 0.05)))
        assert node.leader_fault_active is False
        node.control_loop()
        assert arm.servo_commands == [tuple(init + 0.05)]
    finally:
        node.destroy_node()
        rclpy.shutdown()

def test_reset_lift_commands_tcp_up_before_joint_motion(monkeypatch) -> None:
    """A reset first lifts TCP z by reset_lift_mm, then drives joints home."""

    import xarm_ros2_teleop.xarm.xarm_controller as controller_module
    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_lift_mm:=100.0",
            "-p",
            "reset_lift_speed_mm_s:=50.0",
            "-p",
            "reset_enabled:=true",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        node.latest_error_code = 0
        node.session_ready = True
        arm.mode_commands.clear()
        arm.state_commands.clear()

        # Coordinator publishes active + the reset target; the first control
        # ticks must lift TCP z by 100 mm and suppress joint commands.
        node.reset_active_callback(Bool(data=True))
        target = JointState()
        target.position = list(np.asarray(node.init_qpos))
        node.reset_target_callback(target)
        node.control_loop()
        assert len(arm.position_commands) == 1
        assert arm.position_commands[0] == 300.0  # 200 mm start + 100 mm lift
        assert arm.mode_commands == [0]
        assert arm.state_commands == [0]
        assert arm.servo_commands == []

        # While TCP z is far from the lift target, joints stay suppressed.
        arm.tcp_pose[2] = 250.0
        node.control_loop()
        assert arm.servo_commands == []

        # Once the lift height is reached (past the 0.3 s check interval),
        # joint motion toward home begins.
        # Seed the fake clock from the real monotonic value so the lift's
        # 0.3 s check interval compares against a consistent timeline.
        fake_clock = {"value": controller_module.time.monotonic() + 1.0}
        monkeypatch.setattr(
            controller_module.time, "monotonic", lambda: fake_clock["value"]
        )
        arm.tcp_pose[2] = 300.0
        fake_clock["value"] += 1.0
        node.control_loop()
        assert arm.mode_commands == [0, 6]
        assert arm.state_commands == [0, 0]
        fake_clock["value"] += 1.0
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(np.asarray(node.init_qpos))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_lift_is_skipped_above_configured_tcp_height() -> None:
    """A reset starting at or above 400 mm must go directly to joint motion."""

    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_lift_mm:=100.0",
            "-p",
            "reset_lift_skip_above_z_mm:=400.0",
            "-p",
            "reset_enabled:=true",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        node.latest_error_code = 0
        node.session_ready = True
        arm.position_commands.clear()
        arm.servo_commands.clear()
        arm.tcp_pose[2] = 450.0

        node.reset_active_callback(Bool(data=True))
        target = JointState()
        target.position = list(np.asarray(node.init_qpos))
        node.reset_target_callback(target)
        node.control_loop()
        assert arm.position_commands == []
        assert node.reset_lift_pending is False

        node.control_loop()
        assert arm.servo_commands[-1] == tuple(np.asarray(node.init_qpos))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_end_waits_for_zero_delta_before_accepting_leader() -> None:
    """Queued pre-rebase deltas cannot move xArm on the first active frame."""

    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_enabled:=true",
            "-p",
            "leader_watchdog_enabled:=true",
        ]
    )
    node = XarmNode()
    try:
        init = np.asarray(node.init_qpos)
        node.factr_connected_callback(Bool(data=True))
        node.arm_callback(_leader_message(np.zeros(7)))
        assert np.allclose(node.target_arm_pos, init)

        node.reset_active_callback(Bool(data=True))
        node.arm_callback(_leader_message(np.full(7, 0.9)))
        assert np.allclose(node.target_arm_pos, init)

        node.reset_active_callback(Bool(data=False))
        assert node.awaiting_fresh_leader is True
        assert node.last_accepted_leader_target is None
        # A queued nonzero delta is ignored until FACTR confirms its new origin.
        node.arm_callback(_leader_message(np.full(7, 0.9)))
        assert np.allclose(node.target_arm_pos, init)
        assert node.awaiting_fresh_leader is True

        node.arm_callback(_leader_message(np.zeros(7)))
        assert np.allclose(node.target_arm_pos, init)
        assert node.awaiting_fresh_leader is False
        assert node.leader_fault_active is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_end_accepts_first_absolute_policy_target_without_factr_rebase() -> None:
    """Inference mode accepts an absolute seven-joint target immediately after reset."""

    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_enabled:=true",
            "-p",
            "leader_watchdog_enabled:=false",
            "-p",
            "absolute_command_enabled:=true",
            "-p",
            "max_policy_step_rad:=0.5",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        init = np.asarray(node.init_qpos)
        node.latest_error_code = 0
        arm.servo_commands.clear()
        node.absolute_target_callback(_leader_message(init))
        pre_reset_policy_target = init + 0.484
        node.absolute_target_callback(_leader_message(pre_reset_policy_target))
        assert np.allclose(node.target_arm_pos, pre_reset_policy_target)
        node.reset_active_callback(Bool(data=True))
        reset_target = JointState()
        reset_target.position = init.tolist()
        node.reset_target_callback(reset_target)
        node.reset_active_callback(Bool(data=False))

        assert node.awaiting_fresh_leader is False
        assert np.allclose(node.target_arm_pos, init)
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(init)
        policy_target = init + 0.05
        node.absolute_target_callback(_leader_message(policy_target))
        assert np.allclose(node.target_arm_pos, policy_target)
        assert node.awaiting_fresh_leader is False
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(policy_target)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_episode_gate_holds_mode_zero_until_capture_starts() -> None:
    """Between episodes xArm holds mode 0 while reset targets remain allowed."""

    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_enabled:=true",
            "-p",
            "reset_lift_mm:=0.0",
            "-p",
            "recording_gate_enabled:=true",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        node.latest_error_code = 0
        arm.mode_commands.clear()
        arm.state_commands.clear()
        arm.servo_commands.clear()

        node.episode_active_callback(Bool(data=True))
        assert arm.mode_commands == [6]
        node.arm_callback(_leader_message(np.zeros(7)))
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(np.asarray(node.init_qpos))

        node.episode_active_callback(Bool(data=False))
        assert arm.mode_commands[-1] == 0
        arm.servo_commands.clear()
        node.arm_callback(_leader_message(np.full(7, 0.1)))
        node.control_loop()
        assert arm.servo_commands == []

        node.reset_active_callback(Bool(data=True))
        target = JointState()
        target.position = list(np.asarray(node.init_qpos))
        node.reset_target_callback(target)
        node.control_loop()
        assert arm.servo_commands[-1] == tuple(np.asarray(node.init_qpos))
        node.reset_active_callback(Bool(data=False))
        assert arm.mode_commands[-1] == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_missing_factr_commands_hold_xarm_but_reset_still_moves() -> None:
    """Leader timeout selects mode 0 while coordinated reset retains priority."""

    from xarm_ros2_teleop.xarm.xarm_controller import XarmNode

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "ip:=127.0.0.1",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "reset_enabled:=true",
            "-p",
            "reset_lift_mm:=0.0",
            "-p",
            "leader_watchdog_enabled:=true",
            "-p",
            "leader_command_timeout_sec:=0.25",
        ]
    )
    node = XarmNode()
    arm = _FakeXArmAPI.instance
    try:
        node.latest_error_code = 0
        node.factr_connected_callback(Bool(data=True))
        node.arm_callback(_leader_message(np.zeros(7)))
        node.control_loop()
        assert len(arm.servo_commands) == 1

        node.last_leader_command_time -= 1.0
        node.control_loop()
        assert node.leader_stale is True
        assert arm.mode_commands[-1] == 0
        assert len(arm.servo_commands) == 1

        node.reset_active_callback(Bool(data=True))
        target = JointState()
        target.position = list(np.asarray(node.init_qpos))
        node.reset_target_callback(target)
        node.control_loop()
        assert len(arm.servo_commands) == 2
    finally:
        node.destroy_node()
        rclpy.shutdown()
