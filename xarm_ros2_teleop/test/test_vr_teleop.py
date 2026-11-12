"""VR clutch, semantic recording and native TCP tests without physical devices."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from xarm_ros2_teleop.recording.control_contract import (
    JOINT_ACTION,
    TCP_ACTION,
    make_contract,
    validate_contract,
    write_contract,
)
from xarm_ros2_teleop.recording.synchronizer import (
    FrameSynchronizer,
    SynchronizationError,
    XarmState,
)
from xarm_ros2_teleop.recording.timestamp_buffer import TimestampedSample
from xarm_ros2_teleop.system.config import load_system_config, load_system_mapping
from xarm_ros2_teleop.vr.config import VRConfig
from xarm_ros2_teleop.vr.mapping import RelativeTeleop
from xarm_ros2_teleop.vr.pose_math import from_rotvec, matrix_quaternion, rotation_matrix, rotvec
from xarm_ros2_teleop.vr.protocol import parse_pose_frame
from xarm_ros2_teleop.xarm.tcp_control import TCPControl, robot_geometry

ROOT = Path(__file__).parents[2]
PROFILE = ROOT / "xarm_ros2_teleop/config/teleop_vr.yaml"
IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def vr_config(**values):
    """Return conservative test-only workspace limits, in metres."""
    return VRConfig.from_mapping(
        {"workspace_lower_m": [-1.0, -1.0, -1.0], "workspace_upper_m": [1.0, 1.0, 1.0], **values}
    )


def engage(mapper, pose=IDENTITY, robot=IDENTITY, trigger=0.0, gripper=0.4, start=1.0):
    """Feed fresh released samples for .4 s then press, returning the first target."""
    for offset in (0.0, 0.1, 0.2, 0.4):
        mapper.update(
            pose, trigger, False, True, ("session", 0), start + offset, robot, gripper, True
        )
    return mapper.update(
        pose, trigger, True, True, ("session", 0), start + 0.41, robot, gripper, True
    )


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_fixed_space_translation_and_zero_engagement(axis, sign):
    """Six hand translation directions preserve base axes despite hand orientation."""
    mapper = RelativeTeleop(vr_config(position_scale=0.5))
    hand = np.r_[[0.0, 0.0, 0.0], from_rotvec([0.0, 0.0, np.pi / 2])]
    initial, grip = engage(mapper, hand)
    np.testing.assert_allclose(initial, IDENTITY, atol=1e-12)
    assert grip == pytest.approx(0.4)
    moved = hand.copy()
    moved[axis] += sign * 0.2
    actual, _ = mapper.update(moved, 0.2, True, True, ("session", 0), 1.5, IDENTITY, 0.4, True)
    expected = np.zeros(3)
    expected[axis] = sign * 0.1
    np.testing.assert_allclose(actual[:3], expected, atol=1e-12)


def test_rotation_clutch_and_gripper_rebase():
    """Known rotation, repeated clutch and trigger origins never jump the robot/gripper."""
    mapper = RelativeTeleop(vr_config())
    engage(mapper)
    hand = np.r_[[0.1, 0.0, 0.0], from_rotvec([0.0, 0.0, 0.4])]
    target, gripper = mapper.update(hand, 0.3, True, True, ("session", 0), 1.5, IDENTITY, 0.4, True)
    np.testing.assert_allclose(rotvec(target[3:]), [0.0, 0.0, 0.4], atol=1e-12)
    assert gripper == pytest.approx(0.7)
    mapper.update(hand, 0.0, False, True, ("session", 0), 1.6, target, 0.7, True)
    rebase, closure = mapper.update(hand, 0.0, True, True, ("session", 0), 1.7, target, 0.7, True)
    np.testing.assert_allclose(rebase, target, atol=1e-12)
    assert closure == pytest.approx(0.7)


@pytest.mark.parametrize("loss", ["tracking", "origin", "gap", "reset"])
def test_loss_requires_release_not_just_held_grip(loss):
    """Loss, epoch changes, long gaps and reset cannot silently restore control."""
    mapper = RelativeTeleop(vr_config())
    engage(mapper)
    generation = ("session", 0)
    now = 1.5
    if loss == "tracking":
        mapper.update(IDENTITY, 0.0, True, False, generation, now, IDENTITY, 0.0, True)
    elif loss == "origin":
        generation = ("session", 1)
    elif loss == "gap":
        now = 2.0
    else:
        mapper.invalidate("reset active")
    for index in range(6):
        assert (
            mapper.update(
                IDENTITY, 0.0, True, True, generation, now + 0.1 * index, IDENTITY, 0.0, True
            )
            is None
        )
    assert not mapper.following


def test_start_held_grip_and_workspace_rejection():
    """Starting with grip held is disarmed; out-of-box targets are not clipped."""
    mapper = RelativeTeleop(vr_config())
    for index in range(8):
        assert (
            mapper.update(
                IDENTITY, 0.0, True, True, ("session", 0), 0.1 * index, IDENTITY, 0.0, True
            )
            is None
        )
    engage(mapper, start=2.0)
    outside = IDENTITY.copy()
    outside[0] = 2.0
    assert mapper.update(outside, 1.0, True, True, ("session", 0), 2.5, IDENTITY, 0.0, True) is None
    assert not mapper.following
    assert "outside" in mapper.reason


def test_early_press_is_not_queued_until_recovery_is_stable():
    """Holding a premature grip press must not become automatic engagement later."""
    mapper = RelativeTeleop(vr_config())
    mapper.update(IDENTITY, 0.0, False, True, ("s", 0), 1.0, IDENTITY, 0.0, True)
    for now in (1.1, 1.2, 1.3, 1.4, 1.5):
        assert mapper.update(IDENTITY, 0.0, True, True, ("s", 0), now, IDENTITY, 0.0, True) is None
    mapper.update(IDENTITY, 0.0, False, True, ("s", 0), 1.6, IDENTITY, 0.0, True)
    assert mapper.update(IDENTITY, 0.0, True, True, ("s", 0), 1.7, IDENTITY, 0.0, True) is not None


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0], [0.2, -0.7, 1.2]])
def test_quaternion_matrix_roundtrip(vector):
    """Pure quaternion conversions preserve zero, pi and general rotations."""
    expected = rotation_matrix(from_rotvec(vector))
    np.testing.assert_allclose(rotation_matrix(matrix_quaternion(expected)), expected, atol=1e-12)


class FakeTCPArm:
    """Record SDK calls; no socket, ROS node or robot is started."""

    version_number = (2, 5, 0)
    version = "2.5.0"
    axis = 7
    sn = "test-xarm7"
    tcp_offset = [0.0] * 6
    world_offset = [0.0] * 6

    def __init__(self):
        """Initialize fake measured TCP and configurable command/read failure codes."""
        self.pose = [200.0, 0.0, 300.0, 0.0, 0.0, 0.0]
        self.commands = []
        self.stops = []
        self.code = 0
        self.read_code = 0

    def get_position_aa(self, **kwargs):
        """Return controller-native axis-angle millimetres/radians."""
        return self.read_code, self.pose.copy()

    def set_position_aa(self, pose, **kwargs):
        """Retain exact submitted pose/limits and return the configured SDK code."""
        self.commands.append((pose, kwargs))
        return self.code

    def set_state(self, state):
        """Capture fail-closed stop requests."""
        self.stops.append(state)
        return 0

    def get_dh_params(self):
        """Return synthetic opaque geometry for contract tests only."""
        return 0, [0.0] * 28


def test_tcp_latest_units_release_hold_and_sdk_failure():
    """Native targets use mm/rad and only successful submissions become labels."""
    arm = FakeTCPArm()
    control = TCPControl(arm, vr_config())
    measured = control.read_feedback(1.0)
    control.set_following(True, 1.0)
    control.accept(np.r_[[0.3, 0.0, 0.3], IDENTITY[3:]], 1.0)
    control.accept(np.r_[[0.4, 0.0, 0.3], IDENTITY[3:]], 1.0)
    code, target = control.command(1.01)
    assert code == 0
    assert target[0] == pytest.approx(0.4)
    assert arm.commands[-1][0][0] == pytest.approx(400.0)
    assert arm.commands[-1][1]["wait"] is False
    assert arm.commands[-1][1]["speed"] == pytest.approx(100.0)
    control.set_following(False, 1.02)
    _, hold = control.command(1.03)
    np.testing.assert_allclose(hold, measured)
    arm.code = 22
    code, target = control.command(1.04)
    assert code == 22 and target is None
    assert arm.stops == [4]
    assert control.command(1.05)[1] is None


def test_tcp_stale_feedback_and_stale_deadman():
    """Old input holds measured TCP; old robot feedback triggers SDK stop, not motion."""
    arm = FakeTCPArm()
    control = TCPControl(arm, vr_config())
    measured = control.read_feedback(1.0)
    control.set_following(True, 0.5)
    control.accept(np.r_[[0.7, 0.0, 0.3], IDENTITY[3:]], 0.5)
    np.testing.assert_allclose(control.command(1.01)[1], measured)
    assert control.command(1.5)[1] is None
    assert arm.stops == [4]
    control.clear()
    assert not control.following and control.target is None


def test_tcp_angular_rate_and_quaternion_sign():
    """A large requested rotation is geodesically rate limited, not Euler interpolated."""
    control = TCPControl(FakeTCPArm(), vr_config())
    control.read_feedback(1.0)
    control.set_following(True, 1.0)
    control.accept(np.r_[[0.2, 0.0, 0.3], -from_rotvec([0.0, 0.0, 1.0])], 1.0)
    _, target = control.command(1.01)
    assert np.linalg.norm(rotvec(target[3:])) == pytest.approx(0.005)


def test_driver_watchdog_cannot_resume_from_repeated_true():
    """Even a ROS-only transport stall requires a fresh false->true deadman edge."""
    control = TCPControl(FakeTCPArm(), vr_config())
    measured = control.read_feedback(1.0)
    control.set_following(True, 0.5)
    target = np.r_[[0.7, 0.0, 0.3], IDENTITY[3:]]
    control.accept(target, 0.5)
    np.testing.assert_allclose(control.command(1.01)[1], measured)
    assert control.waiting_release
    control.set_following(True, 1.02)
    control.accept(target, 1.02)
    np.testing.assert_allclose(control.command(1.03)[1], measured)
    control.set_following(False, 1.04)
    control.set_following(True, 1.05)
    control.accept(target, 1.05)
    assert control.command(1.06)[1][0] == pytest.approx(0.7)


def test_geometry_snapshot_and_resume_contract(tmp_path):
    """Equal 8D shapes cannot mix joint/TCP semantics or changed tool geometry."""
    geometry = robot_geometry(FakeTCPArm())
    contract = make_contract(TCP_ACTION, geometry)
    write_contract(tmp_path, contract)
    validate_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="mismatch"):
        validate_contract(tmp_path, make_contract(JOINT_ACTION))
    changed = json.loads(json.dumps(contract))
    changed["robot_geometry"]["tcp_offset_mm_rad"][2] = 100.0
    with pytest.raises(ValueError, match="robot_geometry"):
        validate_contract(tmp_path, changed)
    legacy = tmp_path / "legacy"
    (legacy / "meta").mkdir(parents=True)
    (legacy / "meta/info.json").write_text("{}")
    validate_contract(legacy, make_contract(JOINT_ACTION))
    with pytest.raises(ValueError, match="legacy"):
        validate_contract(legacy, contract)
    arm = FakeTCPArm()
    arm.world_offset = [1.0] * 6
    with pytest.raises(ValueError, match="world_offset"):
        robot_geometry(arm)


def test_vr_sync_causal_command_and_joint_state():
    """State stays measured joints; action is prior accepted TCP, never future TCP."""
    config = load_system_config(PROFILE).recording
    config = replace(config, streams=(config.streams[0],), anchor_key=config.streams[0].key)
    sync = FrameSynchronizer(config)
    anchor = TimestampedSample(1.0, np.zeros((4, 4, 3), dtype=np.uint8))
    sync.xarm_state.add(1.0, XarmState(np.arange(7), np.zeros(7), np.zeros(7)))
    sync.robotiq_state.add(1.0, [0.4])
    accepted = np.r_[[0.2, 0.1, 0.3], IDENTITY[3:]]
    sync.xarm_action.add(0.99, accepted)
    sync.xarm_action.add(1.01, np.ones(7))
    sync.robotiq_action.add(0.99, [0.6])
    sync.teleop_status.add(1.0, [1.0, 0.0])
    frame = sync.assemble(anchor, "insert")
    np.testing.assert_allclose(frame["action"], np.r_[accepted, 0.6])
    np.testing.assert_allclose(frame["observation.state"], np.r_[np.arange(7), 0.4])
    np.testing.assert_array_equal(frame["observation.teleop_status"], [1.0, 0.0])
    sync.clear()
    assert sync.xarm_action.newest() is None
    with pytest.raises(SynchronizationError):
        sync.assemble(anchor, "insert")


def test_config_profile_preview_and_incompatible_routes(tmp_path):
    """VR inherits cameras and preserves TCP semantics when RL is selected."""
    system = load_system_config(PROFILE)
    assert system.recording.action_format == TCP_ACTION
    assert len(system.recording.streams) == 3
    with pytest.raises(ValueError, match="workspace"):
        VRConfig.from_mapping(system.teleop["vr"]).require_workspace()
    raw = load_system_mapping(PROFILE)
    raw["rl"]["enabled"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_system_config(path).recording.action_format == TCP_ACTION


def test_schema3_requires_coherent_buttons_and_legacy_is_diagnostic_only():
    """New APK declares epochs/inputs; v2 remains parseable with input_valid=false."""
    sample = {
        "hand": "left",
        "tracked": True,
        "positionMeters": {"x": 0, "y": 0, "z": 0},
        "rotationXyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
        "inputValid": True,
        "grip": True,
        "trigger": 0.7,
    }
    payload = {
        "schemaVersion": 3,
        "frameId": "quest_floor",
        "timestampSeconds": 1.0,
        "sessionId": "s",
        "originRevision": 0,
        "left": sample,
        "right": {**sample, "hand": "right"},
    }
    assert parse_pose_frame(json.dumps(payload)).right.trigger == 0.7
    payload["schemaVersion"] = 2
    assert not parse_pose_frame(json.dumps(payload)).right.input_valid
    payload["schemaVersion"] = 3
    payload["right"]["trigger"] = float("nan")
    with pytest.raises(ValueError, match="trigger"):
        parse_pose_frame(json.dumps(payload))


def test_recording_script_vr_dry_run_has_no_hardware():
    """The shared entry point resolves the VR profile without ROS/USB startup."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/record_data.sh"), "--source", "vr", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "no hardware was started" in result.stdout


def test_synthetic_tcp_capture_four_category_real_writers(tmp_path):
    """Fake SDK -> shared synchronizer -> real LeRobot four-category save/resume.

    Tiny CPU videos and Parquet files are created only in pytest's temporary
    directory. This verifies data ownership without ROS hardware or downloads.
    """
    from xarm_ros2_teleop.recording.lerobot_writer import LeRobotWriter

    system = load_system_config(PROFILE)
    config = replace(
        system.recording,
        streams=(system.recording.streams[0],),
        anchor_key=system.recording.streams[0].key,
    )
    sync = FrameSynchronizer(config)
    arm = FakeTCPArm()
    control = TCPControl(arm, vr_config())
    control.read_feedback(1.0)
    _, accepted = control.command(1.01)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    anchor = TimestampedSample(1.02, image)
    sync.images[config.anchor_key].add(anchor.timestamp, image)
    sync.xarm_state.add(1.02, XarmState(np.zeros(7), np.zeros(7), np.zeros(7)))
    sync.robotiq_state.add(1.02, [0.2])
    sync.xarm_action.add(1.01, accepted)
    sync.robotiq_action.add(1.01, [0.2])
    sync.teleop_status.add(1.02, [1.0, 0.0])
    frame = sync.assemble(anchor, "synthetic insert")
    contract = make_contract(TCP_ACTION, robot_geometry(arm))
    writers = {
        category: LeRobotWriter(
            root=tmp_path / category,
            repo_id=f"local/{category}",
            robot_type="xarm7_vr_tcp",
            fps=30,
            vcodec="h264",
            image_writer_threads=1,
            streaming_encoding=False,
            control_contract=contract,
        )
        for category in ("success", "optimal", "failure", "crash")
    }
    try:
        for writer in writers.values():
            writer.prepare(frame)
        source = writers["success"]
        for destination in writers.values():
            source.add_frame(frame)
            source.add_frame(frame)
            source.save_episode_to(destination)
        assert [writer.episode_count for writer in writers.values()] == [1, 1, 1, 1]
        for category in writers:
            validate_contract(tmp_path / category, contract)
            assert list((tmp_path / category / "data").rglob("*.parquet"))
            assert list((tmp_path / category / "videos").rglob("*.mp4"))
    finally:
        for writer in writers.values():
            writer.finalize()


@pytest.mark.parametrize("tcp_rl", [False, True])
def test_native_driver_mode_reset_and_no_stale_tcp_replay(tmp_path, monkeypatch, tcp_rl):
    """The real ROS driver selects mode 7, yields to joint reset, then forgets VR targets."""
    import time

    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from xarm_ros2_teleop.vr.pose_math import pose_message
    from xarm_ros2_teleop.xarm import xarm_controller

    class DriverArm(FakeTCPArm):
        """SDK test double implementing the common driver's connection and reset API."""

        def __init__(self, *args, **kwargs):
            """Initialize disconnected without opening a physical transport."""
            super().__init__()
            self.connected = False
            self.modes, self.joints = [], []

        def connect(self):
            """Mark the fake connection online."""
            self.connected = True

        def disconnect(self):
            """Mark the fake connection offline."""
            self.connected = False

        def motion_enable(self, enabled):
            """Acknowledge motor enable without hardware."""
            return 0

        def set_mode(self, mode):
            """Capture actual driver mode selection."""
            self.modes.append(mode)
            return 0

        def set_servo_angle(self, **kwargs):
            """Capture the shared reset's joint target."""
            self.joints.append(kwargs["angle"])
            return 0

        def is_joint_limit(self, *args, **kwargs):
            """Treat test reset joints as valid."""
            return 0, False

    raw = load_system_mapping(PROFILE)
    raw["teleop"]["vr"].update(
        workspace_lower_m=[-1.0, -1.0, -1.0], workspace_upper_m=[1.0, 1.0, 1.0]
    )
    path = tmp_path / "driver.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(xarm_controller, "XArmAPI", DriverArm)
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"config_path:={path}",
            "-p",
            "defer_initial_motion:=true",
            "-p",
            "recording_gate_enabled:=true",
            "-p",
            "reset_enabled:=true",
            "-p",
            "reset_lift_mm:=0.0",
            "-p",
            f"tcp_rl_enabled:={str(tcp_rl).lower()}",
        ]
    )
    node = xarm_controller.XarmNode()
    try:
        topics = {subscription.topic_name for subscription in node.subscriptions}
        assert ("/rl/episode_active" if tcp_rl else "/teleop/episode_active") in topics
        assert ("/xarm/tcp_following" if tcp_rl else "/vr/following") in topics
        node.latest_error_code = 0
        node.episode_active_callback(Bool(data=True))
        assert node.arm.modes[-1] == 7
        node.tcp.read_feedback(time.monotonic())
        node.tcp_follow_callback(Bool(data=True))
        message = pose_message(
            np.r_[[0.4, 0.0, 0.3], IDENTITY[3:]], node.get_clock().now().to_msg()
        )
        node.tcp_callback(message)
        node.control_loop()
        assert node.arm.commands[-1][0][0] == pytest.approx(400.0)
        assert not node.arm.joints
        previous = len(node.arm.commands)
        node.reset_active_callback(Bool(data=True))
        assert node.arm.modes[-1] == 6
        node.tcp_callback(message)
        node.reset_target_callback(JointState(position=[0.0] * 7))
        node.control_loop()
        assert len(node.arm.commands) == previous
        assert len(node.arm.joints) == 1
        node.reset_active_callback(Bool(data=False))
        assert node.arm.modes[-1] == 7
        assert node.tcp.target is None and not node.tcp.following
        node.tcp.read_feedback(time.monotonic())
        node.control_loop()
        assert node.arm.commands[-1][0][0] == pytest.approx(200.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_vr_preview_launch_creates_no_actuator_or_recorder(monkeypatch):
    """Build launch actions without executing them; preview never creates SDK owners."""
    import importlib.util

    from launch import LaunchContext
    from launch_ros.actions import Node

    path = ROOT / "xarm_ros2_teleop/launch/teleop.launch.py"
    spec = importlib.util.spec_from_file_location("vr_launch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_share = module.get_package_share_directory
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda name: (
            str(ROOT / "xarm_ros2_teleop") if name == "xarm_ros2_teleop" else original_share(name)
        ),
    )
    context = LaunchContext()
    context.launch_configurations.update(
        config=str(PROFILE),
        inference="false",
        inference_motion="false",
        rl="false",
        rl_motion="false",
        vr_motion="false",
        vr_rviz="false",
    )
    actions = module._launch_setup(context)
    executables = [action.node_executable for action in actions if isinstance(action, Node)]
    assert "vr_teleop" in executables and "quest_pose_node" in executables
    assert not {
        "factr_teleop_node",
        "xarm_node",
        "robotiq_node",
        "reset_node",
        "episode_recorder",
        "session_coordinator",
    }.intersection(executables)
