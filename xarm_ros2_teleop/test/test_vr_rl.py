"""TCP-policy/VR ownership, SDK provenance and deployment tests without devices."""

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import yaml
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from xarm_ros2_teleop.inference.contracts import ACTION_FORMAT, validate_policy_metadata
from xarm_ros2_teleop.inference.remote_policy_node import RemotePolicyNode
from xarm_ros2_teleop.inference.tcp_contracts import (
    TCP_ACTION_FORMAT,
    interpolate_tcp_chunk,
    validate_tcp_chunk,
)
from xarm_ros2_teleop.recording.synchronizer import SynchronizationError
from xarm_ros2_teleop.rl.human_intervention_node import HumanInterventionNode
from xarm_ros2_teleop.rl.intervention import InterventionState
from xarm_ros2_teleop.rl.recording import RLFrameHistory, write_episode_label
from xarm_ros2_teleop.system.config import load_system_mapping
from xarm_ros2_teleop.vr.config import VRConfig
from xarm_ros2_teleop.vr.pose_math import from_rotvec, pose_array, pose_message, rotvec
from xarm_ros2_teleop.vr.teleop_node import VRTeleopNode
from xarm_ros2_teleop.xarm.tcp_control import TCPControl

ROOT = Path(__file__).parents[2]
PROFILE = ROOT / "xarm_ros2_teleop/config/teleop_vr_rl.yaml"
POSE = np.array([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
ACTION = np.r_[POSE, 0.4].astype(np.float32)


class Publisher:
    """Collect typed ROS messages without DDS or hardware side effects."""

    def __init__(self):
        """Initialize an empty message list."""
        self.messages = []

    def publish(self, message):
        """Retain one ROS message for deterministic assertion/routing."""
        self.messages.append(message)


def tcp_config():
    """Return a test-only metre workspace; never use as a real robot envelope."""
    return VRConfig.from_mapping(
        {
            "workspace_lower_m": [-1, -1, -1],
            "workspace_upper_m": [1, 1, 1],
        }
    )


def metadata():
    """Return the required explicit TCP deployment handshake mapping."""
    return {
        "protocol_version": 1,
        "state_dim": 8,
        "action_dim": 8,
        "action_format": TCP_ACTION_FORMAT,
        "state_format": ACTION_FORMAT,
        "action_frame": "xarm_base",
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "image_keys": ["front"],
        "execution_horizon": 25,
        "action_frequency_hz": 30.0,
    }


@pytest.mark.parametrize(
    "field",
    [
        "protocol_version",
        "action_format",
        "state_format",
        "action_frame",
        "position_unit",
        "quaternion_order",
        "action_dim",
        "action_frequency_hz",
    ],
)
def test_tcp_requires_explicit_metadata(field):
    """TCP cannot inherit the legacy joint server's silent defaults."""
    info = metadata()
    assert validate_policy_metadata(info, ["front"], 30, TCP_ACTION_FORMAT) == []
    del info[field]
    with pytest.raises(ValueError):
        validate_policy_metadata(info, ["front"], 30, TCP_ACTION_FORMAT)


def test_tcp_rejects_old_act_and_invalid_targets():
    """Eight coordinates do not make a joint ACT or malformed quaternion compatible."""
    with pytest.raises(ValueError, match="action_format"):
        validate_policy_metadata(
            {**metadata(), "action_format": ACTION_FORMAT}, ["front"], 30, TCP_ACTION_FORMAT
        )
    for column, value in ((0, 2.0), (3, float("nan")), (6, 0), (7, 1.5)):
        candidate = ACTION.copy()
        candidate[column] = value
        with pytest.raises(ValueError):
            validate_tcp_chunk([candidate], tcp_config())


def test_tcp_interpolation_uses_shortest_arc_and_exact_chunk_duration():
    """Opposite quaternion signs cannot cause zero norm; angle uses SLERP."""
    first, last = ACTION.copy(), ACTION.copy()
    last[:3] += 0.1
    last[3:7] = -from_rotvec([0, 0, 1.2])
    actions = validate_tcp_chunk([first, last], tcp_config())
    middle = interpolate_tcp_chunk(actions, 0.5 / 30, 30)
    np.testing.assert_allclose(middle[:3], first[:3] + 0.05, atol=1e-7)
    np.testing.assert_allclose(rotvec(middle[3:7]), [0, 0, 0.6], atol=1e-7)
    assert np.linalg.norm(middle[3:7]) == pytest.approx(1)
    assert interpolate_tcp_chunk(actions, 2 / 30, 30) is None


@pytest.fixture
def nodes(tmp_path, monkeypatch):
    """Construct real ROS nodes with no sockets/SDK, replacing only output publishers."""
    raw = load_system_mapping(PROFILE)
    raw["teleop"]["vr"].update(
        workspace_lower_m=[-1, -1, -1],
        workspace_upper_m=[1, 1, 1],
    )
    raw["rl"]["keyboard_enabled"] = False
    path = tmp_path / "vr_rl.yaml"
    path.write_text(yaml.safe_dump(raw))
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"config_path:={path}",
            "-p",
            "motion_enabled:=true",
            "-p",
            "rl_enabled:=true",
        ]
    )
    arbiter, vr = HumanInterventionNode(), VRTeleopNode()
    for node, names in (
        (
            arbiter,
            (
                "tcp_pub",
                "tcp_follow_pub",
                "gripper_pub",
                "command_event_pub",
                "human_enable_pub",
                "policy_paused_pub",
                "reward_pub",
            ),
        ),
        (vr, ("target_pub", "follow_pub", "gripper_pub", "candidate_pub", "status_pub")),
    ):
        for name in names:
            setattr(node, name, Publisher())
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    arbiter._arm_callback(JointState(position=[0.1] * 7, velocity=[0.0] * 7))
    arbiter._tcp_pose_callback(pose_message(POSE, arbiter.get_clock().now().to_msg()))
    arbiter._gripper_callback(JointState(position=[0.4]))
    arbiter._control_callback(Bool(data=True))
    arbiter._vr_status_callback(String(data=json.dumps({"ready": True})))
    arbiter._update_human_ready()
    arbiter._episode_callback(Bool(data=True))
    vr.pose_callback(pose_message(POSE, vr.get_clock().now().to_msg()))
    vr.gripper_callback(JointState(position=[0.4]))
    vr.health_callback(Bool(data=True))
    vr.gate_callback("episode_active", True)
    try:
        yield arbiter, vr, now
    finally:
        arbiter.destroy_node()
        vr.destroy_node()
        rclpy.shutdown()


def candidate(node, action, **extra):
    """Return a fresh explicit TCP String candidate carrying float action [8]."""
    return String(
        data=json.dumps(
            {
                "action_format": TCP_ACTION_FORMAT,
                "timestamp": node.get_clock().now().nanoseconds / 1e9,
                "action": action.tolist() if action is not None else None,
                **extra,
            }
        )
    )


def feed_vr(vr, now, grip, x=0.0):
    """Feed one coherent valid schema-3 frame and run its timer, without hardware."""
    vr.sample_callback(
        String(
            data=json.dumps(
                {
                    "schema_version": 3,
                    "generation": ["test", 0, 1],
                    "timestamp_seconds": now,
                    "hands": {
                        "right": {
                            "pose": [x, 0, 0, 0, 0, 0, 1],
                            "trigger": 0.2,
                            "grip": grip,
                            "tracked": True,
                            "input_valid": True,
                        }
                    },
                }
            )
        )
    )
    vr.tick()


def test_real_nodes_switch_clutch_hold_and_fresh_resume(nodes):
    """Real arbiter/VR code shares one TCP publisher; takeover requires a new clutch."""
    arbiter, vr, now = nodes
    target = ACTION.copy()
    target[0] += 0.1
    arbiter._tcp_policy_callback(candidate(arbiter, target))
    action, source = arbiter._selected_action()
    assert source == "policy"
    arbiter._apply_action(action, source)
    np.testing.assert_allclose(pose_array(arbiter.tcp_pub.messages[-1]), target[:7])
    assert arbiter._handle_key("switch")[0]
    assert arbiter.controller.state == InterventionState.STOPPING
    arbiter._advance_stopping()
    # Fresh velocity/TCP/gripper and input are required for the full stop window.
    now[0] += 0.31
    arbiter.robot_state_received_at = arbiter.gripper_state_received_at = now[0]
    arbiter.velocity_received_at = now[0]
    arbiter._advance_stopping()
    assert arbiter.controller.state == InterventionState.HUMAN_CONTROL
    vr.gate_callback("human_enabled", arbiter.human_enable_pub.messages[-1].data)
    for _ in range(5):
        now[0] += 0.1
        vr.robot_at = vr.gripper_at = now[0]
        feed_vr(vr, now[0], True)
        assert not vr.mapper.following
    now[0] += 0.01
    feed_vr(vr, now[0], False)
    now[0] += 0.01
    feed_vr(vr, now[0], True)
    assert vr.mapper.following
    arbiter._vr_action_callback(vr.candidate_pub.messages[-1])
    action, source = arbiter._selected_action()
    assert source == "human"
    np.testing.assert_allclose(action, ACTION)
    now[0] += 0.01
    feed_vr(vr, now[0], False)
    arbiter._vr_action_callback(vr.candidate_pub.messages[-1])
    action, source = arbiter._selected_action()
    assert source == "hold"
    arbiter.robot_state_received_at = arbiter.gripper_state_received_at = now[0]
    arbiter._apply_action(action, source)
    assert not arbiter.tcp_follow_pub.messages[-1].data
    assert not vr.target_pub.messages and not vr.follow_pub.messages and not vr.gripper_pub.messages
    assert arbiter._handle_key("switch")[0]
    assert arbiter.controller.state == InterventionState.POLICY
    assert arbiter.policy_action is None
    assert not arbiter.human_enable_pub.messages[-1].data
    assert not arbiter.policy_paused_pub.messages[-1].data


def test_stale_wrong_owner_and_reset_targets_rejected(nodes):
    """Queued pre-handoff targets, wrong action semantics and reset cannot move TCP."""
    arbiter, vr, _ = nodes
    payload = json.loads(candidate(arbiter, ACTION).data)
    payload["timestamp"] -= 1
    arbiter._tcp_policy_callback(String(data=json.dumps(payload)))
    assert arbiter.policy_action is None
    arbiter._vr_action_callback(candidate(arbiter, ACTION, following=True))
    assert arbiter.vr_action is None
    payload["action_format"] = ACTION_FORMAT
    arbiter._tcp_policy_callback(String(data=json.dumps(payload)))
    assert arbiter.controller.state == InterventionState.PAUSED
    arbiter._reset_active_callback(Bool(data=True))
    assert arbiter._selected_action() == (None, "hold")
    assert not arbiter.human_enable_pub.messages[-1].data
    vr.gate_callback("reset_active", True)
    assert not vr.mapper.following


def test_human_loss_pauses_and_reward_ends_episode(nodes):
    """Disconnect leaves the episode open, while an explicit reward ends it."""
    arbiter, _, now = nodes
    arbiter.controller.state = InterventionState.HUMAN_CONTROL
    now[0] += 0.26
    arbiter._update_human_ready()
    assert not arbiter.human_ready
    _, source = arbiter._selected_action()
    assert source == "hold" and arbiter.episode_active
    assert arbiter.controller.state == InterventionState.PAUSED
    arbiter._publish_reward("failure")
    assert not arbiter.episode_active
    assert json.loads(arbiter.reward_pub.messages[-1].data)["reward"] == 0
    assert not arbiter.human_enable_pub.messages[-1].data


def test_tcp_sdk_limited_pose_joins_original_selection_and_hold(tmp_path):
    """Actual SDK output can differ from selected policy, with auditable source linkage."""

    class Arm:
        """Fake native TCP SDK connection; no physical transport."""

        def get_position_aa(self, **kwargs):
            """Return measured TCP mm/axis-angle [6]."""
            return 0, [200, 0, 300, 0, 0, 0]

        def set_position_aa(self, pose, **kwargs):
            """Accept absolute TCP [6]; SDK arguments tested by original VR tests."""
            return 0

    control = TCPControl(Arm(), tcp_config())
    control.read_feedback(1)
    target = ACTION.copy()
    target[3:7] = from_rotvec([0, 0, 0.8])
    history = RLFrameHistory(2, TCP_ACTION_FORMAT)
    history.add(
        {
            "timestamp": 10.0,
            "source": "policy",
            "action_format": TCP_ACTION_FORMAT,
            "selected_action": target,
            "policy_action": target,
            "policy_valid": True,
        }
    )
    control.set_following(True, 1)
    control.accept(target[:7], 1, selection=10.0)
    code, actual = control.command(1.01)
    assert code == 0 and not np.allclose(actual, target[:7])
    history.add_acceptance(
        {
            "timestamp": 10.01,
            "selection_timestamp": control.command_selection,
            "accepted_pose": actual,
        }
    )
    frame = {"action": np.r_[actual, 0.4].astype(np.float32)}
    frame["action"][3:7] *= -1  # Recorder may choose the equivalent quaternion sign.
    history.annotate(frame, 10.02, (10.01, 10.01))
    assert frame["action.source"][0] == 0
    np.testing.assert_allclose(frame["action.selected"], target)
    control.set_following(False, 1.02)
    _, held = control.command(1.02)
    history.add_acceptance({"timestamp": 10.02, "selection_timestamp": None, "accepted_pose": held})
    frame = {"action": np.r_[held, 0.4]}
    history.annotate(frame, 10.03, (10.02, 10.02))
    assert frame["action.source"][0] == 2 and not frame["action.policy_valid"][0]
    with pytest.raises(SynchronizationError):
        history.annotate(frame, 10.04, (10.035, 10.035))
    write_episode_label(
        tmp_path,
        0,
        {
            "outcome": "success",
            "reward": 1,
            "terminated": True,
            "truncated": False,
            "action_format": TCP_ACTION_FORMAT,
        },
    )
    assert (
        json.loads((tmp_path / "rl/episode_000000.json").read_text())["action_format"]
        == TCP_ACTION_FORMAT
    )


def test_tcp_remote_emits_no_joint_commands():
    """TCP remote inference produces candidates only, never invokes joint conversion."""
    node = SimpleNamespace(
        tcp_mode=True,
        policy_action_pub=Publisher(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100)),
    )
    RemotePolicyNode._execute_action(node, ACTION)
    result = json.loads(node.policy_action_pub.messages[-1].data)
    assert result["action_format"] == TCP_ACTION_FORMAT
    np.testing.assert_allclose(result["action"], ACTION)


def test_vr_rl_launch_has_one_arbiter_and_no_factr(tmp_path, monkeypatch):
    """Resolve launch without execution and check episode gates/actuator ownership."""
    from launch import LaunchContext
    from launch_ros.actions import Node

    path = ROOT / "xarm_ros2_teleop/launch/teleop.launch.py"
    spec = importlib.util.spec_from_file_location("vr_rl_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_package_share_directory", lambda name: str(ROOT / name))
    raw = load_system_mapping(PROFILE)
    raw["teleop"]["vr"].update(workspace_lower_m=[-1, -1, -1], workspace_upper_m=[1, 1, 1])
    profile = tmp_path / "launch.yaml"
    profile.write_text(yaml.safe_dump(raw))
    context = LaunchContext()
    context.launch_configurations.update(
        config=str(profile),
        inference="false",
        inference_motion="false",
        rl="true",
        rl_motion="true",
        vr_motion="false",
        vr_rviz="false",
    )
    actions = module._launch_setup(context)
    nodes = {action.node_executable: action for action in actions if isinstance(action, Node)}
    assert {
        "vr_teleop",
        "human_intervention",
        "remote_policy_inference",
        "rl_episode_recorder",
        "xarm_node",
        "robotiq_node",
        "reset_node",
    }.issubset(nodes)
    assert "factr_teleop_node" not in nodes
    # Humble exposes no public raw parameter accessor. Evaluate its normalized
    # dictionaries, without executing the launch action or starting a process.
    from launch.utilities import perform_substitutions
    from launch_ros.utilities import evaluate_parameters

    assert evaluate_parameters(context, nodes["vr_teleop"]._Node__parameters)[0]["rl_enabled"]
    assert evaluate_parameters(context, nodes["xarm_node"]._Node__parameters)[0]["tcp_rl_enabled"]
    remaps = [
        (perform_substitutions(context, source), perform_substitutions(context, target))
        for source, target in nodes["robotiq_node"]._Node__remappings
    ]
    assert ("/teleop/episode_active", "/rl/episode_active") in remaps


def test_vr_rl_script_dry_run_never_starts_devices():
    """The public uv/no-sync entry resolves the TCP RL profile without connecting."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/record_data.sh"), "--source", "vr", "--rl", "--dry-run"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "no hardware was started" in result.stdout
