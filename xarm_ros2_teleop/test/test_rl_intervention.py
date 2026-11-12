"""Validate single-button human takeover without ROS hardware."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from xarm_ros2_teleop.rl.human_intervention_node import (
    RL_SERVICE_COMMANDS,
    HumanInterventionNode,
    decode_rl_key,
)
from xarm_ros2_teleop.rl.intervention import (
    InterventionController,
    InterventionState,
    RewardEvent,
)


def test_automatic_takeover_is_continuous_and_relative() -> None:
    """Automatic engagement must start exactly at the measured robot pose."""

    robot = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.5], dtype=np.float32)
    factr = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.2], dtype=np.float32)
    controller = InterventionController(gripper_delta_scale=-2.0)

    assert np.array_equal(controller.begin_takeover(robot), robot)
    assert controller.state == InterventionState.STOPPING
    assert np.array_equal(controller.engage_human(robot, factr), robot)

    moved = factr.copy()
    moved[:7] += 0.02
    moved[7] += 0.1
    target = controller.human_action(moved)
    assert np.allclose(target[:7], robot[:7] + 0.02)
    assert target[7] == pytest.approx(0.3)

    controller.pause(target)
    assert controller.state == InterventionState.PAUSED
    controller.resume_policy()
    assert controller.state == InterventionState.POLICY


def test_takeover_rejects_out_of_order_engagement() -> None:
    """Relative control cannot start before stopping and alignment phases."""

    state = np.zeros(8, dtype=np.float32)
    controller = InterventionController()
    with pytest.raises(RuntimeError, match="cannot engage human"):
        controller.engage_human(state, state)


def test_reward_event_and_keyboard_contract() -> None:
    """Keyboard labels and JSON reward transport stay stable for RL clients."""

    assert decode_rl_key(b"T") == "switch"
    assert decode_rl_key(b"\r") is None
    assert decode_rl_key(b"Y") == "success"
    assert decode_rl_key(b"N") == "failure"
    assert decode_rl_key(b"A") is None
    assert decode_rl_key(b"R") is None
    assert decode_rl_key(b"\x1b") == "stop"
    assert decode_rl_key(b"?") is None

    payload = json.loads(RewardEvent(1.25, 1.0, "task_success", True, False).to_json())
    assert payload == {
        "event": "task_success",
        "reward": 1.0,
        "source": "human",
        "terminated": True,
        "timestamp": 1.25,
        "truncated": False,
    }


def test_web_services_and_keyboard_share_one_ownership_switch() -> None:
    """One press waits for stopping; repeated presses cannot bypass that gate."""

    published = []
    robot = np.zeros(8, dtype=np.float32)
    human = np.ones(8, dtype=np.float32)
    node = SimpleNamespace(
        controller=InterventionController(), reset_in_progress=False, episode_active=True,
        robot_state=robot, human_ready=True, executed_action=None, stable_since=None,
        policy_action=np.ones(8, dtype=np.float32), policy_paused_pub="paused",
        _publish_bool=lambda publisher, value: published.append((publisher, value)),
        _publish_state=lambda detail: published.append(("status", detail)),
        _human_state=lambda: human,
    )
    assert HumanInterventionNode._handle_key(node, "switch")[0]
    assert node.controller.state == InterventionState.STOPPING
    assert not HumanInterventionNode._handle_key(node, "switch")[0]
    node.controller.engage_human(robot, human)  # Simulate the measured stop gate.
    assert HumanInterventionNode._handle_key(node, "switch")[0]
    assert node.controller.state == InterventionState.POLICY
    assert node.policy_action is None
    assert published[-2] == ("paused", False)
    assert RL_SERVICE_COMMANDS == {
        "start": "/rl/start", "discard": "/rl/discard", "finish": "/rl/finish",
        "switch": "/rl/switch_control", "success": "/rl/success",
        "failure": "/rl/failure", "stop": "/rl/stop",
    }


def test_rl_status_uses_stable_state_token_separator() -> None:
    """Dashboard consumers receive a plain enum token before the status detail."""

    messages = []
    node = SimpleNamespace(
        controller=InterventionController(),
        status_pub="status",
        _publish_string=lambda publisher, value: messages.append((publisher, value)),
    )
    HumanInterventionNode._publish_state(node, "ready")
    assert messages == [("status", "POLICY|ready")]


def test_removed_reset_and_abort_commands_do_not_change_ownership() -> None:
    """Per-episode save/discard and the shared recorder now own all reset requests."""

    node = SimpleNamespace(reset_in_progress=False, episode_active=True)
    for command in ("reset", "abort", "resume", "takeover"):
        accepted, detail = HumanInterventionNode._handle_key(node, command)
        assert not accepted
        assert "unknown RL command" in detail


def test_rl_stop_pauses_at_measured_pose_and_remains_recoverable() -> None:
    """RL stop becomes a measured hold that still permits takeover or resume."""

    published = []
    robot = np.arange(8, dtype=np.float32) / 10.0
    node = SimpleNamespace(
        robot_state=robot,
        controller=InterventionController(),
        executed_action=None,
        policy_action=np.ones(8, dtype=np.float32),
        stable_since=1.0,
        policy_paused_pub="paused",
        _publish_bool=lambda publisher, value: published.append((publisher, value)),
        _publish_state=lambda detail: published.append(("status", detail)),
        get_logger=lambda: SimpleNamespace(warn=lambda message: None),
    )

    HumanInterventionNode._pause_policy(node, "operator paused policy control")

    assert node.controller.state == InterventionState.PAUSED
    np.testing.assert_array_equal(node.controller.hold_action, robot)
    np.testing.assert_array_equal(node.executed_action, robot)
    assert node.policy_action is None
    assert node.stable_since is None
    assert published == [
        ("paused", True),
        ("status", "operator paused policy control"),
    ]
    node.controller.begin_takeover(robot)
    assert node.controller.state == InterventionState.STOPPING


def test_rl_controller_fault_pauses_policy_for_coordinated_reset() -> None:
    """A collision/fault pauses policy output and leaves reset available."""

    reasons = []
    node = SimpleNamespace(
        motion_enabled=True,
        control_ok=True,
        _pause_policy=reasons.append,
    )

    from std_msgs.msg import Bool

    HumanInterventionNode._control_callback(node, Bool(data=False))

    assert reasons == ["xArm control health lost; waiting for coordinated reset"]
