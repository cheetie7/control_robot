"""Readiness, automatic no-jump engagement and reset exclusion without hardware."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from xarm_ros2_teleop.rl.human_input import HumanInput
from xarm_ros2_teleop.rl.human_intervention_node import HumanInterventionNode
from xarm_ros2_teleop.rl.intervention import InterventionController, InterventionState


def _node() -> SimpleNamespace:
    """Return an active arbiter with float32 [8] robot/input state and no ROS publishers."""

    node = SimpleNamespace(
        tcp_mode=False,
        controller=InterventionController(), episode_active=True, reset_in_progress=False,
        human_ready=True, robot_state=np.zeros(8, dtype=np.float32),
        robot_velocity=np.zeros(7), robot_state_received_at=time.monotonic(),
        gripper_state_received_at=time.monotonic(),
        executed_action=None, policy_action=np.ones(8), stable_since=None,
        control_ok=True, policy_timeout_sec=0.25, stop_velocity_rad_s=0.03, stop_stable_sec=0.3,
        policy_paused_pub="paused", _human_state=lambda: np.ones(8, dtype=np.float32),
        _publish_bool=lambda _publisher, _value: None, _publish_state=lambda _detail: None,
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )
    node._pause_policy = lambda reason: HumanInterventionNode._pause_policy(node, reason)
    return node


def test_one_press_waits_for_measured_stop_then_automatically_engages(monkeypatch) -> None:
    """The engagement origin follows the final settled pose, not the request-time pose."""

    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    node = _node()
    assert HumanInterventionNode._handle_key(node, "switch")[0]
    node.robot_velocity[:] = 0.1
    HumanInterventionNode._advance_stopping(node)
    assert node.stable_since is None
    node.robot_velocity[:] = 0
    HumanInterventionNode._advance_stopping(node)
    now[0] += 0.31
    node.robot_state_received_at = now[0]
    node.gripper_state_received_at = now[0]
    node.robot_state[:] = 0.02
    HumanInterventionNode._advance_stopping(node)
    assert node.controller.state == InterventionState.HUMAN_CONTROL
    np.testing.assert_array_equal(node.executed_action, node.robot_state)


@pytest.mark.parametrize("fault", ["human", "velocity", "robot_stale", "gripper_stale", "control"])
def test_switch_fails_closed_if_inputs_or_robot_health_are_lost(fault: str) -> None:
    """No automatic engagement on stale state, lost device or missing stop feedback."""

    node = _node()
    HumanInterventionNode._handle_key(node, "switch")
    if fault == "human":
        node._human_state = lambda: None
    elif fault == "velocity":
        node.robot_velocity = None
    elif fault == "robot_stale":
        node.robot_state_received_at -= 1
    elif fault == "gripper_stale":
        node.gripper_state_received_at -= 1
    else:
        node.control_ok = False
    HumanInterventionNode._advance_stopping(node)
    assert node.controller.state == InterventionState.PAUSED
    assert node.episode_active


def test_device_readiness_requires_both_channels_continuously_fresh() -> None:
    """Readiness permits hand motion but resets after either channel stalls."""

    source = HumanInput()
    assert not source.ready(10, 0.25, 0.3)
    source.arm = np.ones(7, dtype=np.float32)
    source.gripper = 0.2
    source.connected = True
    source.arm_received_at = source.gripper_received_at = 10
    assert not source.ready(10, 0.25, 0.3)
    source.arm_received_at = source.gripper_received_at = 10.31
    assert source.ready(10.31, 0.25, 0.3)
    source.arm_received_at = 10.1
    assert not source.ready(10.4, 0.25, 0.3)
    source.arm_received_at = source.gripper_received_at = 10.4
    assert not source.ready(10.4, 0.25, 0.3)


def test_switch_rejects_unready_input_without_ending_episode() -> None:
    """A missing human device never acquires control or invents an episode outcome."""

    node = _node()
    node.human_ready = False
    assert not HumanInterventionNode._handle_key(node, "switch")[0]
    assert node.controller.state == InterventionState.POLICY
    assert node.episode_active


def test_readiness_heartbeat_clears_when_device_stops_publishing(monkeypatch) -> None:
    """A stale latched True cannot keep new episodes enabled after device loss."""

    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    source = HumanInput(
        arm=np.ones(7), gripper=0.2, connected=True,
        arm_received_at=10, gripper_received_at=10, ready_since=9.6,
    )
    published = []
    node = SimpleNamespace(
        human_input=source, human_input_timeout_sec=0.25, human_ready_stable_sec=0.3,
        tcp_mode=False,
        human_ready=False, human_ready_published_at=0.0, human_ready_pub="ready",
        _publish_bool=lambda _topic, value: published.append(value),
        _publish_state=lambda _detail: None,
    )
    HumanInterventionNode._update_human_ready(node)
    assert node.human_ready
    now[0] += 0.11
    HumanInterventionNode._update_human_ready(node)
    now[0] += 0.3
    HumanInterventionNode._update_human_ready(node)
    assert published == [True, True, False]
    assert not node.human_ready


def test_idle_and_reset_never_allow_switching_or_action_forwarding() -> None:
    """The shared save/reset transaction remains the exclusive motion owner."""

    node = _node()
    node.episode_active = False
    assert not HumanInterventionNode._handle_key(node, "switch")[0]
    assert HumanInterventionNode._selected_action(node) == (None, "hold")
    node.episode_active = True
    node.reset_in_progress = True
    assert not HumanInterventionNode._handle_key(node, "switch")[0]
    assert HumanInterventionNode._selected_action(node) == (None, "hold")
