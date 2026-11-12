"""Pure interface and joint-mapping tests for coordinated episode reset."""

import time

import numpy as np
import pytest
import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rclpy.action import GoalResponse
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint
from xarm_ros2_teleop.reset.mapping import (
    factr_target_from_xarm,
    rebase_factr_origin,
)
from xarm_ros2_teleop.reset.reset_coordinator import (
    JOINT_NAMES,
    ResetCoordinator,
    normalize_reset_goal,
)


def _goal(names=JOINT_NAMES, positions=None):
    """Create one position-only reset goal for seven joints in radians."""

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(names)
    point = JointTrajectoryPoint()
    values = range(7) if positions is None else positions
    point.positions = [float(value) for value in values]
    goal.trajectory.points = [point]
    return goal


def test_goal_reorders_named_joints_and_applies_tolerance_override() -> None:
    """Action input order may differ while internal arrays stay joint1-to-joint7."""

    names = tuple(reversed(JOINT_NAMES))
    goal = _goal(names=names, positions=range(7))
    tolerance = JointTolerance()
    tolerance.name = "joint3"
    tolerance.position = 0.01
    goal.goal_tolerance = [tolerance]
    target, tolerances, timeout = normalize_reset_goal(goal, 0.03, 15.0)
    assert target.tolist() == list(reversed(range(7)))
    assert tolerances[2] == pytest.approx(0.01)
    assert timeout == pytest.approx(15.0)


@pytest.mark.parametrize(
    "goal",
    [
        _goal(names=JOINT_NAMES[:-1], positions=range(6)),
        _goal(names=(*JOINT_NAMES[:-1], "joint6"), positions=range(7)),
    ],
)
def test_goal_rejects_invalid_joint_arrays(goal) -> None:
    """Missing joints and non-finite radians must fail before hardware motion."""

    with pytest.raises(ValueError):
        normalize_reset_goal(goal, 0.03, 15.0)


def test_goal_rejects_non_finite_joint_position() -> None:
    """NaN reset targets must fail before any hardware command is published."""

    values = [0.0] * 7
    values[3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalize_reset_goal(_goal(positions=values), 0.03, 15.0)


def test_completed_automatic_reset_does_not_reject_the_next_goal() -> None:
    """A COMPLETE status without an active Action handle must not add a dead window."""

    rclpy.init()
    node = ResetCoordinator()
    try:
        node.phase = "COMPLETE"
        node.active_goal_handle = None
        assert node.goal_callback(_goal()) == GoalResponse.ACCEPT
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_factr_rebase_makes_first_post_reset_command_continuous() -> None:
    """Reverse alignment followed by rebasing reproduces actual xArm angles."""

    xarm_initial = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    xarm_actual = xarm_initial + np.array([0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.05])
    old_origin = np.array([-1.5, 0.1, 1.4, -0.2, -0.1, 1.5, -1.6])
    factr_target = factr_target_from_xarm(old_origin, xarm_actual, xarm_initial)
    new_origin = rebase_factr_origin(factr_target, xarm_actual, xarm_initial)
    first_xarm_command = xarm_initial + (factr_target - new_origin)
    np.testing.assert_allclose(new_origin, old_origin)
    np.testing.assert_allclose(first_xarm_command, xarm_actual)


def test_factr_rebase_respects_cross_arm_joint_directions() -> None:
    """Reverse reset and normal deltas use the same per-joint direction map."""

    xarm_initial = np.zeros(7)
    xarm_actual = np.arange(7, dtype=float) / 10.0
    origin = np.ones(7)
    signs = np.array([1, -1, 1, -1, 1, -1, 1])
    target = factr_target_from_xarm(origin, xarm_actual, xarm_initial, signs)
    rebased = rebase_factr_origin(target, xarm_actual, xarm_initial, signs)
    recovered_xarm = xarm_initial + signs * (target - rebased)
    np.testing.assert_allclose(rebased, origin)
    np.testing.assert_allclose(recovered_xarm, xarm_actual)


def test_coordinator_completes_only_after_all_devices_are_stable() -> None:
    """xArm arrival alone is insufficient until FACTR and Robotiq also report ready."""

    rclpy.init()
    node = ResetCoordinator()
    target = np.asarray(node.initial_joint_positions)
    try:
        with node.lock:
            node.latest_error_code = 0
            node.control_ok = True
            node.latest_xarm_position = target.copy()
            node.latest_xarm_velocity = np.zeros(7)
            node.latest_xarm_time = time.monotonic()
            node.robotiq_position = 0
            node.robotiq_state_time = time.monotonic()
            node._begin_reset(
                target,
                np.full(7, node.position_tolerance),
                node.timeout_sec,
            )
            node.xarm_target_valid = True
        node.control_loop()
        node.xarm_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "ALIGNING_FACTR"
        assert node.reset_active is True
        node.factr_aligned = True
        node.gripper_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "COMPLETE"
        assert node.reset_active is False
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_evaluation_reset_completes_without_factr_alignment() -> None:
    """Policy evaluation resets only xArm and Robotiq before the next rollout."""

    rclpy.init()
    node = ResetCoordinator()
    target = np.asarray(node.initial_joint_positions)
    try:
        node.require_factr_alignment = False
        with node.lock:
            node.latest_error_code = 0
            node.control_ok = True
            node.latest_xarm_position = target.copy()
            node.latest_xarm_velocity = np.zeros(7)
            node.latest_xarm_time = time.monotonic()
            node.robotiq_position = 0
            node.robotiq_state_time = time.monotonic()
            node._begin_reset(
                target,
                np.full(7, node.position_tolerance),
                node.timeout_sec,
            )
            node.xarm_target_valid = True
        node.control_loop()
        node.xarm_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "OPENING_GRIPPER"
        node.gripper_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "COMPLETE"
        assert node.reset_active is False
        assert node.terminal_message == "xArm and Robotiq reset complete"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_reset_waits_without_timeout_until_factr_reconnects() -> None:
    """xArm may reset offline, but completion remains gated by FACTR recovery."""

    rclpy.init()
    node = ResetCoordinator()
    target = np.asarray(node.initial_joint_positions)
    try:
        node.leader_stale_callback(Bool(data=True))
        assert node.factr_connected is False
        with node.lock:
            node.latest_error_code = 0
            node.control_ok = True
            node.latest_xarm_position = target.copy()
            node.latest_xarm_velocity = np.zeros(7)
            node.latest_xarm_time = time.monotonic()
            node.robotiq_position = 0
            node.robotiq_state_time = time.monotonic()
            node._begin_reset(
                target,
                np.full(7, node.position_tolerance),
                node.timeout_sec,
            )
            node.xarm_target_valid = True
        node.control_loop()
        node.xarm_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "WAITING_FACTR"

        # Simulate an offline interval longer than the amount by which the
        # original reset deadline has passed. Reconnect must add it back.
        node.reset_deadline = time.monotonic() - 1.0
        node.factr_wait_started = time.monotonic() - 2.0
        node.control_loop()
        assert node.phase == "WAITING_FACTR"

        node.factr_connected_callback(Bool(data=True))
        assert node.phase == "ALIGNING_FACTR"
        assert node.reset_deadline > time.monotonic()
        node.factr_aligned = True
        node.gripper_stable_since = time.monotonic() - node.stable_time_sec
        node.control_loop()
        assert node.phase == "COMPLETE"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_preexisting_error_does_not_replace_manual_goal() -> None:
    """An error present before reset is cleared before pursuing the requested goal."""

    rclpy.init()
    node = ResetCoordinator()
    requested = np.asarray(node.initial_joint_positions) + 0.1
    try:
        with node.lock:
            node.latest_error_code = 12
            node.active_goal_handle = object()
            node._begin_reset(
                requested,
                np.full(7, node.position_tolerance),
                node.timeout_sec,
            )
        node.control_loop()
        np.testing.assert_allclose(node.reset_target, requested)
        assert node.phase == "CLEARING_ERROR"
        assert node.error_during_goal is False
    finally:
        node.active_goal_handle = None
        node.destroy_node()
        rclpy.shutdown()


def test_error_after_motion_started_replaces_goal_with_initial_pose() -> None:
    """A new moving-phase error must retreat to the configured initial pose."""

    rclpy.init()
    node = ResetCoordinator()
    requested = np.asarray(node.initial_joint_positions) + 0.1
    try:
        with node.lock:
            node.latest_error_code = 0
            node.active_goal_handle = object()
            node._begin_reset(
                requested,
                np.full(7, node.position_tolerance),
                node.timeout_sec,
            )
            node.latest_error_code = 12
        node.control_loop()
        np.testing.assert_allclose(node.reset_target, node.initial_joint_positions)
        assert node.error_during_goal is True
    finally:
        node.active_goal_handle = None
        node.destroy_node()
        rclpy.shutdown()
