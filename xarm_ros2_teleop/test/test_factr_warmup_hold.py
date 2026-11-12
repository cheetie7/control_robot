"""Pure policy tests for FACTR warm-up hold, free-reset and origin rebase."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarm_ros2_teleop.teleop.controller as controller_module
from std_msgs.msg import Bool
from xarm_ros2_teleop.teleop.controller import (
    FACTRTeleop,
    interpolated_pd_torque,
)
from xarm_ros2_teleop.teleop.leader_node import Factr_teleop_Ros2, should_hold_factr


def test_factr_holds_before_session_ready_and_during_reset() -> None:
    """Warm-up and every reset phase retain current-servo ownership."""

    assert should_hold_factr(True, False, True, False) is True
    assert should_hold_factr(True, True, True, False) is True
    assert should_hold_factr(True, True, True, True) is True


def test_factr_releases_hold_only_for_ready_normal_teleoperation() -> None:
    """Normal gravity-compensated teleoperation resumes only after readiness."""

    assert should_hold_factr(True, False, True, True) is False
    assert should_hold_factr(True, False, False, True) is False
    assert should_hold_factr(False, False, True, False) is False


def test_gripper_limit_barrier_can_be_disabled_without_disabling_arm_limits() -> None:
    """The eighth motor stays current-free while seven-axis limit protection remains."""

    controller = SimpleNamespace(
        arm_joint_limits_max=np.ones(7),
        arm_joint_limits_min=-np.ones(7),
        joint_limit_kp=1.0,
        joint_limit_kd=0.1,
        gripper_limit_max=0.8,
        gripper_limit_min=-0.8,
        enable_gripper_limit_barrier=False,
    )

    arm_torque, gripper_torque = FACTRTeleop.joint_limit_barrier(
        controller,
        np.asarray([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(7),
        1.2,
        0.0,
    )

    assert arm_torque[0] == pytest.approx(-0.2)
    assert gripper_torque == 0.0


def test_episode_activation_rebases_once_but_staged_start_preserves_origin() -> None:
    """Only a false-to-true control gate resets the FACTR relative origin."""

    controller = SimpleNamespace(episode_active=False, origin_rebase_pending=False)

    Factr_teleop_Ros2.episode_active_callback(controller, Bool(data=True))
    assert controller.origin_rebase_pending is True

    # Simulate the next control tick consuming the pending rebase. Repeated
    # True is staging -> recording and must not create a second pose change.
    controller.origin_rebase_pending = False
    Factr_teleop_Ros2.episode_active_callback(controller, Bool(data=True))
    assert controller.origin_rebase_pending is False

    Factr_teleop_Ros2.episode_active_callback(controller, Bool(data=False))
    Factr_teleop_Ros2.episode_active_callback(controller, Bool(data=True))
    assert controller.origin_rebase_pending is True


def test_reset_holds_gravity_only_and_acknowledges_alignment_at_once() -> None:
    """Reset free-holds the leader on gravity and never blocks on alignment."""

    gravity = np.full(7, 0.05)

    class FakeDriver:
        """Record torque commands; no mode-switch methods exist at all."""

        def __init__(self) -> None:
            self.commands = []

        def set_torque(self, torques, max_current_ma=900.0):
            """Store one N·m command array and its current ceiling."""

            self.commands.append((np.asarray(torques, dtype=float), max_current_ma))

    driver = FakeDriver()
    controller = SimpleNamespace(
        reset_enabled=True,
        session_gate_enabled=True,
        session_ready=False,
        reset_active=True,
        reset_aligned=False,
        factr_connected=True,
        num_arm_joints=7,
        enable_gravity_comp=True,
        gravity_compensation=lambda pos, vel, **kwargs: gravity,
        reset_max_current_ma=500.0,
        joint_signs=np.ones(8),
        driver=driver,
    )
    controller.set_leader_joint_torque = _bind(
        FACTRTeleop.set_leader_joint_torque, controller
    )
    aligned_events = []
    controller._publish_reset_aligned = aligned_events.append

    owned = Factr_teleop_Ros2.run_reset_control(
        controller, np.full(7, 0.3), np.zeros(7), 0.4, 0.0
    )

    assert owned is True
    torques, ceiling = driver.commands[-1]
    assert ceiling == 500.0
    np.testing.assert_allclose(torques[:7], gravity, atol=1e-12)
    assert torques[7] == pytest.approx(0.0)

    controller.reset_align_callback = _bind(
        Factr_teleop_Ros2.reset_align_callback, controller
    )
    controller.reset_align_callback(Bool(data=True))
    assert aligned_events == [True]


def test_update_communication_pins_zero_deltas_then_rebases_origin() -> None:
    """Deltas pin to zero during reset; afterwards the origin re-seeds."""

    class FakePublisher:
        """Collect published position arrays for assertions."""

        def __init__(self) -> None:
            self.frames = []

        def publish(self, msg):
            """Record one published position array."""

            self.frames.append(list(msg.position))

    controller = SimpleNamespace(
        reset_active=True,
        origin_rebase_pending=False,
        origin_zero_frames_remaining=0,
        ORIGIN_REBASE_ZERO_FRAMES=20,
        xarm_to_factr_signs=np.ones(7),
        leader_arm_publish_origin=np.full(7, -1.0),
        num_arm_joints=7,
        arm_pos_publisher=FakePublisher(),
        gripper_pos_publisher=FakePublisher(),
    )
    controller.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)

    # During reset the arm delta is pinned to zero regardless of leader pose.
    Factr_teleop_Ros2.update_communication(
        controller, np.full(7, 1.0), 0.5
    )
    assert controller.arm_pos_publisher.frames == [[0.0] * 7]

    # Reset ended: the first tick re-seeds the origin from the measured pose.
    controller.reset_active = False
    controller.origin_rebase_pending = True
    Factr_teleop_Ros2.update_communication(
        controller, np.full(7, 1.0), 0.5
    )
    np.testing.assert_allclose(controller.leader_arm_publish_origin, np.full(7, 1.0))
    assert controller.arm_pos_publisher.frames[-1] == [0.0] * 7

    # A short zero burst covers cross-topic delivery ordering at activation.
    Factr_teleop_Ros2.update_communication(
        controller, np.full(7, 1.2), 0.5
    )
    assert controller.arm_pos_publisher.frames[-1] == [0.0] * 7
    np.testing.assert_allclose(controller.leader_arm_publish_origin, np.full(7, 1.2))

    # Later deltas are relative to the re-based origin.
    controller.origin_zero_frames_remaining = 0
    Factr_teleop_Ros2.update_communication(controller, np.full(7, 1.4), 0.5)
    assert controller.arm_pos_publisher.frames[-1] == [pytest.approx(0.2)] * 7


def test_factr_reconnect_requires_consecutive_complete_reads() -> None:
    """FACTR health returns only after the configured valid-read streak."""

    class FakeDriver:
        """Record reconnect/read operations from the leader recovery loop."""

        def __init__(self) -> None:
            """Create empty operation counters."""

            self.reconnects = 0
            self.reads = 0

        def reconnect_current_control(self) -> None:
            """Record one serial reopen and mode restoration."""

            self.reconnects += 1

        def get_positions_and_velocities(self, tries=0):
            """Return finite eight-axis arrays for one verification read."""

            self.reads += 1
            return np.zeros(8), np.zeros(8)

        def close(self) -> None:
            """Accept a close operation without hardware."""

    driver = FakeDriver()
    health = []
    alignment = []
    controller = SimpleNamespace(
        driver=driver,
        factr_connected=False,
        factr_reconnect_verifying=False,
        factr_recovery_reads=0,
        factr_next_reconnect_time=0.0,
        reconnect_period_sec=1.0,
        recovery_successful_reads=2,
        origin_rebase_pending=False,
        origin_zero_frames_remaining=0,
        ORIGIN_REBASE_ZERO_FRAMES=20,
        reset_active=True,
        reset_align_requested=True,
        _publish_factr_connected=lambda connected, force=False: health.append(connected),
        _publish_reset_aligned=alignment.append,
        get_logger=lambda: SimpleNamespace(info=lambda *_args: None, warn=lambda *_args: None),
    )
    controller._close_factr_port = _bind(
        Factr_teleop_Ros2._close_factr_port, controller
    )

    Factr_teleop_Ros2._attempt_factr_reconnect(controller, 1.0)
    assert controller.factr_connected is False
    assert health == []

    Factr_teleop_Ros2._attempt_factr_reconnect(controller, 1.01)
    assert controller.factr_connected is True
    assert driver.reconnects == 1
    assert driver.reads == 2
    assert health == [True]
    assert alignment == [True]
    assert controller.origin_zero_frames_remaining == 20


def test_interpolated_pd_torque_limits_step_and_converges() -> None:
    """Far joints advance by kp*step; near joints converge with plain PD."""

    goal = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0])
    position = np.array([0.1, 0.005, -0.3, 1.4, 0.0, 1.56, 0.0])
    velocity = np.zeros(7)
    kp, kd, step = 1.03561658022, 0.04315069084, 0.02

    torque = interpolated_pd_torque(
        position, velocity, goal, kp=kp, kd=kd, step=step
    )
    # Torque points toward the goal, capped at kp*step per cycle.
    np.testing.assert_allclose(
        torque, kp * np.clip(goal - position, -step, step), atol=1e-12
    )


def test_gravity_compensation_uses_the_verified_urdf_coordinate_frame(
    monkeypatch,
) -> None:
    """RNEA positions must be translated from xArm mapping to FACTR model radians."""

    recorded = {}

    def fake_rnea(_model, _data, position, velocity, acceleration):
        """Record Pinocchio arrays and return deterministic seven-axis torque."""

        recorded["position"] = position.copy()
        recorded["velocity"] = velocity.copy()
        recorded["acceleration"] = acceleration.copy()
        return np.arange(1.0, 8.0)

    monkeypatch.setattr(controller_module.pin, "rnea", fake_rnea)
    gravity_offset = np.array([-1.56, 0.03, 1.58, -1.23, 0.03, 0.01, -1.60])
    controller = SimpleNamespace(
        pin_model=SimpleNamespace(nq=7, nv=7),
        pin_data=object(),
        num_arm_joints=7,
        gravity_joint_position_offset=gravity_offset,
        gravity_comp_modifier=0.2,
        tau_g=np.zeros(7),
    )
    teleop_position = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0])
    velocity = np.arange(7, dtype=float) / 10.0

    torque = FACTRTeleop.gravity_compensation(
        controller,
        teleop_position,
        velocity,
    )

    np.testing.assert_allclose(
        recorded["position"],
        [-1.56, 0.03, 1.58, 0.34, 0.03, 1.58, -1.60],
    )
    np.testing.assert_allclose(recorded["velocity"], velocity)
    np.testing.assert_allclose(recorded["acceleration"], np.zeros(7))
    np.testing.assert_allclose(torque, np.arange(1.0, 8.0) * 0.2)


def _bind(function, controller):
    """Return a bound callable for one unbound method on a stand-in object."""

    from functools import partial

    return partial(function, controller)
