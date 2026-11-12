"""Pure tests for the remote policy protocol and hardware command contract."""

from __future__ import annotations

import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Bool
from xarm_ros2_teleop.inference.contracts import (
    build_observation,
    interpolate_action_chunk,
    model_to_ros_commands,
    validate_action_chunk,
)
from xarm_ros2_teleop.inference.remote_policy_node import RemotePolicyNode
from xarm_ros2_teleop.inference.websocket_policy import pack_message, unpack_message


def test_observation_and_msgpack_preserve_native_rgb_and_float_arrays() -> None:
    """OpenPI payloads retain RGB uint8 and physical float32 shapes exactly."""

    images = {"cam_high": np.arange(18, dtype=np.uint8).reshape(2, 3, 3)}
    observation = build_observation(np.arange(7, dtype=np.float32), 0.25, images, "pick")
    restored = unpack_message(pack_message(observation))
    assert restored["state"].dtype == np.float32
    assert restored["state"].shape == (8,)
    assert restored["images"]["cam_high"].dtype == np.uint8
    np.testing.assert_array_equal(restored["images"]["cam_high"], images["cam_high"])


def test_action_conversion_preserves_absolute_xarm_targets() -> None:
    """Model targets remain absolute for xArm and map gripper into legacy radians."""

    action = np.asarray([0.1, 0.0, 0.25, 1.1, 0.3, 1.55, 0.6, 0.75], dtype=np.float32)
    arm_target, gripper = model_to_ros_commands(action)
    np.testing.assert_allclose(arm_target, action[:7])
    assert gripper == pytest.approx(-0.75)
    # Existing Robotiq callback computes clip(-leader * 255), so this is 75% closed.
    assert int(np.clip(-gripper * 255, 0, 255)) == 191


def test_dry_run_publishes_candidate_without_robot_commands() -> None:
    """Dry-run exposes a candidate for arbitration but emits no robot command."""

    class Publisher:
        """Count messages that would reach one physical command topic."""

        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            """Record one unexpected command message."""

            self.messages.append(message)

    class Logger:
        """Accept the informational action log emitted by dry-run mode."""

        def info(self, _message: str) -> None:
            """Discard one human-readable dry-run log line."""

    class DryRunHarness:
        """Provide only the state used by ``RemotePolicyNode._execute_action``."""

        dry_run = True
        tcp_mode = False
        direct_robot_commands = True
        last_commanded_action = np.zeros(8, dtype=np.float32)
        policy_action_pub = Publisher()
        arm_pub = Publisher()
        gripper_pub = Publisher()

        class Clock:
            """Supply a minimal timestamp compatible with JointState."""

            class Now:
                """Represent one test clock instant."""

                @staticmethod
                def to_msg():
                    """Return a valid ROS Time message."""

                    from builtin_interfaces.msg import Time

                    return Time()

            @staticmethod
            def now():
                """Return the current synthetic clock instant."""

                return DryRunHarness.Clock.Now()

        def get_clock(self) -> Clock:
            """Return the synthetic ROS-compatible clock."""

            return self.Clock()

        def get_logger(self) -> Logger:
            """Return the no-op test logger."""

            return Logger()

    harness = DryRunHarness()
    RemotePolicyNode._execute_action(harness, np.ones(8, dtype=np.float32) * 0.1)
    assert len(harness.policy_action_pub.messages) == 1
    assert harness.policy_action_pub.messages[0].position == pytest.approx([0.1] * 8)
    assert harness.arm_pub.messages == []
    assert harness.gripper_pub.messages == []


def test_action_chunk_rejects_nonfinite_limits_gripper_and_large_steps() -> None:
    """Every untrusted server output guard fails closed before command publication."""

    current = np.zeros(8, dtype=np.float32)
    lower = np.full(7, -1.0, dtype=np.float32)
    upper = np.full(7, 1.0, dtype=np.float32)
    valid = np.zeros((2, 8), dtype=np.float32)
    valid[:, 7] = [0.25, 0.5]
    assert validate_action_chunk(valid, current, lower, upper, 0.15).shape == (2, 8)

    invalid = valid.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_action_chunk(invalid, current, lower, upper, 0.15)
    invalid = valid.copy()
    invalid[0, 0] = 1.1
    with pytest.raises(ValueError, match="joint limits"):
        validate_action_chunk(invalid, current, lower, upper, 2.0)
    invalid = valid.copy()
    invalid[0, 7] = -0.1
    with pytest.raises(ValueError, match="gripper"):
        validate_action_chunk(invalid, current, lower, upper, 0.15)
    invalid = valid.copy()
    invalid[0, 0] = 0.2
    with pytest.raises(ValueError, match="joint step"):
        validate_action_chunk(invalid, current, lower, upper, 0.15)


def test_action_chunk_interpolation_preserves_30_hz_timeline_at_100_hz() -> None:
    """Low-rate waypoints are interpolated smoothly and retain their duration."""

    chunk = np.zeros((25, 8), dtype=np.float32)
    chunk[:, 0] = np.arange(25, dtype=np.float32)
    chunk[:, 7] = np.linspace(0.0, 1.0, 25, dtype=np.float32)

    np.testing.assert_allclose(interpolate_action_chunk(chunk, 0.0, 30.0), chunk[0])
    np.testing.assert_allclose(
        interpolate_action_chunk(chunk, 1.5 / 30.0, 30.0),
        (chunk[1] + chunk[2]) / 2.0,
    )
    np.testing.assert_allclose(
        interpolate_action_chunk(chunk, 24.5 / 30.0, 30.0), chunk[-1]
    )
    assert interpolate_action_chunk(chunk, 25.0 / 30.0, 30.0) is None


def test_action_chunk_interpolation_rejects_invalid_time_and_frequency() -> None:
    """The executor rejects invalid clocks rather than publishing an unsafe target."""

    chunk = np.zeros((25, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        interpolate_action_chunk(chunk, -0.01, 30.0)
    with pytest.raises(ValueError, match="positive finite"):
        interpolate_action_chunk(chunk, 0.0, 0.0)


def test_rl_resume_rearms_failed_chunk_from_active_episode_gate() -> None:
    """Resuming a recoverable request failure must not leave inference gated forever."""

    starts = []
    node = SimpleNamespace(
        policy_paused=True,
        action_chunk=np.zeros((25, 8), dtype=np.float32),
        inference_future=None,
        episode_gated=True,
        external_episode_active=True,
        episode_active=False,
        _episode_active_callback=lambda message: starts.append(message.data),
        _publish_status=lambda _status: None,
    )
    RemotePolicyNode._policy_paused_callback(node, Bool(data=False))
    assert starts == [True]
    assert node.action_chunk is None


def test_observation_state_does_not_replace_last_commanded_action() -> None:
    """Measured lag is model input, not the chunk-boundary command reference."""

    now = time.monotonic()
    commanded = np.asarray([0.0, 0.0, 0.0, 1.4, 0.0, 1.4, 0.0, 0.5], dtype=np.float32)
    measured = np.asarray([0.0, 0.0, 0.0, 1.2, 0.0, 1.2, 0.0], dtype=np.float32)
    node = SimpleNamespace(
        config=SimpleNamespace(
            images=[SimpleNamespace(key="front")],
            observation_timeout_sec=0.25,
            max_camera_skew_sec=0.08,
            prompt="insert the block",
        ),
        images={"front": (now, np.zeros((2, 2, 3), dtype=np.uint8))},
        joints=(now, measured),
        gripper=(now, 0.4),
        dry_run=False,
        initial_joints=np.zeros(7, dtype=np.float32),
        last_commanded_action=commanded.copy(),
    )

    observation = RemotePolicyNode._build_fresh_observation(node)

    np.testing.assert_allclose(observation["state"][:7], measured)
    np.testing.assert_allclose(node.last_commanded_action, commanded)


def test_chunk_safety_uses_the_measured_state_submitted_for_inference() -> None:
    """A returned chunk is compared with its input state, not an older command."""

    future = Future()
    chunk = np.zeros((25, 8), dtype=np.float32)
    chunk[:, 0] = 0.4
    future.set_result(chunk)
    statuses = []
    node = SimpleNamespace(
        inference_future=future,
        tcp_mode=False,
        discard_inference_result=False,
        inference_state=np.zeros(8, dtype=np.float32),
        last_commanded_action=np.full(8, -1.0, dtype=np.float32),
        config=SimpleNamespace(
            joint_lower_rad=(-2.0,) * 7,
            joint_upper_rad=(2.0,) * 7,
            max_joint_step_rad=0.5,
        ),
        episode_gated=True,
        dry_run=False,
        action_chunk=None,
        chunk_count=0,
        _publish_status=statuses.append,
    )

    RemotePolicyNode._finish_inference(node)

    np.testing.assert_allclose(node.action_chunk, chunk)
    assert node.chunk_count == 1
    assert statuses == ["EXECUTING"]


def test_motion_stop_holds_measured_pose_without_hardware_emergency_stop() -> None:
    """An armed inference stop publishes one measured hold and ends the rollout."""

    statuses = []
    executed = []
    joints = np.arange(7, dtype=np.float32) / 10.0
    node = SimpleNamespace(
        stopped=False,
        tcp_mode=False,
        dry_run=False,
        action_chunk=np.zeros((25, 8), dtype=np.float32),
        joints=(1.0, joints),
        gripper=(1.0, 0.4),
        _execute_action=lambda action: executed.append(action),
        _publish_status=statuses.append,
        get_logger=lambda: SimpleNamespace(error=lambda message: None),
    )

    RemotePolicyNode._stop(node, "test stop")

    assert node.stopped is True
    assert node.action_chunk is None
    assert len(executed) == 1
    np.testing.assert_allclose(executed[0], np.concatenate([joints, [0.4]]))
    assert statuses == ["STOPPED: test stop"]


def test_evaluation_gate_reuses_policy_process_across_episodes() -> None:
    """Closing one episode discards its chunk without permanently stopping policy."""

    statuses = []
    node = SimpleNamespace(
        episode_gated=True,
        stopped=False,
        episode_active=True,
        action_chunk=np.ones((25, 8), dtype=np.float32),
        inference_future=None,
        dry_run=True,
        control_ok=True,
        joints=None,
        gripper=None,
        chunk_count=4,
        started_at=0.0,
        _publish_status=statuses.append,
    )
    node._pause_episode = lambda reason: RemotePolicyNode._pause_episode(node, reason)

    RemotePolicyNode._episode_active_callback(node, SimpleNamespace(data=False))
    assert node.episode_active is False
    assert node.action_chunk is None
    assert node.stopped is False
    assert statuses[-1].startswith("WAITING_FOR_EPISODE")

    RemotePolicyNode._episode_active_callback(node, SimpleNamespace(data=True))
    assert node.episode_active is True
    assert node.chunk_count == 0
    assert node.stopped is False
    assert statuses[-1] == "WAITING_FOR_OBSERVATION"


def test_policy_safety_failure_holds_and_keeps_evaluation_service_reusable() -> None:
    """An invalid ACT chunk fails one episode without latching the server stopped."""

    statuses = []
    executed = []
    errors = []
    joints = np.arange(7, dtype=np.float32) / 10.0
    node = SimpleNamespace(
        episode_active=True,
        stopped=False,
        tcp_mode=False,
        dry_run=False,
        action_chunk=np.ones((25, 8), dtype=np.float32),
        inference_future=None,
        discard_inference_result=False,
        joints=(1.0, joints),
        gripper=(1.0, 0.4),
        _execute_action=lambda action: executed.append(action),
        _publish_status=statuses.append,
        get_logger=lambda: SimpleNamespace(error=errors.append),
    )

    RemotePolicyNode._fail_episode(node, "unsafe action")

    assert node.episode_active is False
    assert node.stopped is False
    assert node.action_chunk is None
    np.testing.assert_allclose(executed[0], np.concatenate([joints, [0.4]]))
    assert statuses == ["EPISODE_FAILED: unsafe action"]
    assert errors == ["evaluation episode failed: unsafe action"]
