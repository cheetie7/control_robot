"""Launch policy tests for Xense tactile nodes."""

from launch.actions import TimerAction
from xarm_ros2_teleop.system.config import TactileConfig
from xarm_ros2_teleop.tactile.driver_actions import build_tactile_actions


def _device(*, required: bool, startup_delay_sec: float = 0.0) -> TactileConfig:
    """Build one validated scalar Xense launch configuration."""

    return TactileConfig.from_mapping(
        {
            "name": "gripper_tactile",
            "serial_number": "OP000064",
            "topic_prefix": "tactile/gripper",
            "required": required,
            "startup_delay_sec": startup_delay_sec,
            "outputs": {"rectify": True},
        }
    )


def test_required_tactile_retries_without_launch_shutdown() -> None:
    """Required Xense recovery must match optional sensor process behavior."""

    actions = build_tactile_actions(_device(required=True))
    assert len(actions) == 1
    assert isinstance(actions[0], TimerAction)


def test_optional_tactile_only_creates_delayed_node() -> None:
    """An optional Xense node may exit without shutting down teleoperation."""

    actions = build_tactile_actions(_device(required=False))
    assert len(actions) == 1
    assert isinstance(actions[0], TimerAction)


def test_tactile_uses_configured_absolute_start_delay() -> None:
    """Each tactile action must own its absolute delay from launch start."""

    actions = build_tactile_actions(_device(required=False, startup_delay_sec=4.0))
    assert actions[0].period == 4.0


def test_tactile_node_respawns_after_a_crash(monkeypatch) -> None:
    """A transient Xense open failure retries instead of exiting the launch."""

    from launch_ros.actions import Node

    captured = {}
    monkeypatch.setattr(
        "xarm_ros2_teleop.tactile.driver_actions.Node",
        lambda **kwargs: captured.update(kwargs) or Node(**kwargs),
    )
    build_tactile_actions(_device(required=False))
    assert captured["respawn"] is True
    assert captured["respawn_delay"] == 3.0
