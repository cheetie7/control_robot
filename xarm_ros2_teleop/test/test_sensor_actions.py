"""Tests for the thin unified sensor launch composition layer."""

from types import SimpleNamespace

from xarm_ros2_teleop.system import sensor_actions


def test_only_enabled_sensors_contribute_their_own_actions(monkeypatch) -> None:
    """The system composer must filter devices without adding readiness logic."""

    cameras = [
        SimpleNamespace(name="left", enabled=True),
        SimpleNamespace(name="disabled_camera", enabled=False),
    ]
    tactile_devices = [
        SimpleNamespace(name="left_tactile", enabled=True),
        SimpleNamespace(name="disabled_tactile", enabled=False),
    ]
    system = SimpleNamespace(
        cameras_enabled=True,
        cameras=cameras,
        tactile_enabled=True,
        tactile_devices=tactile_devices,
    )
    monkeypatch.setattr(
        sensor_actions,
        "build_camera_actions",
        lambda camera: [f"camera:{camera.name}"],
    )
    monkeypatch.setattr(
        sensor_actions,
        "build_tactile_actions",
        lambda device: [f"tactile:{device.name}"],
    )

    assert sensor_actions.build_sensor_actions(system) == [
        "camera:left",
        "tactile:left_tactile",
    ]


def test_global_sensor_switches_disable_whole_device_families(monkeypatch) -> None:
    """A disabled sensor family must not construct any physical driver action."""

    system = SimpleNamespace(
        cameras_enabled=False,
        cameras=[SimpleNamespace(name="left", enabled=True)],
        tactile_enabled=False,
        tactile_devices=[SimpleNamespace(name="left_tactile", enabled=True)],
    )
    monkeypatch.setattr(
        sensor_actions,
        "build_camera_actions",
        lambda camera: [f"camera:{camera.name}"],
    )
    monkeypatch.setattr(
        sensor_actions,
        "build_tactile_actions",
        lambda device: [f"tactile:{device.name}"],
    )

    assert sensor_actions.build_sensor_actions(system) == []
