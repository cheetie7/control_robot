"""Unit tests for the unified camera configuration contract."""

from pathlib import Path

import pytest
import yaml
from xarm_ros2_teleop.system.config import CameraConfig, load_system_config


def _camera(**overrides):
    """Return a complete scalar camera mapping with selected test overrides."""

    camera = {
        "name": "front_camera",
        "vendor": "realsense",
        "model": "d435",
        "serial_number": "123456",
        "topic_prefix": "camera/front",
        "required": True,
        "color": {"enabled": True},
        "depth": {"enabled": False},
        "point_cloud": {"enabled": False},
    }
    camera.update(overrides)
    return camera


def _write_config(path: Path, devices: list, *, cameras_enabled: bool = True) -> Path:
    """Write a test system YAML with a one-dimensional device list."""

    content = {
        "system": {"name": "test_system"},
        "teleop": {"enabled": False},
        "xarm": {"enabled": False},
        "cameras": {"enabled": cameras_enabled, "devices": devices},
    }
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_point_cloud_requires_explicit_color_and_depth() -> None:
    """A cloud may not silently override explicitly disabled input streams."""

    with pytest.raises(ValueError, match="Colored point cloud requires"):
        CameraConfig.from_mapping(
            _camera(
                color={"enabled": False},
                depth={"enabled": False},
                point_cloud={"enabled": True},
            )
        )


def test_aligned_depth_requires_explicit_color() -> None:
    """Aligned depth may not generate color after color was explicitly disabled."""

    with pytest.raises(ValueError, match="Aligned depth requires"):
        CameraConfig.from_mapping(_camera(color={"enabled": False}, depth={"enabled": True}))


def test_femto_bolt_is_supported() -> None:
    """The second supported hardware profile is Orbbec Femto Bolt."""

    camera = CameraConfig.from_mapping(_camera(vendor="orbbec", model="femto_bolt"))
    assert (camera.vendor, camera.model) == ("orbbec", "femto_bolt")


@pytest.mark.parametrize(
    "field,value",
    [
        ("serial_number", ""),
        ("name", "front-camera"),
        ("topic_prefix", "camera//front"),
        ("model", "unsupported"),
    ],
)
def test_invalid_camera_fields_are_rejected(field: str, value: str) -> None:
    """Invalid physical identifiers and ROS names must fail before launch."""

    with pytest.raises(ValueError):
        CameraConfig.from_mapping(_camera(**{field: value}))


def test_partial_stream_profile_is_rejected() -> None:
    """A vendor profile override requires width, height and fps together."""

    with pytest.raises(ValueError, match="configured together"):
        CameraConfig.from_mapping(_camera(color={"enabled": True, "width": 640}))


@pytest.mark.parametrize(
    "duplicate_field,duplicate_value",
    [
        ("name", "front_camera"),
        ("topic_prefix", "camera/front"),
        ("serial_number", "123456"),
    ],
)
def test_duplicate_camera_identity_is_rejected(
    tmp_path: Path, duplicate_field: str, duplicate_value: str
) -> None:
    """Multiple devices may not share a logical or physical identity."""

    second = _camera(
        name="side_camera",
        serial_number="654321",
        topic_prefix="camera/side",
    )
    second[duplicate_field] = duplicate_value
    config_path = _write_config(tmp_path / "system.yaml", [_camera(), second])
    with pytest.raises(ValueError, match="Duplicate camera"):
        load_system_config(config_path)


def test_disabled_camera_remains_available_for_diagnostics(tmp_path: Path) -> None:
    """A disabled camera remains in config for UI display but creates no launch action."""

    config_path = _write_config(tmp_path / "system.yaml", [_camera(enabled=False)])
    system = load_system_config(config_path)
    assert len(system.cameras) == 1
    assert system.cameras[0].enabled is False


def test_infrared_stream_parses_with_dedicated_switch() -> None:
    """The stereo infrared pair has its own enabled flag and profile."""

    from xarm_ros2_teleop.system.config import CameraConfig

    camera = CameraConfig.from_mapping(
        {
            "name": "left_camera",
            "vendor": "realsense",
            "model": "d435",
            "serial_number": "s",
            "topic_prefix": "camera/left",
            "color": {"enabled": True},
            "infrared": {"enabled": True, "width": 640, "height": 480, "fps": 30},
            "depth": {"enabled": False},
        }
    )
    assert camera.infrared_enabled is True
    assert (camera.infrared.width, camera.infrared.height, camera.infrared.fps) == (
        640,
        480,
        30,
    )
    assert camera.depth_enabled is False
