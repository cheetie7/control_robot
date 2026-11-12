"""Unit tests for Xense tactile configuration validation."""

from pathlib import Path

import pytest
import yaml
from xarm_ros2_teleop.system.config import TactileConfig, load_system_config


def _tactile(**overrides) -> dict:
    """Return a complete Xense YAML mapping with selected scalar overrides."""

    device = {
        "name": "gripper_tactile",
        "serial_number": "OP000064",
        "topic_prefix": "tactile/gripper",
        "required": True,
        "publish_rate_hz": 30.0,
        "outputs": {"rectify": True},
    }
    device.update(overrides)
    return device


def _write_system(path: Path, devices: list[dict]) -> Path:
    """Write a disabled-robot system YAML containing Xense device mappings."""

    content = {
        "teleop": {"enabled": False},
        "xarm": {"enabled": False},
        "cameras": {"enabled": False, "devices": []},
        "tactile": {"enabled": True, "devices": devices},
    }
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_tactile_defaults_to_rectified_image() -> None:
    """An omitted output mapping must publish only the native rectified image."""

    device = TactileConfig.from_mapping(_tactile(outputs={}))
    assert device.outputs == ("rectify",)


def test_tactile_supports_all_documented_outputs() -> None:
    """Every implemented Xense SDK output must preserve its configured order."""

    outputs = {
        "rectify": True,
        "difference": True,
        "depth": True,
        "marker_2d": True,
        "force": True,
        "force_norm": True,
        "force_resultant": True,
    }
    device = TactileConfig.from_mapping(_tactile(outputs=outputs))
    assert device.outputs == tuple(outputs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"serial_number": ""},
        {"publish_rate_hz": 0},
        {"outputs": {"rectify": False}},
        {"outputs": {"unknown": True}},
    ],
)
def test_invalid_tactile_fields_are_rejected(overrides: dict) -> None:
    """Unsafe device identity, rate or output selections must fail pre-launch."""

    with pytest.raises(ValueError):
        TactileConfig.from_mapping(_tactile(**overrides))


def test_duplicate_tactile_serial_is_rejected(tmp_path: Path) -> None:
    """Two enabled nodes may not open the same physical Xense serial number."""

    second = _tactile(name="second_tactile", topic_prefix="tactile/second")
    config_path = _write_system(tmp_path / "system.yaml", [_tactile(), second])
    with pytest.raises(ValueError, match="Duplicate tactile serial_number"):
        load_system_config(config_path)
