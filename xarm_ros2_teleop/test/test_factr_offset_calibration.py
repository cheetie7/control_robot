"""Pure tests for persistent FACTR encoder offsets and arm reference mapping."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from xarm_ros2_teleop.teleop.offset_calibration import (
    calculate_gravity_joint_offset,
    calculate_joint_offsets,
    derive_factr_home,
    load_offset_calibration,
    logical_joint_positions,
    save_offset_calibration,
)


def _metadata() -> dict:
    """Return one consistent eight-motor FACTR calibration description."""

    return {
        "dynamixel_port": "/dev/serial/by-id/factr",
        "servo_types": ["XL330"] * 8,
        "joint_signs": np.ones(8),
        "calibration_joint_positions": np.arange(7, dtype=float) / 10.0,
        "operating_mode": 0,
        "xarm_to_factr_signs": np.ones(7),
    }


def test_factr_home_is_derived_from_the_current_xarm_initial_pose() -> None:
    """Changing xArm initial joints shifts FACTR home through the reference pair."""

    factr_reference = np.array([-1.6, 0.1, 1.6, -0.1, -0.2, 1.5, -1.6])
    xarm_reference = np.array([0.0, 0.0, 0.0, 0.88, 0.0, 0.88, 0.0])
    xarm_initial = xarm_reference + np.array([0.1, -0.2, 0.0, 0.3, 0.0, 0.1, -0.1])
    signs = np.array([1, -1, 1, 1, -1, 1, 1])
    expected = factr_reference + signs * (xarm_initial - xarm_reference)
    np.testing.assert_allclose(
        derive_factr_home(factr_reference, xarm_initial, xarm_reference, signs),
        expected,
    )


def test_known_start_pose_calculates_arm_unwrap_and_open_gripper_offset() -> None:
    """Calibration selects arm pi/2 periods and makes the open gripper exact zero."""

    start = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0])
    expected_offsets = np.array([np.pi / 2, np.pi, np.pi, 0.0, 2 * np.pi, 2 * np.pi, 0.0, 3.52])
    raw = expected_offsets + start
    offsets = calculate_joint_offsets(raw, start, np.ones(8))
    logical = logical_joint_positions(raw, offsets, np.ones(8))

    np.testing.assert_allclose(offsets, expected_offsets)
    np.testing.assert_allclose(logical, start)


def test_gravity_offset_exactly_reproduces_old_pi_over_two_calibration() -> None:
    """The dynamics frame must match the old raw-encoder branch selection exactly."""

    raw = np.array([1.56, 3.05, 9.34, 2.88, 9.41, 3.06, 1.56, 3.52])
    signs = np.ones(8)
    teleop_start = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0])
    model_reference = np.array([-1.56, 0.03, 1.58, 0.34, 0.03, 1.58, -1.60])
    teleop_offsets = calculate_joint_offsets(raw, teleop_start, signs)
    model_offsets = calculate_joint_offsets(
        raw,
        np.append(model_reference, 0.0),
        signs,
    )
    teleop_position = logical_joint_positions(raw, teleop_offsets, signs)[:7]
    old_model_position = logical_joint_positions(raw, model_offsets, signs)[:7]

    coordinate_offset = calculate_gravity_joint_offset(
        raw,
        teleop_offsets,
        signs,
        model_reference,
    )

    np.testing.assert_allclose(teleop_position + coordinate_offset, old_model_position)


def test_two_capture_gravity_offset_is_independent_of_stage_one_placement() -> None:
    """Stage-one matching sloppiness must not leak into the URDF dynamics frame."""

    signs = np.ones(8)
    teleop_start = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0])
    model_reference = np.array([-1.56, 0.03, 1.58, 0.34, 0.03, 1.58, -1.60])

    # The physical folded pose produces fixed encoder readings; only the home
    # matching pose differs between the two calibration attempts.
    folded_raw = np.array([1.56, 3.05, 9.34, 2.88, 9.41, 3.06, 1.56, 3.52])
    sloppy_home_raw = np.array([1.60, 3.20, 9.30, 3.10, 9.45, 3.10, 1.60, 3.52])
    careful_home_raw = np.array([1.57, 3.15, 9.42, 2.90, 9.42, 3.07, 1.57, 3.52])

    gravity_offsets = []
    for home_raw in (sloppy_home_raw, careful_home_raw):
        teleop_offsets = calculate_joint_offsets(home_raw, teleop_start, signs)
        gravity_offsets.append(
            calculate_gravity_joint_offset(folded_raw, teleop_offsets, signs, model_reference)
        )

    # Both attempts map arbitrary later encoder readings into the same URDF frame.
    np.testing.assert_allclose(gravity_offsets[0], gravity_offsets[1], atol=1e-12)
    teleop_offsets = calculate_joint_offsets(careful_home_raw, teleop_start, signs)
    later_logical = logical_joint_positions(
        folded_raw, teleop_offsets, signs
    )[:7]
    model_offsets = calculate_joint_offsets(
        folded_raw, np.append(model_reference, 0.0), signs
    )
    urdf_position = logical_joint_positions(folded_raw, model_offsets, signs)[:7]
    np.testing.assert_allclose(later_logical + gravity_offsets[0], urdf_position)


def test_offset_yaml_round_trip_keeps_encoder_calibration(tmp_path: Path) -> None:
    """The saved YAML reloads all eight raw-to-logical Dynamixel offsets."""

    path = tmp_path / "offset.yaml"
    metadata = _metadata()
    offsets = np.arange(8, dtype=float) * np.pi / 2.0
    raw_positions = np.arange(8, dtype=float) / 4.0 + offsets
    logical_positions = logical_joint_positions(
        raw_positions, offsets, metadata["joint_signs"]
    )
    reference = logical_positions[:7]
    gravity_offset = np.arange(7, dtype=float) / 10.0 - 0.3
    save_offset_calibration(
        path,
        **metadata,
        joint_offsets=offsets,
        raw_positions=raw_positions,
        logical_positions=logical_positions,
        gravity_joint_position_offset=gravity_offset,
        factr_reference=reference,
        xarm_reference=np.zeros(7),
    )
    loaded = load_offset_calibration(path, **metadata)
    np.testing.assert_allclose(loaded.joint_offsets_rad, offsets)
    np.testing.assert_allclose(loaded.raw_positions_rad, raw_positions)
    np.testing.assert_allclose(loaded.logical_positions_rad, logical_positions)
    np.testing.assert_allclose(loaded.factr_reference_joint_positions, reference)
    np.testing.assert_allclose(loaded.xarm_reference_joint_positions, np.zeros(7))
    np.testing.assert_allclose(
        loaded.gravity_joint_position_offset_rad, gravity_offset
    )
    assert path.read_text(encoding="utf-8").startswith("schema_version: 3")


def test_offset_yaml_rejects_changed_joint_direction(tmp_path: Path) -> None:
    """A changed raw motor direction invalidates the persisted encoder offsets."""

    path = tmp_path / "offset.yaml"
    metadata = _metadata()
    raw_positions = np.zeros(8)
    save_offset_calibration(
        path,
        **metadata,
        joint_offsets=np.zeros(8),
        raw_positions=raw_positions,
        logical_positions=raw_positions,
        gravity_joint_position_offset=np.zeros(7),
        factr_reference=raw_positions[:7],
        xarm_reference=np.zeros(7),
    )
    changed = dict(metadata)
    changed["joint_signs"] = np.array([-1.0, *([1.0] * 7)])
    with pytest.raises(ValueError, match="joint_signs"):
        load_offset_calibration(path, **changed)


def test_offset_yaml_rejects_legacy_schema_and_stale_mapping(tmp_path: Path) -> None:
    """Runtime rejects old files and mappings that differ from current directions."""

    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text("schema_version: 1\nfactr: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_offset_calibration(legacy_path, **_metadata())

    path = tmp_path / "mapping.yaml"
    metadata = _metadata()
    raw_positions = np.zeros(8)
    save_offset_calibration(
        path,
        **metadata,
        joint_offsets=np.zeros(8),
        raw_positions=raw_positions,
        logical_positions=raw_positions,
        gravity_joint_position_offset=np.zeros(7),
        factr_reference=raw_positions[:7],
        xarm_reference=np.zeros(7),
    )
    changed = dict(metadata)
    changed["xarm_to_factr_signs"] = np.array([-1.0, *([1.0] * 6)])
    with pytest.raises(ValueError, match="xarm_to_factr_signs"):
        load_offset_calibration(path, **changed)


def test_schema_two_uses_measured_factr_pose_over_old_requested_reference(
    tmp_path: Path,
) -> None:
    """Existing schema-2 files use the physical logical sample as FACTR reference."""

    path = tmp_path / "old_schema_two.yaml"
    metadata = _metadata()
    raw_positions = np.arange(8, dtype=float) / 10.0
    save_offset_calibration(
        path,
        **metadata,
        joint_offsets=np.zeros(8),
        raw_positions=raw_positions,
        logical_positions=raw_positions,
        gravity_joint_position_offset=np.zeros(7),
        factr_reference=raw_positions[:7],
        xarm_reference=np.zeros(7),
    )
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    stored["joint_mapping"]["factr_reference_joint_positions_rad"] = [0.0] * 7
    path.write_text(yaml.safe_dump(stored, sort_keys=False), encoding="utf-8")

    loaded = load_offset_calibration(path, **metadata)
    np.testing.assert_allclose(
        loaded.factr_reference_joint_positions, raw_positions[:7]
    )
