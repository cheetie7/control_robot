"""Persist FACTR encoder offsets and derive its home from xArm joint references."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class FactrOffsetCalibration:
    """Loaded FACTR encoder calibration.

    Args:
        joint_offsets_rad: Raw-to-logical Dynamixel offsets, shape ``(8,)`` in rad.
        raw_positions_rad: Raw Dynamixel sample used for calibration, shape ``(8,)``.
        logical_positions_rad: Logical pose reconstructed from the sample, shape ``(8,)``.
        factr_reference_joint_positions: FACTR arm pose at calibration, shape ``(7,)``.
        xarm_reference_joint_positions: Matching xArm pose, shape ``(7,)``.
        xarm_to_factr_signs: Cross-arm joint directions, shape ``(7,)``.
        gravity_joint_position_offset_rad: Measured constant translation from
            xArm-aligned logical coordinates into the URDF dynamics frame,
            shape ``(7,)`` in rad. Captured from a second known-pose sample so
            gravity compensation never depends on a guessed reference pose.
    """

    joint_offsets_rad: np.ndarray
    raw_positions_rad: np.ndarray
    logical_positions_rad: np.ndarray
    factr_reference_joint_positions: np.ndarray
    xarm_reference_joint_positions: np.ndarray
    xarm_to_factr_signs: np.ndarray
    gravity_joint_position_offset_rad: np.ndarray


def calculate_joint_offsets(
    raw_positions: np.ndarray,
    start_joints: np.ndarray,
    joint_signs: np.ndarray,
) -> np.ndarray:
    """Calculate eight FACTR raw-to-logical offsets at a known physical pose.

    Args:
        raw_positions: Raw Dynamixel angles, float array shape ``(8,)`` in rad.
        start_joints: Known FACTR pose, float array shape ``(8,)`` in rad; the
            last value is the open-gripper logical position.
        joint_signs: Raw-to-logical direction signs, float array shape ``(8,)``
            containing only ``-1`` or ``1``.

    Returns:
        Float offset array shape ``(8,)`` in rad. Arm offsets are the nearest
        multiples of pi/2, matching the FACTR horn assembly convention. The
        gripper offset is exact so its current open pose equals start joint 8.
    """

    raw = _finite_array(raw_positions, "raw_positions", 8)
    start = _finite_array(start_joints, "start_joints", 8)
    signs = _finite_array(joint_signs, "joint_signs", 8)
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("joint_signs values must be -1 or 1")
    quarter_turn = np.pi / 2.0
    offsets = np.empty(8, dtype=float)
    offsets[:7] = np.round((raw[:7] - start[:7] / signs[:7]) / quarter_turn) * quarter_turn
    offsets[7] = raw[7] - start[7] / signs[7]
    return offsets


def logical_joint_positions(
    raw_positions: np.ndarray,
    joint_offsets: np.ndarray,
    joint_signs: np.ndarray,
) -> np.ndarray:
    """Convert eight raw encoder angles into signed FACTR logical radians.

    All inputs and the returned array have shape ``(8,)``. The conversion is
    ``q_logical = (q_raw - offset) * sign``.
    """

    raw = _finite_array(raw_positions, "raw_positions", 8)
    offsets = _finite_array(joint_offsets, "joint_offsets", 8)
    signs = _finite_array(joint_signs, "joint_signs", 8)
    return (raw - offsets) * signs


def calculate_gravity_joint_offset(
    raw_positions: np.ndarray,
    teleop_joint_offsets: np.ndarray,
    joint_signs: np.ndarray,
    model_reference_joint_positions: np.ndarray,
) -> np.ndarray:
    """Reproduce the old FACTR encoder frame used by gravity compensation.

    Args:
        raw_positions: Calibration encoder sample, array shape ``(8,)`` in rad.
        teleop_joint_offsets: Persisted xArm-aligned offsets, shape ``(8,)`` in rad.
        joint_signs: Raw-to-logical direction signs, shape ``(8,)``.
        model_reference_joint_positions: Old FACTR URDF reference, shape ``(7,)``
            in rad at the same physical calibration pose.

    Returns:
        Constant position translation, float array shape ``(7,)`` in rad. Adding
        it to xArm-aligned FACTR joints exactly reproduces the old ``pi/2``
        encoder-branch selection used as Pinocchio RNEA input.
    """

    raw = _finite_array(raw_positions, "raw_positions", 8)
    teleop_offsets = _finite_array(
        teleop_joint_offsets, "teleop_joint_offsets", 8
    )
    signs = _finite_array(joint_signs, "joint_signs", 8)
    model_reference = _finite_array(
        model_reference_joint_positions,
        "model_reference_joint_positions",
        7,
    )
    model_start = np.append(model_reference, 0.0)
    model_offsets = calculate_joint_offsets(raw, model_start, signs)
    return (teleop_offsets[:7] - model_offsets[:7]) * signs[:7]


def derive_factr_home(
    factr_reference: np.ndarray,
    xarm_initial: np.ndarray,
    xarm_reference: np.ndarray,
    xarm_to_factr_signs: np.ndarray,
) -> np.ndarray:
    """Map one xArm initial pose into the corresponding FACTR joint target.

    All inputs are finite arrays with shape ``(7,)`` in radians. The returned
    FACTR target also has shape ``(7,)`` and follows
    ``q_f_ref + signs * (q_x_initial - q_x_ref)``.
    """

    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (
            factr_reference,
            xarm_initial,
            xarm_reference,
            xarm_to_factr_signs,
        )
    )
    if any(value.shape != (7,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("FACTR/xArm joint mapping arrays must contain 7 finite values")
    if not np.all(np.isin(arrays[3], (-1.0, 1.0))):
        raise ValueError("xarm_to_factr_signs values must be -1 or 1")
    return arrays[0] + arrays[3] * (arrays[1] - arrays[2])


def load_offset_calibration(
    path: str | Path,
    *,
    dynamixel_port: str,
    servo_types: list[str],
    joint_signs: np.ndarray,
    calibration_joint_positions: np.ndarray,
    operating_mode: int,
    xarm_to_factr_signs: np.ndarray,
) -> FactrOffsetCalibration:
    """Load and validate one persistent FACTR encoder-offset YAML.

    Device identity and raw-to-logical conventions must match the running
    configuration. The return contains the encoder offsets, audit samples and
    authoritative FACTR/xArm reference mapping used to derive FACTR home.
    """

    offset_path = Path(path).expanduser()
    raw = yaml.safe_load(offset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("FACTR offset file has an unsupported schema_version")
    factr = _mapping(raw.get("factr"), "factr")
    expected_scalars = {
        "dynamixel_port": str(dynamixel_port),
        "servo_types": list(servo_types),
        "operating_mode": int(operating_mode),
    }
    for key, expected in expected_scalars.items():
        if factr.get(key) != expected:
            raise ValueError(f"FACTR offset file {key} does not match current configuration")
    _require_matching_array(factr, "joint_signs", joint_signs, 8)
    _require_matching_array(
        factr,
        "calibration_joint_positions_rad",
        calibration_joint_positions,
        7,
    )
    offsets = _finite_array(factr.get("joint_offsets_rad"), "joint_offsets_rad", 8)
    raw_positions = _finite_array(factr.get("raw_positions_rad"), "raw_positions_rad", 8)
    logical_positions = _finite_array(
        factr.get("logical_positions_rad"), "logical_positions_rad", 8
    )
    gravity_offset = _finite_array(
        factr.get("gravity_joint_position_offset_rad"),
        "gravity_joint_position_offset_rad",
        7,
    )
    reconstructed = logical_joint_positions(raw_positions, offsets, joint_signs)
    if not np.allclose(logical_positions, reconstructed, atol=1e-9, rtol=0.0):
        raise ValueError("FACTR offset file raw/logical position samples are inconsistent")

    joint_mapping = _mapping(raw.get("joint_mapping"), "joint_mapping")
    _finite_array(
        joint_mapping.get("factr_reference_joint_positions_rad"),
        "factr_reference_joint_positions_rad",
        7,
    )
    xarm_reference = _finite_array(
        joint_mapping.get("xarm_reference_joint_positions_rad"),
        "xarm_reference_joint_positions_rad",
        7,
    )
    mapping_signs = _finite_array(
        joint_mapping.get("xarm_to_factr_signs"), "xarm_to_factr_signs", 7
    )
    expected_mapping_signs = _finite_array(
        xarm_to_factr_signs, "xarm_to_factr_signs", 7
    )
    if not np.array_equal(mapping_signs, expected_mapping_signs):
        raise ValueError("FACTR offset file xarm_to_factr_signs does not match configuration")
    # Early schema-2 files stored the requested coarse calibration pose here.
    # The measured logical sample is the physical pose that actually matched xArm.
    factr_reference = logical_positions[:7].copy()
    return FactrOffsetCalibration(
        joint_offsets_rad=offsets,
        raw_positions_rad=raw_positions,
        logical_positions_rad=logical_positions,
        factr_reference_joint_positions=factr_reference,
        xarm_reference_joint_positions=xarm_reference,
        xarm_to_factr_signs=mapping_signs,
        gravity_joint_position_offset_rad=gravity_offset,
    )


def save_offset_calibration(
    path: str | Path,
    *,
    dynamixel_port: str,
    servo_types: list[str],
    joint_signs: np.ndarray,
    calibration_joint_positions: np.ndarray,
    joint_offsets: np.ndarray,
    raw_positions: np.ndarray,
    logical_positions: np.ndarray,
    gravity_joint_position_offset: np.ndarray,
    operating_mode: int,
    factr_reference: np.ndarray,
    xarm_reference: np.ndarray,
    xarm_to_factr_signs: np.ndarray,
) -> None:
    """Atomically save one FACTR calibration and its reference mapping.

    Joint arrays use radians. Encoder data has shape ``(8,)``; arm reference
    arrays have shape ``(7,)``. ``gravity_joint_position_offset`` is the
    measured ``(7,)`` translation into the URDF dynamics frame from the
    second calibration capture. The saved mapping is authoritative for
    runtime FACTR home derivation and is replaced only by a new calibration.
    """

    offsets = _finite_array(joint_offsets, "joint_offsets", 8)
    raw_sample = _finite_array(raw_positions, "raw_positions", 8)
    logical_sample = _finite_array(logical_positions, "logical_positions", 8)
    factr_ref = _finite_array(factr_reference, "factr_reference", 7)
    xarm_ref = _finite_array(xarm_reference, "xarm_reference", 7)
    gravity_offset = _finite_array(
        gravity_joint_position_offset, "gravity_joint_position_offset", 7
    )
    mapping_signs = _finite_array(xarm_to_factr_signs, "xarm_to_factr_signs", 7)
    calibration_pose = _finite_array(
        calibration_joint_positions, "calibration_joint_positions", 7
    )
    signs = _finite_array(joint_signs, "joint_signs", 8)
    if not isinstance(operating_mode, int) or isinstance(operating_mode, bool):
        raise ValueError("operating_mode must be an integer")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("joint_signs values must be -1 or 1")
    if not np.all(np.isin(mapping_signs, (-1.0, 1.0))):
        raise ValueError("xarm_to_factr_signs values must be -1 or 1")
    if not np.allclose(factr_ref, logical_sample[:7], atol=1e-9, rtol=0.0):
        raise ValueError("factr_reference must equal the measured FACTR logical pose")
    reconstructed = logical_joint_positions(raw_sample, offsets, signs)
    if not np.allclose(logical_sample, reconstructed, atol=1e-9, rtol=0.0):
        raise ValueError("raw_positions, offsets and logical_positions are inconsistent")
    data = {
        "schema_version": SCHEMA_VERSION,
        "factr": {
            "dynamixel_port": str(dynamixel_port),
            "servo_types": list(servo_types),
            "operating_mode": int(operating_mode),
            "joint_signs": signs.tolist(),
            "calibration_joint_positions_rad": calibration_pose.tolist(),
            "joint_offsets_rad": offsets.tolist(),
            "raw_positions_rad": raw_sample.tolist(),
            "logical_positions_rad": logical_sample.tolist(),
            "gravity_joint_position_offset_rad": gravity_offset.tolist(),
        },
        "joint_mapping": {
            "factr_reference_joint_positions_rad": factr_ref.tolist(),
            "xarm_reference_joint_positions_rad": xarm_ref.tolist(),
            "xarm_to_factr_signs": mapping_signs.tolist(),
        },
    }
    offset_path = Path(path).expanduser()
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = offset_path.with_suffix(f"{offset_path.suffix}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, offset_path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return one YAML mapping or raise a field-specific error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"FACTR offset file {name} must be a mapping")
    return value


def _finite_array(value: Any, name: str, count: int) -> np.ndarray:
    """Return one finite float array with shape ``(count,)``."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FACTR offset file {name} must contain {count} finite values"
        ) from exc
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"FACTR offset file {name} must contain {count} finite values")
    return array


def _require_matching_array(
    mapping: Mapping[str, Any], name: str, expected: np.ndarray, count: int
) -> None:
    """Reject one calibration metadata array that differs from runtime config."""

    stored = _finite_array(mapping.get(name), name, count)
    if not np.allclose(stored, np.asarray(expected, dtype=float), atol=1e-9, rtol=0.0):
        raise ValueError(f"FACTR offset file {name} does not match current configuration")
