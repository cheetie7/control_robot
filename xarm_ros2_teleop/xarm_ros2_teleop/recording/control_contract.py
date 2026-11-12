"""Semantic dataset guard: equal vector dimensions do not imply equal actions."""

import json
import os
import tempfile
from pathlib import Path

JOINT_ACTION = "absolute_joint_rad_plus_gripper"
TCP_ACTION = "absolute_tcp_pose_xyzw_plus_gripper"
STATE_FIELDS = [f"joint{i}" for i in range(1, 8)] + ["gripper"]


def make_contract(action_format, geometry=None, capture_config=None):
    """Return JSON-safe state/action provenance; TCP requires complete robot geometry."""
    tcp = action_format == TCP_ACTION
    if tcp and geometry is None:
        raise ValueError("VR recording waits for complete robot geometry")
    return {
        "format_version": 1,
        "source": "vr" if tcp else "factr",
        "state_fields": STATE_FIELDS,
        "state_units": ["rad"] * 7 + ["closure_0_1"],
        "action_format": action_format,
        "action_fields": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
        if tcp
        else STATE_FIELDS,
        "action_units": ["m"] * 3 + ["unit_quaternion_xyzw"] * 4 + ["closure_0_1"]
        if tcp
        else ["rad"] * 7 + ["closure_0_1"],
        "action_frame": "xarm_base" if tcp else "joint",
        "robot_geometry": geometry,
        "capture_config": capture_config,
    }


def validate_contract(root, contract):
    """Reject incompatible resume before opening writers or removing temporary frames.

    Missing contracts are allowed only for legacy joint datasets; TCP datasets
    never inherit an ambiguous old float32[8] action schema.
    """
    path = Path(root) / "meta/control_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = contract or make_contract(JOINT_ACTION)
        for key in expected.keys() - {"capture_config"}:
            if existing.get(key) != expected[key]:
                raise ValueError(f"Dataset control contract mismatch: {key} at {root}")
    elif (Path(root) / "meta/info.json").exists() and contract:
        if contract["action_format"] == TCP_ACTION:
            raise ValueError(f"Cannot append TCP actions to legacy dataset: {root}")


def write_contract(root, contract):
    """Atomically save one immutable contract; never overwrite a prior capture configuration."""
    path = Path(root) / "meta/control_contract.json"
    if contract is None or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".control_contract-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(contract, output, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
