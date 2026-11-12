"""Pure observation, action validation and robot-command conversions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

STATE_DIM = 8
JOINT_COUNT = 7
PROTOCOL_VERSION = 1
ACTION_FORMAT = "absolute_joint_rad_plus_gripper"


def validate_policy_metadata(
    metadata: Mapping[str, Any], image_keys: list[str], action_frequency_hz: float,
    action_format: str = ACTION_FORMAT,
) -> list[str]:
    """Check deployment compatibility before arming a remote policy.

    Args:
        metadata: Server handshake; state is [8], actions are absolute [H,8].
        image_keys: Ordered configured camera names, not ROS topic names.
        action_frequency_hz: Local waypoint rate in Hz, not the driver rate.
        action_format: Required action semantics. TCP requires explicit SI/base
            frame/xyzw metadata; only joint policies allow legacy defaults.
    Returns:
        Warnings for absent legacy timing/version. Explicit version, dimension,
        camera, format or timing mismatches raise ValueError before execution.
    """

    # TCP has no legacy fallback: dimensions alone cannot distinguish it from
    # the old joint ACT. Require an explicit versioned, base-frame SI contract.
    if action_format != ACTION_FORMAT:
        required = {
            "protocol_version": PROTOCOL_VERSION, "state_dim": 8, "action_dim": 8,
            "action_format": action_format, "action_frame": "xarm_base",
            "position_unit": "m", "quaternion_order": "xyzw",
            "state_format": ACTION_FORMAT,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"TCP policy metadata {key} must be {expected!r}")
        if "action_frequency_hz" not in metadata:
            raise ValueError("TCP policy metadata missing action_frequency_hz")
    version = metadata.get("protocol_version")
    warnings = []
    if version is None:
        warnings.append("unversioned policy server; deployment contract cannot be fully checked")
    elif version != PROTOCOL_VERSION:
        raise ValueError("unsupported policy protocol version")
    if version is not None:
        for key in ("state_dim", "action_format", "image_keys", "execution_horizon"):
            if key not in metadata:
                raise ValueError(f"policy metadata missing {key}")
        if int(metadata["execution_horizon"]) < 1:
            raise ValueError("policy execution horizon must be positive")
    if int(metadata.get("state_dim", STATE_DIM)) != STATE_DIM:
        raise ValueError("policy state dimension does not match this 8-D robot")
    if metadata.get("action_format", ACTION_FORMAT) != action_format:
        raise ValueError(f"policy action_format must be {action_format}")
    if "image_keys" in metadata and list(metadata["image_keys"]) != list(image_keys):
        raise ValueError("policy camera names/order do not match configured inference images")
    frequency = metadata.get("action_frequency_hz")
    if frequency is None:
        warnings.append("checkpoint has no action frequency; using explicit local inference rate")
    elif not np.isfinite(frequency) or not np.isclose(frequency, action_frequency_hz):
        raise ValueError("policy action frequency does not match configured waypoint rate")
    return warnings


def build_observation(
    joints_rad: np.ndarray,
    gripper_normalized: float,
    images: Mapping[str, np.ndarray],
    prompt: str,
) -> dict[str, Any]:
    """Build one native OpenPI observation from synchronized robot sensors.

    Args:
        joints_rad: Measured absolute xArm angles, float array shape ``(7,)`` in radians.
        gripper_normalized: Measured Robotiq position in ``[0,1]``; zero is open.
        images: RGB uint8 arrays keyed by model camera name, each ``(H,W,3)``.
        prompt: Natural-language task instruction.

    Returns:
        Mapping with ``state`` float32 shape ``(8,)``, RGB images and prompt.
    """

    joints = np.asarray(joints_rad, dtype=np.float32)
    if joints.shape != (JOINT_COUNT,) or not np.all(np.isfinite(joints)):
        raise ValueError("joints_rad must be finite shape (7,)")
    gripper = float(gripper_normalized)
    if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
        raise ValueError("gripper_normalized must be finite and in [0,1]")
    rgb_images: dict[str, np.ndarray] = {}
    for key, value in images.items():
        image = np.asarray(value)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image {key} must have dtype uint8 and shape (H,W,3)")
        rgb_images[str(key)] = np.ascontiguousarray(image)
    if not rgb_images:
        raise ValueError("at least one RGB image is required")
    state = np.concatenate([joints, np.asarray([gripper], dtype=np.float32)])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "state": state,
        "images": rgb_images,
        "prompt": str(prompt),
    }


def validate_action_chunk(
    actions: Any,
    current_state: np.ndarray,
    joint_lower_rad: np.ndarray,
    joint_upper_rad: np.ndarray,
    max_joint_step_rad: float,
) -> np.ndarray:
    """Validate absolute policy targets before any hardware command is published.

    Args:
        actions: Candidate float array shape ``(T,8)`` with ``T >= 1``.
        current_state: Current command/measured state, float shape ``(8,)``.
        joint_lower_rad: Seven absolute lower joint limits in radians.
        joint_upper_rad: Seven absolute upper joint limits in radians.
        max_joint_step_rad: Maximum joint change from measured state to the first
            target or between consecutive model waypoints.

    Returns:
        Contiguous float32 action chunk shape ``(T,8)``. Invalid values raise.
    """

    chunk = np.asarray(actions, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    lower = np.asarray(joint_lower_rad, dtype=np.float32)
    upper = np.asarray(joint_upper_rad, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != STATE_DIM:
        raise ValueError(f"actions must have shape (T,{STATE_DIM}), got {chunk.shape}")
    if (
        state.shape != (STATE_DIM,)
        or lower.shape != (JOINT_COUNT,)
        or upper.shape != (JOINT_COUNT,)
    ):
        raise ValueError("state must be (8,) and joint limits must be (7,)")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("actions must contain only finite values")
    if np.any(chunk[:, :JOINT_COUNT] < lower) or np.any(chunk[:, :JOINT_COUNT] > upper):
        raise ValueError("action chunk exceeds configured absolute joint limits")
    if np.any(chunk[:, JOINT_COUNT] < 0.0) or np.any(chunk[:, JOINT_COUNT] > 1.0):
        raise ValueError("gripper actions must be in [0,1]")
    path = np.vstack([state[:JOINT_COUNT], chunk[:, :JOINT_COUNT]])
    absolute_steps = np.abs(np.diff(path, axis=0))
    transition_index, joint_index = np.unravel_index(
        int(np.argmax(absolute_steps)), absolute_steps.shape
    )
    largest_step = float(absolute_steps[transition_index, joint_index])
    if largest_step > float(max_joint_step_rad):
        transition = (
            "measured_state->action[0]"
            if transition_index == 0
            else f"action[{transition_index - 1}]->action[{transition_index}]"
        )
        raise ValueError(
            f"action joint step {largest_step:.4f} rad exceeds "
            f"{max_joint_step_rad:.4f} rad at joint{joint_index + 1} ({transition})"
        )
    return np.ascontiguousarray(chunk)


def interpolate_action_chunk(
    actions: np.ndarray,
    elapsed_sec: float,
    waypoint_frequency_hz: float,
) -> np.ndarray | None:
    """Sample a low-rate action chunk on the local high-rate control clock.

    Args:
        actions: Absolute model waypoints, float array shape ``(T,8)``.
        elapsed_sec: Monotonic seconds since execution of this chunk began.
        waypoint_frequency_hz: Training/action sampling frequency in hertz.

    Returns:
        Linearly interpolated absolute target with shape ``(8,)`` while the
        chunk is active. ``None`` marks the exact ``T / frequency`` boundary.

    Notes:
        Waypoint ``i`` is anchored at ``i / frequency``. The final waypoint is
        held for one sampling interval, so 25 actions recorded at 30 Hz occupy
        exactly ``25 / 30`` seconds without requiring 30 model forward passes.
    """

    chunk = np.asarray(actions, dtype=np.float32)
    elapsed = float(elapsed_sec)
    frequency = float(waypoint_frequency_hz)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != STATE_DIM:
        raise ValueError(f"actions must have shape (T,{STATE_DIM}), got {chunk.shape}")
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_sec must be a finite non-negative number")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("waypoint_frequency_hz must be a positive finite number")
    if elapsed >= chunk.shape[0] / frequency:
        return None

    phase = elapsed * frequency
    lower_index = min(int(np.floor(phase)), chunk.shape[0] - 1)
    if lower_index == chunk.shape[0] - 1:
        return np.ascontiguousarray(chunk[-1])
    fraction = np.float32(phase - lower_index)
    target = (np.float32(1.0) - fraction) * chunk[lower_index] + fraction * chunk[
        lower_index + 1
    ]
    return np.ascontiguousarray(target, dtype=np.float32)


def model_to_ros_commands(action: np.ndarray) -> tuple[np.ndarray, float]:
    """Split one absolute model target into xArm and Robotiq ROS commands.

    Args:
        action: Model target shape ``(8,)``: absolute radians then gripper ``[0,1]``.

    Returns:
        ``(arm_target_rad, leader_gripper_rad)``. The xArm target remains absolute
        with shape ``(7,)``; Robotiq maps ``-leader*255`` to device position bits.
    """

    target = np.asarray(action, dtype=np.float32)
    if target.shape != (STATE_DIM,):
        raise ValueError("action must be shape (8,)")
    return np.ascontiguousarray(target[:JOINT_COUNT]), -float(target[JOINT_COUNT])
