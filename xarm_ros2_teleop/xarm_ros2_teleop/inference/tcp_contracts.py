"""TCP policy boundary: joint observations, metre xyz/xyzw/gripper actions."""

import numpy as np

from xarm_ros2_teleop.vr.pose_math import quaternion

TCP_ACTION_FORMAT = "absolute_tcp_pose_xyzw_plus_gripper"


def validate_tcp_chunk(actions, config):
    """Return contiguous float32 [H,8] base-frame TCP targets, or raise ValueError.

    xyz uses metres; xyzw must already be unit length (small rounding allowed).
    Gripper is closure [0,1]. Workspace is checked without silently clipping.
    Quaternion signs are made continuous; driver owns physical speed limits.
    """
    chunk = np.array(actions, dtype=np.float32, copy=True)
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 8:
        raise ValueError("TCP actions must have shape (H,8)")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("TCP actions must be finite")
    if np.any((chunk[:, 7] < 0) | (chunk[:, 7] > 1)):
        raise ValueError("TCP gripper actions must be in [0,1]")
    if not np.allclose(np.linalg.norm(chunk[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError("TCP xyzw quaternion must be normalized")
    for index, action in enumerate(chunk):
        config.check_workspace(action[:3])
        action[3:7] = quaternion(action[3:7])
        if index and action[3:7] @ chunk[index - 1, 3:7] < 0:
            action[3:7] *= -1
    return np.ascontiguousarray(chunk)


def interpolate_tcp_chunk(actions, elapsed_sec, frequency_hz):
    """Sample TCP [H,8] at elapsed seconds; use shortest-arc SLERP, not xyz-w lerp.

    Translation/gripper interpolate linearly. Final waypoint is held for one
    interval, so H waypoints occupy H/frequency_hz seconds; returns [8] or None.
    Input is a validated, sign-continuous chunk.
    """
    if (
        not np.isfinite(elapsed_sec)
        or elapsed_sec < 0
        or not np.isfinite(frequency_hz)
        or frequency_hz <= 0
    ):
        raise ValueError("invalid TCP interpolation time/rate")
    phase = elapsed_sec * frequency_hz
    if phase >= len(actions):
        return None
    index = int(phase)
    if index == len(actions) - 1:
        return actions[-1].copy()
    fraction = phase - index
    first, second = actions[index : index + 2]
    target = (1 - fraction) * first + fraction * second
    start, end = first[3:7], second[3:7].copy()
    dot = float(start @ end)
    if dot < 0:
        end, dot = -end, -dot
    if dot < 0.9995:
        angle = np.arccos(np.clip(dot, -1, 1))
        target[3:7] = (
            np.sin((1 - fraction) * angle) * start + np.sin(fraction * angle) * end
        ) / np.sin(angle)
    target[3:7] = quaternion(target[3:7])
    return np.ascontiguousarray(target, dtype=np.float32)
