"""Pure pose arithmetic; positions are metres and quaternions are xyzw."""

import numpy as np


def quaternion(value) -> np.ndarray:
    """Return a finite unit xyzw quaternion, float64 [4]; reject invalid input."""
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("quaternion must be finite xyzw [4]")
    norm = np.linalg.norm(result)
    if norm < 1e-8:
        raise ValueError("quaternion must be nonzero")
    return result / norm


def multiply(left, right) -> np.ndarray:
    """Compose two xyzw rotations [4], returning a unit quaternion [4]."""
    a, b = quaternion(left), quaternion(right)
    return quaternion(
        np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]
    )


def rotation_matrix(value) -> np.ndarray:
    """Convert xyzw [4] to the active rotation matrix float64 [3,3]."""
    x, y, z, w = quaternion(value)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quaternion(matrix) -> np.ndarray:
    """Convert a proper rotation [3,3] to xyzw [4] via symmetric eigendecomposition."""
    m = np.asarray(matrix, dtype=float)
    k = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], m[1, 0] + m[0, 1], m[2, 0] + m[0, 2], m[2, 1] - m[1, 2]],
            [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], m[2, 1] + m[1, 2], m[0, 2] - m[2, 0]],
            [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
            [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], np.trace(m)],
        ]
    )
    result = np.linalg.eigh(k)[1][:, -1]
    return result if result[3] >= 0 else -result


def rotvec(value) -> np.ndarray:
    """Convert xyzw [4] to shortest axis-angle vector [3], in radians."""
    q = quaternion(value)
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return q[:3] * (2 * np.arctan2(length, q[3]) / length) if length > 1e-9 else 2 * q[:3]


def from_rotvec(value) -> np.ndarray:
    """Convert axis-angle radians [3] to unit xyzw [4]."""
    vector = np.asarray(value, dtype=float)
    angle = np.linalg.norm(vector)
    return quaternion(
        np.r_[vector * (np.sin(angle / 2) / angle if angle > 1e-9 else 0.5), np.cos(angle / 2)]
    )


def pose_array(message) -> np.ndarray:
    """Read PoseStamped into finite xyz+xyzw [7] in metres; normalize orientation."""
    p, q = message.pose.position, message.pose.orientation
    result = np.r_[[p.x, p.y, p.z], quaternion([q.x, q.y, q.z, q.w])]
    if not np.all(np.isfinite(result)):
        raise ValueError("TCP pose must be finite")
    return result


def pose_message(value, stamp, frame_id="xarm_base"):
    """Build PoseStamped from xyz+xyzw [7], metres, ROS stamp and frame name."""
    from geometry_msgs.msg import PoseStamped

    result = PoseStamped()
    result.header.stamp, result.header.frame_id = stamp, frame_id
    result.pose.position.x, result.pose.position.y, result.pose.position.z = map(float, value[:3])
    q = quaternion(value[3:])
    (
        result.pose.orientation.x,
        result.pose.orientation.y,
        result.pose.orientation.z,
        result.pose.orientation.w,
    ) = map(float, q)
    return result
