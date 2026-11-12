"""Parse Quest pose frames and convert Unity coordinates to ROS FLU coordinates."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector3:
    """Three-dimensional vector.

    Attributes:
        x: First coordinate in metres.
        y: Second coordinate in metres.
        z: Third coordinate in metres.
    """

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion in xyzw component order.

    Attributes:
        x: Quaternion x component.
        y: Quaternion y component.
        z: Quaternion z component.
        w: Quaternion scalar component.
    """

    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True)
class ControllerPose:
    """One controller pose relative to the floor-level tracking origin.

    Attributes:
        hand: Logical side, either ``left`` or ``right``.
        tracked: Whether the Quest runtime reports a valid six-DoF pose.
        position: Unity-coordinate position in metres, shape ``(3,)``.
        rotation: Unity-coordinate unit quaternion in xyzw order, shape ``(4,)``.
    """

    hand: str
    tracked: bool
    position: Vector3
    rotation: Quaternion
    input_valid: bool = False
    grip: bool = False
    trigger: float = 0.0


@dataclass(frozen=True)
class PoseFrame:
    """Coherent left and right controller samples from one Unity frame.

    Attributes:
        timestamp_seconds: Monotonic time since the Quest application started.
        left: Left controller pose relative to the floor origin.
        right: Right controller pose relative to the floor origin.
    """

    timestamp_seconds: float
    left: ControllerPose
    right: ControllerPose
    schema_version: int = 2
    session_id: str = ""
    origin_revision: int = 0


def parse_pose_frame(line: str) -> PoseFrame:
    """Parse one version-2 newline-delimited floor-relative Quest pose frame.

    Args:
        line: UTF-8 JSON object produced by the Unity application.

    Returns:
        Parsed frame containing two floor-relative controller poses.

    Raises:
        ValueError: The JSON schema, hand labels, or numeric values are invalid.
        json.JSONDecodeError: ``line`` is not valid JSON.
    """

    payload = json.loads(line)
    version = payload.get("schemaVersion")
    if version not in {2, 3}:
        raise ValueError(
            f"Unsupported Quest pose schema: {payload.get('schemaVersion')}; "
            "rebuild and install the floor-level Quest APK"
        )
    if payload.get("frameId") != "quest_floor":
        raise ValueError(f"Expected quest_floor pose frame, received {payload.get('frameId')}")

    timestamp_seconds = float(payload["timestampSeconds"])
    if not math.isfinite(timestamp_seconds):
        raise ValueError("Quest pose timestamp must be finite")

    session_id = str(payload.get("sessionId", ""))
    revision = payload.get("originRevision", 0)
    if version == 3 and (not session_id or type(revision) is not int or revision < 0):
        raise ValueError("VR control schema requires sessionId and originRevision")
    left = _parse_controller_pose(payload["left"], "left", version)
    right = _parse_controller_pose(payload["right"], "right", version)
    return PoseFrame(timestamp_seconds, left, right, version, session_id, revision)


def unity_pose_to_ros(position: Vector3, rotation: Quaternion) -> tuple[Vector3, Quaternion]:
    """Convert a Unity floor-relative pose to ROS forward-left-up coordinates.

    Unity uses ``(+x right, +y up, +z forward)``. The returned ROS pose uses
    ``(+x forward, +y left, +z up)``. Input and output positions have shape
    ``(3,)`` in metres; quaternions have shape ``(4,)`` in xyzw order.

    Args:
        position: Unity-coordinate position in metres.
        rotation: Unity-coordinate orientation quaternion in xyzw order.

    Returns:
        Tuple ``(position, rotation)`` in the ROS ``quest_floor`` frame; no length scaling.
    """

    ros_position = Vector3(x=position.z, y=-position.x, z=position.y)
    ros_rotation = Quaternion(
        x=-rotation.z,
        y=rotation.x,
        z=-rotation.y,
        w=rotation.w,
    )
    return ros_position, ros_rotation


def _parse_controller_pose(payload: dict, expected_hand: str, version: int = 2) -> ControllerPose:
    """Validate and parse one controller object from a network frame.

    Args:
        payload: JSON dictionary containing hand, tracked, position and rotation.
        expected_hand: Required logical side for this field.

    Returns:
        Validated controller pose with finite numeric components.
    """

    hand = str(payload["hand"])
    if hand != expected_hand:
        raise ValueError(f"Expected {expected_hand} controller, received {hand}")

    position = _parse_vector3(payload["positionMeters"])
    rotation = _parse_quaternion(payload["rotationXyzw"])
    input_valid, grip, trigger = False, False, 0.0
    if type(payload["tracked"]) is not bool:
        raise ValueError("tracked must be boolean")
    if version == 3:
        input_valid, grip = payload["inputValid"], payload["grip"]
        trigger = float(payload["trigger"])
        if type(input_valid) is not bool or type(grip) is not bool:
            raise ValueError("inputValid/grip must be boolean")
        if not math.isfinite(trigger) or not 0 <= trigger <= 1:
            raise ValueError("trigger must be finite in [0,1]")
    return ControllerPose(
        hand=hand,
        tracked=bool(payload["tracked"]),
        position=position,
        rotation=rotation,
        input_valid=input_valid,
        grip=grip,
        trigger=trigger,
    )


def _parse_vector3(payload: dict) -> Vector3:
    """Parse a finite three-element JSON vector measured in metres.

    Args:
        payload: Dictionary with scalar ``x``, ``y`` and ``z`` entries.

    Returns:
        Finite vector with shape ``(3,)``.
    """

    vector = Vector3(x=float(payload["x"]), y=float(payload["y"]), z=float(payload["z"]))
    if not all(math.isfinite(value) for value in (vector.x, vector.y, vector.z)):
        raise ValueError("Quest pose position must contain finite values")
    return vector


def _parse_quaternion(payload: dict) -> Quaternion:
    """Parse a finite, non-zero JSON quaternion in xyzw order.

    Args:
        payload: Dictionary with scalar ``x``, ``y``, ``z`` and ``w`` entries.

    Returns:
        Quaternion with shape ``(4,)`` in xyzw order.
    """

    quaternion = Quaternion(
        x=float(payload["x"]),
        y=float(payload["y"]),
        z=float(payload["z"]),
        w=float(payload["w"]),
    )
    components = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    if not all(math.isfinite(value) for value in components):
        raise ValueError("Quest pose quaternion must contain finite values")
    if sum(value * value for value in components) < 1e-12:
        raise ValueError("Quest pose quaternion must be non-zero")
    return quaternion
