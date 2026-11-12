"""Tests for the Meta Quest pose wire protocol and ROS coordinate conversion."""

import json

import pytest
from xarm_ros2_teleop.vr.protocol import Quaternion, Vector3, parse_pose_frame, unity_pose_to_ros


def _frame_json() -> str:
    """Return one valid two-controller Unity JSON frame for protocol tests."""

    return json.dumps(
        {
            "schemaVersion": 2,
            "frameId": "quest_floor",
            "timestampSeconds": 12.5,
            "left": {
                "hand": "left",
                "tracked": True,
                "positionMeters": {"x": 1.0, "y": 2.0, "z": 3.0},
                "rotationXyzw": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
            },
            "right": {
                "hand": "right",
                "tracked": False,
                "positionMeters": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotationXyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
    )


def test_parse_pose_frame_preserves_tracking_and_units() -> None:
    """The floor-frame parser preserves timestamps, metres, and tracking flags."""

    frame = parse_pose_frame(_frame_json())

    assert frame.timestamp_seconds == pytest.approx(12.5)
    assert frame.left.tracked is True
    assert (frame.left.position.x, frame.left.position.y, frame.left.position.z) == (
        1.0,
        2.0,
        3.0,
    )
    assert frame.right.tracked is False


def test_unity_pose_converts_to_ros_forward_left_up() -> None:
    """Unity right-up-forward axes map to ROS forward-left-up axes."""

    frame = parse_pose_frame(_frame_json())
    position, rotation = unity_pose_to_ros(frame.left.position, frame.left.rotation)

    assert (position.x, position.y, position.z) == (3.0, -1.0, 2.0)
    assert (rotation.x, rotation.y, rotation.z, rotation.w) == (-0.3, 0.1, -0.2, 0.9)


@pytest.mark.parametrize("schema_version", [1, 4])
def test_parser_rejects_unknown_schema(schema_version: int) -> None:
    """Old head-relative and unknown schemas cannot masquerade as floor poses."""

    payload = json.loads(_frame_json())
    payload["schemaVersion"] = schema_version

    with pytest.raises(ValueError, match="Unsupported Quest pose schema"):
        parse_pose_frame(json.dumps(payload))


@pytest.mark.parametrize("frame_id", ["quest_head", None])
def test_parser_requires_floor_frame(frame_id: str | None) -> None:
    """Missing or head-relative frame labels are rejected before pose publication."""

    payload = json.loads(_frame_json())
    payload["frameId"] = frame_id
    with pytest.raises(ValueError, match="Expected quest_floor"):
        parse_pose_frame(json.dumps(payload))


def test_floor_vertical_displacement_is_one_to_one() -> None:
    """A 20 cm Unity floor-up movement becomes exactly 20 cm ROS +z movement."""

    orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    initial, _ = unity_pose_to_ros(Vector3(0.3, 1.0, 0.5), orientation)
    moved, _ = unity_pose_to_ros(Vector3(0.3, 1.2, 0.5), orientation)
    assert moved.x == initial.x
    assert moved.y == initial.y
    assert moved.z - initial.z == pytest.approx(0.2)
