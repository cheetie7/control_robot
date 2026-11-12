"""Message conversion and serial-binding tests for the Xense ROS node."""

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from xarm_ros2_teleop.tactile.xense_publisher import (
    XenseTactileNode,
    _array_message,
    _image_message,
)


class _FakeOutputType:
    """Minimal Xense output enum namespace used by node construction tests."""

    Rectify = object()


class _FakeDevice:
    """Record resource release for a fake physical Xense sensor."""

    released = False
    requested_outputs = ()

    def selectSensorInfo(self, *output_types):
        """Record the requested output enums and return one BGR test image."""

        self.requested_outputs = output_types
        return np.zeros((7, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        """Record that node destruction released the sensor and return ``None``."""

        self.released = True


class _FakeSensorApi:
    """Emulate Xense serial scanning and exact device creation."""

    OutputType = _FakeOutputType
    created_serial = ""
    device = _FakeDevice()

    @classmethod
    def scanSerialNumber(cls) -> dict[str, int]:
        """Return one connected serial-to-camera-ID mapping."""

        return {"OP000064": 2}

    @classmethod
    def create(cls, serial_number: str, **kwargs) -> _FakeDevice:
        """Record the requested serial and return a fake device instance."""

        cls.created_serial = serial_number
        assert kwargs == {"check_serial": True}
        return cls.device


def test_image_messages_preserve_xense_encoding_and_shape() -> None:
    """BGR and millimeter depth arrays must map to their ROS image encodings."""

    stamp = Time(sec=1)
    color = _image_message(np.zeros((7, 4, 3)), "rectify", stamp, "tactile_frame")
    depth = _image_message(np.zeros((7, 4)), "depth", stamp, "tactile_frame")
    assert (color.height, color.width, color.encoding, color.step) == (7, 4, "bgr8", 12)
    assert (depth.height, depth.width, depth.encoding, depth.step) == (7, 4, "32FC1", 16)


def test_tensor_message_preserves_force_shape() -> None:
    """A force grid ``(35, 20, 3)`` must retain dimensions after flattening."""

    message = _array_message(np.zeros((35, 20, 3)), "force")
    assert [dimension.size for dimension in message.layout.dim] == [35, 20, 3]
    assert [dimension.stride for dimension in message.layout.dim] == [2100, 60, 3]
    assert len(message.data) == 2100


def test_node_opens_only_the_requested_serial() -> None:
    """The node must open the exact serial and request only enabled outputs."""

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "serial_number:=OP000064",
            "-p",
            "topic_prefix:=tactile/test",
        ]
    )
    node = XenseTactileNode(sensor_api=_FakeSensorApi)
    try:
        assert _FakeSensorApi.created_serial == "OP000064"
        node.output_publishers["rectify"] = _RecordingPublisher()
        node.publish_outputs()
        assert _FakeSensorApi.device.requested_outputs == (_FakeOutputType.Rectify,)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    assert _FakeSensorApi.device.released is True


class _RecordingPublisher:
    """Capture one ROS message without requiring a DDS transport in tests."""

    def __init__(self) -> None:
        """Initialize an empty one-dimensional message list."""

        self.messages = []

    def publish(self, message) -> None:
        """Append one ROS message to the in-memory message list."""

        self.messages.append(message)
