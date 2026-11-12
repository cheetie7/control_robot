"""Publish selected Xense tactile-vision outputs through ROS 2."""

from importlib import import_module
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

_OUTPUT_TYPES = {
    "rectify": ("Rectify", Image),
    "difference": ("Difference", Image),
    "depth": ("Depth", Image),
    "marker_2d": ("Marker2D", Float32MultiArray),
    "force": ("Force", Float32MultiArray),
    "force_norm": ("ForceNorm", Float32MultiArray),
    "force_resultant": ("ForceResultant", Float32MultiArray),
}


class XenseTactileNode(Node):
    """Acquire one serial-bound Xense sensor and publish selected outputs.

    Main attributes:
        sensor: XenseSDK sensor instance opened by exact serial number.
        outputs: Ordered output names sampled from the same physical frame.
        output_publishers: ROS publishers keyed by output name.

    Rectified and difference images are published as BGR8. Depth is published
    as 32FC1 in the SDK's native millimeter unit. Force and marker tensors are
    flattened into ``Float32MultiArray`` while retaining their source shape in
    ``layout.dim``.
    """

    def __init__(self, sensor_api: Any | None = None) -> None:
        """Open the configured device and start its acquisition timer.

        Args:
            sensor_api: Optional Xense ``Sensor`` class used by tests. ``None``
                imports the installed ``xensesdk.Sensor`` implementation.

        Returns:
            ``None``. Missing serial numbers or devices raise immediately.
        """

        super().__init__("xense_tactile")
        self.declare_parameter("sensor_name", "tactile")
        self.declare_parameter("serial_number", "")
        self.declare_parameter("topic_prefix", "tactile")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("config_path", "")
        self.declare_parameter("outputs", ["rectify"])

        self.sensor_name = str(self.get_parameter("sensor_name").value)
        self.serial_number = str(self.get_parameter("serial_number").value).strip()
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).strip("/")
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        config_path = str(self.get_parameter("config_path").value).strip()
        self.outputs = tuple(self.get_parameter("outputs").value)
        self.frame_id = f"{self.sensor_name}_optical_frame"

        if not self.serial_number:
            raise ValueError("serial_number must be non-empty")
        if publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be positive")
        unsupported_outputs = set(self.outputs) - set(_OUTPUT_TYPES)
        if not self.outputs or unsupported_outputs:
            raise ValueError(f"Invalid tactile outputs: {sorted(unsupported_outputs)}")

        sensor_api = sensor_api or _load_sensor_api()
        connected_devices = sensor_api.scanSerialNumber()
        if self.serial_number not in connected_devices:
            raise RuntimeError(
                f"Xense sensor '{self.serial_number}' not found; "
                f"connected serials: {sorted(connected_devices)}"
            )

        create_arguments = {"check_serial": True}
        if config_path:
            create_arguments["config_path"] = config_path
        self.sensor = sensor_api.create(self.serial_number, **create_arguments)
        if not self.sensor:
            raise RuntimeError(f"Failed to open Xense sensor '{self.serial_number}'")

        self.output_enums = tuple(
            getattr(sensor_api.OutputType, _OUTPUT_TYPES[name][0]) for name in self.outputs
        )
        self.output_publishers = {
            name: self.create_publisher(_OUTPUT_TYPES[name][1], f"/{self.topic_prefix}/{name}", 10)
            for name in self.outputs
        }
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_outputs)
        self.get_logger().info(
            f"Opened Xense sensor {self.serial_number}; outputs={list(self.outputs)}"
        )

    def publish_outputs(self) -> None:
        """Acquire one coherent SDK frame and publish every configured output.

        Returns:
            ``None``. SDK acquisition errors propagate so launch can apply the
            configured required/optional process policy.
        """

        values = self.sensor.selectSensorInfo(*self.output_enums)
        if len(self.outputs) == 1 and not isinstance(values, tuple):
            values = (values,)
        if len(values) != len(self.outputs):
            raise RuntimeError("XenseSDK returned an unexpected output count")

        stamp = self.get_clock().now().to_msg()
        for output_name, value in zip(self.outputs, values):
            if value is None:
                self.get_logger().warning(f"Xense output '{output_name}' is unavailable")
                continue
            if output_name in {"rectify", "difference", "depth"}:
                message = _image_message(value, output_name, stamp, self.frame_id)
            else:
                message = _array_message(value, output_name)
            self.output_publishers[output_name].publish(message)

    def destroy_node(self) -> None:
        """Release the physical Xense sensor and destroy this ROS node."""

        try:
            if hasattr(self, "sensor"):
                self.sensor.release()
                del self.sensor
        finally:
            super().destroy_node()


def _load_sensor_api() -> Any:
    """Import and return ``xensesdk.Sensor`` from the uv environment."""

    return import_module("xensesdk").Sensor


def _image_message(value: np.ndarray, output_name: str, stamp: Any, frame_id: str) -> Image:
    """Convert one Xense image array to ``sensor_msgs/Image``.

    Args:
        value: BGR image ``(H, W, 3)`` or depth image ``(H, W)`` in millimeters.
        output_name: SDK-derived name: rectify, difference or depth.
        stamp: ROS builtin time message for the sampled frame.
        frame_id: Unique optical coordinate-frame name.

    Returns:
        ROS image preserving the input height and width.
    """

    if output_name == "depth":
        image = np.ascontiguousarray(value, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Xense depth must have shape (H, W), got {image.shape}")
        encoding = "32FC1"
    else:
        image = np.ascontiguousarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Xense BGR image must have shape (H, W, 3), got {image.shape}")
        encoding = "bgr8"

    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = image.shape[0]
    message.width = image.shape[1]
    message.encoding = encoding
    message.is_bigendian = False
    message.step = image.strides[0]
    message.data = image.tobytes()
    return message


def _array_message(value: np.ndarray, output_name: str) -> Float32MultiArray:
    """Convert an Xense force or marker tensor to a shape-preserving ROS array.

    Args:
        value: Numeric SDK tensor, such as force ``(35, 20, 3)`` or resultant
            force ``(6,)``.
        output_name: Human-readable SDK output name used in dimension labels.

    Returns:
        ``Float32MultiArray`` with row-major flattened data and original shape.
    """

    array = np.ascontiguousarray(value, dtype=np.float32)
    message = Float32MultiArray()
    stride = int(array.size)
    for axis, size in enumerate(array.shape):
        message.layout.dim.append(
            MultiArrayDimension(
                label=f"{output_name}_axis_{axis}",
                size=int(size),
                stride=stride,
            )
        )
        stride //= int(size)
    message.data = array.reshape(-1).tolist()
    return message


def main(args: list[str] | None = None) -> None:
    """Run one serial-configured Xense tactile ROS node.

    Args:
        args: Optional one-dimensional ROS command-line argument list.

    Returns:
        ``None`` after releasing the sensor and shutting down ROS.
    """

    rclpy.init(args=args)
    node = None
    try:
        node = XenseTactileNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
