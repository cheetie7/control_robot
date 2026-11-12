"""Stream configured ROS recording inputs to a local Rerun Web Viewer."""

import re
import threading
from functools import partial
from importlib import import_module

import rclpy
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.recording.message_converters import (
    center_crop_resize,
    joint_values,
    metric_depth,
    rgb_image,
    stamp_seconds,
)
from xarm_ros2_teleop.system.config import load_system_config

_ENTITY_CHARACTER = re.compile(r"[^A-Za-z0-9_-]")


class RerunVisualizer(Node):
    """Visualize live inputs without participating in recording or synchronization."""

    def __init__(self) -> None:
        """Start matching Rerun servers and subscribe to configured ROS data."""

        super().__init__("rerun_visualizer")
        self.declare_parameter("config_path", "")
        self.declare_parameter("operation_mode", "teleop")
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided by teleop.launch.py")
        system = load_system_config(config_path)
        self.operation_mode = str(self.get_parameter("operation_mode").value)
        self.config = system.dashboard.rerun
        self.recording = system.recording
        if not self.config.enabled:
            raise ValueError("RerunVisualizer requires dashboard.rerun.enabled=true")

        self.rr = import_module("rerun")
        self.rrb = import_module("rerun.blueprint")
        self.rr.init("control_robot_recording")
        blueprint = self._blueprint()
        self.server_uri = self.rr.serve_grpc(
            grpc_port=self.config.grpc_port,
            default_blueprint=blueprint,
            server_memory_limit=self.config.server_memory_limit,
        )
        self.rr.serve_web_viewer(
            web_port=self.config.web_port,
            open_browser=False,
            connect_to=self.server_uri,
        )
        self.latest_images: dict[str, tuple[object, Image]] = {}
        self.image_lock = threading.Lock()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for stream in self.recording.streams:
            self.create_subscription(
                Image,
                stream.topic,
                partial(self.image_callback, stream),
                image_qos,
            )
        self.create_timer(1.0 / self.config.max_fps, self.flush_latest_images)
        for topic, callback in (
            ("/xarm/arm_joint_state", partial(self.joint_callback, "state", 7)),
            ("/xarm/commanded_joint_state", partial(self.joint_callback, "action", 7)),
            ("/robotiq/joint_state", partial(self.joint_callback, "state", 1)),
            ("/robotiq/commanded_joint_state", partial(self.joint_callback, "action", 1)),
        ):
            self.create_subscription(JointState, topic, callback, qos_profile_sensor_data)
        policy_action_topic = str(system.rl.get("policy_action_topic", "/control/policy_action"))
        self.create_subscription(
            JointState,
            policy_action_topic,
            partial(self.canonical_action_callback, "policy"),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            "/rl/executed_action",
            partial(self.canonical_action_callback, "executed"),
            qos_profile_sensor_data,
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, "/recording/status", partial(self.state_callback, "recording"), latched_qos
        )
        self.create_subscription(
            String, "/teleop/reset/status", partial(self.state_callback, "reset"), latched_qos
        )
        for topic, source in (
            ("/inference/status", "inference"),
            ("/rl/status", "rl"),
            ("/rl/action_source", "rl_action_source"),
        ):
            self.create_subscription(
                String, topic, partial(self.state_callback, source), latched_qos
            )
        self.create_subscription(
            String,
            "/rl/reward_event",
            partial(self.state_callback, "rl_reward"),
            10,
        )
        rosout_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Log, "/rosout", self.rosout_callback, rosout_qos)
        self.status_pub = self.create_publisher(
            String, "/visualization/rerun/status", 10
        )
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info(
            f"Rerun Web Viewer http://127.0.0.1:{self.config.web_port}, "
            f"source={self.server_uri}"
        )

    def _blueprint(self):
        """Return a fixed camera grid, robot plot and log layout."""

        image_views = [
            self.rrb.Spatial2DView(
                name=stream.key,
                origin=f"/sensors/{stream.key}",
            )
            for stream in self.recording.streams
        ]
        return self.rrb.Blueprint(
            self.rrb.Tabs(
                self.rrb.Grid(*image_views, name="Recording inputs"),
                self.rrb.TimeSeriesView(name="Robot state and action", origin="/robot"),
                self.rrb.TextLogView(name="System events", origin="/logs"),
                name=f"Control robot · {self.operation_mode}",
            ),
            collapse_panels=True,
        )

    def image_callback(self, stream, message: Image) -> None:
        """Replace one stream's pending preview so stale frames never queue."""

        with self.image_lock:
            self.latest_images[stream.key] = (stream, message)

    def flush_latest_images(self) -> None:
        """Log at most one newest frame per camera on each preview timer tick."""

        with self.image_lock:
            pending = self.latest_images
            self.latest_images = {}
        for stream, message in pending.values():
            self._log_image(stream, message)

    def _log_image(self, stream, message: Image) -> None:
        """Downscale and overwrite one camera's live preview frame.

        Args:
            stream: Recording stream configuration containing the entity key,
                image kind and optional depth unit.
            message: ROS image containing one RGB or depth frame.

        Returns:
            ``None``. Images are logged as Rerun static data, so each entity
            keeps only its newest frame instead of building a video timeline.
        """

        entity_path = f"/sensors/{stream.key}"
        if stream.kind == "rgb":
            preview = self._resize_preview(rgb_image(message), nearest=False)
            self.rr.log(
                entity_path,
                self.rr.Image(preview).compress(jpeg_quality=80),
                static=True,
            )
        else:
            depth = metric_depth(message, input_unit=stream.depth_unit)
            preview = self._resize_preview(depth, nearest=True)
            self.rr.log(
                entity_path,
                self.rr.DepthImage(preview, meter=1.0),
                static=True,
            )

    def _resize_preview(self, image, *, nearest: bool):
        """Return one preview with width and height scaled by the configured ratio."""

        height, width = image.shape[:2]
        output_size = (
            max(1, int(round(width * self.config.preview_scale))),
            max(1, int(round(height * self.config.preview_scale))),
        )
        if output_size == (width, height):
            return image
        return center_crop_resize(
            image,
            source_size=(width, height),
            output_size=output_size,
            nearest=nearest,
        )

    def joint_callback(self, value_kind: str, count: int, message: JointState) -> None:
        """Log actual or commanded joint values as independent scalar series."""

        timestamp = stamp_seconds(message)
        values = joint_values(message, count)
        source = "xarm" if count == 7 else "robotiq"
        self.rr.set_time("ros_time", timestamp=timestamp)
        for index, value in enumerate(values):
            joint_name = f"joint_{index + 1}" if count == 7 else "finger"
            self.rr.log(
                f"/robot/{source}/{value_kind}/{joint_name}",
                self.rr.Scalars(float(value)),
            )

    def canonical_action_callback(self, source: str, message: JointState) -> None:
        """Log one canonical policy or executed action with shape ``(8,)``.

        Args:
            source: Scalar series namespace, normally ``policy`` or ``executed``.
            message: JointState whose first eight positions are seven absolute
                arm radians followed by normalized gripper position.

        Returns:
            ``None``. The canonical producer contract guarantees eight positions.
        """

        timestamp = stamp_seconds(message)
        values = joint_values(message, 8)
        self.rr.set_time("ros_time", timestamp=timestamp)
        for index, value in enumerate(values):
            name = f"joint_{index + 1}" if index < 7 else "gripper"
            self.rr.log(f"/robot/{source}/{name}", self.rr.Scalars(float(value)))

    def state_callback(self, source: str, message: String) -> None:
        """Log one recorder or reset state transition as text."""

        self.rr.set_time("ros_time", timestamp=self.get_clock().now().nanoseconds * 1e-9)
        text = (
            InferenceStatus.from_wire(message.data).text if source == "inference" else message.data
        )
        self.rr.log(f"/logs/{source}", self.rr.TextLog(text))

    def rosout_callback(self, message: Log) -> None:
        """Log ROS warning and error messages into the Rerun text view."""

        if message.level < Log.WARN[0]:
            return
        timestamp = float(message.stamp.sec) + float(message.stamp.nanosec) * 1e-9
        self.rr.set_time("ros_time", timestamp=timestamp)
        node_name = _ENTITY_CHARACTER.sub("_", message.name.strip("/")) or "unknown"
        self.rr.log(f"/logs/rosout/{node_name}", self.rr.TextLog(message.msg))

    def publish_status(self) -> None:
        """Publish a software heartbeat while both local servers are alive."""

        message = String()
        message.data = f"OK|{self.server_uri}"
        self.status_pub.publish(message)

    def destroy_node(self) -> None:
        """Disconnect the Rerun recording stream before ROS shutdown."""

        self.rr.disconnect()
        super().destroy_node()


def main(args=None) -> None:
    """Run the independent Rerun visualization process."""

    rclpy.init(args=args)
    node = RerunVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
