"""Speak ROS announcement requests sequentially with a local TTS process."""

import queue
import shutil
import subprocess
import threading

import rclpy
from diagnostic_msgs.msg import KeyValue
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from xarm_ros2_teleop.system.config import load_system_config


class SpeechAnnouncer(Node):
    """Consume text announcements without blocking robot ROS callbacks.

    Requests use ``KeyValue.key`` as an event identifier and ``value`` as the
    spoken UTF-8 text. Completion publishes the same identifier after the local
    TTS process exits, including when it fails, so workflow nodes cannot hang.
    """

    def __init__(self) -> None:
        """Load speech settings and start one ordered background worker."""

        super().__init__("speech_announcer")
        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided by teleop.launch.py")
        self.config = load_system_config(config_path).speech
        executable = shutil.which(self.config.command)
        if executable is None:
            raise FileNotFoundError(f"TTS command not found: {self.config.command}")
        self.tts_executable = executable
        self.requests: queue.Queue[KeyValue | None] = queue.Queue()
        qos = QoSProfile(
            depth=20,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.done_pub = self.create_publisher(String, "/system/announcement/finished", qos)
        self.status_pub = self.create_publisher(String, "/system/speech/status", qos)
        self.create_subscription(
            KeyValue,
            "/system/announcement/request",
            self.announcement_callback,
            qos,
        )
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="speech-announcer",
            daemon=True,
        )
        self.worker.start()
        self._publish_status("OK|ready")
        self.get_logger().info(
            f"语音播报已就绪: command={self.tts_executable}, voice={self.config.voice}"
        )

    def announcement_callback(self, message: KeyValue) -> None:
        """Queue one non-empty event identifier and phrase for ordered speech."""

        if message.key and message.value:
            self.get_logger().info(f"语音播报请求: {message.value}")
            self.requests.put(message)

    def _worker_loop(self) -> None:
        """Run local TTS commands in FIFO order and acknowledge every request."""

        while True:
            request = self.requests.get()
            if request is None:
                return
            try:
                subprocess.run(
                    [
                        self.tts_executable,
                        "-v",
                        self.config.voice,
                        "-s",
                        str(self.config.speed),
                        "-a",
                        str(self.config.volume),
                        request.value,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=self.config.timeout_sec,
                )
                self._publish_status(f"OK|{request.value}")
                self.get_logger().info(f"语音播报完成: {request.value}")
            except (OSError, subprocess.SubprocessError) as exc:
                stderr = getattr(exc, "stderr", b"")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                detail = str(stderr).strip() or str(exc)
                self.get_logger().error(f"语音播报失败: {detail}")
                self._publish_status(f"ERROR|{detail}")
            done = String()
            done.data = request.key
            self.done_pub.publish(done)

    def _publish_status(self, value: str) -> None:
        """Publish one latched scalar speech health and activity string."""

        message = String()
        message.data = value
        self.status_pub.publish(message)

    def destroy_node(self) -> None:
        """Stop the worker before releasing ROS publishers and subscriptions."""

        self.requests.put(None)
        self.worker.join(timeout=1.0)
        super().destroy_node()


def main(args=None) -> None:
    """Run the local speech announcer with ROS launch arguments."""

    rclpy.init(args=args)
    node = SpeechAnnouncer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
