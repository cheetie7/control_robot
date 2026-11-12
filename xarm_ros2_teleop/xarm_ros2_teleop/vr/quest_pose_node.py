"""Receive Meta Quest controller poses over USB and publish ROS 2 topics."""

import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from xarm_ros2_teleop.vr.protocol import PoseFrame, parse_pose_frame, unity_pose_to_ros


class QuestPoseTcpServer:
    """Receive newline-delimited Quest pose frames on a background TCP thread.

    Main attributes:
        port: Host TCP port mapped from the Quest through ``metavr adb reverse``.
        frame_callback: Function receiving validated ``PoseFrame`` values.
        event_callback: Function receiving human-readable connection events.
    """

    def __init__(
        self,
        port: int,
        frame_callback: Callable[[PoseFrame], None],
        event_callback: Callable[[str], None],
    ) -> None:
        """Create a localhost-only pose server.

        Args:
            port: TCP port in the range 1-65535.
            frame_callback: Callback invoked once per valid two-controller frame.
            event_callback: Callback invoked for connection and malformed-frame events.

        Returns:
            ``None``. Call ``start`` before receiving frames.
        """

        if not 1 <= port <= 65535:
            raise ValueError("Quest pose TCP port must be in the range 1-65535")
        self.port = port
        self.frame_callback = frame_callback
        self.event_callback = event_callback
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the local TCP endpoint and start accepting Quest connections."""

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("127.0.0.1", self.port))
        server_socket.listen(1)
        server_socket.settimeout(0.5)
        self._server_socket = server_socket
        self._thread = threading.Thread(
            target=self._serve,
            name="quest-pose-tcp-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop accepting frames and release the localhost TCP port."""

        self._stop_event.set()
        if self._server_socket is not None:
            self._server_socket.close()
            self._server_socket = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _serve(self) -> None:
        """Accept one Quest client at a time until ``stop`` is requested."""

        while not self._stop_event.is_set():
            try:
                client, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                raise

            self.event_callback("Quest pose stream connected")
            with client:
                client.settimeout(0.5)
                self._read_client(client)
            self.event_callback("Quest pose stream disconnected; waiting for reconnect")

    def _read_client(self, client: socket.socket) -> None:
        """Read and parse newline-delimited UTF-8 frames from one TCP client.

        Args:
            client: Connected Quest TCP socket.

        Returns:
            ``None`` when the peer closes or the server stops.
        """

        receive_buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                return
            if not chunk:
                return

            receive_buffer.extend(chunk)
            while b"\n" in receive_buffer:
                raw_line, _, remainder = receive_buffer.partition(b"\n")
                receive_buffer = bytearray(remainder)
                if not raw_line:
                    continue
                try:
                    frame = parse_pose_frame(raw_line.decode("utf-8"))
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    self.event_callback(f"Dropped malformed Quest pose frame: {error}")
                    continue
                self.frame_callback(frame)

            if len(receive_buffer) > 1_000_000:
                self.event_callback("Dropped oversized Quest pose receive buffer")
                receive_buffer.clear()


class QuestPoseNode(Node):
    """Launch the Quest app and publish floor-relative controller poses.

    Published poses use the ROS forward-left-up convention in the ``quest_floor``
    frame. Position arrays have shape ``(3,)`` in metres and rotations are xyzw
    unit quaternions. A pose is published only while its corresponding tracking
    topic is true. RViz receives current controller axes and 10 Hz position
    markers limited to the preceding three seconds.
    """

    def __init__(self) -> None:
        """Configure USB transport, launch the Quest application, and start publishers."""

        super().__init__("quest_pose")
        self.declare_parameter("device_serial", "")
        self.declare_parameter("package_name", "com.controlrobot.questposebridge")
        self.declare_parameter("tcp_port", 7447)
        self.declare_parameter("frame_id", "quest_floor")
        self.declare_parameter("launch_app", True)
        self.declare_parameter("device_wait_timeout_sec", 120.0)
        self.declare_parameter("tracking_timeout_sec", 0.25)
        self.declare_parameter("trajectory_history_sec", 3.0)

        self.device_serial = str(self.get_parameter("device_serial").value).strip()
        self.package_name = str(self.get_parameter("package_name").value).strip()
        self.tcp_port = int(self.get_parameter("tcp_port").value)
        self.frame_id = str(self.get_parameter("frame_id").value).strip()
        self.launch_app = bool(self.get_parameter("launch_app").value)
        self.device_wait_timeout_sec = float(self.get_parameter("device_wait_timeout_sec").value)
        self.tracking_timeout_sec = float(self.get_parameter("tracking_timeout_sec").value)
        self.trajectory_history_sec = float(self.get_parameter("trajectory_history_sec").value)
        if not self.package_name:
            raise ValueError("package_name must be non-empty")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if self.device_wait_timeout_sec <= 0.0:
            raise ValueError("device_wait_timeout_sec must be positive")
        if self.tracking_timeout_sec <= 0.0:
            raise ValueError("tracking_timeout_sec must be positive")
        if self.trajectory_history_sec <= 0.0:
            raise ValueError("trajectory_history_sec must be positive")

        self._latest_frame: PoseFrame | None = None
        self._latest_frame_lock = threading.Lock()
        self._last_frame_received_monotonic: float | None = None
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._last_frame_monotonic: float | None = None
        self._stream_stale = True
        self._tracking_state = {"left": False, "right": False}
        # Sample history at 10 Hz; the position-only marker avoids drawing old pose axes.
        self._trajectory_sample_period_sec = 0.1
        self._last_trajectory_sample_monotonic = {"left": 0.0, "right": 0.0}
        self._trajectories: dict[str, deque[tuple[float, Point]]] = {
            "left": deque(),
            "right": deque(),
        }
        self._current_poses: dict[str, PoseStamped | None] = {"left": None, "right": None}
        self._last_visualization_publish_monotonic = 0.0
        self._visualization_period_sec = 0.1
        self._transport_stop_event = threading.Event()
        self._transport_thread: threading.Thread | None = None

        self.pose_publishers = {
            hand: self.create_publisher(
                PoseStamped,
                f"/quest/{hand}_controller/pose",
                qos_profile_sensor_data,
            )
            for hand in ("left", "right")
        }
        self.tracking_publishers = {
            hand: self.create_publisher(
                Bool,
                f"/quest/{hand}_controller/tracked",
                qos_profile_sensor_data,
            )
            for hand in ("left", "right")
        }
        self.visualization_publisher = self.create_publisher(
            MarkerArray,
            "/quest/visualization/markers",
            1,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(String, "/quest/bridge/status", status_qos)
        # One coherent frame carries pose AND buttons; control never joins
        # independent ROS topics with different arrival times.
        self.sample_publisher = self.create_publisher(String, "/quest/controller_samples", 1)
        self.transport_generation = 0
        self.last_wire_timestamp = None

        self.server = QuestPoseTcpServer(
            port=self.tcp_port,
            frame_callback=self._receive_frame,
            event_callback=self._events.put,
        )
        self.server.start()
        try:
            self.device_serial = self._wait_for_device(self.device_serial)
            self._configure_usb_transport()
            self._start_transport_watchdog()
        except Exception:
            self.server.stop()
            raise

        self.create_timer(1.0 / 120.0, self._publish_latest_frame)
        self._publish_status(
            f"Waiting for Quest pose stream on USB device {self.device_serial}, tcp:{self.tcp_port}"
        )

    def _wait_for_device(self, requested_serial: str) -> str:
        """Wait for one authorized Quest and return its serial number.

        Args:
            requested_serial: Exact serial requested by the ROS parameter, or empty.

        Returns:
            Serial number of the selected device.
        """

        deadline = time.monotonic() + self.device_wait_timeout_sec
        self.get_logger().info(
            f"Waiting up to {self.device_wait_timeout_sec:g} seconds for an authorized Quest"
        )
        while True:
            output = self._run_metavr(["device", "list", "--format", "json"])
            devices = [device for device in json.loads(output) if device.get("state") == "device"]
            serials = [str(device["id"]) for device in devices]
            if requested_serial and requested_serial in serials:
                return requested_serial
            if not requested_serial and len(serials) == 1:
                return serials[0]
            if not requested_serial and len(serials) > 1:
                raise RuntimeError(
                    f"Multiple Quests are online; use --device with one of: {serials}"
                )

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.0:
                target = requested_serial or "one authorized Quest"
                raise RuntimeError(f"Timed out waiting for {target}; online devices: {serials}")
            time.sleep(min(1.0, remaining_seconds))

    def _configure_usb_transport(self) -> None:
        """Wake the Quest, launch the app, and map its TCP port to Ubuntu."""

        self._run_metavr(["device", "wake", "-d", self.device_serial])
        if self.launch_app:
            self._launch_pose_app()
        # App and system-dialog transitions can restart the device transport, so
        # install the reverse mapping afterwards and keep repairing it in the watchdog.
        self._install_reverse_mapping()

    def _install_reverse_mapping(self) -> None:
        """Install the Quest-to-Ubuntu TCP mapping used by the pose stream."""

        port_spec = f"tcp:{self.tcp_port}"
        self._run_metavr(["adb", "reverse", "-d", self.device_serial, port_spec, port_spec])

    def _launch_pose_app(self) -> None:
        """Bring the installed Quest pose application to the foreground."""

        self._run_metavr(
            [
                "app",
                "launch",
                "-d",
                self.device_serial,
                self.package_name,
            ]
        )

    def _start_transport_watchdog(self) -> None:
        """Start a worker that repairs USB mappings lost during system dialogs."""

        self._transport_thread = threading.Thread(
            target=self._maintain_usb_transport,
            name="quest-usb-transport-watchdog",
            daemon=True,
        )
        self._transport_thread.start()

    def _maintain_usb_transport(self) -> None:
        """Relaunch the app and repair its USB mapping after a sustained stream loss."""

        last_error = ""
        last_launch_attempt = 0.0
        while not self._transport_stop_event.wait(2.0):
            with self._latest_frame_lock:
                last_frame_time = self._last_frame_received_monotonic
            now_monotonic = time.monotonic()
            if last_frame_time is not None and now_monotonic - last_frame_time <= 1.0:
                continue

            try:
                if self.launch_app and now_monotonic - last_launch_attempt >= 10.0:
                    last_launch_attempt = now_monotonic
                    self._launch_pose_app()
                self._install_reverse_mapping()
            except RuntimeError as error:
                error_text = str(error)
                if error_text != last_error:
                    self._events.put(f"USB pose mapping repair failed: {error_text}")
                    last_error = error_text
            else:
                last_error = ""

    def _run_metavr(self, arguments: list[str]) -> str:
        """Run one bounded metavr command and return standard output.

        Args:
            arguments: Command arguments excluding the leading ``metavr`` executable.

        Returns:
            UTF-8 standard output with surrounding whitespace removed.
        """

        metavr_path = shutil.which("metavr")
        if metavr_path is None:
            raise RuntimeError("metavr is not installed or is not on PATH")
        command_environment = os.environ.copy()
        adb_path = shutil.which("adb")
        if adb_path is not None:
            command_environment.setdefault("HZDB_ADB_PATH", adb_path)
        completed = subprocess.run(
            [metavr_path, *arguments],
            check=False,
            capture_output=True,
            env=command_environment,
            text=True,
            timeout=30.0,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"metavr {' '.join(arguments)} failed: {details}")
        return completed.stdout.strip()

    def _receive_frame(self, frame: PoseFrame) -> None:
        """Store only the newest network frame for the ROS timer.

        Args:
            frame: Validated two-controller sample from the Quest TCP thread.

        Returns:
            ``None``. Any older unconsumed frame is intentionally replaced.
        """

        with self._latest_frame_lock:
            self._latest_frame = frame
            self._last_frame_received_monotonic = time.monotonic()

    def _publish_latest_frame(self) -> None:
        """Publish the newest frame and enforce a false tracking state when stale."""

        while not self._events.empty():
            event = self._events.get()
            if event == "Quest pose stream connected":
                # Even a reconnect shorter than the stale timeout is a new
                # control epoch and must invalidate any held clutch origin.
                self.transport_generation += 1
            self._publish_status(event)

        with self._latest_frame_lock:
            frame = self._latest_frame
            self._latest_frame = None

        now_monotonic = time.monotonic()
        if frame is not None:
            if self._stream_stale or (self.last_wire_timestamp is not None
                                     and frame.timestamp_seconds < self.last_wire_timestamp):
                self.transport_generation += 1
            self.last_wire_timestamp = frame.timestamp_seconds
            self._last_frame_monotonic = now_monotonic
            self._stream_stale = False
            stamp = self.get_clock().now().to_msg()
            self._publish_controller(frame.left, stamp)
            self._publish_controller(frame.right, stamp)
            samples = {}
            for hand, controller in (("left", frame.left), ("right", frame.right)):
                p, q = unity_pose_to_ros(controller.position, controller.rotation)
                samples[hand] = {
                    "pose": [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
                    "tracked": controller.tracked, "input_valid": controller.input_valid,
                    "grip": controller.grip, "trigger": controller.trigger,
                }
            self.sample_publisher.publish(String(data=json.dumps({
                "schema_version": frame.schema_version,
                "generation": [frame.session_id, frame.origin_revision, self.transport_generation],
                "timestamp_seconds": frame.timestamp_seconds, "hands": samples,
            })))
        elif (
            not self._stream_stale
            and self._last_frame_monotonic is not None
            and now_monotonic - self._last_frame_monotonic > self.tracking_timeout_sec
        ):
            self._stream_stale = True
            self._publish_tracking("left", False)
            self._publish_tracking("right", False)
            self._publish_status("Quest pose stream stale; controller tracking forced false")

        self._publish_visualization(now_monotonic, self.get_clock().now().to_msg())

    def _publish_controller(self, controller, stamp) -> None:
        """Publish one controller state and pose, sampling position history.

        Args:
            controller: Parsed ``ControllerPose`` relative to the floor origin.
            stamp: ROS builtin time message assigned on Ubuntu receipt.

        Returns:
            ``None``. Invalid poses only update the tracking flag.
        """

        self._publish_tracking(controller.hand, controller.tracked)
        if not controller.tracked:
            self._current_poses[controller.hand] = None
            return

        position, rotation = unity_pose_to_ros(controller.position, controller.rotation)
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.pose.position.x = position.x
        message.pose.position.y = position.y
        message.pose.position.z = position.z
        message.pose.orientation.x = rotation.x
        message.pose.orientation.y = rotation.y
        message.pose.orientation.z = rotation.z
        message.pose.orientation.w = rotation.w
        self.pose_publishers[controller.hand].publish(message)
        self._current_poses[controller.hand] = message

        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._last_trajectory_sample_monotonic[controller.hand]
            < self._trajectory_sample_period_sec
        ):
            return

        point = Point(
            x=message.pose.position.x,
            y=message.pose.position.y,
            z=message.pose.position.z,
        )
        self._trajectories[controller.hand].append((now_monotonic, point))
        self._last_trajectory_sample_monotonic[controller.hand] = now_monotonic

    def _publish_visualization(self, now_monotonic: float, stamp) -> None:
        """Publish global axes, current hand axes, and three seconds of position dots.

        Args:
            now_monotonic: Local monotonic time in seconds, used to prune history.
            stamp: ROS builtin timestamp assigned to all markers in this update.

        Returns:
            ``None`` after publishing one ``visualization_msgs/MarkerArray`` at 10 Hz.
        """

        if (
            now_monotonic - self._last_visualization_publish_monotonic
            < self._visualization_period_sec
        ):
            return

        markers = MarkerArray()
        markers.markers.append(self._make_axes_marker("global_axes", 0, None, stamp, 0.30))
        hand_colors = {
            "left": ColorRGBA(r=1.0, g=0.25, b=0.25, a=1.0),
            "right": ColorRGBA(r=0.25, g=1.0, b=0.45, a=1.0),
        }
        for hand, history_id, axes_id in (("left", 1, 3), ("right", 2, 4)):
            history = self._trajectories[hand]
            while history and now_monotonic - history[0][0] > self.trajectory_history_sec:
                history.popleft()

            dots = Marker()
            dots.header.frame_id = self.frame_id
            dots.header.stamp = stamp
            dots.ns = "controller_history"
            dots.id = history_id
            dots.type = Marker.SPHERE_LIST
            dots.action = Marker.ADD
            dots.scale.x = dots.scale.y = dots.scale.z = 0.012
            dots.color = hand_colors[hand]
            dots.points = [point for _, point in history]
            markers.markers.append(dots)

            current_pose = self._current_poses[hand]
            if current_pose is None:
                axes = Marker()
                axes.header.frame_id = self.frame_id
                axes.header.stamp = stamp
                axes.ns = "controller_current"
                axes.id = axes_id
                axes.action = Marker.DELETE
            else:
                axes = self._make_axes_marker(
                    "controller_current",
                    axes_id,
                    current_pose,
                    stamp,
                    0.15,
                )
            markers.markers.append(axes)

        self.visualization_publisher.publish(markers)
        self._last_visualization_publish_monotonic = now_monotonic

    def _make_axes_marker(
        self, namespace: str, marker_id: int, pose, stamp, length: float
    ) -> Marker:
        """Create an RGB coordinate triad, optionally located at a controller pose.

        Args:
            namespace: RViz marker namespace used to update this triad in place.
            marker_id: Stable integer ID unique inside ``namespace``.
            pose: ``PoseStamped`` transform for a controller, or ``None`` for origin.
            stamp: ROS builtin timestamp assigned to the marker header.
            length: Axis length in metres.

        Returns:
            A ``visualization_msgs/Marker`` line list containing three colored axes.
        """

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        if pose is not None:
            marker.pose = pose.pose
        marker.scale.x = 0.006
        origin = Point()
        endpoints = (Point(x=length), Point(y=length), Point(z=length))
        colors = (
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
        )
        for endpoint, color in zip(endpoints, colors, strict=True):
            marker.points.extend((origin, endpoint))
            marker.colors.extend((color, color))
        return marker

    def _publish_tracking(self, hand: str, tracked: bool) -> None:
        """Publish a controller tracking state and retain it for stale detection.

        Args:
            hand: Logical side, ``left`` or ``right``.
            tracked: Whether a current six-DoF pose is valid.

        Returns:
            ``None`` after publishing one ``std_msgs/Bool``.
        """

        self._tracking_state[hand] = tracked
        if not tracked:
            self._current_poses[hand] = None
        self.tracking_publishers[hand].publish(Bool(data=tracked))

    def _publish_status(self, text: str) -> None:
        """Publish and log one bridge status string.

        Args:
            text: Human-readable device, connection, or safety status.

        Returns:
            ``None`` after publishing one latched status message.
        """

        self.status_publisher.publish(String(data=text))
        self.get_logger().info(text)

    def destroy_node(self) -> None:
        """Stop the TCP worker before destroying ROS publishers and timers."""

        self._transport_stop_event.set()
        if self._transport_thread is not None:
            self._transport_thread.join(timeout=2.0)
            self._transport_thread = None
        if hasattr(self, "server"):
            self.server.stop()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Run the Quest-to-ROS pose bridge until interrupted.

    Args:
        args: Optional one-dimensional ROS command-line argument list.

    Returns:
        ``None`` after closing the TCP server and shutting down ROS.
    """

    rclpy.init(args=args)
    node = None
    try:
        node = QuestPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
