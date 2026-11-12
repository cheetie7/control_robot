"""ROS 2 node that records synchronized teleoperation episodes as LeRobot v3."""

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from functools import partial
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Int32MultiArray, String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from xarm_ros2_teleop.evaluation.outcomes import EvaluationOutcomeWriter
from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.recording.control_contract import make_contract, validate_contract
from xarm_ros2_teleop.recording.keyboard_controller import KeyboardController
from xarm_ros2_teleop.recording.lerobot_writer import (
    LeRobotWriter,
    validate_overwrite_target,
)
from xarm_ros2_teleop.recording.message_converters import (
    center_crop_resize,
    depth_preview,
    joint_values,
    rgb_image,
    stamp_seconds,
)
from xarm_ros2_teleop.recording.synchronizer import (
    FrameSynchronizer,
    SynchronizationError,
    XarmState,
)
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.vr.pose_math import pose_array

_TELEOP_EPISODE_CATEGORIES = ("success", "optimal", "failure", "crash")


def decode_evaluation_key(sequence: bytes) -> str | None:
    """Map terminal keys to labeled policy-evaluation episode commands."""

    return {
        b"s": "start",
        b"S": "start",
        b"y": "success",
        b"Y": "success",
        b"n": "failure",
        b"N": "failure",
        b"\x1b[D": "discard",
        b"\x1bOD": "discard",
        b"\x1b": "finish",
    }.get(sequence)


class EpisodeRecorder(Node):
    """Record fixed-schema 30 FPS episodes around the configured camera clock.

    Image observations are arrays with shape ``(H,W,3)``. Robot state and
    action are float32 arrays with shape ``(8,)``: seven xArm joints followed
    by one normalized Robotiq position.
    """

    # Keeps small unit-test harnesses and legacy direct construction in teleop
    # mode; normal ROS construction always replaces this instance attribute.
    evaluation_mode = False
    rl_mode = False
    vr_mode = False
    robot_geometry = None
    vr_ready = False
    vr_status_at = 0.0
    vr_reason = "waiting for VR input"
    last_tcp_orientation = None

    def __init__(self) -> None:
        """Load the unified YAML, subscribe to sensors and expose episode services."""

        super().__init__("episode_recorder")
        self.declare_parameter("config_path", "")
        self.declare_parameter("operation_mode", "teleop")
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided by teleop.launch.py")
        self.system = load_system_config(config_path)
        self.config = self.system.recording
        self.vr_mode = self.system.teleop.get("source", "factr") == "vr"
        self.operation_mode = str(self.get_parameter("operation_mode").value)
        if self.operation_mode not in {"teleop", "evaluation", "rl"}:
            raise ValueError("operation_mode must be teleop, evaluation or rl")
        self.evaluation_mode = self.operation_mode == "evaluation"
        self.rl_mode = self.operation_mode == "rl"
        self.evaluation_config = self.system.evaluation
        if self.evaluation_mode and not bool(self.evaluation_config.get("enabled", False)):
            raise ValueError("evaluation recorder requires evaluation.enabled=true")
        self.target_episodes = int(
            self.evaluation_config.get("max_episodes", 100)
            if self.evaluation_mode
            else self.config.num_episodes
        )
        self.max_episode_sec = float(self.evaluation_config.get("max_episode_sec", 30.0))
        if not self.config.enabled:
            raise ValueError("EpisodeRecorder may only start when recording.enabled=true")

        default_dataset_root = str(
            Path(self.evaluation_config.get("dataset_root", self.config.dataset_root()))
            .expanduser()
            .resolve()
            if self.evaluation_mode
            else self.config.dataset_root()
        )
        if self.rl_mode:
            default_dataset_root = str(self.system.rl.get("dataset_root", "/home/gxy/datasets/rl"))
        self.declare_parameter("session_instruction", self.config.instruction)
        self.declare_parameter("session_dataset_root", default_dataset_root)
        self.declare_parameter("session_write_mode", "resume")
        self.instruction = str(self.get_parameter("session_instruction").value).strip()
        self.dataset_root = Path(
            str(self.get_parameter("session_dataset_root").value)
        ).expanduser()
        self.write_mode = str(self.get_parameter("session_write_mode").value)
        if not self.instruction:
            raise ValueError("session_instruction must be non-empty")
        if not self.dataset_root.is_absolute():
            raise ValueError("session_dataset_root must be an absolute path")
        if self.write_mode not in {"resume", "overwrite"}:
            raise ValueError("session_write_mode must be resume or overwrite")

        self.lock = threading.RLock()
        # Writer workers acquire this lock; sensor callbacks never wait for
        # dataset imports, image copies or video/Parquet I/O.
        self.writer_lock = threading.Lock()
        self.capture_generation = 0
        self.callback_group = ReentrantCallbackGroup()
        self.synchronizer = FrameSynchronizer(self.config)
        self.category_writers = self._new_category_writers()
        self.writer = (
            self._new_writer()
            if self.evaluation_mode
            else self.category_writers["success"]
        )
        self.outcome_writer = EvaluationOutcomeWriter(self.dataset_root)
        self.successful_episodes, self.failed_episodes = self.outcome_writer.counts()
        self.discarded_attempts = 0
        self.pending_outcome: dict | None = None
        self.last_outcome = ""
        self.last_outcome_reason = ""
        self.xarm_control_ok: bool | None = None
        self.inference_ready = not self.evaluation_mode
        self.stream_by_key = {stream.key: stream for stream in self.config.streams}
        self.writer_queue: Queue = Queue(maxsize=self.config.writer_queue_size)
        self.writer_stop = threading.Event()
        self.writer_prepared = False
        self.writer_prepare_requested = False
        self.session_configured = False
        self.writer_queue_dropped_frames = 0
        self.frame_writer_error = ""
        self.captured_frames_in_episode = 0
        self.session_ready = self.evaluation_mode or self.rl_mode or not self.system.warmup.enabled
        self.state = "IDLE" if self.session_ready else "WARMUP"
        self.capture_started_at = None
        self.last_anchor_timestamp = None
        # A missing robot sample drops only the affected dataset row. The
        # counters remain visible in Web metrics without disabling teleoperation.
        self.sync_dropped_frames = 0
        self.last_sync_warning = ""
        self.last_sync_warning_at = 0.0
        self.last_error = ""
        self.reset_continuation = None
        self.reset_deadline = None
        self.reset_goal_handle = None
        self.reset_request_id = 0
        self.post_reset_state = "IDLE"
        self.episode_completion_pending = False
        self.episode_write_done = True
        self.episode_reset_done = True
        self.episode_write_error = ""
        self.episode_reset_error = ""
        self.episode_finish_session = False
        self.episode_saved = False
        self.episode_category = "success"
        self.episode_number = 0
        # A ROS timer consumes Arrow episode commits on the process main
        # thread. libarrow may segfault when a worker thread owns the Parquet
        # commit; reset remains concurrent because it runs in another node.
        self.episode_writer_jobs: Queue = Queue()
        self.finalize_requested = False
        self.error_recovery_pending = False
        self.error_reset_started = False
        self.error_episode_number = 0
        self.factr_connected = True
        self.factr_recovery_pending = False
        self.keyboard_commands = deque()
        self.keyboard = None
        self.auto_start_pending = (
            False if self.evaluation_mode or self.rl_mode else self.config.auto_start_first_episode
        )
        self.add_on_set_parameters_callback(self._configure_session_parameters)
        self.frame_writer_thread = threading.Thread(
            target=self._frame_writer_loop,
            name="lerobot-frame-writer",
            daemon=True,
        )
        self.frame_writer_thread.start()

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # A latest-frame recorder must not drain an old DDS image backlog.
        # BEST_EFFORT matches camera publishers while depth=1 replaces an
        # unread image as soon as a newer one arrives.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.status_pub = self.create_publisher(String, "/recording/status", latched_qos)
        self.teleop_active_pub = self.create_publisher(
            Bool, "/teleop/episode_active", latched_qos
        )
        self.evaluation_active_pub = self.create_publisher(
            Bool, "/evaluation/episode_active", latched_qos
        )
        self.metrics_pub = self.create_publisher(
            DiagnosticArray, "/recording/metrics", latched_qos
        )
        self.announcement_pub = self.create_publisher(
            KeyValue, "/system/announcement/request", latched_qos
        )
        self.create_subscription(
            Bool,
            "/system/session/ready",
            self.session_ready_callback,
            latched_qos,
            callback_group=self.callback_group,
        )
        for stream in self.config.streams:
            self.create_subscription(
                Image,
                stream.topic,
                partial(self.image_callback, stream.key),
                image_qos,
                callback_group=self.callback_group,
            )
        self.create_subscription(
            JointState,
            "/xarm/arm_joint_state",
            self.xarm_state_callback,
            30,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointState,
            "/robotiq/joint_state",
            self.robotiq_state_callback,
            30,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PoseStamped if self.vr_mode else JointState,
            "/xarm/commanded_tcp_pose" if self.vr_mode else "/xarm/commanded_joint_state",
            self.tcp_action_callback if self.vr_mode else self.xarm_action_callback,
            30,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointState,
            "/robotiq/commanded_joint_state",
            self.robotiq_action_callback,
            30,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Int32MultiArray,
            "/xarm/error_warn",
            self.xarm_error_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            "/xarm/control_ok",
            self.control_ok_callback,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            "/inference/status",
            self.inference_status_callback,
            latched_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            "/factr/connected",
            self.factr_connected_callback,
            latched_qos,
            callback_group=self.callback_group,
        )
        if self.vr_mode:
            self.create_subscription(String, "/vr/teleop_status", self.vr_status_callback,
                                     10, callback_group=self.callback_group)
            self.create_subscription(String, "/xarm/geometry", self.geometry_callback,
                                     latched_qos, callback_group=self.callback_group)
        self.create_subscription(
            Bool,
            "/xarm/leader_stale",
            self.leader_stale_callback,
            latched_qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            "/teleop/reset/status",
            self.reset_status_callback,
            # Reset phases are transaction events. Main-thread video/Parquet
            # work can span COMPLETE -> IDLE; retaining their short history
            # prevents a completed automatic reset from leaving save locked.
            QoSProfile(
                depth=16,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
            callback_group=self.callback_group,
        )
        self.reset_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.config.reset_action_name,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/start_staging",
            self.start_staging,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/start_episode",
            self.start_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/stop_episode",
            self.stop_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/save_optimal",
            self.save_optimal_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/save_failure",
            self.save_failure_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/discard_episode",
            self.discard_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/recording/finalize",
            self.finalize_dataset,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/evaluation/success",
            self.success_episode,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "/evaluation/failure",
            self.failure_episode,
            callback_group=self.callback_group,
        )
        self.create_timer(0.01, self.process_pending, callback_group=self.callback_group)
        self.create_timer(
            0.02,
            partial(self.process_writer_job, 0.0),
            callback_group=self.callback_group,
        )
        self.create_timer(0.02, self.process_keyboard, callback_group=self.callback_group)
        self.create_timer(0.5, self.publish_metrics, callback_group=self.callback_group)
        self.teleop_active = False
        self._set_teleop_active(False)
        self._set_status(self.state)
        if self.config.keyboard_enabled and not self.rl_mode:
            self.keyboard = KeyboardController(
                self.config.keyboard_device,
                self.queue_keyboard_command,
                decoder=decode_evaluation_key if self.evaluation_mode else None,
            )
            self.keyboard.start()
            controls = (
                "评测按键: s=开始, y=成功并保存, n=失败并保存, ←=丢弃, Esc=结束"
                if self.evaluation_mode
                else "录制按键: g=预定位, s=开始本条, ←=丢弃/退出预定位, "
                "→=success, o=optimal, f=failure, Esc=结束录制"
            )
            self.get_logger().info(controls)
        self.auto_start_timer = self.create_timer(
            0.25,
            self.auto_start_first_episode,
            callback_group=self.callback_group,
        )

    def _new_category_writers(self) -> dict[str, LeRobotWriter]:
        """Create the four outcome writers used only by teleoperation.

        Returns:
            Mapping from ``success``, ``optimal``, ``failure`` and ``crash``
            to independent LeRobot datasets below the selected parent path.
            Evaluation mode returns an empty mapping and retains its original
            single-dataset behavior.
        """

        if self.evaluation_mode:
            return {}
        return {
            category: self._new_writer(category)
            for category in _TELEOP_EPISODE_CATEGORIES
        }

    def _new_writer(self, category: str | None = None) -> LeRobotWriter:
        """Create one unprepared writer from the active session settings.

        Args:
            category: Teleoperation outcome folder name. ``None`` preserves
                the exact-root writer used by policy evaluation.

        Returns:
            An unprepared writer. Teleoperation categories target
            ``<dataset_root>/<category>``; evaluation targets ``dataset_root``.
        """

        root = self.dataset_root if category is None else self.dataset_root / category
        repo_id = self.config.repo_id if category is None else f"{self.config.repo_id}_{category}"
        return LeRobotWriter(
            root=root,
            repo_id=repo_id,
            robot_type=self.config.robot_type,
            fps=self.config.fps,
            vcodec=self.config.vcodec,
            image_writer_threads=self.config.image_writer_threads,
            streaming_encoding=self.config.streaming_encoding,
            write_mode=self.write_mode,
            control_contract=(None if self.vr_mode else make_contract(self.config.action_format)),
        )

    def _configure_session_parameters(
        self, parameters: list[Parameter]
    ) -> SetParametersResult:
        """Apply task, exact dataset path and write mode before recording.

        Args:
            parameters: ROS parameters. The supported scalar strings are
                ``session_instruction``, ``session_dataset_root`` and
                ``session_write_mode``.

        Returns:
            A ROS parameter result. Settings are accepted only once, before
            schema preparation or the first episode starts.
        """

        supported = {
            "session_instruction",
            "session_dataset_root",
            "session_write_mode",
        }
        updates = {item.name: item.value for item in parameters if item.name in supported}
        if not updates:
            return SetParametersResult(successful=True)
        with self.lock:
            if (
                self.session_configured
                or self.writer_prepare_requested
                or self.writer_prepared
                or self.writer.dataset is not None
                or self.state not in {"WARMUP", "IDLE"}
            ):
                return SetParametersResult(
                    successful=False,
                    reason="recording session settings are already locked",
                )
            instruction = updates.get("session_instruction", self.instruction)
            root_value = updates.get("session_dataset_root", str(self.dataset_root))
            write_mode = updates.get("session_write_mode", self.write_mode)
            if not isinstance(instruction, str) or not instruction.strip():
                return SetParametersResult(
                    successful=False,
                    reason="task instruction must be a non-empty string",
                )
            if not isinstance(root_value, str) or not root_value.strip():
                return SetParametersResult(
                    successful=False,
                    reason="dataset path must be a non-empty string",
                )
            dataset_root = Path(root_value).expanduser()
            if not dataset_root.is_absolute():
                return SetParametersResult(
                    successful=False,
                    reason="dataset path must be absolute or start with '~/'",
                )
            if write_mode not in {"resume", "overwrite"}:
                return SetParametersResult(
                    successful=False,
                    reason="write mode must be resume or overwrite",
                )
            if write_mode == "overwrite":
                try:
                    targets = (
                        [dataset_root]
                        if self.evaluation_mode
                        else [
                            dataset_root / category
                            for category in _TELEOP_EPISODE_CATEGORIES
                        ]
                    )
                    for target in targets:
                        validate_overwrite_target(target)
                except ValueError as exc:
                    return SetParametersResult(successful=False, reason=str(exc))
            self.instruction = instruction.strip()
            self.dataset_root = dataset_root
            self.write_mode = str(write_mode)
            self.category_writers = self._new_category_writers()
            self.writer = (
                self._new_writer()
                if self.evaluation_mode
                else self.category_writers["success"]
            )
            self.outcome_writer = EvaluationOutcomeWriter(self.dataset_root)
            self.successful_episodes, self.failed_episodes = self.outcome_writer.counts()
            self.session_configured = True
            if self.session_ready:
                self._set_status("PREPARING")
        return SetParametersResult(successful=True)

    def image_callback(self, key: str, message: Image) -> None:
        """Buffer one decoded image and trigger assembly from the anchor stream.

        Args:
            key: Configured scalar stream key under ``observation.images``.
            message: ROS image whose decoded value has shape ``(H,W,3)`` and
                whose header timestamp is used for nearest-frame selection.
        """

        stream = self.stream_by_key[key]
        value = (
            rgb_image(message)
            if stream.kind == "rgb"
            else depth_preview(
                message,
                input_unit=stream.depth_unit,
                depth_min_m=stream.depth_min_m,
                depth_max_m=stream.depth_max_m,
            )
        )
        if stream.output_size is not None:
            # Topic frames stay at the driver resolution; only the stored
            # training view is cropped and rescaled, with depth kept nearest-
            # neighbour so invalid pixels are not blended into valid edges.
            value = center_crop_resize(
                value,
                source_size=stream.source_size,
                output_size=stream.output_size,
                nearest=stream.kind == "depth",
            )
        timestamp = stamp_seconds(message)
        with self.lock:
            sample = self.synchronizer.images[key].add(timestamp, value)
            if key != self.config.anchor_key:
                return
            if self.state != "RECORDING":
                if self.session_configured:
                    self._request_writer_prepare()
                return
            minimum_period = 0.75 / self.config.fps
            if self.last_anchor_timestamp is not None:
                period = timestamp - self.last_anchor_timestamp
                if period < minimum_period:
                    return
                # A physical 30 FPS camera may occasionally miss a USB/DDS
                # deadline. Keep the next real frame as the anchor instead of
                # invalidating the complete episode for one scheduling jitter.
            self.last_anchor_timestamp = timestamp
            try:
                frame = self._assemble_frame(sample)
            except SynchronizationError as exc:
                self._record_sync_drop(str(exc), time.monotonic())
                return
            self._enqueue_frame(frame, self.capture_generation)

    def _assemble_frame(self, anchor) -> dict:
        """Return one synchronized row for the camera's timestamped anchor."""

        return self.synchronizer.assemble(anchor, self.instruction)

    def _schema_frame(self) -> dict:
        """Return a non-recorded example used to initialize dataset features."""

        return self.synchronizer.schema_frame(self.instruction)

    def xarm_state_callback(self, message: JointState) -> None:
        """Buffer measured xArm position, velocity and effort arrays of shape ``(7,)``."""

        position = joint_values(message, 7)
        velocity = _optional_joint_field(message.velocity, 7)
        effort = _optional_joint_field(message.effort, 7)
        with self.lock:
            self.synchronizer.xarm_state.add(
                stamp_seconds(message), XarmState(position, velocity, effort)
            )

    def robotiq_state_callback(self, message: JointState) -> None:
        """Buffer one normalized Robotiq position with shape ``(1,)``."""

        with self.lock:
            self.synchronizer.robotiq_state.add(stamp_seconds(message), joint_values(message, 1))

    def xarm_action_callback(self, message: JointState) -> None:
        """Retain a short causal history of accepted seven-axis xArm targets."""

        with self.lock:
            self.synchronizer.xarm_action.add(stamp_seconds(message), joint_values(message, 7))

    def tcp_action_callback(self, message: PoseStamped) -> None:
        """Buffer only SDK-accepted base-frame TCP commands [7], metres/xyzw."""
        if message.header.frame_id == "xarm_base":
            with self.lock:
                pose = pose_array(message)
                if (self.last_tcp_orientation is not None
                        and pose[3:] @ self.last_tcp_orientation < 0):
                    pose[3:] *= -1
                self.last_tcp_orientation = pose[3:].copy()
                self.synchronizer.xarm_action.add(stamp_seconds(message), pose)

    def geometry_callback(self, message: String) -> None:
        """Store the driver's read-only robot geometry snapshot for dataset provenance."""
        with self.lock:
            self.robot_geometry = json.loads(message.data)

    def vr_status_callback(self, message: String) -> None:
        """Record input health and following flags, without ending an interrupted episode."""
        status = json.loads(message.data)
        with self.lock:
            self.vr_ready = bool(status["ready"])
            self.vr_status_at = time.monotonic()
            self.vr_reason = str(status["reason"])
            timestamp = self.get_clock().now().nanoseconds * 1e-9
            self.synchronizer.teleop_status.add(timestamp, np.asarray(
                [status["tracking_valid"], status["following"]], dtype=np.float32))

    def _vr_input_ready(self) -> bool:
        """Require fresh VR readiness heartbeat before a new episode or staging."""
        return self.vr_ready and time.monotonic() - self.vr_status_at <= 0.25

    def robotiq_action_callback(self, message: JointState) -> None:
        """Retain a short causal history of normalized Robotiq commands."""

        with self.lock:
            self.synchronizer.robotiq_action.add(stamp_seconds(message), joint_values(message, 1))

    def xarm_error_callback(self, message: Int32MultiArray) -> None:
        """Leave staging or save active teleoperation capture as ``crash``.

        Args:
            message: ``[error, warning]`` controller codes. A nonzero first
                value while ``STAGING`` or ``RECORDING`` starts recovery.

        Returns:
            Nothing. The reset coordinator independently clears the error and
            returns the mode-specific actuators to their configured reset state.
        """

        if not message.data or int(message.data[0]) == 0:
            return
        with self.lock:
            if self.evaluation_mode and self.state == "RECORDING":
                error_code = int(message.data[0])
                self.last_error = f"xArm error {error_code}; evaluation marked failure"
                self.get_logger().error(self.last_error)
                self._finish_evaluation_episode(
                    outcome="failure",
                    reason=f"xarm_error_{error_code}",
                    terminated=True,
                    truncated=False,
                    automatic_error_reset=True,
                )
                return
            if self.state not in {"STAGING", "RECORDING"} or self.error_recovery_pending:
                return
            was_recording = self.state == "RECORDING"
            error_code = int(message.data[0])
            self.error_episode_number = self.writer.episode_count + 1
            if was_recording and self.captured_frames_in_episode > 0:
                self.last_error = (
                    f"xArm error {error_code}; saving episode as crash before reset"
                )
                self.get_logger().error(self.last_error)
                accepted, detail = self._finish_episode(
                    save=True,
                    finish_session=False,
                    automatic_error_reset=True,
                    category="crash",
                )
                if not accepted:
                    self.get_logger().error(f"crash save request failed: {detail}")
                return
            self.capture_generation += 1
            self._discard_queued_frames()
            if was_recording:
                with self.writer_lock:
                    self.writer.discard_episode()
            self.captured_frames_in_episode = 0
            self.synchronizer.clear()
            self.error_recovery_pending = True
            self.error_reset_started = False
            self.last_error = (
                f"xArm error {error_code}; empty episode {self.error_episode_number} discarded"
                if was_recording
                else f"xArm error {error_code}; staging aborted"
            )
            self._set_teleop_active(False)
            self.get_logger().error(self.last_error)
            if was_recording:
                self._announce(
                    "episode_discarded",
                    self.system.speech.phrase(
                        "episode_discarded", episode=self.error_episode_number
                    ),
                )
            self._set_status("ERROR_RESETTING")

    def control_ok_callback(self, message: Bool) -> None:
        """Save an active evaluation as failure on follower-health loss.

        Args:
            message: Current xArm control-health flag. Startup ``False`` values
                outside a recording episode do not create outcomes.
        """

        with self.lock:
            self.xarm_control_ok = bool(message.data)
            if self.xarm_control_ok or not self.evaluation_mode:
                return
            if self.state != "RECORDING":
                return
            self._finish_evaluation_episode(
                outcome="failure",
                reason="xarm_control_unhealthy",
                terminated=True,
                truncated=False,
                automatic_error_reset=True,
            )

    def inference_status_callback(self, message: String) -> None:
        """Convert a policy rollout failure or fatal server stop into failure."""

        if not self.evaluation_mode:
            return
        with self.lock:
            status = InferenceStatus.from_wire(message.data)
            self.inference_ready = status.phase == "WAITING_FOR_EPISODE"
            failure_status = status.failed
            if not failure_status:
                return
            if self.state != "RECORDING":
                return
            reason = status.reason or "inference_stopped"
            self._finish_evaluation_episode(
                outcome="failure",
                reason=f"inference_{reason}",
                terminated=True,
                truncated=False,
                automatic_error_reset=False,
            )

    def factr_connected_callback(self, message: Bool) -> None:
        """Abort active capture on disconnect and unlock only after reset.

        Args:
            message: Latched FACTR serial health. ``False`` starts recovery;
                ``True`` resumes a paused reset deadline but never starts an
                episode automatically.

        Returns:
            ``None``. Finished episodes already in ``SAVING`` remain intact.
        """

        if self.evaluation_mode:
            return
        if self.vr_mode:
            return
        with self.lock:
            connected = bool(message.data)
            self.factr_connected = connected
            if not connected:
                if self.reset_continuation is not None:
                    self.reset_deadline = None
                self._handle_factr_fault("FACTR serial connection lost")
                return
            if self.reset_continuation is not None and self.reset_deadline is None:
                self.reset_deadline = time.monotonic() + float(
                    self.system.reset.get("timeout_sec", 15.0)
                ) + 2.0
            if (
                self.factr_recovery_pending
                and self.reset_continuation is None
                and self.state == "FACTR_RECONNECTING"
            ):
                self._start_factr_recovery_reset()

    def leader_stale_callback(self, message: Bool) -> None:
        """Treat an xArm leader-command timeout as the same FACTR fault."""

        if self.vr_mode or self.evaluation_mode or not bool(message.data):
            return
        with self.lock:
            self.factr_connected = False
            self._handle_factr_fault("FACTR command watchdog timed out")

    def _handle_factr_fault(self, reason: str) -> None:
        """Discard active capture and request one coordinated recovery reset.

        Args:
            reason: Human-readable serial disconnect or command-timeout cause.

        Returns:
            ``None``. Active data is discarded; an episode already being saved
            continues its writer/reset transaction and only blocks the next one.
        """

        if self.factr_recovery_pending or self.state in {"FINALIZED", "FINALIZING"}:
            return
        previous_state = self.state
        self.factr_recovery_pending = True
        self.post_reset_state = "IDLE"
        self.last_error = str(reason)
        if previous_state in {"RECORDING", "STAGING"}:
            was_recording = previous_state == "RECORDING"
            episode_number = self.writer.episode_count + 1
            self.capture_generation += 1
            self._discard_queued_frames()
            if was_recording:
                with self.writer_lock:
                    self.writer.discard_episode()
            self.captured_frames_in_episode = 0
            self.synchronizer.clear()
            self._set_teleop_active(False)
            if was_recording:
                self._announce(
                    "episode_discarded",
                    self.system.speech.phrase(
                        "episode_discarded", episode=episode_number
                    ),
                )
            self.get_logger().error(
                f"{reason}; "
                f"{'episode discarded' if was_recording else 'staging aborted'}"
            )
            self._set_status("FACTR_RECONNECTING")
            self._start_factr_recovery_reset()
            return
        if previous_state in {"IDLE", "FAILED"}:
            self._set_teleop_active(False)
            self._set_status("FACTR_RECONNECTING")
            self._start_factr_recovery_reset()
            return
        self.get_logger().error(f"{reason}; current {previous_state} operation will finish safely")

    def _start_factr_recovery_reset(self) -> None:
        """Start one reset that may wait indefinitely for FACTR reconnection."""

        if self.reset_continuation is not None:
            return
        if not self._start_reset("FACTR_RECOVERY", publish_state=False):
            self._set_status("FACTR_RECONNECTING")
            return
        if not self.factr_connected:
            self.reset_deadline = None
        self._set_status("FACTR_RESETTING")

    def reset_status_callback(self, message: String) -> None:
        """Unlock recording after automatic xArm error recovery finishes.

        Args:
            message: Scalar reset phase published on ``/teleop/reset/status``.

        Returns:
            Nothing. Only an error-triggered recovery owned by this recorder is
            observed; ordinary Action resets keep their existing callbacks.
        """

        with self.lock:
            if not self.error_recovery_pending:
                return
            status = str(message.data)
            if status in {"CLEARING_ERROR", "MOVING_XARM", "ALIGNING_FACTR"}:
                if not self.error_reset_started:
                    self._announce(
                        "reset_started",
                        self.system.speech.phrase("reset_started"),
                    )
                self.error_reset_started = True
                return
            recovery_complete = status == "COMPLETE" or (
                status == "IDLE" and self.error_reset_started
            )
            if recovery_complete:
                self.error_recovery_pending = False
                self.error_reset_started = False
                if self.factr_recovery_pending and self.factr_connected:
                    self.factr_recovery_pending = False
                    self.last_error = ""
                self._announce(
                    "reset_finished",
                    self.system.speech.phrase("reset_finished"),
                )
                if self.episode_completion_pending:
                    self.episode_reset_done = True
                    self._complete_episode_if_ready()
                else:
                    self._set_status("IDLE")
            elif status == "FAILED":
                self.error_recovery_pending = False
                self.error_reset_started = False
                self.last_error = f"{self.last_error}; automatic reset failed"
                if self.episode_completion_pending:
                    self.episode_reset_done = True
                    self.episode_reset_error = "automatic reset failed"
                    self._complete_episode_if_ready()
                else:
                    self._set_status("FAILED")

    def session_ready_callback(self, message: Bool) -> None:
        """Apply the warm-up gate to recorder state and keyboard/Web controls."""

        with self.lock:
            ready = bool(message.data)
            self.session_ready = ready
            if ready and self.state == "WARMUP":
                self._set_status("IDLE")
            elif not ready and self.state in {"IDLE", "FAILED"}:
                self._set_status("WARMUP")

    def process_pending(self) -> None:
        """Check evaluation and reset deadlines without sensor or disk I/O."""

        with self.lock:
            now = time.monotonic()
            if (
                self.evaluation_mode
                and self.state == "RECORDING"
                and self.capture_started_at is not None
                and now - self.capture_started_at >= self.max_episode_sec
            ):
                self._finish_evaluation_episode(
                    outcome="failure",
                    reason="time_limit",
                    terminated=False,
                    truncated=True,
                    automatic_error_reset=False,
                )
            self._check_reset_timeout(now)

    def _request_writer_prepare(self) -> bool:
        """Queue schema preparation after all configured inputs have samples.

        Returns:
            ``True`` when the writer is prepared or a preparation job is
            queued; ``False`` while required sensor samples are unavailable.
        """

        if self.writer_prepared or self.writer_prepare_requested:
            return True
        if self.vr_mode and self.robot_geometry is None:
            return False
        try:
            example_frame = self._schema_frame()
        except SynchronizationError:
            return False
        self.writer_prepare_requested = True
        self.episode_writer_jobs.put_nowait(("prepare", example_frame))
        return True

    def _prepare_writer(self, example_frame: dict) -> None:
        """Open every active LeRobot schema on the process main thread."""

        error = ""
        try:
            with self.writer_lock:
                writers = (
                    [self.writer]
                    if self.evaluation_mode
                    else list(self.category_writers.values())
                )
                for writer in writers:
                    if self.vr_mode:
                        writer.control_contract = make_contract(
                            self.config.action_format, self.robot_geometry, asdict(self.system))
                for writer in writers:
                    if self.vr_mode and writer.write_mode == "resume":
                        validate_contract(writer.root, writer.control_contract)
                for writer in writers:
                    writer.prepare(example_frame)
        except Exception as exc:
            error = f"dataset writer preparation failed: {exc}"
        with self.lock:
            self.writer_prepare_requested = False
            self.writer_prepared = not error
            if error:
                self.session_configured = False
                self.last_error = error
                self.get_logger().error(error)
                self._set_status("IDLE" if self.session_ready else "WARMUP")
            elif self._episode_limit_reached():
                self._set_status("COMPLETE")
            else:
                self.get_logger().info("LeRobot 数据集已完成初始化")
                if self.state == "PREPARING":
                    self._set_status("IDLE")

    def _enqueue_frame(self, frame: dict, generation: int) -> None:
        """Append one synchronized row without blocking a sensor callback.

        Args:
            frame: Complete LeRobot row containing current images, state and
                causal action arrays.
            generation: Integer episode generation used to reject stale rows.

        Returns:
            ``None``. A full queue drops its oldest unwritten row so the newest
            physical observation remains available.
        """

        item = (int(generation), frame)
        while True:
            try:
                self.writer_queue.put_nowait(item)
                self.captured_frames_in_episode += 1
                return
            except Full:
                try:
                    self.writer_queue.get_nowait()
                except Empty:
                    continue
                self.writer_queue.task_done()
                self.writer_queue_dropped_frames += 1
                self.captured_frames_in_episode = max(
                    0, self.captured_frames_in_episode - 1
                )

    def _frame_writer_loop(self) -> None:
        """Consume synchronized rows on one dedicated LeRobot writer thread."""

        while not self.writer_stop.is_set():
            try:
                generation, frame = self.writer_queue.get(timeout=0.1)
            except Empty:
                continue
            error = ""
            try:
                with self.writer_lock:
                    # Read after taking the writer lock: an error/discard that
                    # invalidated this generation either runs before us (skip)
                    # or waits and clears the row we are about to append.
                    if generation == self.capture_generation:
                        self.writer.add_frame(frame)
            except Exception as exc:
                error = f"dataset writer failed: {exc}"
            finally:
                self.writer_queue.task_done()
            if not error:
                self.writer_prepared = True
                continue
            with self.lock:
                if generation != self.capture_generation:
                    continue
                self.frame_writer_error = error
                if self.state == "RECORDING":
                    self._fail_episode(error)

    def _discard_queued_frames(self) -> None:
        """Remove every unwritten synchronized row from the bounded queue."""

        while True:
            try:
                self.writer_queue.get_nowait()
            except Empty:
                return
            self.writer_queue.task_done()

    def queue_keyboard_command(self, command: str) -> None:
        """Queue one scalar keyboard command for execution on a ROS thread."""

        with self.lock:
            self.keyboard_commands.append(str(command))

    def process_keyboard(self) -> None:
        """Execute queued terminal commands without calling ROS from their reader thread."""

        with self.lock:
            commands = list(self.keyboard_commands)
            self.keyboard_commands.clear()
        for command in commands:
            if self.evaluation_mode and command == "success":
                success, message = self._finish_evaluation_episode(
                    outcome="success",
                    reason="operator_success",
                    terminated=True,
                    truncated=False,
                    automatic_error_reset=False,
                )
            elif self.evaluation_mode and command == "failure":
                success, message = self._finish_evaluation_episode(
                    outcome="failure",
                    reason="operator_failure",
                    terminated=True,
                    truncated=False,
                    automatic_error_reset=False,
                )
            elif command == "staging":
                success, message = self._start_staging()
            elif command == "start":
                success, message = self._start_episode()
            elif command == "save":
                success, message = self._finish_episode(
                    save=True,
                    finish_session=False,
                    category="success",
                )
            elif command == "optimal":
                success, message = self._finish_episode(
                    save=True,
                    finish_session=False,
                    category="optimal",
                )
            elif command == "failure":
                success, message = self._finish_episode(
                    save=True,
                    finish_session=False,
                    category="failure",
                )
            elif command == "discard":
                success, message = self._discard_or_reset()
            else:
                success, message = self._finish_recording()
            log = self.get_logger().info if success else self.get_logger().warn
            log(message)

    def auto_start_first_episode(self) -> None:
        """Start the first configured episode once the reset Action server is ready."""

        if not self.auto_start_pending:
            self.auto_start_timer.cancel()
            return
        if not self.session_ready:
            return
        success, message = self._start_episode()
        if success:
            self.auto_start_pending = False
            self.auto_start_timer.cancel()
            self.get_logger().info(message)
        elif "unavailable" not in message:
            self.auto_start_pending = False
            self.auto_start_timer.cancel()
            self.get_logger().error(f"自动开始第一条失败: {message}")

    def start_episode(self, _request, response):
        """Start capture from either the reset pose or a staged pose."""

        success, message = self._start_episode()
        return _trigger(response, success, message)

    def start_staging(self, _request, response):
        """Enable unrecorded FACTR control after reset through ``Trigger``."""

        success, message = self._start_staging()
        return _trigger(response, success, message)

    def stop_episode(self, _request, response):
        """Stop capture and save one teleoperation episode under ``success``."""

        if self.evaluation_mode:
            return _trigger(
                response,
                False,
                "evaluation episodes must be labeled success or failure before saving",
            )
        success, message = self._finish_episode(
            save=True,
            finish_session=False,
            category="success",
        )
        return _trigger(response, success, message)

    def save_optimal_episode(self, _request, response):
        """Stop capture and save one teleoperation episode under ``optimal``."""

        if self.evaluation_mode:
            return _trigger(response, False, "optimal is only available in teleoperation")
        success, message = self._finish_episode(
            save=True,
            finish_session=False,
            category="optimal",
        )
        return _trigger(response, success, message)

    def save_failure_episode(self, _request, response):
        """Stop capture and save one teleoperation episode under ``failure``."""

        if self.evaluation_mode:
            return _trigger(response, False, "failure is only available in teleoperation")
        success, message = self._finish_episode(
            save=True,
            finish_session=False,
            category="failure",
        )
        return _trigger(response, success, message)

    def discard_episode(self, _request, response):
        """Discard capture, or leave staging and reset without creating data."""

        success, message = self._discard_or_reset()
        return _trigger(response, success, message)

    def success_episode(self, _request, response):
        """Save the active evaluation episode with a terminal success label."""

        success, message = self._finish_evaluation_episode(
            outcome="success",
            reason="operator_success",
            terminated=True,
            truncated=False,
            automatic_error_reset=False,
        )
        return _trigger(response, success, message)

    def failure_episode(self, _request, response):
        """Save the active evaluation episode with a terminal failure label."""

        success, message = self._finish_evaluation_episode(
            outcome="failure",
            reason="operator_failure",
            terminated=True,
            truncated=False,
            automatic_error_reset=False,
        )
        return _trigger(response, success, message)

    def finalize_dataset(self, _request, response):
        """Close LeRobot writers when no episode is being captured or saved."""

        success, message = self._finish_recording()
        return _trigger(response, success, message)

    def _start_episode(self) -> tuple[bool, str]:
        """Start one episode from reset-confirmed ``IDLE`` or ``STAGING``."""

        with self.lock:
            if self.vr_mode and not self._vr_input_ready():
                return False, self.vr_reason
            if (
                not self.evaluation_mode
                and (not self.factr_connected or self.factr_recovery_pending)
            ):
                return False, "FACTR recovery and reset are not complete"
            if not self.session_ready:
                return False, "warm-up is not complete"
            if (self.evaluation_mode or self.vr_mode) and self.xarm_control_ok is not True:
                return False, "xArm control is not healthy"
            if self.state not in {"IDLE", "STAGING"}:
                return False, f"recorder is {self.state}"
            if not self.session_configured:
                return False, "confirm dataset settings before starting an episode"
            if not self.writer_prepared:
                return False, "dataset is still preparing"
            if self._episode_limit_reached():
                return False, "configured episode count is complete"
            with self.writer_lock:
                self.writer.discard_episode()
            self.synchronizer.clear()
            self._discard_queued_frames()
            self.captured_frames_in_episode = 0
            self.last_error = ""
            self._begin_capture()
            self.auto_start_pending = False
            return True, "episode recording started"

    def _start_staging(self) -> tuple[bool, str]:
        """Enter unrecorded teleoperation from reset-confirmed ``IDLE``.

        Returns:
            ``(success, message)``. A successful transition enables FACTR
            follower commands while sensor callbacks remain outside the
            ``RECORDING`` state and therefore write no dataset rows.
        """

        with self.lock:
            if self.vr_mode and not self._vr_input_ready():
                return False, self.vr_reason
            if self.vr_mode and self.xarm_control_ok is not True:
                return False, "waiting for healthy xArm controller"
            if self.evaluation_mode:
                return False, "staging is unavailable during policy evaluation"
            if not self.factr_connected or self.factr_recovery_pending:
                return False, "FACTR recovery and reset are not complete"
            if not self.session_ready:
                return False, "warm-up is not complete"
            if self.state != "IDLE":
                return False, f"recorder is {self.state}"
            if not self.session_configured:
                return False, "confirm dataset settings before staging"
            if not self.writer_prepared:
                return False, "dataset is still preparing"
            if self._episode_limit_reached():
                return False, "configured episode count is complete"
            with self.writer_lock:
                self.writer.discard_episode()
            self.synchronizer.clear()
            self._discard_queued_frames()
            self.captured_frames_in_episode = 0
            self.last_error = ""
            self.auto_start_pending = False
            self._set_teleop_active(True)
            self._set_status("STAGING")
            return True, "staging started; teleoperation is active and not recording"

    def _discard_or_reset(self) -> tuple[bool, str]:
        """Discard an episode or abandon staging and return through reset.

        Returns:
            ``(success, message)`` describing the accepted state transition.
            ``STAGING`` contains no dataset rows, so that branch only disables
            teleoperation and requests the normal coordinated reset.
        """

        with self.lock:
            if self.state != "STAGING":
                if self.evaluation_mode and self.state == "RECORDING":
                    self.discarded_attempts += 1
                    self.pending_outcome = None
                return self._finish_episode(save=False, finish_session=False)
            self.post_reset_state = "IDLE"
            self._set_teleop_active(False)
            if not self._start_reset("STAGING_ABORT"):
                self._set_teleop_active(True)
                return False, "post-staging reset action server is unavailable"
            return True, "staging abandoned and robot reset started"

    def _finish_evaluation_episode(
        self,
        *,
        outcome: str,
        reason: str,
        terminated: bool,
        truncated: bool,
        automatic_error_reset: bool,
    ) -> tuple[bool, str]:
        """Label, stop and save one policy-evaluation rollout.

        Args:
            outcome: ``success`` or ``failure``.
            reason: Stable machine-readable terminal reason.
            terminated: Whether the task reached an MDP terminal condition.
            truncated: Whether a time limit ended the rollout.
            automatic_error_reset: Whether xArm's error recovery already owns
                reset, so this recorder must observe it rather than send a goal.

        Returns:
            ``(accepted, message)``. Accepted outcomes are written together
            with the episode while reset proceeds concurrently.
        """

        with self.lock:
            if not self.evaluation_mode:
                return False, "evaluation outcome services require evaluation mode"
            if outcome not in {"success", "failure"}:
                return False, f"unsupported evaluation outcome: {outcome}"
            if self.state != "RECORDING":
                return False, f"recorder is {self.state}"
            elapsed = (
                0.0
                if self.capture_started_at is None
                else time.monotonic() - self.capture_started_at
            )
            reward_key = "success_reward" if outcome == "success" else "failure_reward"
            default_reward = 1.0 if outcome == "success" else 0.0
            self.pending_outcome = {
                "outcome": outcome,
                "reason": str(reason),
                "reward": float(
                    self.evaluation_config.get(reward_key, default_reward)
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "duration_sec": float(elapsed),
                "timestamp_unix": float(time.time()),
                "instruction": self.instruction,
            }
            self.last_outcome = outcome
            self.last_outcome_reason = str(reason)
            save_episode = self.captured_frames_in_episode > 0
            if not save_episode:
                # A policy can fail before the first fully synchronized camera
                # row. There is no valid LeRobot episode to commit, but reset is
                # still mandatory before another physical rollout may start.
                self.discarded_attempts += 1
            return self._finish_episode(
                save=save_episode,
                finish_session=False,
                automatic_error_reset=automatic_error_reset,
            )

    def _finish_episode(
        self,
        *,
        save: bool,
        finish_session: bool,
        automatic_error_reset: bool = False,
        category: str = "success",
    ) -> tuple[bool, str]:
        """Write or discard one episode while running the mode-specific reset.

        Args:
            save: Whether buffered rows and videos become a committed episode.
            finish_session: Whether the dataset is finalized after post-episode reset.
            automatic_error_reset: Whether xArm error recovery owns the reset.
            category: Teleoperation destination folder. Evaluation ignores the
                value and retains its original single-dataset behavior.

        Returns:
            ``(success, message)`` describing the requested state transition.
        """

        with self.lock:
            if not self.evaluation_mode and category not in _TELEOP_EPISODE_CATEGORIES:
                return False, f"unsupported episode category: {category}"
            episode_number = self.writer.episode_count + 1
            allowed_states = {"RECORDING"} if save else {"RECORDING", "FAILED"}
            if self.state not in allowed_states:
                return False, f"recorder is {self.state}"
            if save and self.captured_frames_in_episode == 0:
                return False, "cannot save an episode before its first synchronized frame"
            if self.evaluation_mode and save and self.pending_outcome is None:
                return False, "evaluation save requires a success or failure label"
            self.synchronizer.clear()
            self.episode_completion_pending = True
            self.episode_write_done = False
            self.episode_reset_done = False
            self.episode_write_error = ""
            self.episode_reset_error = ""
            self.episode_finish_session = bool(finish_session)
            self.episode_saved = bool(save)
            self.episode_category = str(category)
            self.episode_number = episode_number
            self._set_teleop_active(False)
            self._set_status("SAVING" if save else "DISCARDING")
            if automatic_error_reset:
                self.error_recovery_pending = True
                self.error_reset_started = False
                self._set_status("ERROR_RESETTING")
            elif not self._start_reset("EPISODE_COMPLETION", publish_state=False):
                self.episode_reset_done = True
                self.episode_reset_error = "post-episode reset action server is unavailable"
            self.episode_writer_jobs.put_nowait(("episode", bool(save)))
            operation = f"save as {category}" if save else "discard"
            return True, f"episode {operation} and robot reset started"

    def _finish_recording(self) -> tuple[bool, str]:
        """Handle Escape by completing active work and finalizing the dataset."""

        with self.lock:
            state = self.state
            if state == "FINALIZED":
                return False, "dataset is already finalized"
            if self.episode_completion_pending:
                self.episode_finish_session = True
                return True, "recording will finish after episode write and reset"
            if state == "RESETTING":
                self.reset_continuation = "FINALIZE"
                return True, "recording will finish after the active reset"
            if state in {"PREPARING", "SAVING", "DISCARDING", "FINALIZING"}:
                return False, f"cannot finish while {state}"
        if state == "RECORDING":
            if self.evaluation_mode:
                self.discarded_attempts += 1
                self.pending_outcome = None
                return self._finish_episode(save=False, finish_session=True)
            return self._finish_episode(
                save=self.captured_frames_in_episode > 0,
                finish_session=True,
            )
        with self.lock:
            if state == "FAILED":
                with self.writer_lock:
                    self.writer.discard_episode()
            return self._request_finalize()

    def process_writer_job(self, timeout_sec: float = 0.1) -> bool:
        """Execute one Arrow writer job on the calling process main thread.

        Args:
            timeout_sec: Maximum scalar wait for a queued job, in seconds.

        Returns:
            ``True`` when one prepare, episode, or finalize job was processed;
            otherwise ``False``. The recorder timer calls this method on the
            process main thread because Arrow Parquet commits are not stable on
            ROS executor worker threads.
        """

        try:
            operation, payload = self.episode_writer_jobs.get(timeout=timeout_sec)
        except Empty:
            return False
        try:
            if operation == "prepare":
                self._prepare_writer(payload)
            elif operation == "episode":
                self._write_episode(bool(payload))
            else:
                self._finalize_writer()
        finally:
            self.episode_writer_jobs.task_done()
        return True

    def _write_episode(self, save: bool) -> None:
        """Commit or discard one episode on the recorder's main ROS thread."""

        error = ""
        committed_outcome = None
        try:
            # Capture is already stopped, so FIFO completion means every row
            # accepted before the button press is now owned by LeRobot.
            self.writer_queue.join()
            if save and self.frame_writer_error:
                raise RuntimeError(self.frame_writer_error)
            with self.writer_lock:
                if save:
                    episode_index = int(self.writer.episode_count)
                    if self.evaluation_mode or self.episode_category == "success":
                        self.writer.save_episode()
                    else:
                        destination = self.category_writers[self.episode_category]
                        self.writer.save_episode_to(destination)
                    if self.evaluation_mode:
                        if self.pending_outcome is None:
                            raise RuntimeError("saved evaluation episode has no outcome")
                        committed_outcome = {
                            **self.pending_outcome,
                            "episode_index": episode_index,
                            "frames": int(self.captured_frames_in_episode),
                            "fps": int(self.config.fps),
                        }
                        self.outcome_writer.append(committed_outcome)
                else:
                    self.writer.discard_episode()
        except Exception as exc:
            operation = "save" if save else "discard"
            error = f"episode {operation} failed: {exc}"
        with self.lock:
            if not self.episode_completion_pending:
                return
            self.captured_frames_in_episode = 0
            self.episode_write_done = True
            self.episode_write_error = error
            if not error and committed_outcome is not None:
                if committed_outcome["outcome"] == "success":
                    self.successful_episodes += 1
                else:
                    self.failed_episodes += 1
            self._complete_episode_if_ready()

    def _complete_episode_if_ready(self) -> None:
        """Unlock the next episode only after both write and reset branches end."""

        if not (
            self.episode_completion_pending
            and self.episode_write_done
            and self.episode_reset_done
        ):
            return
        self.episode_completion_pending = False
        errors = [
            message
            for message in (self.episode_write_error, self.episode_reset_error)
            if message
        ]
        if errors:
            self.last_error = "; ".join(errors)
            self.get_logger().error(self.last_error)
            self._set_status("FAILED")
            return
        event = "episode_finished" if self.episode_saved else "episode_discarded"
        self._announce(
            event,
            self.system.speech.phrase(event, episode=self.episode_number),
        )
        completed = self._episode_limit_reached()
        self.post_reset_state = "COMPLETE" if completed else "IDLE"
        self.pending_outcome = None
        if self.episode_finish_session:
            self._request_finalize()
        else:
            self._set_status(self.post_reset_state)

    def _start_reset(self, continuation: str, *, publish_state: bool = True) -> bool:
        """Send the configured seven-axis reset goal and remember the next phase."""

        if not self.reset_client.wait_for_server(timeout_sec=0.0):
            return False
        self.reset_request_id += 1
        request_id = self.reset_request_id
        self.reset_continuation = str(continuation)
        self.reset_deadline = time.monotonic() + float(
            self.system.reset.get("timeout_sec", 15.0)
        ) + 2.0
        self.reset_goal_handle = None
        if publish_state:
            self._set_status("RESETTING")
        self._announce("reset_started", self.system.speech.phrase("reset_started"))
        future = self.reset_client.send_goal_async(self._reset_goal())
        future.add_done_callback(
            partial(self._reset_goal_response, request_id=request_id)
        )
        return True

    def _target_episode_count(self) -> int:
        """Return the configured number of episodes for the active mode.

        Returns:
            Positive integer session limit. Evaluation uses
            ``evaluation.max_episodes``; teleoperation uses
            ``recording.num_episodes``.
        """

        return int(getattr(self, "target_episodes", self.config.num_episodes))

    def _episode_limit_reached(self) -> bool:
        """Return whether this mode's configured episode quota is complete."""

        return self.writer.episode_count >= self._target_episode_count()

    def _check_reset_timeout(self, now: float) -> None:
        """Fail one outstanding reset request after its bounded Action deadline."""

        if self.factr_recovery_pending and not self.factr_connected:
            return
        if self.reset_deadline is None or now < self.reset_deadline:
            return
        if self.reset_goal_handle is not None:
            self.reset_goal_handle.cancel_goal_async()
        self._handle_reset_failure("reset action timed out")

    def _handle_reset_failure(self, message: str) -> None:
        """Route a reset failure to episode completion or normal recorder failure."""

        continuation = self.reset_continuation
        self.reset_continuation = None
        self.reset_deadline = None
        self.reset_goal_handle = None
        if continuation == "EPISODE_COMPLETION" and self.episode_completion_pending:
            self.episode_reset_done = True
            self.episode_reset_error = str(message)
            self._complete_episode_if_ready()
            return
        if continuation == "FACTR_RECOVERY":
            self.last_error = str(message)
            self.get_logger().error(self.last_error)
            self._set_status("FAILED")
            return
        self._fail_episode(message)

    def _finalize_writer(self) -> tuple[bool, str]:
        """Finalize all MP4 and Parquet writers from the process main thread."""

        self._set_teleop_active(False)
        try:
            self.writer_queue.join()
            with self.writer_lock:
                writers = (
                    [self.writer]
                    if self.evaluation_mode
                    else list(self.category_writers.values())
                )
                for writer in writers:
                    writer.finalize()
        except Exception as exc:
            self.finalize_requested = False
            self._fail_episode(f"dataset finalize failed: {exc}")
            return False, self.last_error
        self.finalize_requested = False
        self._set_status("FINALIZED")
        if self.keyboard is not None:
            self.keyboard.close()
        return True, "dataset finalized; recording has ended"

    def _request_finalize(self) -> tuple[bool, str]:
        """Queue dataset finalization for the process main thread.

        Returns:
            ``(success, message)``. Success means the asynchronous finalization
            request was accepted; completion is reported through recorder
            diagnostics as ``FINALIZED`` or ``FAILED``.
        """

        if self.finalize_requested:
            return True, "dataset finalization is already in progress"
        if self.state == "FINALIZED":
            return False, "dataset is already finalized"
        self.finalize_requested = True
        self._set_teleop_active(False)
        self._set_status("FINALIZING")
        self.episode_writer_jobs.put_nowait(("finalize", None))
        return True, "dataset finalization started"

    def _reset_goal(self):
        """Build one position-only seven-axis post-episode reset goal in radians."""

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f"joint{index + 1}" for index in range(7)]
        point = JointTrajectoryPoint()
        point.positions = list(self.config.reset_joint_positions)
        goal.trajectory.points = [point]
        return goal

    def _reset_goal_response(self, future, *, request_id: int) -> None:
        """Wait asynchronously for an accepted reset goal result."""

        try:
            goal_handle = future.result()
        except Exception as exc:
            with self.lock:
                if (
                    request_id == self.reset_request_id
                    and self.reset_continuation is not None
                ):
                    self._handle_reset_failure(f"reset action request failed: {exc}")
            return
        with self.lock:
            if request_id != self.reset_request_id or self.reset_continuation is None:
                if goal_handle.accepted:
                    goal_handle.cancel_goal_async()
                return
        if not goal_handle.accepted:
            with self.lock:
                self._handle_reset_failure("reset action goal was rejected")
            return
        with self.lock:
            self.reset_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._reset_result, request_id=request_id)
        )

    def _reset_result(self, future, *, request_id: int) -> None:
        """Continue capture, idle return or finalization after a successful reset."""

        try:
            wrapped = future.result()
        except Exception as exc:
            with self.lock:
                if (
                    request_id == self.reset_request_id
                    and self.reset_continuation is not None
                ):
                    self._handle_reset_failure(f"reset action result failed: {exc}")
            return
        result = wrapped.result
        with self.lock:
            if request_id != self.reset_request_id or self.reset_continuation is None:
                return
            successful = (
                wrapped.status == GoalStatus.STATUS_SUCCEEDED
                and result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
            )
            if successful:
                self._announce("reset_finished", self.system.speech.phrase("reset_finished"))
                continuation = self.reset_continuation
                self.reset_continuation = None
                self.reset_deadline = None
                self.reset_goal_handle = None
                if self.factr_recovery_pending:
                    self.factr_recovery_pending = False
                    self.last_error = ""
                if continuation == "FINALIZE":
                    self._request_finalize()
                elif continuation == "EPISODE_COMPLETION":
                    self.episode_reset_done = True
                    self._complete_episode_if_ready()
                elif continuation == "FACTR_RECOVERY":
                    self._set_status("IDLE")
                else:
                    self._set_status(self.post_reset_state)
            else:
                self._handle_reset_failure(
                    f"reset failed: code={result.error_code}, {result.error_string}"
                )

    def _begin_capture(self) -> None:
        """Clear reset-era samples and enter anchor-camera-driven recording."""

        self.capture_generation += 1
        self.synchronizer.clear()
        self._discard_queued_frames()
        self.capture_started_at = time.monotonic()
        self.last_anchor_timestamp = None
        self.captured_frames_in_episode = 0
        self.writer_queue_dropped_frames = 0
        self.frame_writer_error = ""
        self.sync_dropped_frames = 0
        self.last_sync_warning = ""
        self.last_sync_warning_at = 0.0
        episode_number = self.writer.episode_count + 1
        self._announce(
            "episode_started",
            self.system.speech.phrase("episode_started", episode=episode_number),
        )
        if self.evaluation_mode or self.rl_mode:
            # Arm capture before policy motion so the initial observation and
            # first interpolated command cannot precede the recording state.
            self._set_status("RECORDING")
            self._set_teleop_active(True)
        else:
            self._set_teleop_active(True)
            self._set_status("RECORDING")

    def _record_sync_drop(self, message: str, now: float) -> None:
        """Count one unsynchronized anchor without stopping robot control.

        Args:
            message: Human-readable missing or stale stream description.
            now: Monotonic event time in seconds.

        Returns:
            ``None``. The current episode remains ``RECORDING``; only this
            candidate row is omitted from the LeRobot dataset.
        """

        self.sync_dropped_frames += 1
        self.last_sync_warning = str(message)
        if now - self.last_sync_warning_at < 1.0:
            return
        self.last_sync_warning_at = now
        self.get_logger().warn(
            f"跳过未同步录制帧（累计 {self.sync_dropped_frames}）: "
            f"{self.last_sync_warning}"
        )

    def _announce(self, event: str, phrase: str) -> None:
        """Publish one optional queued speech event without blocking recording."""

        if not self.system.speech.enabled:
            return
        message = KeyValue()
        message.key = f"recorder:{event}:{time.monotonic_ns()}"
        message.value = phrase
        self.announcement_pub.publish(message)

    def _fail_episode(self, message: str) -> None:
        """Discard incomplete data and publish one recoverable failure reason."""

        self._set_teleop_active(False)
        self.capture_generation += 1
        self._discard_queued_frames()
        with self.writer_lock:
            self.writer.discard_episode()
        self.captured_frames_in_episode = 0
        self.last_error = str(message)
        self.get_logger().error(self.last_error)
        self._set_status("FAILED")

    def _set_status(self, state: str) -> None:
        """Publish one scalar recorder lifecycle state string."""

        self.state = str(state)
        message = String()
        message.data = self.state
        self.status_pub.publish(message)
        self.publish_metrics()

    def _set_teleop_active(self, active: bool) -> None:
        """Latch whether the active teleop or policy episode may drive motion.

        Args:
            active: ``True`` while staging or actively recording.

        Returns:
            ``None``. FACTR state publication is unaffected; only follower
            command consumption is gated by the published boolean.
        """

        self.teleop_active = bool(active)
        active_message = Bool(data=self.teleop_active)
        inactive_message = Bool(data=False)
        if self.evaluation_mode:
            self.teleop_active_pub.publish(inactive_message)
            self.evaluation_active_pub.publish(active_message)
        else:
            self.teleop_active_pub.publish(active_message)
            if hasattr(self, "evaluation_active_pub"):
                self.evaluation_active_pub.publish(inactive_message)

    def publish_metrics(self) -> None:
        """Publish recorder progress and errors as standard ROS diagnostics."""

        if not hasattr(self, "metrics_pub"):
            return
        with self.lock:
            target_episodes = self._target_episode_count()
            elapsed = (
                time.monotonic() - self.capture_started_at
                if self.state == "RECORDING" and self.capture_started_at is not None
                else 0.0
            )
            factr_ready = (
                self._vr_input_ready() and self.xarm_control_ok is True
                if self.vr_mode else self.factr_connected and not self.factr_recovery_pending
            )
            evaluation_ready = (
                self.xarm_control_ok is True
                if self.evaluation_mode or self.rl_mode
                else factr_ready
            )
            start_blocker = ""
            if not self.session_ready:
                start_blocker = "系统尚未就绪"
            elif (
                (self.evaluation_mode or self.rl_mode or self.vr_mode)
                and self.xarm_control_ok is not True
            ):
                start_blocker = "等待 xArm control_ok"
            elif self.state not in {"IDLE", "STAGING"}:
                start_blocker = f"当前状态为 {self.state}"
            elif not self.session_configured:
                start_blocker = "请先应用数据集设置"
            elif not self.writer_prepared:
                start_blocker = ("等待机器人运动学/TCP 配置" if self.vr_mode
                                 and self.robot_geometry is None else "等待数据集初始化")
            elif self.vr_mode and not factr_ready:
                start_blocker = self.vr_reason
            elif self.rl_mode and not self.human_input_ready:
                start_blocker = "等待人工控制设备持续在线、输入就绪"
            elif self.rl_mode and not self.inference_ready:
                start_blocker = "等待策略服务就绪"
            elif self._episode_limit_reached():
                start_blocker = "已达到 episode 数量上限"
            operation_phase, operation_detail = _operation_status(
                state=self.state,
                completion_pending=self.episode_completion_pending,
                write_done=self.episode_write_done,
                reset_done=self.episode_reset_done,
                last_error=self.last_error,
                start_blocker=start_blocker,
            )
            remaining_sec = (
                max(0.0, self.max_episode_sec - elapsed)
                if self.evaluation_mode and self.state == "RECORDING"
                else self.max_episode_sec
            )
            values = {
                "state": self.state,
                "current_episode": self.writer.episode_count,
                "completed_episodes": self.writer.episode_count,
                "success_episodes": self.writer.episode_count,
                "optimal_episodes": self._category_episode_count("optimal"),
                "failure_episodes": self._category_episode_count("failure"),
                "crash_episodes": self._category_episode_count("crash"),
                "target_episodes": target_episodes,
                "frame_count": self.captured_frames_in_episode,
                "writer_queue_depth": self.writer_queue.qsize(),
                "writer_queue_dropped_frames": self.writer_queue_dropped_frames,
                "frame_writer_error": self.frame_writer_error,
                "writer_prepared": self.writer_prepared,
                "elapsed_sec": f"{elapsed:.3f}",
                "fps": self.config.fps,
                "dataset_root": str(self.dataset_root),
                "repo_id": self.config.repo_id,
                "instruction": self.instruction,
                "task": self.instruction,
                "write_mode": self.write_mode,
                "session_configured": self.session_configured,
                "last_error": self.last_error,
                "sync_dropped_frames": self.sync_dropped_frames,
                "last_sync_warning": self.last_sync_warning,
                "session_ready": self.session_ready,
                "factr_connected": self.factr_connected,
                "teleop_source": "vr" if self.vr_mode else "factr",
                "vr_ready": factr_ready if self.vr_mode else False,
                "vr_reason": self.vr_reason if self.vr_mode else "",
                "factr_recovery_pending": self.factr_recovery_pending,
                "xarm_control_ok": self.xarm_control_ok,
                "inference_ready": self.inference_ready,
                "human_input_ready": self.human_input_ready if self.rl_mode else False,
                "teleop_active": self.teleop_active,
                "episode_write_done": self.episode_write_done,
                "episode_reset_done": self.episode_reset_done,
                "operation_mode": self.operation_mode,
                "evaluation_enabled": self.evaluation_mode,
                "max_episode_sec": self.max_episode_sec,
                "remaining_sec": f"{remaining_sec:.3f}",
                "successful_episodes": self.successful_episodes,
                "failed_episodes": self.failed_episodes,
                "discarded_attempts": self.discarded_attempts,
                "last_outcome": self.last_outcome,
                "last_outcome_reason": self.last_outcome_reason,
                "last_episode_reward": getattr(self, "last_rl_reward", 0.0),
                "start_blocker": start_blocker,
                "operation_phase": operation_phase,
                "operation_detail": operation_detail,
                "can_start": self.session_ready
                and evaluation_ready
                and (not self.rl_mode or self.inference_ready)
                and (not self.rl_mode or self.human_input_ready)
                and self.state in {"IDLE", "STAGING"}
                and self.session_configured
                and self.writer_prepared
                and not self._episode_limit_reached(),
                "can_stage": self.session_ready
                and factr_ready
                and not self.evaluation_mode
                and not self.rl_mode
                and self.state == "IDLE"
                and self.session_configured
                and self.writer_prepared
                and not self._episode_limit_reached(),
                "can_configure": self.state in {"WARMUP", "IDLE"}
                and not self.session_configured
                and not self.writer_prepare_requested
                and not self.writer_prepared,
                # Keep save clickable while the first frame lazily creates the
                # dataset. The service gives a precise empty-episode response.
                "can_save": self.session_ready
                and self.state == "RECORDING"
                and not self.evaluation_mode
                and not self.rl_mode,
                "can_label": self.session_ready
                and self.evaluation_mode
                and self.state == "RECORDING",
                "can_discard": self.session_ready
                and self.state in {"STAGING", "RECORDING", "FAILED"},
                "can_finish": self.session_ready
                and self.state
                not in {"PREPARING", "FINALIZED", "FINALIZING", "SAVING", "DISCARDING"},
            }
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "recording"
        status.hardware_id = self.config.repo_id
        status.level = (
            DiagnosticStatus.ERROR if self.state == "FAILED" else DiagnosticStatus.OK
        )
        status.message = self.last_error if self.state == "FAILED" else self.state
        status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
        message.status = [status]
        self.metrics_pub.publish(message)

    def destroy_node(self) -> None:
        """Finalize committed episodes and discard any unfinished episode."""

        # SIGINT may invalidate the ROS context before launch reaches this
        # cleanup hook. Dataset finalization remains valid, but publishing into
        # a dead context would only mask the original shutdown reason.
        if rclpy.ok():
            self._set_teleop_active(False)
        if self.keyboard is not None:
            self.keyboard.close()
        self.writer_queue.join()
        self.writer_stop.set()
        self.frame_writer_thread.join()
        with self.writer_lock:
            writers = (
                [self.writer]
                if self.evaluation_mode
                else list(self.category_writers.values())
            )
            for writer in writers:
                writer.finalize()
        super().destroy_node()

    def _category_episode_count(self, category: str) -> int:
        """Return one teleoperation category count for diagnostics.

        Args:
            category: One of the four physical dataset folder names.

        Returns:
            Number of committed episodes in that category. Evaluation mode
            returns zero because it does not create outcome subfolders.
        """

        writer = self.category_writers.get(category)
        return 0 if writer is None else writer.episode_count


def _optional_joint_field(values, count: int) -> np.ndarray:
    """Return a finite float32 joint field or zeros when feedback omits it."""

    if len(values) < count:
        return np.zeros(count, dtype=np.float32)
    array = np.asarray(values[:count], dtype=np.float32)
    return array if np.all(np.isfinite(array)) else np.zeros(count, dtype=np.float32)


def _operation_status(
    *,
    state: str,
    completion_pending: bool,
    write_done: bool,
    reset_done: bool,
    last_error: str,
    start_blocker: str,
) -> tuple[str, str]:
    """Describe the active recording transaction for the operator dashboard.

    Args:
        state: Recorder lifecycle token such as ``RECORDING`` or ``FINALIZING``.
        completion_pending: Whether episode writing and reset are still joined.
        write_done: Whether the current episode writer branch has completed.
        reset_done: Whether the current robot reset branch has completed.
        last_error: Latest recorder error text, or an empty string.
        start_blocker: Reason the next episode cannot start, or an empty string.

    Returns:
        ``(phase, detail)`` containing a stable phase token and one Chinese
        operator-facing explanation.
    """

    if completion_pending:
        if not write_done and not reset_done:
            return "SAVING_AND_RESETTING", "正在并行保存本条数据并复位机械臂"
        if not write_done:
            return "SAVING", "机械臂复位完成，正在保存本条数据"
        return "RESETTING", "本条数据保存完成，正在复位机械臂"
    fixed_states = {
        "PREPARING": "正在初始化数据集",
        "RECORDING": "正在录制当前 episode",
        "RESETTING": "正在复位机械臂",
        "ERROR_RESETTING": "检测到执行异常，正在保存失败结果并自动复位",
        "FINALIZING": "正在整理并关闭数据集",
        "FINALIZED": "数据集已关闭，评测或录制会话已经结束",
    }
    if state in fixed_states:
        return state, fixed_states[state]
    if state == "FAILED":
        return "FAILED", last_error or "录制流程失败"
    if start_blocker:
        return "BLOCKED", start_blocker
    if state in {"IDLE", "STAGING"}:
        return "READY", "系统已就绪，可以开始下一条"
    return state, f"当前录制状态：{state}"


def _trigger(response, success: bool, message: str):
    """Fill and return one ``std_srvs/Trigger`` response."""

    response.success = bool(success)
    response.message = str(message)
    return response


def main(args=None) -> None:
    """Run callbacks and Arrow commits sequentially on the process main thread."""

    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
