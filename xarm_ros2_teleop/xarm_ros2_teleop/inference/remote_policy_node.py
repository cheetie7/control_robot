"""ROS 2 real-robot evaluator for an OpenPI-compatible remote policy."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from xarm_ros2_teleop.inference.contracts import (
    build_observation,
    interpolate_action_chunk,
    model_to_ros_commands,
    validate_action_chunk,
    validate_policy_metadata,
)
from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.inference.tcp_contracts import (
    TCP_ACTION_FORMAT,
    interpolate_tcp_chunk,
    validate_tcp_chunk,
)
from xarm_ros2_teleop.inference.websocket_policy import PolicyClient
from xarm_ros2_teleop.recording.message_converters import center_crop_resize, rgb_image
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.vr.config import VRConfig


class RemotePolicyNode(Node):
    """Capture RGB/state observations and execute validated absolute action chunks.

    Images are RGB uint8 arrays shaped ``(H,W,3)``. Model state/actions are
    float32 arrays shaped ``(8,)``. State is always seven measured joint radians
    and gripper; action is either joint/gripper (FACTR) or xyz/xyzw/gripper (VR).
    TCP candidates require the RL arbiter and explicit metadata. Network I/O runs on one worker;
    ROS callbacks and the 100 Hz state machine remain responsive meanwhile.
    """

    tcp_mode = False

    def __init__(self) -> None:
        """Load YAML, subscribe to sensors and begin the policy connection."""

        super().__init__("remote_policy_inference")
        self.declare_parameter("config_path", "")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("direct_robot_commands", True)
        self.declare_parameter("episode_gated", False)
        self.declare_parameter("episode_active_topic", "/evaluation/episode_active")
        self.declare_parameter("unlimited_chunks", False)
        self.declare_parameter("policy_action_topic", "/control/policy_action")
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided")
        self.system = load_system_config(config_path)
        self.config = self.system.inference
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.direct_robot_commands = bool(self.get_parameter("direct_robot_commands").value)
        self.tcp_mode = self.system.teleop.get("source", "factr") == "vr"
        self.tcp_config = VRConfig.from_mapping(self.system.teleop.get("vr", {}))
        if self.tcp_mode and self.direct_robot_commands:
            raise ValueError("TCP policy requires the RL ownership arbiter")
        self.episode_gated = bool(self.get_parameter("episode_gated").value)
        self.unlimited_chunks = bool(self.get_parameter("unlimited_chunks").value)
        self.episode_active = not self.episode_gated
        self.external_episode_active = self.episode_active
        self.task_prompt = self.config.prompt
        policy_action_topic = str(self.get_parameter("policy_action_topic").value)
        self.initial_joints = np.asarray(
            self.system.xarm["initial_joint_positions"], dtype=np.float32
        )

        self.images: dict[str, tuple[float, np.ndarray]] = {}
        self.joints: tuple[float, np.ndarray] | None = None
        self.gripper: tuple[float, float] | None = None
        self.control_ok: bool | None = None
        self.reset_active = False
        self.client: PolicyClient | None = None
        self.connect_future: Future[PolicyClient] | None = None
        self.inference_future: Future[np.ndarray] | None = None
        self.inference_started_at = 0.0
        self.last_observation_wait_reason = ""
        self.last_observation_wait_log_at = 0.0
        # One model output is a low-rate trajectory [T,8]. The 100 Hz node
        # samples it continuously instead of treating waypoints as blocking
        # moves that must finish before the next target can be issued.
        self.action_chunk: np.ndarray | None = None
        self.chunk_started_at = 0.0
        # Last absolute [7 joint rad, gripper 0..1] target actually published.
        # This is intentionally distinct from measured observation state because
        # the physical robot can lag a valid commanded trajectory.
        self.last_commanded_action = np.concatenate(
            [self.initial_joints, np.asarray([0.0], dtype=np.float32)]
        )
        # Measured state [8] captured with the observation submitted to ACT.
        # Safety validation compares the returned first target with this state.
        self.inference_state = self.last_commanded_action.copy()
        self.chunk_count = 0
        self.policy_paused = False
        self.discard_inference_result = False
        self.started_at = time.monotonic()
        self.stopped = False
        # ``Node.executor`` is an rclpy-managed property. Network work uses a
        # distinctly named pool so assigning it cannot replace the ROS executor.
        self.policy_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="policy",
        )

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_pub = self.create_publisher(String, "/inference/status", latched)
        self.metadata_pub = self.create_publisher(String, "/inference/policy_metadata", latched)
        self.arm_pub = self.create_publisher(JointState, "/xarm/absolute_target", 10)
        self.gripper_pub = self.create_publisher(JointState, "/xarm/gripper_pos", 10)
        self.policy_action_pub = self.create_publisher(
            String if self.tcp_mode else JointState, policy_action_topic, 30
        )
        for image_config in self.config.images:
            self.create_subscription(
                Image,
                image_config.topic,
                partial(self._image_callback, image_config),
                qos_profile_sensor_data,
            )
        self.create_subscription(JointState, "/xarm/arm_joint_state", self._arm_callback, 30)
        self.create_subscription(JointState, "/robotiq/joint_state", self._gripper_callback, 30)
        self.create_subscription(Bool, "/xarm/control_ok", self._control_callback, 10)
        self.create_subscription(Bool, "/teleop/reset/active", self._reset_callback, latched)
        self.create_subscription(Bool, "/rl/policy_paused", self._policy_paused_callback, latched)
        self.create_subscription(String, "/rl/task", self._task_callback, latched)
        self.create_subscription(
            Bool,
            str(self.get_parameter("episode_active_topic").value),
            self._episode_active_callback,
            latched,
        )
        self.create_service(Trigger, "/inference/stop", self._stop_service)
        self.control_period_sec = float(self.system.xarm.get("control_period", 0.01))
        self.timer = self.create_timer(self.control_period_sec, self._tick)
        self.connect_future = self.policy_executor.submit(
            PolicyClient,
            self.config.host,
            self.config.port,
        )
        self._publish_status("CONNECTING")
        mode = "DRY_RUN_NO_COMMANDS" if self.dry_run else "MOTION_ARMED"
        self.get_logger().warn(f"remote inference starting in {mode} mode")

    def _image_callback(self, image_config, message: Image) -> None:
        """Decode one ROS image to contiguous RGB and store its monotonic arrival time."""

        image = rgb_image(message)
        if image_config.output_size is not None:
            image = center_crop_resize(
                image,
                source_size=image_config.source_size,
                output_size=image_config.output_size,
            )
        self.images[image_config.key] = (time.monotonic(), image)

    def _arm_callback(self, message: JointState) -> None:
        """Store the latest measured seven-axis absolute joint state in radians."""

        values = np.asarray(message.position[:7], dtype=np.float32)
        if values.shape == (7,) and np.all(np.isfinite(values)):
            self.joints = (time.monotonic(), values)

    def _gripper_callback(self, message: JointState) -> None:
        """Store the latest normalized measured Robotiq position in ``[0,1]``."""

        if message.position and np.isfinite(message.position[0]):
            self.gripper = (time.monotonic(), float(message.position[0]))

    def _control_callback(self, message: Bool) -> None:
        """Stop real motion immediately when the follower health flag becomes false."""

        was_healthy = self.control_ok is True
        self.control_ok = bool(message.data)
        if not self.dry_run and was_healthy and not self.control_ok and not self.stopped:
            if self.episode_gated:
                self._pause_episode("xArm control_ok became false", hold=False)
            else:
                self._stop("xArm control_ok became false")

    def _reset_callback(self, message: Bool) -> None:
        """Reject concurrent reset motion while a policy rollout owns command topics."""

        self.reset_active = bool(message.data)
        if not self.dry_run and self.reset_active and not self.stopped:
            if self.episode_gated:
                self._pause_episode("reset active", hold=False)
            else:
                self._stop("reset became active")

    def _episode_active_callback(self, message: Bool) -> None:
        """Start or pause one externally coordinated evaluation rollout.

        Args:
            message: Latched episode gate. ``True`` starts from a fresh
                observation; ``False`` discards the current chunk and holds.

        Returns:
            ``None``. The policy process and WebSocket remain reusable across
            evaluation episodes and post-episode resets.
        """

        if not self.episode_gated or self.stopped:
            return
        active = bool(message.data)
        self.external_episode_active = active
        if active == self.episode_active:
            return
        if not active:
            self._pause_episode("waiting for the next evaluation episode")
            return
        self.episode_active = True
        self.chunk_count = 0
        self.started_at = time.monotonic()
        if (
            not self.dry_run
            and not self.tcp_mode
            and self.control_ok is True
            and self.joints is not None
            and self.gripper is not None
        ):
            # The physical robot holds its measured pose while the first ACT
            # request runs. This is also the first causal executed-action label
            # consumed by the recorder for the new episode.
            measured_hold = np.concatenate(
                [self.joints[1], np.asarray([np.clip(self.gripper[1], 0.0, 1.0)])]
            )
            self._execute_action(measured_hold)
        self._publish_status("WAITING_FOR_OBSERVATION")

    def _tick(self) -> None:
        """Advance networking and sample the active chunk on the xArm control clock."""

        if self.stopped:
            return
        now = time.monotonic()
        if not self.episode_gated and now - self.started_at >= self.config.max_runtime_sec:
            self._stop("maximum runtime reached")
            return
        if self.client is None:
            self._finish_connection()
            return
        if self.inference_future is not None:
            if not self.inference_future.done():
                if now - self.inference_started_at <= self.config.inference_timeout_sec:
                    return
                if self.episode_gated:
                    self._fail_episode("inference timeout")
                else:
                    self._stop("inference timeout")
                return
            self._finish_inference()
            if self.stopped:
                return
            # The newly accepted chunk starts inside ``_finish_inference``;
            # refresh the monotonic sample so elapsed time cannot be negative.
            now = time.monotonic()
        if self.episode_gated and not self.episode_active:
            return
        if self.policy_paused:
            return
        if self.action_chunk is not None:
            interpolate = interpolate_tcp_chunk if self.tcp_mode else interpolate_action_chunk
            target = interpolate(
                self.action_chunk,
                now - self.chunk_started_at,
                self.config.policy_frequency_hz,
            )
            if target is not None:
                self._execute_action(target)
                return
            self.action_chunk = None
        if not self.unlimited_chunks and self.chunk_count >= self.config.max_chunks:
            if self.episode_gated:
                self._fail_episode("maximum action chunks reached")
            else:
                self._stop("maximum action chunks reached")
            return
        self._submit_inference()

    def _finish_connection(self) -> None:
        """Install the completed policy connection or stop on connection failure."""

        if self.connect_future is None or not self.connect_future.done():
            return
        try:
            self.client = self.connect_future.result()
            warnings = validate_policy_metadata(
                self.client.metadata,
                [image.key for image in self.config.images],
                self.config.policy_frequency_hz,
                **({"action_format": TCP_ACTION_FORMAT} if self.tcp_mode else {}),
            )
        except Exception as exc:
            if self.client is not None:
                self.client.close()
                self.client = None
            self._stop(f"policy connection failed: {exc}")
            return
        for warning in warnings:
            self.get_logger().warn(warning)
        self._publish_status("WAITING_FOR_OBSERVATION")
        self.metadata_pub.publish(String(data=json.dumps(self.client.metadata, default=str)))
        self.get_logger().info(f"policy metadata: {self.client.metadata}")
        if self.episode_gated and not self.episode_active:
            self._publish_status("WAITING_FOR_EPISODE")

    def _submit_inference(self) -> None:
        """Snapshot one fresh observation and submit exactly one network request."""

        try:
            observation = self._build_fresh_observation()
        except RuntimeError as exc:
            self._report_observation_wait(str(exc))
            return
        self.last_observation_wait_reason = ""
        self.inference_state = np.asarray(observation["state"], dtype=np.float32).copy()
        self.inference_started_at = time.monotonic()
        self.inference_future = self.policy_executor.submit(
            self.client.infer,
            observation,
        )
        self._publish_status("INFERENCING")

    def _report_observation_wait(self, reason: str) -> None:
        """Publish a throttled explanation for a blocked inference request.

        Args:
            reason: Missing, stale, or unsynchronized observation condition.

        Returns:
            ``None``. Status updates occur when the reason changes and warning
            logs repeat at most once per second while the condition persists.
        """

        now = time.monotonic()
        reason_changed = reason != self.last_observation_wait_reason
        if reason_changed:
            self.last_observation_wait_reason = reason
            self._publish_status(f"WAITING_FOR_OBSERVATION: {reason}")
        if reason_changed or now - self.last_observation_wait_log_at >= 1.0:
            self.last_observation_wait_log_at = now
            self.get_logger().warn(f"policy observation not ready: {reason}")

    def _finish_inference(self) -> None:
        """Validate a returned chunk against limits and its measured input state."""

        assert self.inference_future is not None
        if self.discard_inference_result:
            self.inference_future.exception()
            self.inference_future = None
            self.discard_inference_result = False
            self._publish_status("PAUSED_BY_HUMAN")
            return
        try:
            candidate = self.inference_future.result()
            actions = (
                validate_tcp_chunk(candidate, self.tcp_config)
                if self.tcp_mode
                else validate_action_chunk(
                    candidate,
                    self.inference_state,
                    np.asarray(self.config.joint_lower_rad),
                    np.asarray(self.config.joint_upper_rad),
                    self.config.max_joint_step_rad,
                )
            )
        except Exception as exc:
            self.inference_future = None
            reason = f"invalid policy response: {exc}"
            if self.episode_gated:
                self._fail_episode(reason)
            else:
                self._stop(reason)
            return
        self.inference_future = None
        self.action_chunk = actions
        self.chunk_started_at = time.monotonic()
        self.chunk_count += 1
        self._publish_status("EXECUTING_DRY_RUN" if self.dry_run else "EXECUTING")

    def _policy_paused_callback(self, message: Bool) -> None:
        """Discard queued policy actions while human takeover owns control."""

        self.policy_paused = bool(message.data)
        self.action_chunk = None
        if self.inference_future is not None:
            self.discard_inference_result = True
        if (
            not self.policy_paused
            and self.episode_gated
            and self.external_episode_active
            and not self.episode_active
        ):
            self._episode_active_callback(Bool(data=True))
        self._publish_status("PAUSED_BY_HUMAN" if self.policy_paused else "RESUMING")

    def _task_callback(self, message: String) -> None:
        """Use the operator's RL episode instruction for subsequent policy requests."""

        if not self.direct_robot_commands and message.data.strip():
            self.task_prompt = message.data.strip()

    def _pause_episode(self, reason: str, *, hold: bool = True) -> None:
        """Stop consuming the current chunk while keeping the server reusable.

        Args:
            reason: Human-readable pause cause published through inference status.
            hold: Whether to publish one measured hold before reset owns motion.

        Returns:
            ``None``. Any in-flight response is discarded when it arrives.
        """

        self.episode_active = False
        self.action_chunk = None
        if self.inference_future is not None:
            self.discard_inference_result = True
        if (
            hold
            and not self.dry_run
            and not self.tcp_mode
            and self.control_ok is True
            and self.joints is not None
            and self.gripper is not None
        ):
            measured_hold = np.concatenate(
                [self.joints[1], np.asarray([np.clip(self.gripper[1], 0.0, 1.0)])]
            )
            self._execute_action(measured_hold)
        self._publish_status(f"WAITING_FOR_EPISODE: {reason}")

    def _fail_episode(self, reason: str) -> None:
        """Fail one evaluation rollout without permanently stopping policy service.

        Args:
            reason: Human-readable policy or safety failure recorded with the
                current evaluation episode.

        Returns:
            ``None``. Current commands are replaced by a measured hold and an
            ``EPISODE_FAILED`` status asks the recorder to save and reset.
        """

        if not self.episode_active:
            return
        self.episode_active = False
        self.action_chunk = None
        if self.inference_future is not None:
            self.discard_inference_result = True
        if (
            not self.tcp_mode
            and not self.dry_run
            and self.joints is not None
            and self.gripper is not None
        ):
            measured_hold = np.concatenate(
                [self.joints[1], np.asarray([np.clip(self.gripper[1], 0.0, 1.0)])]
            )
            self._execute_action(measured_hold)
        self._publish_status(f"EPISODE_FAILED: {reason}")
        self.get_logger().error(f"evaluation episode failed: {reason}")

    def _build_fresh_observation(self) -> dict:
        """Return a fresh observation or raise ``RuntimeError`` while sensors are incomplete."""

        now = time.monotonic()
        missing = [item.key for item in self.config.images if item.key not in self.images]
        if missing:
            raise RuntimeError(f"waiting for images: {', '.join(missing)}")
        image_times = {item.key: self.images[item.key][0] for item in self.config.images}
        oldest_key = min(image_times, key=image_times.get)
        newest_key = max(image_times, key=image_times.get)
        oldest_age = now - image_times[oldest_key]
        if oldest_age > self.config.observation_timeout_sec:
            raise RuntimeError(
                f"waiting for fresh images: {oldest_key} age={oldest_age:.3f}s "
                f"> {self.config.observation_timeout_sec:.3f}s"
            )
        camera_skew = image_times[newest_key] - image_times[oldest_key]
        if camera_skew > self.config.max_camera_skew_sec:
            raise RuntimeError(
                f"waiting for synchronized images: skew={camera_skew:.3f}s "
                f"> {self.config.max_camera_skew_sec:.3f}s "
                f"({oldest_key}->{newest_key})"
            )

        if self.joints is None or self.gripper is None:
            if not self.dry_run:
                raise RuntimeError("waiting for measured robot state")
            joints = self.initial_joints
            gripper = 0.0
        else:
            state_times = (self.joints[0], self.gripper[0])
            if not self.dry_run and now - min(state_times) > self.config.observation_timeout_sec:
                raise RuntimeError("waiting for fresh robot state")
            joints = self.joints[1]
            gripper = self.gripper[1]
        images = {item.key: self.images[item.key][1].copy() for item in self.config.images}
        return build_observation(
            joints, gripper, images, getattr(self, "task_prompt", self.config.prompt)
        )

    def _execute_action(self, action: np.ndarray) -> None:
        """Log one target in dry-run or publish converted arm and gripper commands."""

        self.last_commanded_action = np.asarray(action, dtype=np.float32).copy()
        if self.tcp_mode:
            # Candidate only: the arbiter is the sole TCP/gripper publisher.
            self.policy_action_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "action_format": TCP_ACTION_FORMAT,
                            "timestamp": self.get_clock().now().nanoseconds / 1e9,
                            "action": self.last_commanded_action.tolist(),
                        }
                    )
                )
            )
            return
        policy_message = JointState()
        policy_message.header.stamp = self.get_clock().now().to_msg()
        policy_message.name = [f"joint{index + 1}" for index in range(7)] + ["robotiq_finger"]
        policy_message.position = self.last_commanded_action.astype(float).tolist()
        self.policy_action_pub.publish(policy_message)
        if self.dry_run or not self.direct_robot_commands:
            if self.dry_run:
                self.get_logger().info(
                    "dry-run action " + np.array2string(self.last_commanded_action, precision=4)
                )
            return
        if self.control_ok is not True or self.reset_active:
            self._stop("motion interlock is not healthy")
            return
        arm_target, leader_gripper = model_to_ros_commands(action)
        stamp = self.get_clock().now().to_msg()
        arm_message = JointState()
        arm_message.header.stamp = stamp
        arm_message.name = [f"joint{index + 1}" for index in range(7)]
        arm_message.position = arm_target.astype(float).tolist()
        gripper_message = JointState()
        gripper_message.header.stamp = stamp
        gripper_message.name = ["leader_gripper"]
        gripper_message.position = [leader_gripper]
        self.arm_pub.publish(arm_message)
        self.gripper_pub.publish(gripper_message)

    def _stop_service(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Stop this rollout; the response reports that no further targets will publish."""

        if self.episode_gated:
            self._pause_episode("operator stop service")
        else:
            self._stop("operator stop service")
        response.success = True
        response.message = (
            "evaluation rollout paused and current measured target held"
            if self.episode_gated
            else "remote inference stopped and current measured target held"
        )
        return response

    def _stop(self, reason: str) -> None:
        """Latch rollout termination and hold the measured pose once."""

        if self.stopped:
            return
        self.stopped = True
        self.action_chunk = None
        if (
            not self.tcp_mode
            and not self.dry_run
            and self.joints is not None
            and self.gripper is not None
        ):
            hold = np.concatenate(
                [self.joints[1], np.asarray([np.clip(self.gripper[1], 0.0, 1.0)])]
            )
            self._execute_action(hold)
        self._publish_status(f"STOPPED: {reason}")
        self.get_logger().error(f"remote inference stopped: {reason}")

    def _publish_status(self, value: str) -> None:
        """Publish a versioned phase/reason status without changing control flow."""

        message = String()
        message.data = InferenceStatus.from_wire(value).to_json()
        self.status_pub.publish(message)

    def destroy_node(self) -> None:
        """Close network resources without issuing any additional motion."""

        self.stopped = True
        if self.client is not None:
            self.client.close()
        self.policy_executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None) -> None:
    """Run the remote policy node; ``args`` is an optional ROS argument sequence."""

    rclpy.init(args=args)
    node = RemotePolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
