"""Manual real-robot RL episodes built on the shared LeRobot capture lifecycle."""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32MultiArray, String
from std_srvs.srv import Trigger

from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.recording.episode_recorder import EpisodeRecorder, _trigger
from xarm_ros2_teleop.rl.episode_services import RL_EPISODE_SERVICES
from xarm_ros2_teleop.rl.recording import (
    ACTION_SOURCES,
    RLFrameHistory,
    rl_schema_fields,
    write_episode_label,
)


class RLEpisodeRecorder(EpisodeRecorder):
    """Record images, measured state, accepted commands and manual RL labels.

    The inherited writer/reset transaction keeps the next episode locked until
    save and reset finish. Only this mode adds RL fields and outcome files;
    it has no duration limit, episode quota, replay or learner dependency.
    """

    def __init__(self) -> None:
        """Initialize shared capture plus the episode gate and RL subscriptions."""

        # The base publishes initial metrics before this subclass creates subscriptions.
        self.human_ready = False
        self.human_ready_received_at = 0.0
        super().__init__()
        if not self.rl_mode:
            raise ValueError("rl_episode_recorder requires operation_mode=rl")
        if self.config.streaming_encoding:
            raise ValueError("RL outcome folders require recording.video.streaming_encoding=false")
        self.declare_parameter("require_remote_policy", True)
        self.require_remote_policy = bool(self.get_parameter("require_remote_policy").value)
        self.rl_history = RLFrameHistory(self.config.action_history_sec, self.config.action_format)
        self.policy_metadata: dict = {}
        self.policy_connected = False
        self.last_policy_status = ""
        self.inference_ready = not self.require_remote_policy
        self.pending_rl_label: dict | None = None
        self.last_rl_reward = 0.0
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.rl_active_pub = self.create_publisher(Bool, "/rl/episode_active", latched)
        self.task_pub = self.create_publisher(String, "/rl/task", latched)
        self.rl_active_pub.publish(Bool(data=False))
        self.create_subscription(String, "/rl/command_event", self.command_event_callback, 100)
        if self.vr_mode:
            self.create_subscription(String, "/xarm/tcp_command_event", self.tcp_ack_callback, 100)
        self.create_subscription(String, "/rl/reward_event", self.reward_event_callback, 10)
        self.create_subscription(Bool, "/rl/human_ready", self.human_ready_callback, latched)
        self.create_subscription(
            String, "/inference/policy_metadata", self.policy_metadata_callback, latched
        )
        self.create_service(Trigger, RL_EPISODE_SERVICES["start"], self.start_episode)
        self.create_service(Trigger, RL_EPISODE_SERVICES["discard"], self.discard_episode)
        self.create_service(Trigger, RL_EPISODE_SERVICES["finish"], self.finalize_dataset)

    @property
    def human_input_ready(self) -> bool:
        """Require a fresh readiness heartbeat, not a stale latched True after node loss."""

        timeout = float(self.system.rl.get("human_input_timeout_sec", 0.25))
        return self.human_ready and time.monotonic() - self.human_ready_received_at <= timeout

    def human_ready_callback(self, message: Bool) -> None:
        """Track the device-independent readiness gate; never end an active episode."""

        with self.lock:
            self.human_ready = bool(message.data)
            self.human_ready_received_at = time.monotonic()

    def command_event_callback(self, message: String) -> None:
        """Buffer a ROS-clock arbiter event containing selected/policy actions [8]."""

        try:
            event = json.loads(message.data)
            with self.lock:
                self.rl_history.add(event)
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f"Invalid RL command provenance: {exc}")

    def policy_metadata_callback(self, message: String) -> None:
        """Store the policy server's metadata for the next saved episode label."""

        metadata = json.loads(message.data)
        if not isinstance(metadata, dict):
            raise ValueError("policy metadata must be a JSON object")
        self.policy_metadata = metadata
        self.policy_connected = True
        if self.require_remote_policy:
            self.inference_ready = InferenceStatus.from_wire(self.last_policy_status).reusable

    def tcp_ack_callback(self, message: String) -> None:
        """Store successful TCP SDK [7] command/selection linkage on the ROS clock."""
        try:
            with self.lock:
                self.rl_history.add_acceptance(json.loads(message.data))
        except (ValueError, KeyError, TypeError) as error:
            self.get_logger().error(f"Invalid TCP acknowledgement: {error}")

    def _configure_session_parameters(self, parameters):
        """Apply the shared dataset settings and publish the task used by policy inference."""

        result = super()._configure_session_parameters(parameters)
        if result.successful and self.session_configured:
            self.task_pub.publish(String(data=self.instruction))
        return result

    def reward_event_callback(self, message: String) -> None:
        """End one active episode on a human success or failure event."""

        try:
            event = json.loads(message.data)
            reward = float(event["reward"])
            if not np.isfinite(reward):
                raise ValueError("reward must be finite")
            outcome = {
                "task_success": "success",
                "task_failure": "failure",
            }[event["event"]]
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f"Invalid RL reward event: {exc}")
            return
        with self.lock:
            if self.state != "RECORDING":
                return
            self._finish_rl_episode(
                category=outcome,
                reward=reward,
                reason=str(event.get("reason", event["event"])),
                terminated=bool(event["terminated"]),
                truncated=bool(event["truncated"]),
                reward_source=str(event.get("source", "human")),
                reward_event=event,
            )

    def _schema_frame(self) -> dict:
        """Extend the shared non-recorded schema example with fixed RL fields."""

        return {**super()._schema_frame(), **rl_schema_fields()}

    def _assemble_frame(self, anchor) -> dict:
        """Attach matched action provenance to one synchronized camera row."""

        frame = super()._assemble_frame(anchor)
        arm = self.synchronizer.xarm_action.latest_before(anchor.timestamp)
        gripper = self.synchronizer.robotiq_action.latest_before(anchor.timestamp)
        return self.rl_history.annotate(frame, anchor.timestamp, (arm.timestamp, gripper.timestamp))

    def _episode_limit_reached(self) -> bool:
        """RL episodes are manually ended and never limited by a quota."""

        return False

    def _target_episode_count(self) -> int:
        """Return zero to denote an unlimited, manually operated RL session."""

        return 0

    def _start_episode(self) -> tuple[bool, str]:
        """Start recording before enabling policy motion from fresh observations."""

        with self.lock:
            if self.xarm_control_ok is not True:
                return False, "xArm control is not healthy"
            if self.state != "IDLE" or self.error_recovery_pending:
                return False, f"recorder is {self.state}"
            if not self.session_configured or not self.writer_prepared:
                return False, "apply dataset settings and wait for initialization"
            if not self.human_input_ready:
                return False, "wait for the human control device to run continuously and stay ready"
            if not self.inference_ready:
                return False, "policy service is not ready"
            with self.writer_lock:
                self.writer.discard_episode()
            self.rl_history.clear()
            self.pending_rl_label = None
            self.last_error = ""
            self._begin_capture()
            return True, "RL episode recording started; policy enabled"

    def _start_staging(self) -> tuple[bool, str]:
        """RL manual takeover uses its own controls instead of teleop staging."""

        return False, "use RL takeover during an active episode"

    def stop_episode(self, _request, response):
        """Require an explicit RL reward instead of an unlabeled generic save."""

        return _trigger(response, False, "use RL success or failure to label this episode")

    def _set_teleop_active(self, active: bool) -> None:
        """Publish only the RL gate; disable the teleop/evaluation motion gates."""

        self.teleop_active = bool(active)
        self.teleop_active_pub.publish(Bool(data=False))
        self.evaluation_active_pub.publish(Bool(data=False))
        if hasattr(self, "rl_active_pub"):
            self.rl_active_pub.publish(Bool(data=bool(active)))

    def _finish_episode(self, *, save: bool, **kwargs) -> tuple[bool, str]:
        """Reject unlabeled generic saves before entering the shared writer/reset path."""

        if save and self.pending_rl_label is None:
            return False, "RL save requires a success, failure or crash label"
        return super()._finish_episode(save=save, **kwargs)

    def _finish_rl_episode(
        self,
        *,
        category: str,
        reward: float,
        reason: str,
        terminated: bool,
        truncated: bool,
        reward_source: str,
        automatic_error_reset: bool = False,
        reward_event: dict | None = None,
        finish_session: bool = False,
    ) -> tuple[bool, str]:
        """Latch an outcome and request the shared save/reset transaction.

        Args:
            category: Success, failure or crash dataset destination.
            reward: Finite terminal scalar; ordinary recorded rows have zero.
            reason: Human command or concrete controller error code.
            terminated: Whether a task/hardware terminal outcome was reached.
            truncated: Whether an operator ended an unfinished attempt.
            reward_source: ``human`` or ``system`` provenance.
            automatic_error_reset: Use the coordinator's existing error recovery.
            reward_event: Original human JSON event, including its event time.
            finish_session: Close all writers after this episode's reset.

        Returns:
            ``(accepted, message)`` from the shared transaction. An empty
            attempt still resets but cannot form a LeRobot episode.
        """

        if self.state != "RECORDING":
            return False, f"recorder is {self.state}"
        self.pending_rl_label = {
            "format_version": 1,
            "action_format": self.config.action_format,
            "outcome": category,
            "reason": reason,
            "reward": float(reward),
            "reward_source": reward_source,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reward_event": reward_event,
            "duration_sec": time.monotonic() - self.capture_started_at,
            "ended_at_ros_s": self.get_clock().now().nanoseconds / 1e9,
            "timestamp_unix": time.time(),
            "instruction": self.instruction,
            "policy_metadata": dict(self.policy_metadata),
            "policy_endpoint": {
                "host": self.system.inference.host,
                "port": self.system.inference.port,
                "waypoint_frequency_hz": self.system.inference.policy_frequency_hz,
            },
            "action_source_codes": ACTION_SOURCES,
        }
        self.last_outcome = category
        self.last_outcome_reason = reason
        self.last_rl_reward = float(reward)
        save = self.captured_frames_in_episode > 0
        if not save:
            self.discarded_attempts += 1
        return self._finish_episode(
            save=save,
            finish_session=finish_session,
            category=category,
            automatic_error_reset=automatic_error_reset,
        )

    def _write_episode(self, save: bool) -> None:
        """Commit the final reward, action provenance and durable episode label."""

        error = ""
        try:
            self.writer_queue.join()
            if save and self.frame_writer_error:
                raise RuntimeError(self.frame_writer_error)
            with self.writer_lock:
                if save:
                    if self.pending_rl_label is None:
                        raise RuntimeError("RL episode cannot be saved without an outcome")
                    label = dict(self.pending_rl_label)
                    buffer = self.writer.dataset.episode_buffer
                    size = int(buffer["size"])
                    # Labels live on the final recorded row, with the exact
                    # button/error time retained separately in the JSON label.
                    for key in ("reward", "terminated", "truncated"):
                        buffer[key][-1] = np.asarray([label[key]], dtype=np.float32)
                    sources = np.asarray(buffer["action.source"]).reshape(-1)
                    label.update(
                        frames=size,
                        fps=self.config.fps,
                        terminal_frame_index=size - 1,
                        source_frame_counts={
                            name: int(np.count_nonzero(sources == code))
                            for name, code in ACTION_SOURCES.items()
                        },
                        sync_dropped_frames=self.sync_dropped_frames,
                        queue_dropped_frames=self.writer_queue_dropped_frames,
                    )
                    destination = self.category_writers[self.episode_category]
                    episode_index = destination.episode_count
                    self.writer.save_episode_to(destination)
                    destination.flush_for_readers()
                    write_episode_label(destination.root, episode_index, label)
                else:
                    self.writer.discard_episode()
        except Exception as exc:
            error = f"RL episode {'save' if save else 'discard'} failed: {exc}"
        with self.lock:
            self.captured_frames_in_episode = 0
            self.episode_write_done = True
            self.episode_write_error = error
            self._complete_episode_if_ready()

    def xarm_error_callback(self, message: Int32MultiArray) -> None:
        """Save an active episode as crash on a nonzero hardware error code."""

        if not message.data or int(message.data[0]) == 0:
            return
        with self.lock:
            if self.state == "RECORDING":
                self._finish_rl_episode(
                    category="crash",
                    reward=float(self.system.rl.get("crash_reward", 0.0)),
                    reason=f"xarm_error_{int(message.data[0])}",
                    terminated=True,
                    truncated=False,
                    reward_source="system",
                    automatic_error_reset=True,
                )

    def control_ok_callback(self, message: Bool) -> None:
        """Track driver health; concrete error codes own automatic crash labeling."""

        with self.lock:
            self.xarm_control_ok = bool(message.data)

    def inference_status_callback(self, message: String) -> None:
        """Track policy readiness; inference faults leave labeling to the operator."""

        status = InferenceStatus.from_wire(message.data)
        if not self.require_remote_policy:
            return
        with self.lock:
            self.last_policy_status = message.data
            if status.phase == "STOPPED":
                self.policy_connected = False
            self.inference_ready = self.policy_connected and status.reusable
            if self.state == "RECORDING" and status.failed:
                self.last_error = f"inference_{status.reason}"
                self.get_logger().warn(
                    f"{self.last_error}; episode remains open for manual labeling"
                )

    def factr_connected_callback(self, message: Bool) -> None:
        """Track leader availability without discarding an ongoing RL attempt."""

        self.factr_connected = bool(message.data)

    def leader_stale_callback(self, message: Bool) -> None:
        """The RL arbiter, rather than the teleop watchdog, owns human command health."""

    def session_ready_callback(self, message: Bool) -> None:
        """RL startup is manually gated and does not depend on teleop warm-up."""

        self.session_ready = True

    def _finish_recording(self) -> tuple[bool, str]:
        """Save an unfinished attempt as failure before closing the manual RL session."""

        with self.lock:
            if self.state == "RECORDING":
                return self._finish_rl_episode(
                    category="failure",
                    reward=float(self.system.rl.get("failure_reward", 0.0)),
                    reason="operator_finish_session",
                    terminated=True,
                    truncated=False,
                    reward_source="human",
                    finish_session=True,
                )
            return super()._finish_recording()


def main(args=None) -> None:
    """Run shared main-thread LeRobot commits for manual RL data collection."""

    rclpy.init(args=args)
    node = RLEpisodeRecorder()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.process_writer_job(0.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
