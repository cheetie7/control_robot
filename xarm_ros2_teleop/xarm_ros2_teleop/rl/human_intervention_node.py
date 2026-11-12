"""ROS 2 action arbiter for single-button policy/human switching and rewards."""

from __future__ import annotations

import json
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from xarm_ros2_teleop.inference.contracts import model_to_ros_commands, validate_action_chunk
from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.inference.tcp_contracts import TCP_ACTION_FORMAT, validate_tcp_chunk
from xarm_ros2_teleop.recording.keyboard_controller import KeyboardController
from xarm_ros2_teleop.rl.episode_services import RL_EPISODE_SERVICES
from xarm_ros2_teleop.rl.human_input import HumanInput
from xarm_ros2_teleop.rl.intervention import (
    RL_SERVICE_COMMANDS,
    InterventionController,
    InterventionState,
    RewardEvent,
)
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.vr.config import VRConfig
from xarm_ros2_teleop.vr.pose_math import pose_array, pose_message


def decode_rl_key(sequence: bytes) -> str | None:
    """Map terminal bytes to ownership switch, reward and episode commands."""

    return {
        b"s": "start",
        b"S": "start",
        b"q": "finish",
        b"Q": "finish",
        b"\x1b[D": "discard",
        b"\x1bOD": "discard",
        b"t": "switch",
        b"T": "switch",
        b"y": "success",
        b"Y": "success",
        b"n": "failure",
        b"N": "failure",
        b"\x1b": "stop",
    }.get(sequence)


class HumanInterventionNode(Node):
    """Arbitrate policy and relative human input before the robot driver.

    FACTR uses joint [8] actions and relative joint input. VR uses base-frame
    metre xyz + xyzw + gripper [8] actions; its existing clutch mapper owns SE(3)
    origins, not the joint displacement controller. Joint velocities still
    confirm stopping. Policy observations remain measured joint state [8].
    One arbiter owns actuator topics; returning to policy requests fresh input.
    """

    tcp_mode = False

    def __init__(self) -> None:
        """Load shared YAML, create ROS interfaces and start keyboard handling."""

        super().__init__("human_intervention")
        self.declare_parameter("config_path", "")
        self.declare_parameter("motion_enabled", False)
        config_path = str(self.get_parameter("config_path").value)
        if not config_path:
            raise ValueError("config_path must be provided")
        self.system = load_system_config(config_path)
        self.config = self.system.rl
        self.tcp_mode = self.system.teleop.get("source", "factr") == "vr"
        self.tcp_config = VRConfig.from_mapping(self.system.teleop.get("vr", {}))
        self.vr_ready = False
        self.vr_status_at = 0.0
        self.vr_action = None
        self.vr_action_at = 0.0
        self.velocity_received_at = 0.0
        self.policy_enabled_at = self.get_clock().now().nanoseconds / 1e9
        self.human_enabled_at = self.policy_enabled_at
        self.motion_enabled = bool(self.get_parameter("motion_enabled").value)
        policy_action_topic = str(self.config.get("policy_action_topic", "/control/policy_action"))
        human_arm_topic = str(self.config.get("human_arm_topic", "/control/factr_arm"))
        human_gripper_topic = str(self.config.get("human_gripper_topic", "/control/factr_gripper"))
        self.initial_joints = np.asarray(
            self.system.xarm["initial_joint_positions"], dtype=np.float32
        )
        self.controller = InterventionController(
            gripper_delta_scale=float(self.config.get("gripper_delta_scale", -1.0))
        )
        self.controller.state = InterventionState.PAUSED
        self.episode_active = False
        self.recorder_state = "IDLE"
        self.robot_state: np.ndarray | None = None
        self.robot_velocity: np.ndarray | None = None
        if not self.motion_enabled and not self.tcp_mode:
            self.robot_state = np.concatenate(
                [self.initial_joints, np.asarray([0.0], dtype=np.float32)]
            )
            self.robot_velocity = np.zeros(7, dtype=np.float32)
        self.human_input = HumanInput()
        self.human_ready = False
        self.human_ready_published_at = 0.0
        self.human_input_timeout_sec = float(self.config.get("human_input_timeout_sec", 0.25))
        self.human_ready_stable_sec = float(self.config.get("human_ready_stable_sec", 0.3))
        self.robot_state_received_at = 0.0
        self.gripper_state_received_at = 0.0
        self.policy_action: np.ndarray | None = None
        self.policy_received_at = 0.0
        self.executed_action: np.ndarray | None = None
        self.stable_since: float | None = None
        self.control_ok: bool | None = None
        self.keys: deque[str] = deque()
        self.key_lock = threading.Lock()
        self.keyboard: KeyboardController | None = None
        self.reset_in_progress = False

        self.stop_velocity_rad_s = float(self.config.get("stop_velocity_rad_s", 0.03))
        self.stop_stable_sec = float(self.config.get("stop_stable_sec", 0.3))
        self.policy_timeout_sec = float(self.config.get("policy_timeout_sec", 0.25))
        self.max_human_joint_step_rad = float(self.config.get("max_human_joint_step_rad", 0.05))
        self.joint_lower = np.asarray(self.system.inference.joint_lower_rad)
        self.joint_upper = np.asarray(self.system.inference.joint_upper_rad)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_pub = self.create_publisher(String, "/rl/status", latched)
        self.source_pub = self.create_publisher(String, "/rl/action_source", latched)
        self.intervention_pub = self.create_publisher(Bool, "/rl/intervention_active", latched)
        self.reward_pub = self.create_publisher(String, "/rl/reward_event", 10)
        self.executed_pub = self.create_publisher(JointState, "/rl/executed_action", 30)
        self.command_event_pub = self.create_publisher(String, "/rl/command_event", 100)
        self.policy_paused_pub = self.create_publisher(Bool, "/rl/policy_paused", latched)
        self.human_ready_pub = self.create_publisher(Bool, "/rl/human_ready", latched)
        self.arm_pub = self.create_publisher(JointState, "/xarm/absolute_target", 10)
        self.gripper_pub = self.create_publisher(JointState, "/xarm/gripper_pos", 10)
        if self.tcp_mode:
            self.tcp_pub = self.create_publisher(PoseStamped, "/xarm/tcp_target", 1)
            self.tcp_follow_pub = self.create_publisher(Bool, "/xarm/tcp_following", 1)
            self.human_enable_pub = self.create_publisher(Bool, "/rl/human_enabled", latched)
            self.create_subscription(String, policy_action_topic, self._tcp_policy_callback, 1)
            self.create_subscription(String, "/control/vr_action", self._vr_action_callback, 1)
            self.create_subscription(String, "/vr/teleop_status", self._vr_status_callback, latched)
            self.create_subscription(PoseStamped, "/xarm/tcp_pose", self._tcp_pose_callback, 1)
            self.create_subscription(String, "/xarm/tcp_status", self._tcp_status_callback, latched)
        else:
            self.create_subscription(JointState, policy_action_topic, self._policy_callback, 30)
            self.create_subscription(JointState, human_arm_topic, self._human_arm_callback, 30)
            self.create_subscription(
                JointState, human_gripper_topic, self._human_gripper_callback, 30
            )
        self.create_subscription(JointState, "/xarm/arm_joint_state", self._arm_callback, 30)
        self.create_subscription(JointState, "/robotiq/joint_state", self._gripper_callback, 30)
        self.create_subscription(Bool, "/xarm/control_ok", self._control_callback, 10)
        self.create_subscription(Bool, "/rl/episode_active", self._episode_callback, latched)
        self.create_subscription(String, "/recording/status", self._recorder_callback, latched)
        self.create_subscription(String, "/inference/status", self._inference_callback, latched)
        if not self.tcp_mode:
            self.create_subscription(
                Bool,
                str(self.config.get("human_connected_topic", "/factr/connected")),
                self._human_connected_callback,
                latched,
            )
        self.create_subscription(Bool, "/teleop/reset/active", self._reset_active_callback, latched)
        self.operator_services = [
            self.create_service(Trigger, service_name, self._service_callback(command))
            for command, service_name in RL_SERVICE_COMMANDS.items()
            if command not in RL_EPISODE_SERVICES
        ]
        self.episode_clients = {
            command: self.create_client(Trigger, service_name)
            for command, service_name in RL_EPISODE_SERVICES.items()
        }
        self.timer = self.create_timer(0.01, self._tick)

        if bool(self.config.get("keyboard_enabled", True)):
            self.keyboard = KeyboardController(
                str(self.config.get("keyboard_device", "/dev/tty")),
                self._queue_key,
                decoder=decode_rl_key,
            )
            self.keyboard.start()
        self._publish_bool(self.policy_paused_pub, True)
        self._publish_bool(self.human_ready_pub, False)
        self._publish_state("waiting for human device; S=start, T=switch, Y/N=end, Esc=pause")

    def _arm_callback(self, message: JointState) -> None:
        """Store measured absolute arm position and velocity, each shape ``(7,)``."""

        position = np.asarray(message.position[:7], dtype=np.float32)
        if position.shape != (7,) or not np.all(np.isfinite(position)):
            return
        if not self.tcp_mode:
            gripper = 0.0 if self.robot_state is None else float(self.robot_state[7])
            self.robot_state = np.concatenate([position, np.asarray([gripper], np.float32)])
        velocity = np.asarray(message.velocity[:7], dtype=np.float32)
        self.robot_velocity = (
            velocity if velocity.shape == (7,) and np.all(np.isfinite(velocity)) else None
        )
        self.velocity_received_at = time.monotonic()
        if not self.tcp_mode:
            self.robot_state_received_at = self.velocity_received_at

    def _gripper_callback(self, message: JointState) -> None:
        """Store measured normalized gripper position in the canonical state."""

        if not message.position or not np.isfinite(message.position[0]):
            return
        if self.tcp_mode and self.robot_state is None:
            return
        arm = self.initial_joints if self.robot_state is None else self.robot_state[:7]
        self.robot_state = np.concatenate(
            [arm, np.asarray([np.clip(message.position[0], 0.0, 1.0)], np.float32)]
        )
        self.gripper_state_received_at = time.monotonic()

    def _tcp_pose_callback(self, message: PoseStamped) -> None:
        """Cache measured base TCP [7] metres/xyzw for holds, never as policy state."""
        if message.header.frame_id != "xarm_base":
            return
        closure = 0.0 if self.robot_state is None else self.robot_state[7]
        self.robot_state = np.r_[pose_array(message), closure].astype(np.float32)
        self.robot_state_received_at = time.monotonic()

    def _tcp_policy_callback(self, message: String) -> None:
        """Validate explicit TCP candidate [8]; joint-format input is never accepted."""
        try:
            payload = json.loads(message.data)
            if payload["action_format"] != TCP_ACTION_FORMAT:
                raise ValueError("policy candidate is not a TCP action")
            stamp = float(payload["timestamp"])
            age = self.get_clock().now().nanoseconds / 1e9 - stamp
            if (
                not self.episode_active
                or self.controller.state != InterventionState.POLICY
                or not self.policy_enabled_at <= stamp
                or not 0 <= age <= self.policy_timeout_sec
            ):
                return
            self.policy_action = validate_tcp_chunk([payload["action"]], self.tcp_config)[0]
            self.policy_received_at = time.monotonic()
        except (ValueError, KeyError, TypeError) as error:
            self._pause_policy(f"invalid TCP policy: {error}")

    def _vr_action_callback(self, message: String) -> None:
        """Cache coherent VR TCP+gripper [8], clearing the target on clutch release."""
        try:
            payload = json.loads(message.data)
            if payload["action_format"] != TCP_ACTION_FORMAT:
                raise ValueError("VR candidate is not a TCP action")
            stamp = float(payload["timestamp"])
            age = self.get_clock().now().nanoseconds / 1e9 - stamp
            if (
                self.controller.state != InterventionState.HUMAN_CONTROL
                or not self.human_enabled_at <= stamp
                or not 0 <= age <= self.tcp_config.timeout
            ):
                return
            self.vr_action = (
                validate_tcp_chunk([payload["action"]], self.tcp_config)[0]
                if payload["following"]
                else None
            )
            self.vr_action_at = time.monotonic()
        except (ValueError, KeyError, TypeError) as error:
            self.vr_action = None
            self._pause_policy(f"invalid VR candidate: {error}")

    def _vr_status_callback(self, message: String) -> None:
        """Track continuous VR readiness independently of who currently owns motion."""
        self.vr_ready = bool(json.loads(message.data)["ready"])
        self.vr_status_at = time.monotonic()

    def _tcp_status_callback(self, message: String) -> None:
        """Pause after a TCP SDK/watchdog fault; never replay a stale policy chunk."""
        if message.data != "ready" and self.episode_active:
            self._pause_policy(message.data)

    def _human_arm_callback(self, message: JointState) -> None:
        """Store the continuous, sign-mapped FACTR arm coordinates shape ``(7,)``."""

        values = np.asarray(message.position[:7], dtype=np.float32)
        if values.shape == (7,) and np.all(np.isfinite(values)):
            self.human_input.arm = values
            self.human_input.arm_received_at = time.monotonic()

    def _human_gripper_callback(self, message: JointState) -> None:
        """Store the continuous FACTR gripper coordinate in radians."""

        if message.position and np.isfinite(message.position[0]):
            self.human_input.gripper = float(message.position[0])
            self.human_input.gripper_received_at = time.monotonic()

    def _episode_callback(self, message: Bool) -> None:
        """Enable policy only after capture starts; remove motion when it ends."""

        self.episode_active = bool(message.data)
        if self.tcp_mode:
            self.vr_action = None
            self.policy_enabled_at = self.get_clock().now().nanoseconds / 1e9
        self.policy_action = None
        if self.robot_state is not None:
            self.controller.pause(self.robot_state)
            self.executed_action = self.robot_state.copy()
        if self.episode_active:
            self.controller.state = InterventionState.PAUSED
            self.controller.resume_policy()
        self._publish_bool(self.policy_paused_pub, not self.episode_active)
        self._publish_bool(self.intervention_pub, False)
        self._publish_string(self.source_pub, "hold")
        self._publish_state(
            "episode started" if self.episode_active else "waiting for next episode"
        )

    def _recorder_callback(self, message: String) -> None:
        """Keep keyboard operations locked during data saves and robot reset."""

        self.recorder_state = str(message.data)

    def _inference_callback(self, message: String) -> None:
        """Pause a failed policy request while leaving the episode available for human control."""

        status = InferenceStatus.from_wire(message.data)
        if self.episode_active and status.failed:
            self._pause_policy(f"{status.text}; switch to human control or label failure")

    def _reset_active_callback(self, message: Bool) -> None:
        """Never forward policy/hold commands while the reset coordinator owns motion."""

        self.reset_in_progress = bool(message.data)
        if self.reset_in_progress and self.robot_state is not None:
            self._pause_policy("coordinated reset active")

    def _human_connected_callback(self, message: Bool) -> None:
        """Pause on device loss; reconnection requires a fresh stability window."""

        self.human_input.connected = bool(message.data)
        if not self.human_input.connected:
            self.human_input.ready_since = None
            if self.controller.state == InterventionState.HUMAN_CONTROL:
                self._pause_policy("human device disconnected; episode remains open")

    def _policy_callback(self, message: JointState) -> None:
        """Store one canonical absolute policy target shape ``(8,)``."""

        action = np.asarray(message.position[:8], dtype=np.float32)
        if action.shape == (8,) and np.all(np.isfinite(action)):
            self.policy_action = action
            self.policy_received_at = time.monotonic()

    def _control_callback(self, message: Bool) -> None:
        """Pause policy after an established healthy controller becomes unhealthy."""

        was_healthy = self.control_ok is True
        self.control_ok = bool(message.data)
        if self.motion_enabled and was_healthy and not self.control_ok:
            self._pause_policy("xArm control health lost; waiting for coordinated reset")

    def _queue_key(self, command: str) -> None:
        """Move one keyboard command from its reader thread into the ROS timer."""

        with self.key_lock:
            self.keys.append(str(command))

    def _tick(self) -> None:
        """Process operator events, stopping stability and the active action source."""

        self._update_human_ready()
        with self.key_lock:
            commands = list(self.keys)
            self.keys.clear()
        for command in commands:
            accepted, message = self._handle_key(command)
            if not accepted:
                self.get_logger().warn(message)
        if self.controller.state == InterventionState.STOPPING:
            self._advance_stopping()
        action, source = self._selected_action()
        if action is not None:
            self._apply_action(action, source)

    def _service_callback(self, command: str):
        """Return a Trigger callback that applies one named operator command.

        Args:
            command: Scalar command name from ``RL_SERVICE_COMMANDS``.

        Returns:
            ROS Trigger callback reporting whether the current state accepted
            the command and the resulting human-readable controller state.
        """

        def callback(_request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            """Apply the captured command and populate one Trigger response."""

            response.success, response.message = self._handle_key(command)
            return response

        return callback

    def _handle_key(self, command: str) -> tuple[bool, str]:
        """Apply one keyboard or Web command through the shared RL state machine.

        Args:
            command: One scalar operator command such as ``switch`` or ``success``.

        Returns:
            ``(accepted, message)`` describing whether the current controller
            state allowed the command and the state after processing it.
        """

        if command in RL_EPISODE_SERVICES:
            client = self.episode_clients[command]
            if not client.wait_for_service(timeout_sec=0.0):
                return False, "RL episode recorder is unavailable"
            client.call_async(Trigger.Request()).add_done_callback(self._episode_service_result)
            return True, f"episode {command} requested"
        if self.reset_in_progress:
            return False, "episode reset is in progress; wait for completion"
        if not self.episode_active:
            return False, "start an RL episode before policy or reward commands"

        if command == "switch":
            if self.robot_state is None:
                return False, "robot state is not available"
            if self.controller.state == InterventionState.HUMAN_CONTROL:
                self.controller.pause(self.robot_state)
                self.controller.resume_policy()
                self.executed_action = self.robot_state.copy()
                self.policy_action = None
                if getattr(self, "tcp_mode", False):
                    self.vr_action = None
                    self.policy_enabled_at = self.get_clock().now().nanoseconds / 1e9
                    self._publish_bool(self.human_enable_pub, False)
                    self._publish_bool(self.tcp_follow_pub, False)
                self._publish_bool(self.policy_paused_pub, False)
                self._publish_state("policy requested from a fresh observation")
                return True, "control switched to policy"
            if self.controller.state not in {InterventionState.POLICY, InterventionState.PAUSED}:
                return False, "control switch is already in progress"
            if not self.human_ready or self._human_state() is None:
                return False, "human input is not ready"
            self.controller.begin_takeover(self.robot_state)
            self.executed_action = self.robot_state.copy()
            self.policy_action = None
            self.stable_since = None
            self._publish_bool(self.policy_paused_pub, True)
            self._publish_state("switching to human; waiting for the robot to stop")
            return True, "control switch requested; human engagement is automatic"
        elif command in {"success", "failure"}:
            self._publish_reward(command)
            return True, f"human reward published: {command}"
        elif command == "stop":
            if self.robot_state is None:
                return False, "cannot pause before robot state is available"
            self._pause_policy("operator paused policy control")
            return (
                True,
                "policy paused at the measured pose; switch control or end the episode",
            )
        return False, f"unknown RL command: {command}"

    def _episode_service_result(self, future) -> None:
        """Report acceptance or rejection of a keyboard episode service request."""

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"RL episode request failed: {exc}")
            return
        log = self.get_logger().info if response.success else self.get_logger().warn
        log(response.message)

    def _update_human_ready(self) -> None:
        """Publish a fresh 10 Hz readiness heartbeat, independent of episode ownership."""

        now = time.monotonic()
        ready = (
            self.vr_ready and now - self.vr_status_at <= self.human_input_timeout_sec
            if self.tcp_mode
            else self.human_input.ready(
                now, self.human_input_timeout_sec, self.human_ready_stable_sec
            )
        )
        changed = ready != self.human_ready
        self.human_ready = ready
        if changed or now - self.human_ready_published_at >= 0.1:
            self._publish_bool(self.human_ready_pub, ready)
            self.human_ready_published_at = now
        if changed:
            self._publish_state("human device ready" if ready else "waiting for fresh human input")

    def _advance_stopping(self) -> None:
        """Automatically engage a fresh human origin after measured stop stability."""

        if not self.episode_active or self.reset_in_progress:
            return
        human = self._human_state()
        if human is None or not self.human_ready:
            self._pause_policy("human input lost during control switch")
            return
        now = time.monotonic()
        if (
            self.robot_velocity is None
            or now - min(self.robot_state_received_at, self.gripper_state_received_at)
            > self.policy_timeout_sec
            or self.control_ok is not True
            or (self.tcp_mode and now - self.velocity_received_at > self.policy_timeout_sec)
        ):
            self._pause_policy("fresh healthy robot position/velocity required for control switch")
            return
        if float(np.max(np.abs(self.robot_velocity))) > self.stop_velocity_rad_s:
            self.stable_since = None
            return
        if self.stable_since is None:
            self.stable_since = now
        elif now - self.stable_since >= self.stop_stable_sec:
            # Snapshot both current origins now, not at the initial button press:
            # any device motion while ACT stops must not become a target jump.
            if self.tcp_mode:
                # VR owns its SE(3) origin/clutch math; never add quaternion
                # coordinates as if they were FACTR joint displacements.
                self.vr_action = None
                self.human_enabled_at = self.get_clock().now().nanoseconds / 1e9
                self.controller.state = InterventionState.HUMAN_CONTROL
                self._publish_state("VR control enabled; release then grip; T returns to policy")
            else:
                self.executed_action = self.controller.engage_human(self.robot_state, human)
                self._publish_state("human control engaged; press T to switch back to policy")

    def _selected_action(self) -> tuple[np.ndarray | None, str]:
        """Return the current canonical target and its provenance label."""

        if not self.episode_active or self.reset_in_progress:
            return None, "hold"
        state = self.controller.state
        if state == InterventionState.POLICY:
            fresh = time.monotonic() - self.policy_received_at <= self.policy_timeout_sec
            if self.policy_action is not None and fresh:
                return self.policy_action.copy(), "policy"
            return self.robot_state, "hold"
        if state == InterventionState.HUMAN_CONTROL:
            human = self._human_state()
            if human is not None:
                if self.tcp_mode:
                    if (
                        self.vr_action is not None
                        and time.monotonic() - self.vr_action_at <= self.tcp_config.timeout
                    ):
                        return self.vr_action.copy(), "human"
                    hold = self.robot_state.copy()
                    if self.executed_action is not None:
                        hold[7] = self.executed_action[7]
                    return hold, "hold"
                return self.controller.human_action(human), "human"
            self._pause_policy("human input stale; episode remains open")
            return self.robot_state, "hold"
        return self.controller.hold_action, "hold"

    def _human_state(self) -> np.ndarray | None:
        """Return fresh sign-mapped input-device coordinates shaped [8], or None."""

        if self.tcp_mode:
            return self.robot_state if self.human_ready else None
        return self.human_input.sample(time.monotonic(), self.human_input_timeout_sec)

    def _apply_action(self, action: np.ndarray, source: str) -> None:
        """Validate, record and optionally publish one canonical absolute target."""

        if self.tcp_mode:
            self._apply_tcp_action(action, source)
            return
        reference = self.robot_state if self.executed_action is None else self.executed_action
        if reference is None:
            return
        max_step = (
            self.max_human_joint_step_rad
            if source == "human"
            else self.system.inference.max_joint_step_rad
        )
        try:
            target = validate_action_chunk(
                np.asarray(action)[None], reference, self.joint_lower, self.joint_upper, max_step
            )[0]
        except ValueError as exc:
            self._pause_policy(f"unsafe {source} action rejected: {exc}")
            return
        self.executed_action = target.copy()
        self._publish_string(self.source_pub, source)
        self._publish_bool(
            self.intervention_pub, self.controller.state == InterventionState.HUMAN_CONTROL
        )
        if not self.motion_enabled:
            return
        if self.control_ok is not True:
            return
        stamp = self.get_clock().now()
        policy_valid = (
            self.controller.state == InterventionState.POLICY
            and self.policy_action is not None
            and time.monotonic() - self.policy_received_at <= self.policy_timeout_sec
        )
        event = {
            "timestamp": stamp.nanoseconds / 1e9,
            "source": source,
            "selected_action": target.tolist(),
            "policy_action": self.policy_action.tolist() if policy_valid else [0.0] * 8,
            "policy_valid": bool(policy_valid),
        }
        self._publish_string(self.command_event_pub, json.dumps(event, separators=(",", ":")))
        arm_target, leader_gripper = model_to_ros_commands(target)
        self.arm_pub.publish(_joint_state(arm_target, self.get_clock().now().to_msg(), count=7))
        self.gripper_pub.publish(
            _joint_state(np.asarray([leader_gripper]), self.get_clock().now().to_msg(), count=1)
        )
        # This topic means forwarded ROS target, not SDK acknowledgement. The
        # recorder's action labels come from the two driver confirmation topics.
        self.executed_pub.publish(_joint_state(target, stamp.to_msg()))

    def _apply_tcp_action(self, action: np.ndarray, source: str) -> None:
        """Forward TCP [8] through the sole arbiter, leaving SDK limits/ACKs to driver.

        Holds disable following so the driver overrides old trajectories with
        measured TCP. Command provenance joins SDK acknowledgements by selection
        timestamp, not equality of requested and angular-rate-limited poses.
        """
        try:
            # A measured hold may be outside the configured working box (e.g.
            # the reset/home pose). Never turn that into a commanded relocation.
            target = (
                np.asarray(action, dtype=np.float32).copy()
                if source == "hold"
                else validate_tcp_chunk([action], self.tcp_config)[0]
            )
        except ValueError as error:
            self._pause_policy(f"unsafe {source} TCP target: {error}")
            return
        self.executed_action = target.copy()
        self._publish_string(self.source_pub, source)
        self._publish_bool(self.intervention_pub, source == "human")
        if not self.motion_enabled or self.control_ok is not True:
            return
        if (
            time.monotonic() - min(self.robot_state_received_at, self.gripper_state_received_at)
            > self.tcp_config.timeout
        ):
            self._publish_bool(self.tcp_follow_pub, False)
            self._pause_policy("fresh TCP/gripper feedback required")
            return
        stamp = self.get_clock().now()
        policy_valid = source == "policy" and self.policy_action is not None
        self._publish_string(
            self.command_event_pub,
            json.dumps(
                {
                    "timestamp": stamp.nanoseconds / 1e9,
                    "source": source,
                    "action_format": TCP_ACTION_FORMAT,
                    "selected_action": target.tolist(),
                    "policy_action": self.policy_action.tolist() if policy_valid else [0.0] * 8,
                    "policy_valid": policy_valid,
                }
            ),
        )
        self._publish_bool(self.tcp_follow_pub, source != "hold")
        if source != "hold":
            self.tcp_pub.publish(pose_message(target[:7], stamp.to_msg()))
        self.gripper_pub.publish(_joint_state(np.array([-target[7]]), stamp.to_msg(), count=1))

    def _publish_reward(self, command: str, *, reason: str | None = None) -> None:
        """Publish one sparse human outcome and pause at the measured pose."""

        values = {
            "success": (float(self.config.get("success_reward", 1.0)), True, False),
            "failure": (float(self.config.get("failure_reward", 0.0)), True, False),
        }
        reward, terminated, truncated = values[command]
        event = RewardEvent(time.monotonic(), reward, f"task_{command}", terminated, truncated)
        payload = json.loads(event.to_json())
        payload["reason"] = reason or f"operator_{command}"
        payload["timestamp_ros_s"] = self.get_clock().now().nanoseconds / 1e9
        self._publish_string(self.reward_pub, json.dumps(payload, separators=(",", ":")))
        self.episode_active = False
        if self.robot_state is not None:
            self.controller.pause(self.robot_state)
            self.executed_action = self.robot_state.copy()
        self._publish_bool(self.policy_paused_pub, True)
        self._publish_state(f"human reward: {event.to_json()}")

    def _pause_policy(self, reason: str) -> None:
        """Hold the measured pose; ownership switching and episode labeling remain available.

        Args:
            reason: Human-readable cause published in the RL status.
        """

        if self.robot_state is None:
            self.get_logger().warn(f"cannot pause before robot state is available: {reason}")
            return
        self.controller.pause(self.robot_state)
        self.executed_action = self.robot_state.copy()
        self.policy_action = None
        self.stable_since = None
        self._publish_bool(self.policy_paused_pub, True)
        self._publish_state(reason)
        self.get_logger().warn(reason)

    def _publish_state(self, detail: str) -> None:
        """Publish the current controller state plus one human-readable detail."""

        if getattr(self, "tcp_mode", False):
            if (
                not self.episode_active
                or self.reset_in_progress
                or self.controller.state in {InterventionState.PAUSED, InterventionState.STOPPING}
            ):
                # Revoke mode-7 following now, not at the next timer/watchdog.
                self.vr_action = None
                self._publish_bool(self.tcp_follow_pub, False)
            self._publish_bool(
                self.human_enable_pub,
                self.episode_active
                and not self.reset_in_progress
                and self.controller.state == InterventionState.HUMAN_CONTROL,
            )
        self._publish_string(self.status_pub, f"{self.controller.state.value}|{detail}")

    @staticmethod
    def _publish_string(publisher, value: str) -> None:
        """Publish one scalar ROS String value."""

        publisher.publish(String(data=str(value)))

    @staticmethod
    def _publish_bool(publisher, value: bool) -> None:
        """Publish one scalar ROS Bool value."""

        publisher.publish(Bool(data=bool(value)))

    def destroy_node(self) -> None:
        """Restore terminal settings before destroying ROS resources."""

        if self.keyboard is not None:
            self.keyboard.close()
        if self.tcp_mode:
            self._publish_bool(self.human_enable_pub, False)
            self._publish_bool(self.tcp_follow_pub, False)
        super().destroy_node()


def _joint_state(values: np.ndarray, stamp, *, count: int = 8) -> JointState:
    """Return a stamped JointState carrying one canonical or component action."""

    message = JointState()
    message.header.stamp = stamp
    names = [f"joint{index + 1}" for index in range(min(count, 7))]
    if count == 8:
        names.append("robotiq_finger")
    elif count == 1:
        names = ["leader_gripper"]
    message.name = names
    message.position = np.asarray(values, dtype=float).reshape(-1)[:count].tolist()
    return message


def main(args=None) -> None:
    """Run the human-intervention arbiter with optional ROS arguments."""

    rclpy.init(args=args)
    node = HumanInterventionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
