"""Convert coherent Quest input into relative TCP targets without owning hardware."""

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from xarm_ros2_teleop.inference.tcp_contracts import TCP_ACTION_FORMAT
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.vr.config import VRConfig
from xarm_ros2_teleop.vr.mapping import RelativeTeleop
from xarm_ros2_teleop.vr.pose_math import pose_array, pose_message, quaternion


class VRTeleopNode(Node):
    """Own the clutch and mapping, sharing episode/reset gates with the recorder.

    Publishes base-frame metre TCP poses, a deadman heartbeat, scalar gripper
    closure and explicit readiness/status. Preview never publishes actuator
    commands and uses a labelled virtual TCP origin, not invented feedback.
    """

    def __init__(self):
        """Load the unified profile and subscribe to coherent input and feedback."""
        super().__init__("vr_teleop")
        self.declare_parameter("config_path", "")
        self.declare_parameter("motion_enabled", False)
        self.declare_parameter("rl_enabled", False)
        system = load_system_config(str(self.get_parameter("config_path").value))
        self.motion = bool(self.get_parameter("motion_enabled").value)
        self.rl_mode = bool(self.get_parameter("rl_enabled").value)
        self.human_enabled = not self.rl_mode
        self.config = VRConfig.from_mapping(system.teleop.get("vr", {}))
        self.mapper = RelativeTeleop(self.config)
        self.sample = None
        self.received_at = 0.0
        self.robot = None
        self.robot_at = 0.0
        self.gripper = None
        self.gripper_at = 0.0
        self.episode_active = False
        self.session_ready = self.rl_mode
        self.reset_active = False
        self.control_ok = False
        self.last_generation = None
        self.last_wire_time = None
        self.input_stable_since = None
        self.last_status = ""
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.target_pub = self.create_publisher(PoseStamped, "/xarm/tcp_target", 1)
        self.preview_pub = self.create_publisher(PoseStamped, "/vr/preview_target", 1)
        self.follow_pub = self.create_publisher(Bool, "/vr/following", 1)
        self.status_pub = self.create_publisher(String, "/vr/teleop_status", qos)
        self.gripper_pub = self.create_publisher(JointState, "/xarm/gripper_pos", 1)
        self.candidate_pub = self.create_publisher(String, "/control/vr_action", 1)
        if self.rl_mode:
            self.create_subscription(
                Bool,
                "/rl/human_enabled",
                lambda msg: self.gate_callback("human_enabled", msg.data),
                qos,
            )
        self.create_subscription(String, "/quest/controller_samples", self.sample_callback, 1)
        self.create_subscription(String, "/quest/bridge/status", self.bridge_callback, qos)
        self.create_subscription(PoseStamped, "/xarm/tcp_pose", self.pose_callback, 1)
        self.create_subscription(JointState, "/robotiq/joint_state", self.gripper_callback, 1)
        self.create_subscription(Bool, "/xarm/control_ok", self.health_callback, 1)
        self.create_subscription(String, "/xarm/tcp_status", self.tcp_status_callback, qos)
        for topic, key in (
            ("/rl/episode_active" if self.rl_mode else "/teleop/episode_active", "episode_active"),
            ("/system/session/ready", "session_ready"),
            ("/teleop/reset/active", "reset_active"),
        ):
            self.create_subscription(
                Bool, topic, lambda msg, name=key: self.gate_callback(name, msg.data), qos
            )
        if not self.motion:
            self.robot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            self.gripper = 0.0
            self.get_logger().info("VR PREVIEW: virtual TCP origin; robot commands disabled")
        self.create_timer(0.01, self.tick)

    def sample_callback(self, message):
        """Validate one ROS-side coherent frame and retain only its newest sample."""
        try:
            frame = json.loads(message.data)
            if frame["schema_version"] != 3:
                raise ValueError("old pose-only APK; rebuild/install schema 3 APK")
            sample = frame["hands"][self.config.hand]
            pose = np.asarray(sample["pose"], dtype=float)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise ValueError("invalid controller pose")
            pose[3:] = quaternion(pose[3:])
            trigger = float(sample["trigger"])
            if not np.isfinite(trigger) or not 0 <= trigger <= 1:
                raise ValueError("invalid trigger")
            generation = tuple(frame["generation"])
            if generation != self.last_generation:
                self.input_stable_since = None
            wire_time = float(frame["timestamp_seconds"])
            if (
                generation == self.last_generation
                and self.last_wire_time is not None
                and wire_time <= self.last_wire_time
            ):
                raise ValueError("VR sample clock did not advance")
            self.last_generation, self.last_wire_time = generation, wire_time
            self.sample = (
                pose,
                trigger,
                sample["grip"],
                sample["tracked"] and sample["input_valid"],
                generation,
            )
            self.received_at = time.monotonic()
        except (ValueError, KeyError, TypeError) as error:
            self.sample = None
            self.mapper.invalidate(str(error))
            self.input_stable_since = None

    def pose_callback(self, message):
        """Cache measured xarm_base TCP xyz+xyzw [7] metres with arrival time."""
        if message.header.frame_id == "xarm_base":
            self.robot = pose_array(message)
            self.robot_at = time.monotonic()

    def bridge_callback(self, message):
        """Disarm on a reported TCP disconnect immediately, not only after timeout."""
        if message.data.startswith(("Quest pose stream disconnected", "Quest pose stream stale;")):
            self.sample = None
            self.input_stable_since = None
            self.mapper.invalidate("VR disconnected; waiting for stable input and a new grip")

    def gripper_callback(self, message):
        """Cache finite measured gripper closure fraction [0,1]."""
        if message.position and np.isfinite(message.position[0]):
            self.gripper = float(np.clip(message.position[0], 0, 1))
            self.gripper_at = time.monotonic()

    def health_callback(self, message):
        """Invalidate clutch immediately on xArm controller-health loss."""
        self.control_ok = bool(message.data)
        if not self.control_ok:
            self.mapper.invalidate("xArm not healthy; waiting for reset/recovery")

    def tcp_status_callback(self, message):
        """Disarm the clutch if the sole SDK owner reports a fault/watchdog stop."""
        if message.data != "ready":
            self.mapper.invalidate(message.data)

    def gate_callback(self, name, value):
        """Clear origins on shared session, reset or episode gate transitions."""
        if getattr(self, name) != bool(value):
            self.mapper.invalidate(f"{name} changed; release grip")
            self.held_gripper = None
        setattr(self, name, bool(value))

    def tick(self):
        """Publish latest targets, heartbeat and a precise readiness reason at 100 Hz."""
        now = time.monotonic()
        fresh = self.sample is not None and now - self.received_at <= self.config.timeout
        valid = fresh and bool(self.sample[3])
        if not valid:
            self.input_stable_since = None
            if self.sample is not None:
                self.mapper.invalidate("VR tracking/input stale or invalid; release grip")
        elif self.input_stable_since is None:
            self.input_stable_since = now
        ready = valid and now - self.input_stable_since >= self.config.stable_sec
        feedback_fresh = (
            self.robot is not None
            and self.gripper is not None
            and now - min(self.robot_at, self.gripper_at) <= self.config.timeout
        )
        active = not self.motion or (
            self.episode_active
            and self.human_enabled
            and self.session_ready
            and not self.reset_active
            and self.control_ok
            and feedback_fresh
        )
        if self.rl_mode:
            active = active and self.human_enabled and self.episode_active and not self.reset_active
        result = None
        # Mapping consumes each input ONCE; the timer only sends current control
        # heartbeats. Reusing a sample must not conceal a network gap.
        if fresh and self.received_at != getattr(self, "consumed_at", None):
            self.consumed_at = self.received_at
            pose, trigger, grip, sample_valid, generation = self.sample
            result = self.mapper.update(
                pose, trigger, grip, sample_valid, generation, now, self.robot, self.gripper, active
            )
            if result is not None:
                self.last_target = result
        if not active or not valid:
            reason = self.mapper.reason if not valid else "waiting for episode/feedback"
            self.mapper.invalidate(reason)
        following = self.mapper.following and active and valid
        if not self.rl_mode:
            self.follow_pub.publish(Bool(data=bool(following and self.motion)))
        if following and hasattr(self, "last_target"):
            target, closure = self.last_target
            stamp = self.get_clock().now().to_msg()
            self.preview_pub.publish(pose_message(target, stamp))
            if self.motion and not self.rl_mode:
                self.target_pub.publish(pose_message(target, stamp))
                # Existing gripper driver accepts negative closure coordinates.
                self.held_gripper = closure
        if self.rl_mode:
            # Never publish to actuator topics in RL: ownership is exclusive in
            # HumanInterventionNode, including while the VR device stays online.
            action = np.r_[self.last_target[0], self.last_target[1]].tolist() if following else None
            self.candidate_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "action_format": TCP_ACTION_FORMAT,
                            "timestamp": self.get_clock().now().nanoseconds / 1e9,
                            "action": action,
                            "following": bool(following),
                        }
                    )
                )
            )
        if self.motion and active and not self.rl_mode:
            if getattr(self, "held_gripper", None) is None:
                self.held_gripper = self.gripper
            self.gripper_pub.publish(JointState(position=[-float(self.held_gripper)]))
        status = {
            "ready": bool(ready),
            "tracking_valid": bool(valid),
            "following": bool(following),
            "preview": not self.motion,
            "reason": self.mapper.reason,
        }
        self.status_pub.publish(String(data=json.dumps(status)))
        if self.mapper.reason != self.last_status:
            self.get_logger().info(self.mapper.reason)
            self.last_status = self.mapper.reason

    def destroy_node(self):
        """Drop the deadman heartbeat before releasing ROS interfaces; never call the SDK."""
        if not self.rl_mode:
            self.follow_pub.publish(Bool(data=False))
        super().destroy_node()


def main(args=None):
    """Run the VR adapter; optional args are ROS arguments, not motion commands."""
    rclpy.init(args=args)
    node = VRTeleopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
