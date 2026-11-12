"""Native xArm TCP execution using the existing driver's sole SDK connection."""

from importlib.metadata import version

import numpy as np

from xarm_ros2_teleop.vr.pose_math import from_rotvec, multiply, quaternion, rotvec


def robot_geometry(arm) -> dict:
    """Read immutable robot/kinematic provenance; never invent missing geometry.

    arm is an already connected XArmAPI. Returns JSON-compatible metadata;
    get_dh_params is retained verbatim, not reinterpreted as a generic URDF.
    """
    if tuple(arm.version_number) < (2, 0, 0):
        raise ValueError("VR recording requires firmware >=2.0.0 for DH parameter export")
    code, dh = arm.get_dh_params()
    tcp, world = np.asarray(arm.tcp_offset, float), np.asarray(arm.world_offset, float)
    if (
        code != 0
        or np.asarray(dh).shape != (28,)
        or tcp.shape != (6,)
        or world.shape != (6,)
        or not np.all(np.isfinite(np.r_[dh, tcp, world]))
        or not arm.sn
        or int(arm.axis) != 7
    ):
        raise ValueError("Cannot read complete xArm7 DH/TCP/world configuration")
    if np.any(np.abs(world) > 1e-8):
        raise ValueError("VR xarm_base action requires zero SDK world_offset; configure explicitly")
    return {
        "model": "xarm7",
        "serial": str(arm.sn),
        "firmware": str(arm.version),
        "sdk_version": version("xarm-python-sdk"),
        "dh_params_raw": list(map(float, dh)),
        "dh_encoding": "unmodified XArmAPI.get_dh_params/controller-native response",
        "tcp_offset_mm_rad": tcp.tolist(),
        "world_offset_mm_rad": world.tolist(),
    }


class TCPControl:
    """Latest-only TCP target/hold controller; all physical commands return SDK codes.

    Pose inputs and successful command outputs are float64 xyz+xyzw [7], metres.
    Following and feedback watchdogs use monotonic seconds. Faults latch until
    the shared reset calls clear(), never until the next network packet.
    """

    def __init__(self, arm, config):
        """Bind the sole SDK connection and VRConfig with an explicit workspace."""
        config.require_workspace()
        self.arm, self.config = arm, config
        self.feedback = None
        self.feedback_at = 0.0
        self.clear()

    def clear(self):
        """Invalidate pre-reset references and stop following; retain measured feedback."""
        self.following = False
        self.follow_at = 0.0
        self.target = None
        self.target_at = 0.0
        self.hold = None
        self.last_command = None
        self.last_at = None
        self.fault = ""
        self.waiting_release = False
        self.target_selection = None
        self.command_selection = None

    def read_feedback(self, now):
        """Read SDK TCP axis-angle mm/rad, returning pose [7] metres or None."""
        code, pose = self.arm.get_position_aa(is_radian=True)
        if code or np.asarray(pose).shape != (6,) or not np.all(np.isfinite(pose)):
            return None
        self.feedback = np.r_[np.asarray(pose[:3]) / 1000, from_rotvec(pose[3:])]
        self.feedback_at = now
        return self.feedback.copy()

    def set_following(self, following, now):
        """Refresh deadman heartbeat; each transition invalidates buffered targets."""
        if self.waiting_release:
            if following:
                return
            self.waiting_release = False
        if bool(following) != self.following:
            self.target = None
            self.hold = None
            self.last_command = None
        self.following, self.follow_at = bool(following), now

    def accept(self, pose, now, selection=None):
        """Accept TCP [7] and optional causal ROS selection timestamp (seconds)."""
        if not self.following or self.fault:
            return
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("TCP target must be finite xyz+xyzw [7]")
        self.config.check_workspace(pose[:3])
        self.target = np.r_[pose[:3], quaternion(pose[3:])]
        self.target_at = now
        self.target_selection = selection

    def stop(self, reason):
        """Latch a fault and request SDK stop state; return its integer result."""
        self.fault = str(reason)
        self.target = None
        code = self.arm.set_state(4)
        if code:
            self.fault += f"; STOP NOT CONFIRMED (SDK {code})"
        return code

    def command(self, now):
        """Submit one bounded latest/hold target; return (SDK code, accepted pose or None).

        Fresh measured feedback is mandatory even while holding. On release or
        loss, overwrite the old remote target with the measured TCP, rather
        than merely ceasing to publish and letting mode 7 chase the old target.
        """
        if self.fault:
            return -1, None
        if self.feedback is None or now - self.feedback_at > self.config.timeout:
            self.stop("TCP feedback stale; reset required")
            return -1, None
        if self.following and (
            now - self.follow_at > self.config.timeout
            or (self.target is not None and now - self.target_at > self.config.timeout)
        ):
            # The driver has its own watchdog: a recovered ROS stream cannot
            # revive motion just by repeating True and a formerly held target.
            self.waiting_release = True
            self.following = False
            self.target = None
            self.hold = None
        fresh = (
            self.following
            and now - self.follow_at <= self.config.timeout
            and self.target is not None
            and now - self.target_at <= self.config.timeout
        )
        if fresh:
            target = self.target.copy()
            self.command_selection = self.target_selection
            self.hold = None
        else:
            self.command_selection = None
            if self.hold is None:
                self.hold = self.feedback.copy()
                self.last_command = None
            target = self.hold.copy()
        previous = self.feedback if self.last_command is None else self.last_command
        dt = min(0.05, max(0, now - self.last_at)) if self.last_at is not None else 0.01
        # The controller handles linear acceleration; software bounds angular
        # increments because SDK Cartesian speed is not an angular-speed limit.
        inverse = np.r_[-previous[3:6], previous[6]]
        delta = rotvec(multiply(target[3:], inverse))
        angle = np.linalg.norm(delta)
        limit = self.config.angular_speed * dt
        if fresh and angle > limit:
            target[3:] = multiply(from_rotvec(delta * limit / angle), previous[3:])
        if target[3:] @ previous[3:] < 0:
            target[3:] *= -1
        code = self.arm.set_position_aa(
            np.r_[target[:3] * 1000, rotvec(target[3:])].tolist(),
            speed=self.config.linear_speed * 1000,
            mvacc=self.config.linear_acceleration * 1000,
            is_radian=True,
            is_tool_coord=False,
            relative=False,
            wait=False,
        )
        if code:
            self.stop(f"TCP SDK command failed ({code}); reset required")
            return code, None
        self.last_command, self.last_at = target.copy(), now
        return 0, target
