"""Hardware-independent clutch state and fixed-space relative TCP mapping."""

import numpy as np

from xarm_ros2_teleop.vr.pose_math import matrix_quaternion, rotation_matrix


class RelativeTeleop:
    """Map coherent VR samples to TCP targets; never automatically re-engage.

    Input/output poses are xyz+xyzw float [7], metres. Gripper and trigger are
    closure fractions [0,1]. The caller supplies monotonic arrival times.
    """

    def __init__(self, config):
        """Store VRConfig and initialize a disarmed clutch."""
        self.config = config
        self.generation = None
        self.last_sample_time = None
        self.invalidate("waiting for valid VR input")

    def invalidate(self, reason):
        """Clear all pose origins after loss, reset, release or a rejected target."""
        self.following = False
        self.released = False
        self.valid_since = None
        self.reason = reason
        self.vr_origin = None
        self.robot_origin = None
        self.previous_quaternion = None

    def update(self, pose, trigger, grip, valid, generation, now, robot_pose, gripper, active):
        """Return optional (TCP [7], gripper float) for one fresh coherent sample.

        generation identifies transport/origin epochs; active is the shared
        episode/staging gate. A valid release after activation is mandatory.
        """
        if (
            generation != self.generation
            or self.last_sample_time is None
            or now - self.last_sample_time > self.config.timeout
        ):
            self.invalidate("VR reconnected/origin changed; release grip")
            self.generation = generation
        self.last_sample_time = now
        if not valid:
            self.invalidate("VR tracking/input invalid; waiting for recovery")
            return None
        if self.valid_since is None:
            self.valid_since = now
        if not active or robot_pose is None or gripper is None:
            self.following = False
            self.released = False
            self.reason = "waiting for active episode and fresh robot feedback"
            return None
        if not grip:
            self.following = False
            self.released = True
            self.reason = "ready; hold grip to follow"
            return None
        if now - self.valid_since < self.config.stable_sec:
            # A press made before recovery is stable must not become a queued
            # automatic engagement when the timer later reaches stable_sec.
            self.released = False
            self.reason = "waiting for stable input; release then press grip"
            return None
        if not self.released:
            self.reason = "waiting for stable input and a new grip press"
            return None
        if not self.following:
            self.vr_origin, self.robot_origin = pose.copy(), robot_pose.copy()
            self.trigger_origin, self.gripper_origin = trigger, gripper
            self.following = True
        c = self.config.rotation
        position = self.robot_origin[:3] + self.config.position_scale * c @ (
            pose[:3] - self.vr_origin[:3]
        )
        rotation = (
            c
            @ rotation_matrix(pose[3:])
            @ rotation_matrix(self.vr_origin[3:]).T
            @ c.T
            @ rotation_matrix(self.robot_origin[3:])
        )
        orientation = matrix_quaternion(rotation)
        if self.previous_quaternion is not None and orientation @ self.previous_quaternion < 0:
            orientation = -orientation
        self.previous_quaternion = orientation
        try:
            self.config.check_workspace(position)
        except ValueError as error:
            self.invalidate(str(error))
            return None
        self.reason = "following"
        return np.r_[position, orientation], float(
            np.clip(self.gripper_origin + trigger - self.trigger_origin, 0, 1)
        )
