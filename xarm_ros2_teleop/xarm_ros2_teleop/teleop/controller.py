# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from ament_index_python.packages import get_package_share_directory
from factr_teleop.dynamixel.driver import DynamixelDriver
from rclpy.node import Node

from xarm_ros2_teleop.teleop.offset_calibration import (
    derive_factr_home,
    load_offset_calibration,
)


class FACTRTeleop(Node, ABC):
    """
    Base class for implementing the FACTR low-cost force-feedback teleoperation system
    for a follower arm.

    This class implements the control loop for the leader teleoperation arm, including
    features such as gravity compensation, null-space regulation, friction compensation,
    and force-feedback.

    Note that this class should be used as a parent class, where the defined abstract
    methods must be implemented by subclasses for handling communication between the
    leader and follower arms, as well as force-feedback for the leader gripper.
    """

    def __init__(self):
        """Load configuration and initialize the physical leader controller.

        Scalar gains and one-dimensional joint arrays come from the configured
        YAML. The Dynamixel bus is left in current-control mode.
        """

        super().__init__("factr_teleop")

        config_file_name = (
            self.declare_parameter("config_file", "").get_parameter_value().string_value
        )
        self.declare_parameter("offset_file", "")
        self.declare_parameter("xarm_to_factr_signs", [1.0] * 7)
        self.offset_file = Path(str(self.get_parameter("offset_file").value)).expanduser()
        self.xarm_to_factr_signs = np.asarray(
            self.get_parameter("xarm_to_factr_signs").value,
            dtype=float,
        )
        config_path = os.path.join(
            get_package_share_directory("xarm_ros2_teleop"),
            "config",
            config_file_name,
        )
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.name = self.config["name"]
        self.dt = 1 / self.config["controller"]["frequency"]

        self._hardware_closed = False
        self._prepare_dynamixel()
        self._prepare_inverse_dynamics()

        # leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        )
        self.arm_joint_limits_min = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        )
        self.calibration_start_joints = np.asarray(
            self.config["arm_teleop"]["initialization"]["start_joints"], dtype=float
        )
        self.calibration_joint_pos = self.calibration_start_joints[:7]
        assert (
            self.num_arm_joints == len(self.arm_joint_limits_max) == len(self.arm_joint_limits_min)
        ), "num_arm_joints and the length of arm joint limits must be the same"
        assert self.calibration_start_joints.shape == (self.num_motors,), (
            "start_joints must contain seven FACTR arm joints and one gripper joint"
        )

        # leader gripper parameters
        self.gripper_limit_min = self.config["gripper_teleop"]["actuation_range_min"]
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # gravity comp
        gravity_config = self.config["controller"]["gravity_comp"]
        self.enable_gravity_comp = gravity_config["enable"]
        self.gravity_comp_modifier = gravity_config["gain"]
        # Constant xArm-aligned -> URDF translation; measured by the second
        # factr_calibrate capture at a known folded pose, not derived here.
        self.gravity_joint_position_offset = np.zeros(self.num_arm_joints)
        self.tau_g = np.zeros(self.num_arm_joints)
        # friction comp
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"][
            "enable_speed"
        ]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)
        # joint limit barrier:
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]
        self.enable_gripper_limit_barrier = bool(
            self.config["controller"]["joint_limit_barrier"].get("gripper_enable", True)
        )
        # reset-time software PD servo (current mode; no Dynamixel mode switch)
        position_config = self.config["controller"]["joint_position_control"]
        self.position_kp = float(position_config["kp"])
        self.position_kd = float(position_config["kd"])
        self.interpolation_step_size = float(position_config["interpolation_step_size"])
        if min(self.position_kp, self.position_kd, self.interpolation_step_size) <= 0:
            raise ValueError("joint_position_control kp, kd and step must be positive")
        # null space regulation
        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"]["null_space_joint_target"]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]
        # torque feedback
        self.enable_torque_feedback = self.config["controller"]["torque_feedback"]["enable"]
        self.torque_feedback_gain = self.config["controller"]["torque_feedback"]["gain"]
        self.torque_feedback_motor_scalar = self.config["controller"]["torque_feedback"][
            "motor_scalar"
        ]
        self.torque_feedback_damping = self.config["controller"]["torque_feedback"]["damping"]
        # gripper feedback
        self.enable_gripper_feedback = self.config["controller"]["gripper_feedback"]["enable"]

        # needs to be implemented to establish communication between the leader and the follower
        self.set_up_communication()

        self._initialize_persistent_offsets()
        # start the control loop
        self.timer = self.create_timer(self.dt, self.control_loop_callback)

    def _prepare_dynamixel(self):
        """
        Instantiates driver for interfacing with Dynamixel servos.
        """
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        assert self.num_motors == len(self.joint_signs), (
            "The number of motors and the number of joint signs must be the same"
        )
        self.dynamixel_port = self.config["dynamixel"]["dynamixel_port"]
        self.dynamixel_baudrate = int(self.config["dynamixel"].get("baudrate", 4000000))

        joint_ids = np.arange(self.num_motors) + 1
        try:
            self.driver = DynamixelDriver(
                joint_ids, self.servo_types, self.dynamixel_port, baudrate=self.dynamixel_baudrate
            )
        except FileNotFoundError:
            self.get_logger().info(
                f"Port {self.dynamixel_port} not found. Please check the connection."
            )
            return
        self.driver.enter_current_control()

    def _prepare_inverse_dynamics(self):
        """Load the leader URDF model used for joint-space inverse dynamics.

        Returns:
            Nothing. ``pin_model`` contains the URDF joint and inertia data and
            ``pin_data`` contains its runtime workspace. Visual and collision
            meshes are intentionally not loaded because control only uses joint
            dynamics, arrays of shape ``(pin_model.nq,)``.
        """

        urdf_model_dir = os.path.join(get_package_share_directory("xarm_ros2_teleop"), "urdf")
        urdf_model_path = os.path.join(urdf_model_dir, self.config["arm_teleop"]["leader_urdf"])
        self.leader_urdf = urdf_model_path
        self.pin_model = pin.buildModelFromUrdf(urdf_model_path)
        self.pin_data = self.pin_model.createData()

    def _initialize_persistent_offsets(self):
        """Load encoder offsets and the authoritative cross-arm reference pair."""

        if not self.offset_file.is_file():
            raise FileNotFoundError(
                f"FACTR offset 不存在: {self.offset_file}。"
                "请先运行 ros2 run xarm_ros2_teleop factr_calibrate"
            )
        calibration = load_offset_calibration(
            self.offset_file,
            dynamixel_port=self.dynamixel_port,
            servo_types=self.servo_types,
            joint_signs=self.joint_signs,
            calibration_joint_positions=self.calibration_joint_pos,
            operating_mode=0,
            xarm_to_factr_signs=self.xarm_to_factr_signs,
        )
        self.joint_offsets = calibration.joint_offsets_rad.copy()
        # Exact eight-servo raw pose captured while xArm was at its calibration pose.
        self.calibrated_raw_positions = calibration.raw_positions_rad.copy()
        self.calibrated_logical_positions = calibration.logical_positions_rad.copy()
        # Gravity translation measured at a known folded reference pose during
        # the second calibration capture; keeps RNEA in the verified URDF frame.
        self.gravity_joint_position_offset = (
            calibration.gravity_joint_position_offset_rad.copy()
        )
        self.raw_joint_positions = None
        self.xarm_to_factr_signs = calibration.xarm_to_factr_signs.copy()
        self.leader_arm_publish_origin = derive_factr_home(
            calibration.factr_reference_joint_positions,
            self.xarm_initial_joint_positions,
            calibration.xarm_reference_joint_positions,
            self.xarm_to_factr_signs,
        )
        self.get_logger().info(
            f"已加载 FACTR offset 与参考映射: {self.offset_file}；"
            "重力补偿使用折叠位实测的 URDF 坐标平移，等待 warm-up 自动回位"
        )

    def shut_down(self):
        """De-energize every Dynamixel and close the bus, retrying via port reset.

        Returns:
            ``None``. A failed zero-current or torque-off write reopens the
            serial port and retries; if every attempt fails the operator is
            told loudly to cut the actuator power, because the motors would
            otherwise keep the last commanded torque.
        """

        if self._hardware_closed:
            return
        self._hardware_closed = True
        if not self.driver.emergency_disable():
            self.get_logger().error(
                "FACTR 断电失败！电机可能仍保持力矩，请立即切断 FACTR 电源。"
            )
        self.driver.close()

    def get_leader_joint_states(self):
        """
        Returns the current joint positions and velocities of the leader arm and gripper,
        aligned with the joint conventions (range and direction) of the follower arm.
        """
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()
        self.raw_joint_positions = joint_pos.copy()
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints] - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        joint_vel_arm = (
            joint_vel[0 : self.num_arm_joints] * self.joint_signs[0 : self.num_arm_joints]
        )

        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def set_leader_joint_torque(
        self, arm_torque, gripper_torque, max_current_ma=900.0
    ):
        """
        Apply arm and gripper torque with a symmetric motor-current limit.
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque(
            arm_gripper_torque * self.joint_signs,
            max_current_ma=max_current_ma,
        )

    def update_state_communication(
        self, leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
    ):
        """Allow subclasses to publish measured leader state without control changes."""

    def run_reset_control(
        self, leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
    ):
        """Return true when a subclass applied a reset-specific torque command."""

        return False

    def joint_limit_barrier(
        self, arm_joint_pos, arm_joint_vel, gripper_joint_pos, gripper_joint_vel
    ):
        """
        Computes joint limit repulsive torque to prevent the leader arm and gripper from
        exceeding the physical joint limits of the follower arm.

        This method implements a simplified control law compared to the one described in
        Section IX.B of the paper, while achieving the same protective effect. It applies
        repulsive torques proportional to the distance from the joint limits and the joint
        velocity when limits are approached or exceeded.
        """
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_min_mask

        if not self.enable_gripper_limit_barrier:
            tau_l_gripper = 0.0
        elif gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max)
                - self.joint_limit_kd * gripper_joint_vel
            )
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min)
                - self.joint_limit_kd * gripper_joint_vel
            )
        else:
            tau_l_gripper = 0.0
        return tau_l, tau_l_gripper

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """Compute gravity torque in the FACTR URDF joint-coordinate frame.

        Args:
            arm_joint_pos: xArm-aligned FACTR positions, array shape ``(7,)`` in rad.
            arm_joint_vel: FACTR velocities, array shape ``(7,)`` in rad/s.

        Returns:
            Gravity compensation torque, float array shape ``(7,)`` in N·m,
            scaled by ``gravity_comp.gain``. Position coordinates are translated
            into the measured URDF frame before Pinocchio RNEA.
        """
        pin_joint_pos = np.zeros(self.pin_model.nq)
        pin_joint_vel = np.zeros(self.pin_model.nv)
        pin_joint_acc = np.zeros(self.pin_model.nv)

        pin_joint_pos[: self.num_arm_joints] = (
            np.asarray(arm_joint_pos, dtype=float)
            + self.gravity_joint_position_offset
        )
        pin_joint_vel[: self.num_arm_joints] = arm_joint_vel

        tau_g_full = pin.rnea(
            self.pin_model, self.pin_data, pin_joint_pos, pin_joint_vel, pin_joint_acc
        )
        self.tau_g = tau_g_full[: self.num_arm_joints] * self.gravity_comp_modifier
        return self.tau_g

    def friction_compensation(self, arm_joint_vel):
        """
        Compute joint torques to compensate for static friction during teleoperation.

        This method implements static friction compensation as described in Equation 7,
        Section IX.A of the paper. It omits kinetic friction compensation, which was
        necessary in earlier hardware versions to achieve smooth teleoperation, but has
        since become unnecessary due to hardware improvements, such as weight reduction.
        """
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def null_space_regulation(self, arm_joint_pos, arm_joint_vel):
        """
        Computes joint torques to perform null-space regulation for redundancy resolution
        of the leader arm.

        This method enables the specification of a desired null-space joint configuration
        via `self.null_space_joint_target`. It implements the control strategy described
        in Equation 3 of Section III.B in the paper, projecting a PD control law into
        the null space of the task Jacobian to achieve secondary objectives without
        affecting the primary task.
        """
        if self.null_space_kp == 0.0 and self.null_space_kd == 0.0:
            return np.zeros(self.num_arm_joints)

        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (
            -self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel
        )
        return tau_n

    def torque_feedback(self, external_torque, arm_joint_vel):
        """
        Computes joint torque for the leader arm to achieve force-feedback based on
        the external joint torque from the follower arm.

        This method implements Equation 1 in Section III.A of the paper.
        """
        tau_ff = (
            -1.0 * self.torque_feedback_gain / self.torque_feedback_motor_scalar * external_torque
        )
        tau_ff -= self.torque_feedback_damping * arm_joint_vel
        return tau_ff

    def control_loop_callback(self):
        """
        Runs the main control loop of the leader arm.

        Note that while the control loop can run at up to 500 Hz, lower frequencies
        such as 200 Hz can still yield comparable performance, although they may
        require additional tuning of control parameters. For Dynamixel servos to
        support a 500 Hz control frequency, ensure that the Baud Rate is set to 4 Mbps
        and the Return Delay Time is set to 0 using the Dynamixel Wizard software.
        """
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = (
            self.get_leader_joint_states()
        )
        self.update_state_communication(
            leader_arm_pos,
            leader_arm_vel,
            leader_gripper_pos,
            leader_gripper_vel,
        )
        if self.run_reset_control(
            leader_arm_pos,
            leader_arm_vel,
            leader_gripper_pos,
            leader_gripper_vel,
        ):
            return

        torque_arm = np.zeros(self.num_arm_joints)
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l
        torque_arm += self.null_space_regulation(leader_arm_pos, leader_arm_vel)

        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self.friction_compensation(leader_arm_vel)

        if self.enable_torque_feedback:
            external_joint_torque = self.get_leader_arm_external_joint_torque()
            torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)

        if self.enable_gripper_feedback:
            gripper_feedback = self.get_leader_gripper_feedback()
            torque_gripper += self.gripper_feedback(
                leader_gripper_pos, leader_gripper_vel, gripper_feedback
            )

        self.set_leader_joint_torque(torque_arm, torque_gripper)
        self.update_communication(leader_arm_pos, leader_gripper_pos)

    @abstractmethod
    def set_up_communication(self):
        """
        This method should be implemented to set up communication between the leader arm
        and the follower arm for bilateral teleoperation. This method is called once
        in the __init__ method.

        For example, a subscriber can  be set up to receive external joint torque from
        the leader arm and a publisher can be set up to send joint position target commands
        to the follower arm. Publishers and subscribers can also be set up to record
        the follower arm's joint states

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def get_leader_arm_external_joint_torque(self):
        """
        This method should retrieve the current external joint torque from the follower arm.
        This is used to compute force-feedback in the leader arm. This method is called at
        every iteration of the control loop if self.enable_torque_feedback is set to True.

        Returns:
            np.ndarray: A NumPy array of shape (num_arm_joints,) containing the external
            joint torques.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def get_leader_gripper_feedback(self):
        """
        This method should retrieve any data from the follower gripper that might be required
        to achieve force-feedback in the leader gripper. For example, this method can be used
        to get the current position of the follower gripper for position-position force-feedback
        or the current force of the follower gripper for position-force force-feedback in the
        leader gripper. This method is called at every iteration of the control loop if
        self.enable_gripper_feedback is set to True.

        Returns:
            Any: Feedback data required by the leader gripper. This can be a NumPy array, a
            scalar, or any other data type depending on the implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        """
        Processes feedback data from the follower gripper. This method is intended to compute
        force-feedback for the leader gripper. This method is called at every iteration of the
        control loop if self.enable_gripper_feedback is set to True.

        Args:
            leader_gripper_pos (float): Leader gripper position. Can be used to provide force-
            feedback for the gripper.
            leader_gripper_vel (float): Leader gripper velocity. Can be used to provide force-
            feedback for the gripper.
            gripper_feedback (Any): Feedback data from the gripper. The format can vary depending
            on the implementation, such as a NumPy array, scalar, or custom object.

        Returns:
            float: The computed joint torque value to apply force-feedback to the leader gripper.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    @abstractmethod
    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        """
        This method is intended to be called at every iteration of the control loop to transmit
        relevant data, such as joint position targets, from the leader to the follower arm.

        Args:
            leader_arm_pos (np.ndarray): Joint positions of the leader arm,
                with shape ``(num_arm_joints,)``.
            leader_gripper_pos (np.ndarray): Position of the leader gripper,
                with shape ``(1,)``.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass


def interpolated_pd_torque(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    *,
    kp: float,
    kd: float,
    step: float,
) -> np.ndarray:
    """Return one current-mode PD torque sample toward an interpolated target.

    Implements the upstream FACTR ``set_leader_joint_pos`` law without its
    blocking loop: the target advances from the measured position toward the
    goal by at most ``step`` rad per call, then a PD torque tracks it.

    Args:
        position: Measured joint positions, shape ``(N,)`` in rad.
        velocity: Measured joint velocities, shape ``(N,)`` in rad/s.
        goal: Fixed absolute joint goal, shape ``(N,)`` in rad.
        kp: Proportional gain in N·m/rad.
        kd: Derivative gain in N·m·s/rad.
        step: Maximum per-cycle target advance in rad.

    Returns:
        Torque sample, float array shape ``(N,)`` in N·m. Far joints yield
        ``kp*step*sign(goal - q)`` plus velocity damping; near joints converge
        with plain PD on the remaining error.
    """

    next_target = np.where(
        np.abs(position - goal) > step,
        position + step * np.sign(goal - position),
        goal,
    )
    return -kp * (position - next_target) - kd * velocity
