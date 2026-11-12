"""Control the xArm 7 follower and publish its measured state."""

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32MultiArray, String
from xarm.wrapper import XArmAPI

from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.vr.config import VRConfig
from xarm_ros2_teleop.vr.pose_math import pose_array, pose_message
from xarm_ros2_teleop.xarm.tcp_control import TCPControl, robot_geometry

JOINT_COUNT = 7
JOINT_NAMES = tuple(f"joint{index + 1}" for index in range(JOINT_COUNT))
XARM_NAMESPACE = "/xarm"
USE_RADIANS = True


class XarmNode(Node):
    """Follow leader joint deltas with an xArm 7.

    ROS joint arrays have fixed shape ``(7,)`` and contain radians. Joint speed
    and acceleration limits are configured in radians per second and radians
    per second squared respectively. Controller feedback is read at
    ``state_read_rate_hz`` independently from the motion-command timer.

    Leader samples that jump further than ``max_leader_step_rad`` from the last
    accepted target are dropped as serial glitches while the previous target
    keeps being commanded; rejections sustained longer than
    ``max_leader_reject_sec`` latch a fault that stops following until
    ``LEADER_RESUME_STREAK`` consecutive consistent samples arrive.
    """

    ERROR_RECOVERY_DELAY_NS = 1_000_000_000
    # Consistent leader samples required before following resumes after a fault.
    LEADER_RESUME_STREAK = 5
    # FACTR publishes an exact zero delta after each control-gate origin rebase.
    LEADER_REBASE_TOLERANCE_RAD = 1e-3
    tcp = None
    tcp_mode = False

    def __init__(self):
        """Connect hardware, create ROS interfaces and start the control timer."""

        super().__init__("xarm_node")
        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value)
        system = load_system_config(config_path) if config_path else None
        self.tcp_mode = system is not None and system.teleop.get("source", "factr") == "vr"
        vr_config = VRConfig.from_mapping(system.teleop.get("vr", {})) if self.tcp_mode else None
        if vr_config is not None:
            vr_config.require_workspace()
        self.tcp = None
        self.declare_parameter("ip", "")
        self.declare_parameter("dt", 0.01)
        self.declare_parameter("state_read_rate_hz", 60.0)
        self.declare_parameter("initial_joint_speed_rad_s", 0.5)
        self.declare_parameter("max_joint_speed_rad_s", 1.0)
        self.declare_parameter("max_joint_acceleration_rad_s2", 0.8)
        self.declare_parameter("max_leader_step_rad", 0.2)
        self.declare_parameter("max_policy_step_rad", 0.5)
        self.declare_parameter("max_leader_reject_sec", 0.5)
        self.declare_parameter("leader_watchdog_enabled", False)
        self.declare_parameter("absolute_command_enabled", False)
        self.declare_parameter("leader_command_timeout_sec", 0.25)
        self.declare_parameter("reset_lift_mm", 100.0)
        self.declare_parameter("reset_lift_speed_mm_s", 50.0)
        self.declare_parameter("reset_lift_skip_above_z_mm", 400.0)
        self.declare_parameter(
            "initial_joint_positions",
            [-0.04887, 0.68941, 0.07679, 2.21357, -0.06283, 1.50796, 0.00873],
        )
        self.declare_parameter("reset_enabled", False)
        self.declare_parameter("session_gate_enabled", False)
        self.declare_parameter("recording_gate_enabled", False)
        self.declare_parameter("tcp_rl_enabled", False)
        self.declare_parameter("defer_initial_motion", False)
        self.ip = self.get_parameter("ip").value
        self.dt = self.get_parameter("dt").value
        self.state_read_rate_hz = self.get_parameter("state_read_rate_hz").value
        self.initial_joint_speed = self.get_parameter("initial_joint_speed_rad_s").value
        self.max_joint_speed = self.get_parameter("max_joint_speed_rad_s").value
        self.max_joint_acceleration = self.get_parameter("max_joint_acceleration_rad_s2").value
        self.max_leader_step_rad = self.get_parameter("max_leader_step_rad").value
        self.max_policy_step_rad = self.get_parameter("max_policy_step_rad").value
        self.max_leader_reject_sec = self.get_parameter("max_leader_reject_sec").value
        self.leader_watchdog_enabled = bool(
            self.get_parameter("leader_watchdog_enabled").value
        )
        self.absolute_command_enabled = bool(
            self.get_parameter("absolute_command_enabled").value
        )
        self.leader_command_timeout_sec = float(
            self.get_parameter("leader_command_timeout_sec").value
        )
        self.reset_lift_mm = float(self.get_parameter("reset_lift_mm").value)
        self.reset_lift_speed_mm_s = float(
            self.get_parameter("reset_lift_speed_mm_s").value
        )
        self.reset_lift_skip_above_z_mm = float(
            self.get_parameter("reset_lift_skip_above_z_mm").value
        )
        if self.reset_lift_mm < 0:
            raise ValueError("reset_lift_mm must be non-negative")
        if self.reset_lift_speed_mm_s <= 0:
            raise ValueError("reset_lift_speed_mm_s must be positive")
        if self.reset_lift_skip_above_z_mm <= 0:
            raise ValueError("reset_lift_skip_above_z_mm must be positive")
        # TCP-lift sequencing state for one reset; armed by the reset target.
        self.reset_lift_pending = False
        self.reset_lift_started = False
        self.reset_lift_target_z = None
        self.reset_lift_next_check = 0.0
        self.reset_lift_deadline = None
        self.init_qpos = np.asarray(
            self.get_parameter("initial_joint_positions").value, dtype=float
        )
        self.reset_enabled = bool(self.get_parameter("reset_enabled").value)
        self.session_gate_enabled = bool(self.get_parameter("session_gate_enabled").value)
        self.recording_gate_enabled = bool(
            self.get_parameter("recording_gate_enabled").value
        )
        self.defer_initial_motion = bool(self.get_parameter("defer_initial_motion").value)
        self.session_ready = not self.session_gate_enabled
        self.episode_active = not self.recording_gate_enabled
        self._validate_motion_parameters()
        self.get_logger().info(f"正在连接,ip:{self.ip}")
        self.target_arm_pos = self.init_qpos.copy()
        self.reset_target_arm_pos = None
        self.reset_active = False
        self.awaiting_fresh_leader = False
        self.requires_reset_after_error = False
        # Leader glitch-guard state. ``last_accepted_leader_target`` is the last
        # consistent absolute target in radians (``None`` accepts the next sample
        # unconditionally), ``leader_reject_started`` the monotonic time of the
        # first rejection in the current streak, and the fault flag plus accept
        # streak gate the return to normal following.
        self.last_accepted_leader_target = None
        self.leader_reject_started = None
        self.leader_fault_active = False
        self.leader_accept_streak = 0
        self.rejection_source = "leader"
        self.factr_connected = not self.leader_watchdog_enabled
        self.last_leader_command_time = None
        self.leader_stale = False
        # Monotonic deadline in nanoseconds. Commands pause during error recovery,
        # while ROS callbacks continue recording the newest leader targets.
        self.error_recovery_deadline_ns = None
        # ``None`` pauses motion until the first successful controller-status read.
        self.latest_error_code = None
        self.last_command_error_code = 0
        # 初始化
        self.arm = XArmAPI(self.ip, is_radian=USE_RADIANS, do_not_open=True)
        self.connect_and_initarm()
        if self.tcp_mode:
            try:
                self.geometry = robot_geometry(self.arm)
                self.tcp = TCPControl(self.arm, vr_config)
            except Exception:
                # Initialization failure must release the hardware connection;
                # no unverified TCP geometry is permitted to arm motion.
                self.arm.set_state(4)
                self.arm.disconnect()
                raise
        self.arm_joint_pub = self.create_publisher(
            JointState, f"{XARM_NAMESPACE}/arm_joint_state", 10
        )
        self.commanded_joint_pub = self.create_publisher(
            JointState, f"{XARM_NAMESPACE}/commanded_joint_state", 10
        )
        self.error_warn_pub = self.create_publisher(
            Int32MultiArray, f"{XARM_NAMESPACE}/error_warn", 10
        )
        self.control_ok_pub = self.create_publisher(
            Bool, f"{XARM_NAMESPACE}/control_ok", 10
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.leader_stale_pub = self.create_publisher(
            Bool, f"{XARM_NAMESPACE}/leader_stale", latched_qos
        )
        if self.tcp_mode:
            self.tcp_state_pub = self.create_publisher(PoseStamped, "/xarm/tcp_pose", 10)
            self.tcp_command_pub = self.create_publisher(
                PoseStamped, "/xarm/commanded_tcp_pose", 10
            )
            self.geometry_pub = self.create_publisher(String, "/xarm/geometry", latched_qos)
            self.geometry_pub.publish(String(data=json.dumps(self.geometry)))
            self.tcp_status_pub = self.create_publisher(String, "/xarm/tcp_status", latched_qos)
            self.tcp_event_pub = self.create_publisher(String, "/xarm/tcp_command_event", 100)
            self.create_subscription(PoseStamped, "/xarm/tcp_target", self.tcp_callback, 1)
            self.create_subscription(
                Bool, "/xarm/tcp_following" if self.get_parameter("tcp_rl_enabled").value
                else "/vr/following", self.tcp_follow_callback, 1,
            )
        self.reset_target_valid_pub = self.create_publisher(
            Bool, "/teleop/reset/xarm_target_valid", 10
        )
        self.arm_pos_sub = self.create_subscription(
            JointState, f"{XARM_NAMESPACE}/arm_pos", self.arm_callback, 10
        )
        self.absolute_target_sub = self.create_subscription(
            JointState,
            f"{XARM_NAMESPACE}/absolute_target",
            self.absolute_target_callback,
            10,
        )
        self.reset_active_sub = self.create_subscription(
            Bool, "/teleop/reset/active", self.reset_active_callback, 10
        )
        self.reset_target_sub = self.create_subscription(
            JointState, "/teleop/reset/xarm_target", self.reset_target_callback, 10
        )
        self.session_ready_sub = self.create_subscription(
            Bool, "/system/session/ready", self.session_ready_callback, 10
        )
        self.factr_connected_sub = self.create_subscription(
            Bool,
            "/factr/connected",
            self.factr_connected_callback,
            latched_qos,
        )
        self.episode_active_sub = self.create_subscription(
            Bool,
            "/rl/episode_active" if self.get_parameter("tcp_rl_enabled").value
            else "/teleop/episode_active",
            self.episode_active_callback,
            latched_qos,
        )
        self.state_timer = self.create_timer(1.0 / self.state_read_rate_hz, self.state_read_loop)
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self._publish_leader_stale(False)

    def _validate_motion_parameters(self):
        """Validate xArm timing, speed and acceleration scalar limits.

        Returns:
            Nothing. Non-positive seconds, hertz, rad/s or rad/s² values
            raise ``ValueError`` before the robot connection is opened.
        """

        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.state_read_rate_hz <= 0:
            raise ValueError("state_read_rate_hz must be positive")
        if self.initial_joint_speed <= 0:
            raise ValueError("initial_joint_speed_rad_s must be positive")
        if self.max_joint_speed <= 0:
            raise ValueError("max_joint_speed_rad_s must be positive")
        if self.max_joint_acceleration <= 0:
            raise ValueError("max_joint_acceleration_rad_s2 must be positive")
        if self.max_leader_step_rad <= 0:
            raise ValueError("max_leader_step_rad must be positive")
        if self.max_policy_step_rad <= 0:
            raise ValueError("max_policy_step_rad must be positive")
        if self.max_leader_reject_sec <= 0:
            raise ValueError("max_leader_reject_sec must be positive")
        if self.leader_command_timeout_sec <= 0:
            raise ValueError("leader_command_timeout_sec must be positive")
        if self.init_qpos.shape != (JOINT_COUNT,) or not np.all(np.isfinite(self.init_qpos)):
            raise ValueError("initial_joint_positions must contain finite values for every joint")

    def connect_and_initarm(self):
        """Connect the robot and optionally defer motion until warm-up reset."""

        self.arm.connect()
        if self.arm.connected:
            self.get_logger().info("连接成功")
        else:
            raise ConnectionError(f"xArm 连接失败: {self.ip}")
        self._require_sdk_success("motion_enable", self.arm.motion_enable(True))
        self._require_sdk_success("set_mode(0)", self.arm.set_mode(0))
        self._require_sdk_success("set_state(0)", self.arm.set_state(0))
        if not self.defer_initial_motion:
            self._require_sdk_success(
                "初始化姿态运动",
                self.arm.set_servo_angle(
                    angle=self.init_qpos.tolist(),
                    speed=self.initial_joint_speed,
                    is_radian=USE_RADIANS,
                    wait=True,
                ),
            )
        initial_mode = 0 if self.tcp_mode else 6
        self._require_sdk_success("initial control mode", self.arm.set_mode(initial_mode))
        self._require_sdk_success("set_state(0)", self.arm.set_state(0))
        if self.defer_initial_motion:
            self.get_logger().info("机械臂已连接，等待 warm-up reset 后再移动")
        else:
            self.get_logger().info("机械臂已到达初始化姿态，等待 leader 指令")

    def _require_sdk_success(self, operation, code):
        """Raise when one scalar xArm SDK operation does not return success.

        Args:
            operation: Human-readable physical operation name as a string.
            code: Integer xArm SDK API return code; zero means success.

        Returns:
            Nothing. A nonzero result raises ``RuntimeError`` and prevents later
            initialization operations from issuing additional motion commands.
        """

        if code != 0:
            raise RuntimeError(f"xArm {operation} 失败, SDK code={code}")

    def arm_callback(self, msg):
        """Apply one glitch-guarded ``JointState.position`` delta, shape ``(7,)`` radians.

        A sample whose largest joint step from the last accepted target exceeds
        ``max_leader_step_rad`` (or that contains non-finite values) is dropped;
        ``control_loop`` keeps sending the previously accepted target meanwhile.
        """

        if self.reset_active or not self.episode_active or self.absolute_command_enabled:
            return
        if self.leader_watchdog_enabled and not self.factr_connected:
            return
        if self.tcp_mode:
            return
        if len(msg.position) < JOINT_COUNT:
            self.get_logger().warn("关节数不匹配")
            return
        angles = np.array(msg.position[:JOINT_COUNT])
        if not np.all(np.isfinite(angles)):
            self._register_leader_rejection()
            return
        if self.awaiting_fresh_leader and float(np.max(np.abs(angles))) > (
            self.LEADER_REBASE_TOLERANCE_RAD
        ):
            # The FACTR and xArm receive the latched activation independently.
            # Ignore any queued pre-activation delta until the leader confirms
            # its new origin with a zero command, avoiding a one-frame jump.
            return
        candidate = self.init_qpos + angles
        if self.last_accepted_leader_target is not None:
            step = float(np.max(np.abs(candidate - self.last_accepted_leader_target)))
            if step > self.max_leader_step_rad:
                self._register_leader_rejection(step)
                return
        self._accept_leader_target(candidate)

    def absolute_target_callback(self, msg: JointState) -> None:
        """Accept one policy/RL absolute seven-joint target in radians.

        Args:
            msg: ROS ``JointState`` whose ``position`` contains absolute xArm
                joint angles with shape ``(7,)`` in radians.

        Returns:
            ``None``. Targets are ignored during reset, outside an active control
            episode, or while the node is configured for FACTR teleoperation.
        """

        if (self.tcp_mode or self.reset_active or not self.episode_active
                or not self.absolute_command_enabled):
            return
        if len(msg.position) < JOINT_COUNT:
            self.get_logger().warn("绝对关节目标数量不匹配")
            return
        candidate = np.asarray(msg.position[:JOINT_COUNT], dtype=float)
        if not np.all(np.isfinite(candidate)):
            self._register_leader_rejection(
                source="policy", max_step_rad=self.max_policy_step_rad
            )
            return
        if self.last_accepted_leader_target is not None:
            step = float(np.max(np.abs(candidate - self.last_accepted_leader_target)))
            if step > self.max_policy_step_rad:
                self._register_leader_rejection(
                    step,
                    source="policy",
                    max_step_rad=self.max_policy_step_rad,
                )
                return
        self._accept_leader_target(candidate)

    def _register_leader_rejection(
        self,
        step: float | None = None,
        *,
        source: str = "leader",
        max_step_rad: float | None = None,
    ) -> None:
        """Record one dropped command sample and latch a fault when drops persist.

        Args:
            step: Optional joint-space step magnitude of the dropped sample in
                radians; ``None`` marks a non-finite sample.
            source: Human-readable command source used in diagnostics.
            max_step_rad: Source-specific discontinuity limit in radians. ``None``
                selects the FACTR leader limit.
        """

        self.leader_accept_streak = 0
        self.rejection_source = str(source)
        step_limit = (
            self.max_leader_step_rad if max_step_rad is None else float(max_step_rad)
        )
        now = time.monotonic()
        if self.leader_reject_started is None:
            self.leader_reject_started = now
            reason = (
                f"{source} 目标跳变 {step:.3f} rad 超过上限 {step_limit} rad"
                if step is not None
                else f"{source} 关节数值非有限"
            )
            self.get_logger().warn(f"{reason}，丢弃该帧并维持上一目标")
        elif (
            not self.leader_fault_active
            and now - self.leader_reject_started > self.max_leader_reject_sec
        ):
            self.leader_fault_active = True
            self.get_logger().error(
                f"{source} 目标持续异常超过 {self.max_leader_reject_sec} s，停止跟随"
            )

    def _accept_leader_target(self, candidate: np.ndarray) -> None:
        """Store one consistent leader target and resume following after faults.

        Args:
            candidate: Absolute joint target in radians with shape ``(7,)``.
        """

        self.last_accepted_leader_target = candidate
        self.leader_reject_started = None
        if self.leader_fault_active:
            self.leader_accept_streak += 1
            if self.leader_accept_streak < self.LEADER_RESUME_STREAK:
                return
            self.leader_fault_active = False
            self.leader_accept_streak = 0
            self.get_logger().info(f"{self.rejection_source} 目标恢复一致，重新开始跟随")
        self.target_arm_pos = candidate
        self.awaiting_fresh_leader = False
        self.last_leader_command_time = time.monotonic()
        if self.leader_stale:
            if not self._switch_motion_mode(6, "FACTR 恢复后重新启用关节跟随模式"):
                return
            self.leader_stale = False
            self._publish_leader_stale(False)

    def factr_connected_callback(self, message: Bool) -> None:
        """Store FACTR health and immediately hold on a live disconnection.

        Args:
            message: Latched scalar health from the FACTR serial controller.

        Returns:
            ``None``. Recovery still requires a newly rebased leader command;
            a health ``True`` alone never resumes follower motion.
        """

        self.factr_connected = bool(message.data)
        if (
            self.leader_watchdog_enabled
            and not self.factr_connected
            and self.episode_active
            and not self.reset_active
        ):
            self._latch_leader_stale("FACTR 报告串口断开")

    def _latch_leader_stale(self, reason: str) -> None:
        """Stop following the last leader target and publish a fault latch.

        Args:
            reason: Human-readable timeout or hardware-health explanation.

        Returns:
            ``None``. The follower switches once to mode 0 position hold;
            coordinated reset remains allowed and has higher control priority.
        """

        if self.leader_stale:
            return
        self.leader_stale = True
        self.awaiting_fresh_leader = True
        self.last_accepted_leader_target = None
        self._switch_motion_mode(0, "leader 超时切换位置保持模式")
        self._publish_leader_stale(True)
        self.get_logger().error(f"{reason}，xArm 已保持当前位置并请求 episode 复位")

    def _publish_leader_stale(self, stale: bool) -> None:
        """Publish one latched scalar leader-watchdog state."""

        message = Bool()
        message.data = bool(stale)
        self.leader_stale_pub.publish(message)

    def reset_active_callback(self, msg):
        """Switch between reset targets and fresh post-reset leader commands.

        Args:
            msg: ROS ``Bool``; true pauses normal teleoperation or policy commands.

        Returns:
            Nothing. Leaving reset clears buffered targets. FACTR control waits
            for a zero-origin sample; policy control accepts its next command.
        """

        active = bool(msg.data) and self.reset_enabled
        if self.tcp is not None and active != self.reset_active:
            self.tcp.clear()
            if active:
                self._switch_motion_mode(6, "TCP 控制交还共享关节复位")
        if active and not self.reset_active and self.reset_lift_mm > 0:
            # Every actual reset lifts straight up before moving to its joint
            # target. Starting an episode no longer creates a reset at all.
            self.reset_lift_pending = True
            self.reset_lift_started = False
            self.reset_lift_target_z = None
            self.reset_lift_deadline = None
        if self.reset_active and not active:
            if self.reset_lift_pending:
                self._switch_motion_mode(6, "reset 中止后恢复关节在线规划模式")
            completed_reset_target = (
                None
                if self.reset_target_arm_pos is None
                else self.reset_target_arm_pos.copy()
            )
            self.reset_target_arm_pos = None
            if completed_reset_target is not None:
                # Reset ownership ends before the next policy action exists. Keep
                # the normal control loop on the reached reset pose instead of
                # replaying the stale pre-reset ACT target.
                self.target_arm_pos = completed_reset_target
            # FACTR publishes relative deltas and must rebase to a zero origin
            # after reset. Inference/RL uses the independent absolute-target
            # topic, so it accepts the first newly generated policy target.
            self.awaiting_fresh_leader = not self.absolute_command_enabled
            self.requires_reset_after_error = False
            # Teleoperation waits for FACTR's zero-delta origin confirmation;
            # policy control accepts its first newly generated command directly.
            self.last_accepted_leader_target = None
            self.leader_reject_started = None
            if self.recording_gate_enabled and not self.episode_active:
                self._switch_motion_mode(0, "episode 间隔切换位置保持模式")
        if not active:
            # Any reset end (or disabled reset) clears lift sequencing so a
            # stale partial lift never suppresses later joint commands.
            self.reset_lift_pending = False
            self.reset_lift_started = False
            self.reset_lift_target_z = None
            self.reset_lift_deadline = None
            self.leader_fault_active = False
            self.leader_accept_streak = 0
        self.reset_active = active
        if self.tcp is not None and not active and self.episode_active:
            self._switch_motion_mode(7, "复位结束，等待新的 VR 握持")

    def session_ready_callback(self, msg):
        """Gate normal leader following until the warm-up reset has completed."""

        ready = bool(msg.data) or not self.session_gate_enabled
        if ready != self.session_ready:
            self.awaiting_fresh_leader = ready and not self.absolute_command_enabled
        self.session_ready = ready

    def episode_active_callback(self, msg):
        """Enable leader following during staging or active episode capture.

        Args:
            msg: Latched ROS ``Bool`` published by the recorder. ``False``
                covers saving, discarding, reset and the following idle hold.

        Returns:
            ``None``. Deactivation selects mode 0 position hold; activation
            restores mode 6 and waits for the leader's rebased zero sample.
        """

        active = bool(msg.data) or not self.recording_gate_enabled
        if active == self.episode_active:
            return
        self.episode_active = active
        self.awaiting_fresh_leader = active and not self.absolute_command_enabled
        self.last_accepted_leader_target = None
        self.leader_reject_started = None
        self.leader_fault_active = False
        self.leader_accept_streak = 0
        self.last_leader_command_time = time.monotonic() if active else None
        self.leader_stale = False
        self._publish_leader_stale(False)
        if self.reset_active:
            return
        if self.tcp is not None:
            self.tcp.clear()
        mode = (7 if self.tcp_mode else 6) if active else 0
        operation = "episode 开始恢复关节跟随模式" if active else "episode 结束切换位置保持模式"
        self._switch_motion_mode(mode, operation)

    def reset_target_callback(self, msg):
        """Validate and store one absolute seven-axis reset target in radians.

        Args:
            msg: ``JointState`` with position shape ``(7,)`` in radians.

        Returns:
            ``None``. A validation result is published for the coordinator.
        """

        valid = len(msg.position) == JOINT_COUNT
        target = np.asarray(msg.position, dtype=float)
        valid = valid and target.shape == (JOINT_COUNT,) and np.all(np.isfinite(target))
        if valid:
            code, is_limited = self.arm.is_joint_limit(
                target.tolist(), is_radian=USE_RADIANS
            )
            valid = code == 0 and is_limited is not None and not bool(is_limited)
        if valid:
            self.reset_target_arm_pos = target
        else:
            self.reset_target_arm_pos = None
            self.get_logger().error("拒绝非法或超出 xArm 关节限制的 reset 目标")
        response = Bool()
        response.data = bool(valid)
        self.reset_target_valid_pub.publish(response)

    def read_error_and_publish(self):
        """Read and publish the current xArm error and warning codes.

        Returns:
            The integer xArm error code, or ``None`` when the SDK query fails.
            A nonzero value means motion commands must not be sent.
        """

        code, error_warn = self.arm.get_err_warn_code()
        if code != 0:
            self.get_logger().error(f"读取机械臂错误状态失败, code={code}")
            return None
        msg = Int32MultiArray()
        msg.data = [int(error_warn[0]), int(error_warn[1])]
        self.error_warn_pub.publish(msg)
        return int(error_warn[0])

    def recover_from_error(self, error_code):
        """Clear one xArm error and start a one-second command pause.

        Args:
            error_code: Nonzero integer error reported by the xArm controller.

        Returns:
            Nothing. ``error_recovery_deadline_ns`` records the monotonic time
            after which mode 6 may be restored and leader following resumed.
        """

        clear_code = self.arm.clean_error()
        self.requires_reset_after_error = True
        self.error_recovery_deadline_ns = time.monotonic_ns() + self.ERROR_RECOVERY_DELAY_NS
        if clear_code == 0:
            self.get_logger().warn(
                f"机械臂 error={error_code} 已清除，等待 1 秒后回到配置初始位"
            )
            return
        self.get_logger().error(
            f"机械臂 error={error_code} 清除失败, code={clear_code}，1 秒后重试"
        )

    def resume_after_error(self):
        """Restore xArm online-control mode after the recovery delay.

        Returns:
            ``True`` when motion, mode 6 and state 0 were restored; otherwise
            ``False``. On failure another one-second pause is scheduled.
        """

        result_codes = (
            self.arm.motion_enable(True),
            self.arm.set_mode(6),
            self.arm.set_state(0),
        )
        if all(code == 0 for code in result_codes):
            self.error_recovery_deadline_ns = None
            self.get_logger().info("机械臂控制模式已恢复，等待 reset 初始位命令")
            return True
        self.error_recovery_deadline_ns = time.monotonic_ns() + self.ERROR_RECOVERY_DELAY_NS
        self.get_logger().error(f"机械臂恢复控制模式失败, codes={result_codes}，1 秒后重试")
        return False

    def publish_state(self):
        """Publish current xArm state and return whether the joint read succeeded."""

        if not self.arm.connected:
            self.get_logger().error("还没有连接成功")
            return False
        state_code, joint_states = self.arm.get_joint_states(is_radian=USE_RADIANS)
        joint_state_ok = bool(
            state_code == 0
            and joint_states is not None
            and len(joint_states) >= 3
            and all(
                values is not None and len(values) >= JOINT_COUNT
                for values in joint_states[:3]
            )
        )
        if joint_state_ok:
            position = joint_states[0][:JOINT_COUNT]
            velocity = joint_states[1][:JOINT_COUNT]
            effort = joint_states[2][:JOINT_COUNT]
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(JOINT_NAMES)
            msg.position = [float(v) for v in position]
            msg.velocity = [float(v) for v in velocity]
            msg.effort = [float(v) for v in effort]
            self.arm_joint_pub.publish(msg)
        else:
            self.get_logger().error(f"读取 xArm 关节状态失败, code={state_code}")
        return joint_state_ok

    def state_read_loop(self):
        """Read controller status and joint feedback at the configured rate.

        Returns:
            Nothing. ``latest_error_code`` becomes an integer controller error
            code, or ``None`` when the SDK status request times out. Joint and
            state messages contain one-dimensional arrays of seven values.
        """

        self.latest_error_code = self.read_error_and_publish()
        state_ok = self.publish_state()
        if self.tcp is not None:
            if (
                not np.allclose(self.arm.tcp_offset, self.geometry["tcp_offset_mm_rad"])
                or not np.allclose(self.arm.world_offset, self.geometry["world_offset_mm_rad"])
            ):
                if not self.tcp.fault:
                    self.tcp.stop("TCP/world configuration changed; restart with a new contract")
            pose = self.tcp.read_feedback(time.monotonic())
            state_ok = state_ok and pose is not None and not self.tcp.fault
            if pose is not None:
                self.tcp_state_pub.publish(pose_message(pose, self.get_clock().now().to_msg()))
        msg = Bool()
        msg.data = bool(
            state_ok and self.latest_error_code == 0 and self.last_command_error_code == 0
        )
        self.control_ok_pub.publish(msg)

    def control_loop(self):
        """Recover xArm errors or send the active reset/leader target.

        After the non-blocking one-second recovery delay, normal leader commands
        remain blocked until the coordinator completes an initial-pose reset.
        """

        now_ns = time.monotonic_ns()
        error_code = self.latest_error_code
        recovery_expired = (
            self.error_recovery_deadline_ns is not None
            and now_ns >= self.error_recovery_deadline_ns
        )
        if error_code is None:
            return
        if error_code:
            if self.error_recovery_deadline_ns is None or recovery_expired:
                self.recover_from_error(error_code)
            return
        if self.error_recovery_deadline_ns is not None:
            if not recovery_expired or not self.resume_after_error():
                return

        if self.requires_reset_after_error and not self.reset_active:
            return
        if self.tcp is not None and not self.reset_active:
            if self.session_ready and self.episode_active:
                code, pose = self.tcp.command(time.monotonic())
                if code and code != self.last_command_error_code:
                    self.get_logger().error(self.tcp.fault)
                self.last_command_error_code = code
                if pose is not None:
                    stamp = self.get_clock().now()
                    self.tcp_command_pub.publish(
                        pose_message(pose, stamp.to_msg())
                    )
                    self.tcp_event_pub.publish(String(data=json.dumps({
                        "timestamp": stamp.nanoseconds / 1e9,
                        "selection_timestamp": self.tcp.command_selection,
                        "accepted_pose": pose.tolist(),
                    })))
                detail = self.tcp.fault or (
                    "TCP command watchdog expired; release and re-grip"
                    if self.tcp.waiting_release else "ready"
                )
                self.tcp_status_pub.publish(String(data=detail))
            return
        if self.reset_active:
            if self.reset_target_arm_pos is None:
                return
            if self.reset_lift_pending:
                # Lift phase: raise the end effector by reset_lift_mm straight
                # up (TCP space) before any joint motion toward the reset pose.
                self._execute_reset_lift()
                return
            command_target = self.reset_target_arm_pos
        else:
            if not self.session_ready or not self.episode_active:
                return
            if self.leader_watchdog_enabled:
                leader_age = (
                    float("inf")
                    if self.last_leader_command_time is None
                    else time.monotonic() - self.last_leader_command_time
                )
                if not self.factr_connected or leader_age > self.leader_command_timeout_sec:
                    self._latch_leader_stale(
                        f"超过 {self.leader_command_timeout_sec:.2f} 秒未收到有效 FACTR 指令"
                    )
                    return
            if self.awaiting_fresh_leader:
                return
            if self.leader_fault_active:
                # Sustained leader anomaly: stop sending follower commands so the
                # servo loop decelerates at the current pose instead of chasing a
                # bogus target. Reset targets remain allowed during a fault.
                return
            command_target = self.target_arm_pos

        command_code = self.arm.set_servo_angle(
            angle=command_target,
            speed=self.max_joint_speed,
            mvacc=self.max_joint_acceleration,
            is_radian=USE_RADIANS,
            wait=False,
        )
        if command_code != 0 and command_code != self.last_command_error_code:
            self.get_logger().error(f"xArm 跟随指令失败, SDK code={command_code}")
        elif command_code == 0 and self.last_command_error_code != 0:
            self.get_logger().info("xArm 跟随指令已恢复")
        if command_code == 0:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(JOINT_NAMES)
            msg.position = [float(value) for value in command_target]
            self.commanded_joint_pub.publish(msg)
        self.last_command_error_code = command_code

    def tcp_follow_callback(self, message):
        """Refresh the VR deadman gate; reset/idle cannot accept remote ownership."""
        self.tcp.set_following(
            message.data and self.episode_active and self.session_ready and not self.reset_active,
            time.monotonic(),
        )

    def tcp_callback(self, message):
        """Accept a fresh xarm_base PoseStamped [7] metres only outside reset."""
        if self.reset_active or not self.episode_active or not self.session_ready:
            return
        age = (self.get_clock().now().nanoseconds * 1e-9
               - message.header.stamp.sec - message.header.stamp.nanosec * 1e-9)
        if message.header.frame_id != "xarm_base" or not 0 <= age <= self.tcp.config.timeout:
            self.tcp.set_following(False, time.monotonic())
            return
        try:
            self.tcp.accept(
                pose_array(message), time.monotonic(),
                message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            )
        except ValueError as error:
            self.tcp.stop(str(error))
            self.get_logger().error(str(error))

    def _execute_reset_lift(self):
        """Command and supervise the TCP-space straight-up lift of a reset.

        Returns:
            Nothing. The first call reads the current TCP pose, commands a
            ``set_position`` z-rise of ``reset_lift_mm`` and switches to
            completion polling (5 mm tolerance, checked every 200 ms). The arm
            temporarily switches from joint-online mode 6 to position mode 0,
            then returns to mode 6 before joint commands resume. Any SDK failure
            or lift timeout skips the lift so the reset can still proceed.
        """

        now = time.monotonic()
        if not self.reset_lift_started:
            code, pose = self.arm.get_position(is_radian=USE_RADIANS)
            if code != 0 or pose is None or len(pose) < 6:
                self.get_logger().error(
                    f"reset 抬升失败: 读取 TCP 位姿失败 code={code}，跳过抬升"
                )
                self.reset_lift_pending = False
                return
            current_z = float(pose[2])
            if current_z >= self.reset_lift_skip_above_z_mm:
                self.reset_lift_pending = False
                self.get_logger().info(
                    f"reset 跳过抬升: TCP z={current_z:.1f} mm 已达到 "
                    f"{self.reset_lift_skip_above_z_mm:.1f} mm"
                )
                return
            self.reset_lift_target_z = current_z + self.reset_lift_mm
            if not self._switch_motion_mode(0, "reset 抬升切换位置控制模式"):
                self._finish_reset_lift("reset 抬升模式切换失败，跳过抬升", error=True)
                return
            code = self.arm.set_position(
                x=float(pose[0]),
                y=float(pose[1]),
                z=self.reset_lift_target_z,
                roll=float(pose[3]),
                pitch=float(pose[4]),
                yaw=float(pose[5]),
                speed=self.reset_lift_speed_mm_s,
                mvacc=self.reset_lift_speed_mm_s * 5.0,
                is_radian=USE_RADIANS,
                wait=False,
            )
            if code != 0:
                self._finish_reset_lift(
                    f"reset 抬升失败: set_position code={code}，跳过抬升",
                    error=True,
                )
                return
            self.reset_lift_started = True
            self.reset_lift_next_check = now + 0.3
            expected_duration = self.reset_lift_mm / self.reset_lift_speed_mm_s
            self.reset_lift_deadline = now + max(5.0, 3.0 * expected_duration)
            self.get_logger().info(
                f"reset 抬升开始: TCP z {pose[2]:.1f} -> "
                f"{self.reset_lift_target_z:.1f} mm"
            )
            return
        if now < self.reset_lift_next_check:
            return
        self.reset_lift_next_check = now + 0.2
        if self.reset_lift_deadline is not None and now >= self.reset_lift_deadline:
            self._finish_reset_lift("reset 抬升超时，继续执行关节复位", error=True)
            return
        code, pose = self.arm.get_position(is_radian=USE_RADIANS)
        if code != 0 or pose is None or len(pose) < 3:
            return
        if abs(float(pose[2]) - self.reset_lift_target_z) <= 5.0:
            # Reaching the positional tolerance does not guarantee that the
            # Cartesian command has fully stopped. xArm rejects set_mode(6)
            # with SDK code 10 while state 1 still reports active motion.
            state_code, controller_state = self.arm.get_state()
            if state_code != 0 or controller_state == 1:
                return
            self._finish_reset_lift("reset 抬升完成，开始回初始位")

    def _switch_motion_mode(self, mode, operation):
        """Set one xArm control mode and return the controller to ready state.

        Args:
            mode: Integer xArm mode. Reset uses 0 for Cartesian position
                control and 6 for joint online trajectory planning.
            operation: Human-readable operation name used in error logs.

        Returns:
            ``True`` when both SDK calls return code 0, otherwise ``False``.
        """

        mode_code = self.arm.set_mode(int(mode))
        if mode_code != 0:
            self.get_logger().error(f"{operation}失败: set_mode({mode}) code={mode_code}")
            return False
        state_code = self.arm.set_state(0)
        if state_code != 0:
            self.get_logger().error(f"{operation}失败: set_state(0) code={state_code}")
            return False
        return True

    def _finish_reset_lift(self, message, *, error=False):
        """Restore mode 6 and finish one Cartesian reset-lift stage.

        Args:
            message: Human-readable completion or fallback message.
            error: Boolean selecting error-level rather than info-level logging.

        Returns:
            ``True`` after mode 6 is restored and lift state is cleared;
            ``False`` keeps the lift pending so the next control tick retries.
        """

        if not self._switch_motion_mode(6, "reset 抬升后恢复关节在线规划模式"):
            return False
        self.reset_lift_pending = False
        self.reset_lift_started = False
        self.reset_lift_target_z = None
        self.reset_lift_deadline = None
        log = self.get_logger().error if error else self.get_logger().info
        log(message)
        return True

    def destroy_node(self):
        """Disconnect the physical xArm before destroying the ROS node."""

        if self.tcp is not None and self.arm.connected:
            code = self.tcp.stop("VR driver shutdown")
            if code:
                self.get_logger().error(f"VR shutdown stop failed: SDK code={code}")
        self.arm.disconnect()
        super().destroy_node()


def main(args=None):
    """Run the xArm follower; ``args`` is an optional sequence of ROS arguments."""

    rclpy.init(args=args)
    node = None
    try:
        node = XarmNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
