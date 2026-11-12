"""Launch the complete leader, follower, RGB-D and tactile sensor system."""

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.system.sensor_actions import build_sensor_actions
from xarm_ros2_teleop.teleop.offset_calibration import load_offset_calibration
from xarm_ros2_teleop.vr.config import VRConfig


def _launch_setup(context) -> list:
    """Create all system actions after validating the selected YAML.

    Args:
        context: ROS 2 launch context containing the scalar ``config`` path.

    Returns:
        A one-dimensional list of launch actions. Each action starts a robot or
        sensor process; no image arrays are handled by this function.
    """

    config_path = _resolve_config_path(LaunchConfiguration("config").perform(context))
    system = load_system_config(config_path)
    actions = [LogInfo(msg=f"Starting control_robot system '{system.name}'")]
    inference_requested = _launch_bool(
        LaunchConfiguration("inference").perform(context), "inference"
    )
    inference_motion = _launch_bool(
        LaunchConfiguration("inference_motion").perform(context), "inference_motion"
    )
    rl_requested = _launch_bool(LaunchConfiguration("rl").perform(context), "rl")
    rl_motion = _launch_bool(LaunchConfiguration("rl_motion").perform(context), "rl_motion")
    vr = system.teleop.get("source", "factr") == "vr"
    vr_motion = _launch_bool(LaunchConfiguration("vr_motion").perform(context), "vr_motion")
    inference_enabled = bool(system.inference.enabled or inference_requested)
    rl_enabled = bool(system.rl.get("enabled", False) or rl_requested)
    if vr and inference_enabled and not rl_enabled:
        raise ValueError("VR TCP inference currently requires rl:=true")
    if vr and rl_enabled:
        inference_enabled = True
    if vr and (rl_motion if rl_enabled else vr_motion):
        VRConfig.from_mapping(system.teleop.get("vr", {})).require_workspace()
    evaluation_enabled = bool(
        inference_enabled
        and not rl_enabled
        and inference_motion
        and system.evaluation.get("enabled", False)
    )
    operation_mode = "rl" if rl_enabled else "inference" if inference_enabled else "teleop"
    # VR TCP drivers must be gated even in RL (mode 7 is armed per episode).
    # Their gate is /rl/episode_active, never the inactive FACTR teleop gate.
    recording_gate_enabled = system.recording.enabled and (
        operation_mode == "teleop" or (vr and rl_enabled)
    )
    inference_dry_run = not (rl_motion if rl_enabled else inference_motion)
    actuator_motion_enabled = rl_motion if rl_enabled else not (
        inference_enabled and inference_dry_run
    )
    motion_enabled = rl_motion if rl_enabled else inference_motion if inference_enabled else True
    if vr and not rl_enabled:
        actuator_motion_enabled = motion_enabled = vr_motion
    reset_enabled = bool(system.reset.get("enabled", False))
    warmup_active = bool(system.warmup.enabled and not inference_enabled and not rl_enabled)
    if vr and not vr_motion:
        warmup_active = False
    initial_joint_positions = [
        float(value) for value in system.xarm.get("initial_joint_positions", [])
    ]
    reset_parameters = {
        "action_name": str(system.reset.get("action_name", "/teleop/reset")),
        "timeout_sec": float(system.reset.get("timeout_sec", 15.0)),
        "state_timeout_sec": float(system.reset.get("state_timeout_sec", 0.5)),
        "stable_time_sec": float(system.reset.get("stable_time_sec", 0.3)),
        "xarm_position_tolerance_rad": float(system.reset.get("xarm_position_tolerance_rad", 0.03)),
        "xarm_velocity_tolerance_rad_s": float(
            system.reset.get("xarm_velocity_tolerance_rad_s", 0.05)
        ),
        "robotiq_open_threshold_bits": int(system.reset.get("robotiq_open_threshold_bits", 5)),
        "initial_joint_positions": initial_joint_positions,
        "require_factr_alignment": bool(system.teleop.get("enabled", True))
        and not evaluation_enabled and not vr,
    }

    if (system.recording.enabled or warmup_active) and system.speech.enabled:
        speech_node = Node(
            package="xarm_ros2_teleop",
            executable="speech_announcer",
            name="speech_announcer",
            parameters=[{"config_path": str(config_path)}],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=speech_node,
                        on_exit=[
                            LogInfo(msg="Speech announcer exited; workflow continues silently")
                        ],
                    )
                ),
                speech_node,
            ]
        )

    if (not vr and bool(system.teleop.get("enabled", True))
            and (not inference_enabled or rl_enabled)):
        config_file = str(system.teleop.get("config_file", "gello_grav_comp_demo.yaml"))
        teleop_config_path = _resolve_teleop_config_path(config_file)
        factr_config = _validate_factr_device(teleop_config_path)
        offset_path = Path(str(system.teleop.get("offset_file", ""))).expanduser()
        if not offset_path.is_file():
            raise ValueError(
                f"FACTR offset does not exist: {offset_path}. "
                "Run 'ros2 run xarm_ros2_teleop factr_calibrate' first."
            )
        dynamixel = factr_config["dynamixel"]
        calibration_pose = factr_config["arm_teleop"]["initialization"]["start_joints"]
        load_offset_calibration(
            offset_path,
            dynamixel_port=str(dynamixel["dynamixel_port"]),
            servo_types=list(dynamixel["servo_types"]),
            joint_signs=np.asarray(dynamixel["joint_signs"], dtype=float),
            calibration_joint_positions=np.asarray(calibration_pose[:7], dtype=float),
            operating_mode=0,
            xarm_to_factr_signs=np.asarray(system.teleop["xarm_to_factr_signs"], dtype=float),
        )
        teleop_node = Node(
            package="xarm_ros2_teleop",
            executable="factr_teleop_node",
            name="factr_teleop",
            parameters=[
                {
                    "config_file": config_file,
                    "offset_file": str(offset_path),
                    "xarm_to_factr_signs": [
                        float(value) for value in system.teleop["xarm_to_factr_signs"]
                    ],
                    "reset_enabled": reset_enabled,
                    "session_gate_enabled": warmup_active,
                    "xarm_initial_joint_positions": initial_joint_positions,
                    "reset_state_publish_rate_hz": float(
                        system.reset.get("factr_state_publish_rate_hz", 60.0)
                    ),
                    "reset_max_current_ma": float(system.reset.get("factr_max_current_ma", 150.0)),
                    "reconnect_period_sec": float(
                        system.teleop.get("reconnect_period_sec", 1.0)
                    ),
                    "recovery_successful_reads": int(
                        system.teleop.get("recovery_successful_reads", 5)
                    ),
                    "arm_command_topic": (
                        str(system.rl.get("human_arm_topic", "/control/factr_arm"))
                        if rl_enabled
                        else "/xarm/arm_pos"
                    ),
                    "gripper_command_topic": (
                        str(system.rl.get("human_gripper_topic", "/control/factr_gripper"))
                        if rl_enabled
                        else "/xarm/gripper_pos"
                    ),
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=teleop_node,
                        on_exit=[EmitEvent(event=Shutdown(reason="FACTR leader process exited"))],
                    )
                ),
                teleop_node,
            ]
        )

    if vr:
        vr_options = system.teleop.get("vr", {})
        bridge = Node(
            package="xarm_ros2_teleop", executable="quest_pose_node", name="quest_pose",
            parameters=[{"device_serial": str(vr_options.get("device_serial", "")),
                         "tcp_port": int(vr_options.get("tcp_port", 7447))}],
            prefix=[sys.executable], output="screen",
        )
        adapter = Node(
            package="xarm_ros2_teleop", executable="vr_teleop", name="vr_teleop",
            parameters=[{"config_path": str(config_path), "motion_enabled": motion_enabled,
                         "rl_enabled": rl_enabled}],
            prefix=[sys.executable], output="screen",
        )
        actions.extend(_critical_node_actions(bridge, "Quest bridge process exited"))
        actions.extend(_critical_node_actions(adapter, "VR teleop process exited"))
        actions.append(Node(
            package="rviz2", executable="rviz2", name="vr_tcp_preview",
            arguments=["-d", str(Path(get_package_share_directory("xarm_ros2_teleop"))
                                  / "rviz/vr_teleop.rviz")],
            condition=IfCondition(LaunchConfiguration("vr_rviz")), output="screen",
        ))

    if bool(system.xarm.get("enabled", True)) and actuator_motion_enabled:
        xarm_node = Node(
            package="xarm_ros2_teleop",
            executable="xarm_node",
            name="xarm_node",
            parameters=[
                {
                    "config_path": str(config_path) if vr else "",
                    "tcp_rl_enabled": vr and rl_enabled,
                    "ip": str(system.xarm.get("ip", "")),
                    "dt": float(system.xarm.get("control_period", 0.01)),
                    "state_read_rate_hz": float(system.xarm.get("state_read_rate_hz", 60.0)),
                    "initial_joint_speed_rad_s": float(
                        system.xarm.get("initial_joint_speed_rad_s", 0.5)
                    ),
                    "max_joint_speed_rad_s": float(system.xarm.get("max_joint_speed_rad_s", 1.0)),
                    "max_joint_acceleration_rad_s2": float(
                        system.xarm.get("max_joint_acceleration_rad_s2", 0.8)
                    ),
                    "max_leader_step_rad": float(system.xarm.get("max_leader_step_rad", 0.2)),
                    "max_policy_step_rad": float(system.inference.max_joint_step_rad),
                    "max_leader_reject_sec": float(system.xarm.get("max_leader_reject_sec", 0.5)),
                    "leader_watchdog_enabled": bool(
                        not vr and not inference_enabled and not rl_enabled
                    ),
                    "absolute_command_enabled": bool(inference_enabled or rl_enabled),
                    "leader_command_timeout_sec": float(
                        system.xarm.get("leader_command_timeout_sec", 0.25)
                    ),
                    "reset_lift_mm": float(system.xarm.get("reset_lift_mm", 100.0)),
                    "reset_lift_speed_mm_s": float(system.xarm.get("reset_lift_speed_mm_s", 50.0)),
                    "reset_lift_skip_above_z_mm": float(
                        system.xarm.get("reset_lift_skip_above_z_mm", 400.0)
                    ),
                    "initial_joint_positions": initial_joint_positions,
                    "reset_enabled": reset_enabled,
                    "session_gate_enabled": warmup_active,
                    "recording_gate_enabled": recording_gate_enabled,
                    "defer_initial_motion": warmup_active or vr,
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(_critical_node_actions(xarm_node, "xArm process exited"))

    if bool(system.robotiq.get("enabled", False)) and actuator_motion_enabled:
        robotiq_node = Node(
            package="xarm_ros2_teleop",
            executable="robotiq_node",
            name="robotiq_node",
            parameters=[
                {
                    "com_port": str(system.robotiq.get("com_port", "/dev/robotiq_rs485")),
                    "device_id": int(system.robotiq.get("device_id", 9)),
                    "debug": bool(system.robotiq.get("debug", False)),
                    "command_topic": str(system.robotiq.get("command_topic", "/xarm/gripper_pos")),
                    "control_period": float(system.robotiq.get("control_period", 0.05)),
                    "speed": int(system.robotiq.get("speed", 255)),
                    "force": int(system.robotiq.get("force", 100)),
                    "minimal_motion": int(system.robotiq.get("minimal_motion", 2)),
                    "activate_on_startup": bool(system.robotiq.get("activate_on_startup", True)),
                    "activation_timeout_sec": float(
                        system.robotiq.get("activation_timeout_sec", 15.0)
                    ),
                    "reset_enabled": reset_enabled,
                    "session_gate_enabled": warmup_active,
                    "recording_gate_enabled": recording_gate_enabled,
                }
            ],
            prefix=[sys.executable],
            remappings=[("/teleop/episode_active", "/rl/episode_active")]
            if vr and rl_enabled else [],
            output="screen",
        )
        actions.extend(_critical_node_actions(robotiq_node, "Robotiq process exited"))

    if reset_enabled and actuator_motion_enabled and (
        not inference_enabled or rl_enabled or evaluation_enabled
    ):
        reset_node = Node(
            package="xarm_ros2_teleop",
            executable="reset_node",
            name="reset_coordinator",
            parameters=[reset_parameters],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(_critical_node_actions(reset_node, "Reset coordinator exited"))

    if system.dashboard.enabled or warmup_active:
        monitor_node = Node(
            package="xarm_ros2_teleop",
            executable="system_monitor",
            name="system_monitor",
            parameters=[
                {
                    "config_path": str(config_path),
                    "operation_mode": operation_mode,
                    "motion_enabled": motion_enabled,
                    "inference_enabled": inference_enabled,
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=monitor_node,
                        on_exit=[LogInfo(msg="System monitor exited; robot processes continue")],
                    )
                ),
                monitor_node,
            ]
        )

    # Sensor modules own delayed connection and recovery. The unified monitor
    # observes all configured outputs and is the only readiness authority.
    actions.extend(build_sensor_actions(system))
    if inference_enabled:
        inference_node = Node(
            package="xarm_ros2_teleop",
            executable="remote_policy_inference",
            name="remote_policy_inference",
            parameters=[
                {
                    "config_path": str(config_path),
                    "dry_run": inference_dry_run,
                    "direct_robot_commands": not rl_enabled,
                    "episode_gated": evaluation_enabled or rl_enabled,
                    "episode_active_topic": (
                        "/rl/episode_active" if rl_enabled else "/evaluation/episode_active"
                    ),
                    "unlimited_chunks": rl_enabled,
                    "policy_action_topic": str(
                        system.rl.get("policy_action_topic", "/control/policy_action")
                    ),
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(_critical_node_actions(inference_node, "Remote policy process exited"))
    if rl_enabled:
        intervention_node = Node(
            package="xarm_ros2_teleop",
            executable="human_intervention",
            name="human_intervention",
            parameters=[
                {
                    "config_path": str(config_path),
                    "motion_enabled": rl_motion,
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            _critical_node_actions(intervention_node, "Human intervention process exited")
        )
    recorder_enabled = system.recording.enabled and (not vr or actuator_motion_enabled) and (
        (not inference_enabled and not rl_enabled)
        or (evaluation_enabled and actuator_motion_enabled)
        or (rl_enabled and actuator_motion_enabled)
    )
    if recorder_enabled:
        recorder_node = Node(
            package="xarm_ros2_teleop",
            executable="rl_episode_recorder" if rl_enabled else "episode_recorder",
            name="episode_recorder",
            parameters=[
                {
                    "config_path": str(config_path),
                    "operation_mode": (
                        "rl" if rl_enabled else "evaluation" if evaluation_enabled else "teleop"
                    ),
                    "require_remote_policy": inference_enabled,
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=recorder_node,
                        on_exit=[
                            EmitEvent(event=Shutdown(reason="Episode recorder process exited"))
                        ],
                    )
                ),
                recorder_node,
            ]
        )
    if warmup_active:
        session_node = Node(
            package="xarm_ros2_teleop",
            executable="session_coordinator",
            name="session_coordinator",
            parameters=[{"config_path": str(config_path)}],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=session_node,
                        on_exit=[EmitEvent(event=Shutdown(reason="Warm-up coordinator exited"))],
                    )
                ),
                session_node,
            ]
        )
    if system.dashboard.enabled:
        if system.dashboard.rerun.enabled:
            rerun_node = Node(
                package="xarm_ros2_teleop",
                executable="rerun_visualizer",
                name="rerun_visualizer",
                parameters=[
                    {
                        "config_path": str(config_path),
                        "operation_mode": operation_mode,
                    }
                ],
                prefix=[sys.executable],
                output="screen",
            )
            actions.extend(
                [
                    RegisterEventHandler(
                        OnProcessExit(
                            target_action=rerun_node,
                            on_exit=[LogInfo(msg="Rerun visualizer exited; recording continues")],
                        )
                    ),
                    rerun_node,
                ]
            )
        dashboard_node = Node(
            package="xarm_ros2_teleop",
            executable="dashboard_server",
            name="dashboard_server",
            parameters=[
                {
                    "config_path": str(config_path),
                    "operation_mode": operation_mode,
                    "motion_enabled": motion_enabled,
                    "inference_enabled": inference_enabled,
                }
            ],
            prefix=[sys.executable],
            output="screen",
        )
        actions.extend(
            [
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=dashboard_node,
                        on_exit=[LogInfo(msg="Dashboard exited; recording continues")],
                    )
                ),
                dashboard_node,
            ]
        )
    return actions


def _critical_node_actions(node: Node, reason: str) -> list:
    """Return one node and a launch-wide shutdown handler for its exit."""

    return [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[EmitEvent(event=Shutdown(reason=reason))],
            )
        ),
        node,
    ]


def _launch_bool(value: str, name: str) -> bool:
    """Parse one explicit launch boolean used to arm inference and motion."""

    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


def _resolve_config_path(config_value: str) -> Path:
    """Resolve an absolute path or a path relative to the installed package.

    Args:
        config_value: User-provided UTF-8 YAML path as a string. Relative paths,
            such as ``config/teleop_system.yaml``, are interpreted from the
            ``xarm_ros2_teleop`` package share directory.

    Returns:
        Absolute ``Path`` to the requested YAML file; no tensor data is produced.
    """

    config_path = Path(config_value).expanduser()
    if config_path.is_absolute():
        return config_path
    package_share = Path(get_package_share_directory("xarm_ros2_teleop"))
    return package_share / config_path


def _resolve_teleop_config_path(config_value: str) -> Path:
    """Resolve one FACTR YAML path from a configured absolute path or filename."""

    config_path = Path(config_value).expanduser()
    if config_path.is_absolute():
        return config_path
    package_share = Path(get_package_share_directory("xarm_ros2_teleop"))
    return package_share / "config" / config_path


def _validate_factr_device(config_path: Path) -> Mapping:
    """Fail before actuator startup when the configured FACTR serial port is absent.

    Args:
        config_path: FACTR YAML path containing a scalar Dynamixel device name.

    Returns:
        Parsed FACTR YAML mapping. Missing mappings or serial paths raise before
        any ROS hardware process is created.
    """

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping) or not isinstance(config.get("dynamixel"), Mapping):
        raise ValueError(f"Invalid FACTR configuration: {config_path}")
    port_value = config["dynamixel"].get("dynamixel_port")
    if not isinstance(port_value, str) or not port_value.strip():
        raise ValueError("dynamixel.dynamixel_port must be a non-empty string")
    device_path = Path(port_value).expanduser()
    if not device_path.is_absolute():
        device_path = Path("/dev/serial/by-id") / device_path
    if not device_path.exists():
        raise FileNotFoundError(
            f"FACTR serial device is not connected: {device_path}. "
            "No xArm or Robotiq process was started."
        )
    return config


def generate_launch_description() -> LaunchDescription:
    """Return the only public launch description for the complete system."""

    default_config = (
        Path(get_package_share_directory("xarm_ros2_teleop")) / "config" / "teleop_system.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(default_config),
                description="Absolute path or package-relative path to the unified system YAML",
            ),
            DeclareLaunchArgument(
                "inference",
                default_value="false",
                description="Start remote policy inference instead of FACTR teleoperation",
            ),
            DeclareLaunchArgument(
                "inference_motion",
                default_value="false",
                description="Explicitly allow policy targets to reach xArm and Robotiq",
            ),
            DeclareLaunchArgument(
                "rl",
                default_value="false",
                description="Start the online-RL human-intervention action arbiter",
            ),
            DeclareLaunchArgument(
                "rl_motion",
                default_value="false",
                description="Explicitly allow arbiter targets to reach xArm and Robotiq",
            ),
            DeclareLaunchArgument(
                "vr_motion", default_value="false",
                description="Explicitly enable native TCP actuators for the VR source",
            ),
            DeclareLaunchArgument("vr_rviz", default_value="false"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
