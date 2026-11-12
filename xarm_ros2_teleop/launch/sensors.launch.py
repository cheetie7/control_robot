"""Launch all configured cameras and tactile sensors without any actuator."""

import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.system.sensor_actions import build_sensor_actions


def _launch_setup(context) -> list:
    """Create sensor, diagnostics and local visualization actions only.

    Args:
        context: ROS launch context containing the sensor-only YAML path.

    Returns:
        One-dimensional launch-action list. No actuator, reset, recorder,
        inference or RL process is constructed by this launch file.
    """

    config_path = _resolve_config_path(LaunchConfiguration("config").perform(context))
    system = load_system_config(config_path)
    if any(
        (
            bool(system.teleop.get("enabled", True)),
            bool(system.xarm.get("enabled", True)),
            bool(system.robotiq.get("enabled", False)),
            bool(system.reset.get("enabled", False)),
            system.recording.enabled,
            system.inference.enabled,
            bool(system.rl.get("enabled", False)),
            system.warmup.enabled,
        )
    ):
        raise ValueError(
            "sensors.launch.py requires teleop, xarm, robotiq, reset, recording, "
            "inference, rl and warmup to be disabled"
        )

    enabled_count = sum(camera.enabled for camera in system.cameras) + sum(
        device.enabled for device in system.tactile_devices
    )
    actions = [
        LogInfo(
            msg=(
                f"Starting {enabled_count} sensors from '{system.name}' "
                "(sensor-only: actuator motion is impossible)"
            )
        )
    ]
    monitor_node = Node(
        package="xarm_ros2_teleop",
        executable="system_monitor",
        name="system_monitor",
        parameters=[
            {
                "config_path": str(config_path),
                "operation_mode": "teleop",
                "motion_enabled": False,
                "inference_enabled": False,
            }
        ],
        prefix=[sys.executable],
        output="screen",
    )
    actions.append(monitor_node)
    actions.extend(build_sensor_actions(system))

    if system.dashboard.rerun.enabled:
        actions.append(
            Node(
                package="xarm_ros2_teleop",
                executable="rerun_visualizer",
                name="rerun_visualizer",
                parameters=[
                    {
                        "config_path": str(config_path),
                        "operation_mode": "teleop",
                    }
                ],
                prefix=[sys.executable],
                output="screen",
            )
        )
    if system.dashboard.enabled:
        actions.append(
            Node(
                package="xarm_ros2_teleop",
                executable="dashboard_server",
                name="dashboard_server",
                parameters=[
                    {
                        "config_path": str(config_path),
                        "operation_mode": "teleop",
                        "motion_enabled": False,
                        "inference_enabled": False,
                    }
                ],
                prefix=[sys.executable],
                output="screen",
            )
        )
    return actions


def _resolve_config_path(config_value: str) -> Path:
    """Return an absolute sensor-test YAML path from one launch argument."""

    config_path = Path(config_value).expanduser()
    if config_path.is_absolute():
        return config_path
    package_share = Path(get_package_share_directory("xarm_ros2_teleop"))
    return package_share / config_path


def generate_launch_description() -> LaunchDescription:
    """Return the sensor-only launch description and its config argument."""

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="config/sensors_all.yaml",
                description="Absolute path or package-share-relative sensor-only YAML",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
