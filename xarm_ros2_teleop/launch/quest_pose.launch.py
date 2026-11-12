"""Launch the Meta Quest USB pose bridge as a standalone ROS 2 sensor."""

import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Declare bridge parameters and create the Quest pose node.

    Returns:
        ROS 2 launch description containing one Quest pose publisher process.
    """

    rviz_config = Path(get_package_share_directory("xarm_ros2_teleop")) / "rviz" / "quest_pose.rviz"
    return LaunchDescription(
        [
            DeclareLaunchArgument("device_serial", default_value=""),
            DeclareLaunchArgument("package_name", default_value="com.controlrobot.questposebridge"),
            DeclareLaunchArgument("tcp_port", default_value="7447"),
            DeclareLaunchArgument("frame_id", default_value="quest_floor"),
            DeclareLaunchArgument("launch_app", default_value="true"),
            DeclareLaunchArgument("device_wait_timeout_sec", default_value="120.0"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.25"),
            DeclareLaunchArgument("trajectory_history_sec", default_value="3.0"),
            DeclareLaunchArgument("rviz", default_value="false"),
            Node(
                package="xarm_ros2_teleop",
                executable="quest_pose_node",
                name="quest_pose",
                prefix=[sys.executable],
                output="screen",
                parameters=[
                    {
                        "device_serial": LaunchConfiguration("device_serial"),
                        "package_name": LaunchConfiguration("package_name"),
                        "tcp_port": LaunchConfiguration("tcp_port"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "launch_app": LaunchConfiguration("launch_app"),
                        "device_wait_timeout_sec": LaunchConfiguration("device_wait_timeout_sec"),
                        "tracking_timeout_sec": LaunchConfiguration("tracking_timeout_sec"),
                        "trajectory_history_sec": LaunchConfiguration("trajectory_history_sec"),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="quest_pose_rviz",
                arguments=["-d", str(rviz_config)],
                condition=IfCondition(LaunchConfiguration("rviz")),
                output="screen",
            ),
        ]
    )
