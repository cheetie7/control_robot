"""Create launch actions for serial-bound Xense tactile sensor nodes."""

import sys

from launch.actions import TimerAction
from launch_ros.actions import Node

from xarm_ros2_teleop.system.config import TactileConfig


def build_tactile_actions(device: TactileConfig) -> list:
    """Create launch actions for one configured Xense tactile sensor.

    Args:
        device: Validated scalar Xense device configuration.
    Returns:
        One-dimensional launch-action list. The node retries transient SDK or
        USB failures through ROS launch respawn. Required/optional health is
        decided from published data rather than process-exit timing.
    """

    node = Node(
        package="xarm_ros2_teleop",
        executable="tactile_node",
        name=device.name,
        parameters=[
            {
                "sensor_name": device.name,
                "serial_number": device.serial_number,
                "topic_prefix": device.topic_prefix,
                "publish_rate_hz": device.publish_rate_hz,
                "config_path": device.config_path,
                "outputs": list(device.outputs),
            }
        ],
        # A transient Xense open failure (slow USB enumeration) retries here
        # instead of exiting and tearing the whole launch down.
        respawn=True,
        respawn_delay=3.0,
        prefix=[sys.executable],
        output="screen",
    )
    return [TimerAction(period=device.startup_delay_sec, actions=[node])]
