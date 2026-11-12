"""Build launch actions for the sensors enabled in the unified configuration."""

from xarm_ros2_teleop.camera.driver_actions import build_camera_actions
from xarm_ros2_teleop.tactile.driver_actions import build_tactile_actions


def build_sensor_actions(system) -> list:
    """Return independent launch actions for every enabled physical sensor.

    Args:
        system: Validated ``SystemConfig`` containing camera and tactile device
            definitions. Each device owns its absolute startup delay, driver
            reconnect behavior and published ROS topics.

    Returns:
        One-dimensional launch-action list. The caller only schedules these
        actions; readiness is evaluated centrally from the sensors' real data.
    """

    actions = []
    if system.cameras_enabled:
        for camera in system.cameras:
            if camera.enabled:
                actions.extend(build_camera_actions(camera))
    if system.tactile_enabled:
        for device in system.tactile_devices:
            if device.enabled:
                actions.extend(build_tactile_actions(device))
    return actions
