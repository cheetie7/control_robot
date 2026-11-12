"""Create vendor ROS 2 driver actions with a stable camera topic contract."""

from launch.actions import TimerAction
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from xarm_ros2_teleop.system.config import CameraConfig, StreamConfig


def build_camera_actions(camera: CameraConfig) -> list:
    """Create launch actions for one physical RGB-D camera.

    Args:
        camera: Validated scalar camera configuration. Image stream shapes are
            described by its nested ``StreamConfig`` values.
    Returns:
        A one-dimensional list of launch actions. The list opens exactly the
        configured serial number and publishes canonical color, aligned depth,
        and colored point-cloud topics under ``camera.topic_prefix``. The
        configured delay is absolute from launch start, so USB devices can be
        staggered without coupling one device to another.
    """

    driver = _realsense_action(camera) if camera.vendor == "realsense" else _orbbec_action(camera)
    return [TimerAction(period=camera.startup_delay_sec, actions=[driver])]


def _realsense_action(camera: CameraConfig) -> Node:
    """Create the official RealSense D435 driver node and direct remappings."""

    parameters = {
        "serial_no": camera.serial_number,
        "device_type": "D435",
        # Recover the physical USB device before opening streams, then let the
        # driver wait for and reconnect to the exact configured serial number.
        "initial_reset": True,
        "wait_for_device_timeout": 10.0,
        "reconnect_timeout": 3.0,
        "enable_color": camera.color_enabled,
        "enable_depth": camera.depth_enabled,
        "enable_infra": False,
        "enable_infra1": camera.infrared.enabled,
        "enable_infra2": camera.infrared.enabled,
        "pointcloud.enable": camera.point_cloud.enabled,
        "pointcloud.stream_filter": 2,
        "align_depth.enable": camera.depth_enabled,
        "publish_tf": True,
        "tf_prefix": f"{camera.name}_",
    }
    _set_realsense_profile(parameters, "rgb_camera.color_profile", camera.color)
    _set_realsense_profile(parameters, "depth_module.depth_profile", camera.depth)
    _set_realsense_profile(parameters, "infra_profile", camera.infrared)
    prefix = f"/{camera.topic_prefix}"
    remappings = [
        ("~/color/image_raw", f"{prefix}/color"),
        ("~/color/camera_info", f"{prefix}/color/camera_info"),
        ("~/aligned_depth_to_color/image_raw", f"{prefix}/depth"),
        ("~/aligned_depth_to_color/camera_info", f"{prefix}/depth/camera_info"),
        ("~/infra1/image_raw", f"{prefix}/infra1"),
        ("~/infra1/camera_info", f"{prefix}/infra1/camera_info"),
        ("~/infra2/image_raw", f"{prefix}/infra2"),
        ("~/infra2/camera_info", f"{prefix}/infra2/camera_info"),
        ("~/depth/color/points", f"{prefix}/point_cloud"),
    ]
    return Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace=f"{camera.name}/_driver",
        name="camera",
        parameters=[parameters],
        remappings=remappings,
        # USB cameras can fail their first open (slow enumeration, bandwidth);
        # respawn recovers instead of tearing the whole launch down.
        respawn=True,
        respawn_delay=3.0,
        output="screen",
    )


def _orbbec_action(camera: CameraConfig) -> ComposableNodeContainer:
    """Create the official Femto Bolt component with profile-equivalent defaults."""

    parameters = {
        "camera_name": camera.name,
        "serial_number": camera.serial_number,
        "device_num": 1,
        "enable_color": camera.color_enabled,
        "enable_depth": camera.depth_enabled,
        "enable_ir": False,
        "enable_accel": False,
        "enable_gyro": False,
        "enable_point_cloud": False,
        "enable_colored_point_cloud": camera.point_cloud.enabled,
        "depth_registration": camera.depth_enabled,
        # The Femto Bolt's hardware D2C rejects most depth/color profile
        # combinations ("not support hardware d2c", e.g. 640x576 depth ->
        # 1280x720 color); software alignment supports any profile pair.
        "align_mode": "SW",
        "align_target_stream": "COLOR",
        "enable_frame_sync": True,
        "publish_tf": True,
        "cloud_frame_id": f"{camera.name}_color_optical_frame",
    }
    _set_orbbec_profile(parameters, "color", camera.color)
    _set_orbbec_profile(parameters, "depth", camera.depth)
    prefix = f"/{camera.topic_prefix}"
    remappings = [
        ("color/image_raw", f"{prefix}/color"),
        ("color/camera_info", f"{prefix}/color/camera_info"),
        ("depth/image_raw", f"{prefix}/depth"),
        ("depth/camera_info", f"{prefix}/depth/camera_info"),
        ("depth_registered/points", f"{prefix}/point_cloud"),
    ]
    component = ComposableNode(
        package="orbbec_camera",
        plugin="orbbec_camera::OBCameraNodeDriver",
        name="camera",
        namespace="",
        parameters=[parameters],
        remappings=remappings,
    )
    return ComposableNodeContainer(
        name=f"{camera.name}_container",
        namespace=f"{camera.name}/_driver",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[component],
        # ToF cameras frequently fail their first open; respawn the container
        # so a crashed driver retries instead of tearing the launch down.
        respawn=True,
        respawn_delay=3.0,
        output="screen",
    )


def _set_realsense_profile(parameters: dict, parameter_name: str, stream: StreamConfig) -> None:
    """Add an optional ``width,height,fps`` RealSense stream profile."""

    if stream.width is not None:
        parameters[parameter_name] = f"{stream.width},{stream.height},{stream.fps}"


def _set_orbbec_profile(parameters: dict, stream_name: str, stream: StreamConfig) -> None:
    """Add optional scalar Femto Bolt image dimensions and frame rate."""

    if stream.width is not None:
        parameters.update(
            {
                f"{stream_name}_width": stream.width,
                f"{stream_name}_height": stream.height,
                f"{stream_name}_fps": stream.fps,
            }
        )
