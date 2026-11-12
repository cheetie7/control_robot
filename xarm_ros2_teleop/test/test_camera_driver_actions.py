"""Construction tests for vendor camera launch actions."""

from dataclasses import asdict

from launch.actions import TimerAction
from xarm_ros2_teleop.camera import driver_actions
from xarm_ros2_teleop.camera.driver_actions import build_camera_actions
from xarm_ros2_teleop.system.config import CameraConfig


def _config(
    vendor: str, model: str, *, required: bool, startup_delay_sec: float = 0.0
) -> CameraConfig:
    """Build a scalar camera configuration for action construction tests."""

    return CameraConfig.from_mapping(
        {
            "name": f"{vendor}_camera",
            "vendor": vendor,
            "model": model,
            "serial_number": f"{vendor}-serial",
            "topic_prefix": f"camera/{vendor}",
            "required": required,
            "startup_delay_sec": startup_delay_sec,
            "color": {"enabled": True},
            "depth": {"enabled": False},
            "point_cloud": {"enabled": False},
        }
    )


def test_required_realsense_uses_respawn_without_shutdown_handler() -> None:
    """A required D435 must retry locally instead of shutting down teleoperation."""

    actions = build_camera_actions(_config("realsense", "d435", required=True))
    assert len(actions) == 1
    assert isinstance(actions[0], TimerAction)


def test_optional_orbbec_only_creates_delayed_driver() -> None:
    """An optional Femto Bolt may exit without shutting down teleoperation."""

    actions = build_camera_actions(_config("orbbec", "femto_bolt", required=False))
    assert len(actions) == 1
    assert isinstance(actions[0], TimerAction)


def test_camera_uses_configured_absolute_start_delay() -> None:
    """Each camera action must own its absolute delay from launch start."""

    actions = build_camera_actions(
        _config("realsense", "d435", required=False, startup_delay_sec=3.0)
    )
    assert actions[0].period == 3.0


def test_realsense_colored_cloud_enables_alignment(monkeypatch) -> None:
    """D435 colored cloud must enable color, depth, alignment and exact serial binding."""

    camera = CameraConfig.from_mapping(
        {
            **asdict(_config("realsense", "d435", required=False)),
            "color": {"enabled": True},
            "depth": {"enabled": True},
            "point_cloud": {"enabled": True},
        }
    )
    monkeypatch.setattr(driver_actions, "Node", lambda **kwargs: kwargs)

    node_arguments = driver_actions._realsense_action(camera)
    parameters = node_arguments["parameters"][0]
    assert parameters["serial_no"] == "realsense-serial"
    assert parameters["initial_reset"] is True
    assert parameters["wait_for_device_timeout"] == 10.0
    assert parameters["reconnect_timeout"] == 3.0
    assert parameters["enable_color"] is True
    assert parameters["enable_depth"] is True
    assert parameters["align_depth.enable"] is True
    assert parameters["pointcloud.enable"] is True


def test_realsense_false_streams_disable_driver_generation(monkeypatch) -> None:
    """False D435 outputs must disable their driver streams and computation."""

    camera = CameraConfig.from_mapping(
        {
            **asdict(_config("realsense", "d435", required=False)),
            "color": {"enabled": False},
            "depth": {"enabled": False},
            "point_cloud": {"enabled": False},
        }
    )
    monkeypatch.setattr(driver_actions, "Node", lambda **kwargs: kwargs)

    parameters = driver_actions._realsense_action(camera)["parameters"][0]
    assert parameters["enable_color"] is False
    assert parameters["enable_depth"] is False
    assert parameters["align_depth.enable"] is False
    assert parameters["pointcloud.enable"] is False


def test_orbbec_colored_cloud_uses_hardware_color_alignment(monkeypatch) -> None:
    """Femto Bolt colored cloud must use registration and hardware COLOR alignment."""

    camera = CameraConfig.from_mapping(
        {
            **asdict(_config("orbbec", "femto_bolt", required=False)),
            "color": {"enabled": True},
            "depth": {"enabled": True},
            "point_cloud": {"enabled": True},
        }
    )
    monkeypatch.setattr(driver_actions, "ComposableNode", lambda **kwargs: kwargs)
    monkeypatch.setattr(driver_actions, "ComposableNodeContainer", lambda **kwargs: kwargs)

    container_arguments = driver_actions._orbbec_action(camera)
    component_arguments = container_arguments["composable_node_descriptions"][0]
    parameters = component_arguments["parameters"][0]
    assert parameters["serial_number"] == "orbbec-serial"
    assert parameters["enable_color"] is True
    assert parameters["enable_depth"] is True
    assert parameters["enable_colored_point_cloud"] is True
    assert parameters["depth_registration"] is True
    # Femto Bolt hardware D2C rejects this depth/color profile pair, so the
    # driver must request software alignment.
    assert parameters["align_mode"] == "SW"
    assert parameters["align_target_stream"] == "COLOR"


def test_orbbec_false_streams_disable_driver_generation(monkeypatch) -> None:
    """False Femto Bolt outputs must disable streams, registration and clouds."""

    camera = CameraConfig.from_mapping(
        {
            **asdict(_config("orbbec", "femto_bolt", required=False)),
            "color": {"enabled": False},
            "depth": {"enabled": False},
            "point_cloud": {"enabled": False},
        }
    )
    monkeypatch.setattr(driver_actions, "ComposableNode", lambda **kwargs: kwargs)
    monkeypatch.setattr(driver_actions, "ComposableNodeContainer", lambda **kwargs: kwargs)

    container = driver_actions._orbbec_action(camera)
    parameters = container["composable_node_descriptions"][0]["parameters"][0]
    assert parameters["enable_color"] is False
    assert parameters["enable_depth"] is False
    assert parameters["depth_registration"] is False
    assert parameters["enable_colored_point_cloud"] is False


def test_vendor_drivers_respawn_after_a_crash(monkeypatch) -> None:
    """Both camera vendors restart their driver instead of killing the launch."""

    camera = _config("realsense", "d435", required=False)
    monkeypatch.setattr(driver_actions, "Node", lambda **kwargs: kwargs)
    realsense_kwargs = driver_actions._realsense_action(camera)
    assert realsense_kwargs["respawn"] is True
    assert realsense_kwargs["respawn_delay"] == 3.0

    monkeypatch.setattr(
        driver_actions,
        "ComposableNodeContainer",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        driver_actions,
        "ComposableNode",
        lambda **kwargs: kwargs,
    )
    orbbec_kwargs = driver_actions._orbbec_action(
        _config("orbbec", "femto_bolt", required=False)
    )
    assert orbbec_kwargs["respawn"] is True
    assert orbbec_kwargs["respawn_delay"] == 3.0


def test_realsense_stereo_infrared_enables_infra1_infra2(monkeypatch) -> None:
    """Configured infrared streams enable both IR trackers with one profile."""

    camera = CameraConfig.from_mapping(
        {
            **asdict(_config("realsense", "d435", required=False)),
            "color": {"enabled": True},
            "infrared": {"enabled": True, "width": 640, "height": 480, "fps": 30},
            "depth": {"enabled": False},
        }
    )
    monkeypatch.setattr(driver_actions, "Node", lambda **kwargs: kwargs)

    node_arguments = driver_actions._realsense_action(camera)
    parameters = node_arguments["parameters"][0]
    assert parameters["enable_infra1"] is True
    assert parameters["enable_infra2"] is True
    assert parameters["infra_profile"] == "640,480,30"
    assert parameters["align_depth.enable"] is False
    remappings = dict(node_arguments["remappings"])
    assert remappings["~/infra1/image_raw"] == "/camera/realsense/infra1"
    assert remappings["~/infra2/image_raw"] == "/camera/realsense/infra2"
