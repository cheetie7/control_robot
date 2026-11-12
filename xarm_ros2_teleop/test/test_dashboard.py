"""Pure configuration, diagnostics and static Dashboard interface tests."""

import threading
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import yaml
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import Log
from sensor_msgs.msg import Image
from xarm_ros2_teleop.dashboard.config import DashboardConfig
from xarm_ros2_teleop.dashboard.ros_gateway import (
    _RL_SERVICE_NAMES,
    _SERVICE_NAMES,
    DashboardRosGateway,
)
from xarm_ros2_teleop.dashboard.server import create_app
from xarm_ros2_teleop.monitor.health import ERROR, OK, WARN, ComponentHealth
from xarm_ros2_teleop.monitor.system_monitor import runtime_component_flags
from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.visualization.rerun_visualizer import RerunVisualizer


def _dashboard(**updates) -> DashboardConfig:
    """Return one valid local Dashboard configuration with optional overrides."""

    values = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8080,
        "rerun": {
            "enabled": True,
            "grpc_port": 9876,
            "web_port": 9090,
            "server_memory_limit": "1GiB",
        },
    }
    values.update(updates)
    return DashboardConfig.from_mapping(values)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"host": "0.0.0.0"}, "host"),
        ({"port": 9876}, "ports must be unique"),
        ({"rerun": {"server_memory_limit": "large"}}, "memory_limit"),
        ({"rerun": {"preview_scale": 0}}, "preview_scale"),
        ({"rerun": {"preview_scale": 1.1}}, "preview_scale"),
    ],
)
def test_dashboard_rejects_unsafe_or_ambiguous_network_config(update, error) -> None:
    """The first Web control surface remains local with distinct valid ports."""

    with pytest.raises(ValueError, match=error):
        _dashboard(**update)


def test_component_health_covers_startup_stale_and_recovery() -> None:
    """Required devices become errors after grace and recover on fresh output."""

    tracker = ComponentHealth(
        component_id="camera/front",
        kind="camera",
        enabled=True,
        required=True,
        topics=("/camera/front/color",),
        timeout_sec=1.0,
        startup_grace_sec=2.0,
        started_at=10.0,
    )
    assert tracker.evaluate(11.0).level == WARN
    assert tracker.evaluate(13.0).level == ERROR
    tracker.observe("/camera/front/color", 13.0)
    assert tracker.evaluate(13.1).level == OK
    assert tracker.evaluate(14.1).level == ERROR


def test_rerun_preview_keeps_only_latest_half_size_frame() -> None:
    """Each preview tick logs one newest frame at half width and height."""

    logged = []
    timeline_updates = []

    class FakeImage:
        """Capture the array and JPEG quality passed to Rerun."""

        def __init__(self, array) -> None:
            self.array = array

        def compress(self, jpeg_quality: int):
            return SimpleNamespace(
                shape=self.array.shape,
                value=int(self.array[0, 0, 0]),
                jpeg_quality=jpeg_quality,
            )

    fake_rerun = SimpleNamespace(
        set_time=lambda *args, **kwargs: timeline_updates.append((args, kwargs)),
        Image=FakeImage,
        log=lambda path, value, static=False: logged.append((path, value, static)),
    )
    visualizer = SimpleNamespace(
        config=SimpleNamespace(preview_scale=0.5),
        latest_images={},
        image_lock=threading.Lock(),
        rr=fake_rerun,
    )
    visualizer._resize_preview = MethodType(RerunVisualizer._resize_preview, visualizer)
    visualizer._log_image = MethodType(RerunVisualizer._log_image, visualizer)

    stream = SimpleNamespace(key="front", kind="rgb")

    def message(value: int, second: int) -> Image:
        """Return one 6x4 RGB image filled with a scalar pixel value."""

        frame = Image()
        frame.header.stamp.sec = second
        frame.height = 4
        frame.width = 6
        frame.encoding = "rgb8"
        frame.is_bigendian = 0
        frame.step = 18
        frame.data = bytes([value] * 72)
        return frame

    RerunVisualizer.image_callback(visualizer, stream, message(10, 1))
    RerunVisualizer.image_callback(visualizer, stream, message(20, 2))
    RerunVisualizer.flush_latest_images(visualizer)

    assert len(logged) == 1
    path, preview, static = logged[0]
    assert path == "/sensors/front"
    assert preview.shape == (2, 3, 3)
    assert preview.value == 20
    assert preview.jpeg_quality == 80
    assert static is True
    assert timeline_updates == []
    assert visualizer.latest_images == {}


def test_optional_and_disabled_components_are_not_hard_errors() -> None:
    """Optional stale hardware warns while YAML-disabled hardware stays neutral."""

    optional = ComponentHealth("rerun", "software", True, False, ("status",), 1.0, 0.0, 0.0)
    disabled = ComponentHealth("camera", "camera", False, True, ("image",), 1.0, 0.0, 0.0)
    assert optional.evaluate(2.0).level == WARN
    assert disabled.evaluate(2.0).state == "DISABLED"
    assert disabled.evaluate(2.0).level == OK


def test_gateway_snapshot_separates_hardware_recording_and_viewer() -> None:
    """The public JSON schema keeps UI sections independent and versioned."""

    gateway = DashboardRosGateway.__new__(DashboardRosGateway)
    gateway.lock = threading.RLock()
    gateway.config = _dashboard()
    gateway.system = SimpleNamespace(name="control_robot", dashboard=gateway.config)
    gateway.operation_mode = "rl"
    gateway.motion_enabled = False
    gateway.inference_enabled = True
    gateway.inference_status = "EXECUTING_DRY_RUN"
    gateway.rl_status = "HUMAN_CONTROL|relative FACTR control engaged"
    gateway.rl_action_source = "human"
    gateway.rl_intervention_active = True
    gateway.rl_last_reward = {"event": "task_success", "reward": 1.0}
    gateway.errors = deque([{"timestamp": 1.0, "level": 40, "node": "xarm", "message": "e"}])
    gateway.components = {
        "xarm": {"id": "xarm", "level": 0, "state": "OK", "enabled": True},
        "recording": {
            "id": "recording",
            "level": 0,
            "state": "RECORDING",
            "can_save": True,
        },
        "reset": {"id": "reset", "level": 0, "state": "IDLE"},
        "rerun": {"id": "rerun", "level": 0, "state": "OK"},
        "inference": {"id": "inference", "level": 0, "state": "EXECUTING_DRY_RUN"},
        "rl": {"id": "rl", "level": 0, "state": "HUMAN_CONTROL"},
    }
    snapshot = gateway.snapshot()
    assert snapshot["schema_version"] == 2
    assert snapshot["mode"] == {
        "name": "rl",
        "motion_enabled": False,
        "safety_state": "DRY_RUN",
        "inference_enabled": True,
        "teleop_source": "factr",
    }
    assert snapshot["inference"]["can_stop"] is True
    assert snapshot["rl"]["state"] == "HUMAN_CONTROL"
    assert snapshot["rl"]["can_switch"] is True
    assert snapshot["rl"]["intervention_active"] is True
    assert snapshot["recording"]["state"] == "RECORDING"
    assert snapshot["hardware"][0]["id"] == "xarm"
    assert snapshot["viewer"]["ready"] is True
    assert "rerun%2Bhttp" in snapshot["viewer"]["url"]


def test_web_actions_map_to_existing_recorder_services() -> None:
    """Buttons remain a transport adapter over the established ROS services."""

    assert _SERVICE_NAMES == {
        "staging": "/recording/start_staging",
        "start": "/recording/start_episode",
        "save": "/recording/stop_episode",
        "optimal": "/recording/save_optimal",
        "failure": "/recording/save_failure",
        "discard": "/recording/discard_episode",
        "finish": "/recording/finalize",
    }
    assert _RL_SERVICE_NAMES == {
        "start": "/rl/start",
        "discard": "/rl/discard",
        "finish": "/rl/finish",
        "switch": "/rl/switch_control",
        "success": "/rl/success",
        "failure": "/rl/failure",
        "stop": "/rl/stop",
    }


def test_dashboard_exposes_inference_and_rl_operator_routes() -> None:
    """The local HTTP app must expose every mode-specific Trigger adapter."""

    gateway = SimpleNamespace(config=SimpleNamespace(status_rate_hz=2.0))
    static_root = Path(__file__).parents[1] / "xarm_ros2_teleop" / "dashboard" / "static"
    paths = {route.path for route in create_app(gateway, static_root).routes}
    assert "/api/v1/recording/staging" in paths
    assert "/api/v1/inference/stop" in paths
    assert "/api/v1/evaluation/success" in paths
    assert "/api/v1/evaluation/failure" in paths
    for action in _RL_SERVICE_NAMES:
        assert f"/api/v1/rl/{action}" in paths


def test_runtime_component_flags_match_teleop_inference_and_rl_processes() -> None:
    """Mode-aware monitoring must not report processes that launch deliberately omits."""

    system = SimpleNamespace(
        teleop={"enabled": True},
        xarm={"enabled": True},
        robotiq={"enabled": True},
        reset={"enabled": True},
        evaluation={"enabled": True},
        recording=SimpleNamespace(enabled=True),
        warmup=SimpleNamespace(enabled=True),
        speech=SimpleNamespace(enabled=True),
        dashboard=SimpleNamespace(enabled=True, rerun=SimpleNamespace(enabled=True)),
    )
    teleop = runtime_component_flags(system, "teleop", True, False)
    inference_dry = runtime_component_flags(system, "inference", False, True)
    inference_evaluation = runtime_component_flags(system, "inference", True, True)
    rl_dry = runtime_component_flags(system, "rl", False, True)
    rl_motion = runtime_component_flags(system, "rl", True, False)

    assert all(teleop[name] for name in ("factr", "xarm", "recording", "reset", "session"))
    assert inference_dry["inference"] is True
    assert not any(
        inference_dry[name]
        for name in ("factr", "xarm", "robotiq", "recording", "reset", "session")
    )
    assert inference_evaluation["recording"] is True
    assert inference_evaluation["reset"] is True
    assert inference_evaluation["factr"] is False
    assert rl_dry["factr"] is True and rl_dry["rl"] is True
    assert rl_dry["xarm"] is False and rl_dry["reset"] is False
    assert rl_motion["xarm"] is True and rl_motion["reset"] is True


def test_gateway_decodes_humble_diagnostic_level_bytes() -> None:
    """Dashboard JSON must expose Humble's one-byte level as an integer."""

    gateway = DashboardRosGateway.__new__(DashboardRosGateway)
    gateway.lock = threading.RLock()
    gateway.components = {}
    message = DiagnosticArray()
    status = DiagnosticStatus()
    status.name = "xarm"
    status.level = DiagnosticStatus.ERROR
    status.message = "controller error"
    # Diagnostic values must never override the standard top-level fields.
    status.values = [KeyValue(key="level", value="b'\\x00'")]
    message.status = [status]

    gateway.diagnostics_callback(message)

    assert gateway.components["xarm"]["level"] == 2


def test_rosout_callbacks_compare_integer_messages_with_byte_constants() -> None:
    """Humble Log messages use integer levels while generated constants are bytes."""

    gateway = DashboardRosGateway.__new__(DashboardRosGateway)
    gateway.lock = threading.RLock()
    gateway.minimum_log_level = Log.WARN[0]
    gateway.errors = deque()
    info = Log(level=Log.INFO[0], name="xarm", msg="normal")
    warning = Log(level=Log.WARN[0], name="xarm", msg="warning")
    gateway.rosout_callback(info)
    gateway.rosout_callback(warning)
    assert [event["message"] for event in gateway.errors] == ["warning"]

    rerun_events = []
    fake_rerun = SimpleNamespace(
        set_time=lambda *args, **kwargs: None,
        TextLog=lambda text: text,
        log=lambda path, value: rerun_events.append((path, value)),
    )
    visualizer = SimpleNamespace(rr=fake_rerun)
    RerunVisualizer.rosout_callback(visualizer, info)
    RerunVisualizer.rosout_callback(visualizer, warning)
    assert rerun_events == [("/logs/rosout/xarm", "warning")]


def test_runtime_compatibility_fixes_remain_in_source() -> None:
    """Known Humble and Rerun 0.26 API incompatibilities must not regress."""

    package_root = Path(__file__).parents[1] / "xarm_ros2_teleop"
    launch_source = (Path(__file__).parents[1] / "launch" / "teleop.launch.py").read_text()
    dashboard_source = (package_root / "dashboard" / "ros_gateway.py").read_text()
    monitor_source = (package_root / "monitor" / "system_monitor.py").read_text()
    rerun_source = (package_root / "visualization" / "rerun_visualizer.py").read_text()
    assert "self.service_clients" in dashboard_source
    assert "self.clients =" not in dashboard_source
    assert "status.level = bytes([level])" in monitor_source
    assert "cors_allow_origin" not in rerun_source
    assert '"/rl/executed_action"' in rerun_source
    assert '"/inference/status"' in rerun_source
    assert "canonical_action_callback" in rerun_source
    assert 'operation_mode = "rl" if rl_enabled else "inference"' in launch_source
    assert (
        'operation_mode == "teleop" or (vr and rl_enabled)'
        in launch_source
    )
    assert launch_source.count('"recording_gate_enabled": recording_gate_enabled') == 2
    assert "if system.dashboard.enabled:" in launch_source
    assert "system.dashboard.enabled and not inference_enabled" not in launch_source


def test_static_page_contains_buttons_and_browser_shortcuts() -> None:
    """Packaged UI exposes all four controls and prevents default arrow behavior."""

    static_root = Path(__file__).parents[1] / "xarm_ros2_teleop" / "dashboard" / "static"
    html = (static_root / "index.html").read_text(encoding="utf-8")
    script = (static_root / "app.js").read_text(encoding="utf-8")
    for action in ("start", "save", "optimal", "failure", "discard", "finish"):
        assert f'data-action="{action}"' in html
    for key in ("ArrowLeft", "ArrowRight", "Escape"):
        assert key in script
    assert 'ArrowLeft: "discard"' in script
    assert 'ArrowRight: "save"' in script
    assert '<span>Success 保存</span><kbd>→</kbd>' in html
    assert '<span>Optimal 保存</span><kbd>O</kbd>' in html
    assert '<span>Failure 保存</span><kbd>F</kbd>' in html
    assert '<span>丢弃本条</span><kbd>←</kbd>' in html
    assert 'id="recording-operation"' in html
    assert 'id="evaluation-operation"' in html
    assert "请求已发送，等待系统处理" in script
    assert html.index('data-action="discard"') < html.index('data-action="save"')
    assert "event.preventDefault()" in script
    assert 'id="session-instruction"' in html
    assert 'id="session-dataset-root"' in html
    assert 'value="resume"' in html
    assert 'value="overwrite"' in html
    assert "/api/v1/recording/configure" in script
    assert 'data-action="staging"' in html
    assert 'g: "staging"' in script
    assert "await applySessionSettings()" not in script
    assert 'id="inference-panel"' in html
    assert 'id="rl-panel"' in html
    for action in ("switch", "success", "failure", "stop"):
        assert f'data-rl-action="{action}"' in html
    for removed in ("resume", "abort", "reset", "takeover"):
        assert f'data-rl-action="{removed}"' not in html
    assert "/api/v1/rl/${action}" in script
    assert "/api/v1/inference/stop" in script
    assert "event.target.matches" in script
    assert "window.confirm" in script


def test_rerun_requires_at_least_one_configured_visual_stream(tmp_path: Path) -> None:
    """The visualizer must fail during YAML validation rather than building an empty grid."""

    source = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["recording"]["enabled"] = False
    config["recording"]["streams"] = {}
    target = tmp_path / "rerun_without_streams.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="requires at least one recording stream"):
        load_system_config(target)
