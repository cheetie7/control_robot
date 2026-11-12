"""Thread-safe ROS interface used by the multi-mode HTTP dashboard."""

import json
import threading
import time
from collections import deque
from urllib.parse import quote

from diagnostic_msgs.msg import DiagnosticArray
from rcl_interfaces.msg import Log
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.rl.intervention import RL_SERVICE_COMMANDS
from xarm_ros2_teleop.system.config import load_system_config

_LOG_LEVELS = {
    "debug": Log.DEBUG[0],
    "info": Log.INFO[0],
    "warn": Log.WARN[0],
    "error": Log.ERROR[0],
    "fatal": Log.FATAL[0],
}
_SERVICE_NAMES = {
    "staging": "/recording/start_staging",
    "start": "/recording/start_episode",
    "save": "/recording/stop_episode",
    "optimal": "/recording/save_optimal",
    "failure": "/recording/save_failure",
    "discard": "/recording/discard_episode",
    "finish": "/recording/finalize",
}
_INFERENCE_STOP_SERVICE = "/inference/stop"
_EVALUATION_SERVICE_NAMES = {
    "success": "/evaluation/success",
    "failure": "/evaluation/failure",
}
_RL_SERVICE_NAMES = RL_SERVICE_COMMANDS


class DashboardRosGateway(Node):
    """Convert ROS diagnostics and operator services into Web-safe dictionaries."""

    # Direct snapshot tests do not invoke ``Node.__init__``; runtime instances
    # replace this value from the unified evaluation configuration.
    evaluation_enabled = False

    def __init__(self, system=None) -> None:
        """Create mode-aware subscriptions and clients for all Web operations."""

        super().__init__("dashboard_server")
        self.declare_parameter("config_path", "")
        self.declare_parameter("operation_mode", "teleop")
        self.declare_parameter("motion_enabled", True)
        self.declare_parameter("inference_enabled", False)
        if system is None:
            config_path = str(self.get_parameter("config_path").value)
            if not config_path:
                raise ValueError("config_path must be provided by teleop.launch.py")
            system = load_system_config(config_path)
        self.system = system
        self.config = system.dashboard
        self.operation_mode = str(self.get_parameter("operation_mode").value)
        if self.operation_mode not in {"teleop", "inference", "rl"}:
            raise ValueError("operation_mode must be teleop, inference or rl")
        self.motion_enabled = bool(self.get_parameter("motion_enabled").value)
        self.inference_enabled = bool(self.get_parameter("inference_enabled").value)
        self.evaluation_enabled = bool(
            self.operation_mode == "inference"
            and system.evaluation.get("enabled", False)
        )
        self.lock = threading.RLock()
        self.components: dict[str, dict] = {}
        self.errors = deque(maxlen=self.config.rosout_history_size)
        self.inference_status = "DISABLED"
        self.rl_status = "DISABLED"
        self.rl_action_source = "unknown"
        self.rl_intervention_active = False
        self.rl_last_reward: dict | None = None
        self.minimum_log_level = _LOG_LEVELS[self.config.rosout_min_level]
        # ``Node.clients`` is a read-only rclpy property; keep application
        # service handles under a distinct name.
        self.service_clients = {
            action: self.create_client(Trigger, service_name)
            for action, service_name in _SERVICE_NAMES.items()
        }
        self.session_parameter_client = self.create_client(
            SetParametersAtomically,
            "/episode_recorder/set_parameters_atomically",
        )
        self.inference_stop_client = self.create_client(Trigger, _INFERENCE_STOP_SERVICE)
        self.evaluation_service_clients = {
            action: self.create_client(Trigger, service_name)
            for action, service_name in _EVALUATION_SERVICE_NAMES.items()
        }
        self.rl_service_clients = {
            action: self.create_client(Trigger, service_name)
            for action, service_name in _RL_SERVICE_NAMES.items()
        }
        self.create_subscription(
            DiagnosticArray,
            "/system/diagnostics",
            self.diagnostics_callback,
            10,
        )
        rosout_qos = QoSProfile(
            depth=self.config.rosout_history_size,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Log, "/rosout", self.rosout_callback, rosout_qos)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, "/inference/status", self.inference_status_callback, latched_qos
        )
        self.create_subscription(String, "/rl/status", self.rl_status_callback, latched_qos)
        self.create_subscription(
            String, "/rl/action_source", self.rl_action_source_callback, latched_qos
        )
        self.create_subscription(
            Bool,
            "/rl/intervention_active",
            self.rl_intervention_callback,
            latched_qos,
        )
        self.create_subscription(String, "/rl/reward_event", self.rl_reward_callback, 10)

    def diagnostics_callback(self, message: DiagnosticArray) -> None:
        """Replace component snapshots with one complete diagnostics publication."""

        components = {}
        for status in message.status:
            values = {item.key: _coerce(item.value) for item in status.values}
            components[status.name] = {
                **values,
                "id": status.name,
                "level": _diagnostic_level(status.level),
                "message": status.message,
                "hardware_id": status.hardware_id,
            }
        with self.lock:
            self.components = components

    def rosout_callback(self, message: Log) -> None:
        """Append one configured-level ROS event to the bounded Web history."""

        if int(message.level) < self.minimum_log_level:
            return
        timestamp = float(message.stamp.sec) + float(message.stamp.nanosec) * 1e-9
        event = {
            "timestamp": timestamp,
            "level": int(message.level),
            "node": message.name,
            "message": message.msg,
        }
        with self.lock:
            self.errors.append(event)

    def inference_status_callback(self, message: String) -> None:
        """Store the latest latched remote inference status string."""

        with self.lock:
            self.inference_status = message.data

    def rl_status_callback(self, message: String) -> None:
        """Store the latest stable RL state token and detail string."""

        with self.lock:
            self.rl_status = message.data

    def rl_action_source_callback(self, message: String) -> None:
        """Store the active policy, human or hold action source."""

        with self.lock:
            self.rl_action_source = message.data

    def rl_intervention_callback(self, message: Bool) -> None:
        """Store whether relative FACTR control currently owns the action."""

        with self.lock:
            self.rl_intervention_active = bool(message.data)

    def rl_reward_callback(self, message: String) -> None:
        """Decode and store the most recent sparse human reward event."""

        try:
            reward = json.loads(message.data)
        except (TypeError, ValueError):
            reward = {"event": "invalid", "raw": message.data}
        if not isinstance(reward, dict):
            reward = {"event": "invalid", "raw": message.data}
        with self.lock:
            self.rl_last_reward = reward

    def snapshot(self) -> dict:
        """Return one JSON-serializable system snapshot for HTTP and WebSocket clients."""

        with self.lock:
            components = [dict(value) for value in self.components.values()]
            errors = list(self.errors)
            inference_status = self.inference_status
            rl_status = self.rl_status
            rl_action_source = self.rl_action_source
            rl_intervention_active = self.rl_intervention_active
            rl_last_reward = None if self.rl_last_reward is None else dict(self.rl_last_reward)
        by_id = {component["id"]: component for component in components}
        recording = by_id.get("recording", _unknown_component("recording"))
        reset = by_id.get("reset", _unknown_component("reset"))
        rerun = by_id.get("rerun", _unknown_component("rerun"))
        session = by_id.get("session", _unknown_component("session"))
        speech = by_id.get("speech", _unknown_component("speech"))
        inference_component = by_id.get("inference", _unknown_component("inference"))
        rl_component = by_id.get("rl", _unknown_component("rl"))
        hardware = [
            component
            for component in components
            if component["id"]
            not in {"recording", "reset", "rerun", "session", "speech", "inference", "rl"}
        ]
        active_software = {
            "teleop": [recording, session],
            "inference": (
                [inference_component, recording, reset]
                if self.evaluation_enabled and self.motion_enabled
                else [inference_component]
            ),
            "rl": [rl_component, inference_component, recording, reset],
        }[self.operation_mode]
        health_components = [*hardware, *active_software]
        active_levels = [
            int(component.get("level", 0))
            for component in health_components
            if component.get("enabled", True)
        ]
        overall_level = max(active_levels, default=0)
        viewer_source = f"rerun+http://127.0.0.1:{self.config.rerun.grpc_port}/proxy"
        viewer_url = (
            f"http://127.0.0.1:{self.config.rerun.web_port}/"
            f"?url={quote(viewer_source, safe='')}&theme=light"
        )
        policy_status = InferenceStatus.from_wire(inference_status)
        inference_phase = policy_status.phase
        rl_state, _, rl_detail = rl_status.partition("|")
        rl_state = rl_state.strip()
        rl_enabled = self.operation_mode == "rl"
        evaluation_config = getattr(self.system, "evaluation", {})
        evaluation_duration = float(evaluation_config.get("max_episode_sec", 30.0))
        return {
            "schema_version": 2,
            "timestamp": time.time(),
            "system": {
                "name": self.system.name,
                "level": overall_level,
                "state": _level_name(overall_level),
            },
            "mode": {
                "name": self.operation_mode,
                "motion_enabled": self.motion_enabled,
                "safety_state": "MOTION_ARMED" if self.motion_enabled else "DRY_RUN",
                "inference_enabled": self.inference_enabled,
                "teleop_source": getattr(self.system, "teleop", {}).get("source", "factr"),
            },
            "recording": recording,
            "reset": reset,
            "session": session,
            "speech": speech,
            "inference": {
                "enabled": self.inference_enabled,
                "status": policy_status.text,
                "phase": inference_phase,
                "reason": policy_status.reason,
                "can_stop": self.inference_enabled and policy_status.can_stop,
            },
            "evaluation": {
                "enabled": self.evaluation_enabled and self.motion_enabled,
                "state": recording.get("state", "UNKNOWN"),
                "remaining_sec": recording.get(
                    "remaining_sec",
                    evaluation_duration,
                ),
                "max_episode_sec": evaluation_duration,
                "successful_episodes": recording.get("successful_episodes", 0),
                "failed_episodes": recording.get("failed_episodes", 0),
                "discarded_attempts": recording.get("discarded_attempts", 0),
                "last_outcome": recording.get("last_outcome", ""),
                "last_outcome_reason": recording.get("last_outcome_reason", ""),
                "start_blocker": recording.get("start_blocker", ""),
                "operation_phase": recording.get("operation_phase", "UNKNOWN"),
                "operation_detail": recording.get("operation_detail", "等待录制组件"),
                "can_start": bool(recording.get("can_start", False)),
                "can_label": bool(recording.get("can_label", False)),
                "can_discard": bool(recording.get("can_discard", False)),
                "can_finish": bool(recording.get("can_finish", False)),
            },
            "rl": {
                "enabled": rl_enabled,
                "status": rl_status,
                "state": rl_state,
                "detail": rl_detail.strip(),
                "action_source": rl_action_source,
                "intervention_active": rl_intervention_active,
                "last_reward": (
                    {
                        "event": recording["last_outcome"],
                        "reward": recording.get("last_episode_reward", 0.0),
                    }
                    if recording.get("last_outcome")
                    else rl_last_reward
                ),
                "episode_state": recording.get("state", "UNKNOWN"),
                "frame_count": recording.get("frame_count", 0),
                "elapsed_sec": recording.get("elapsed_sec", 0),
                "success_episodes": recording.get("success_episodes", 0),
                "failure_episodes": recording.get("failure_episodes", 0),
                "crash_episodes": recording.get("crash_episodes", 0),
                "last_outcome": recording.get("last_outcome", ""),
                "last_outcome_reason": recording.get("last_outcome_reason", ""),
                "operation_detail": recording.get("operation_detail", "等待录制组件"),
                "start_blocker": recording.get("start_blocker", ""),
                "can_start": rl_enabled and bool(recording.get("can_start", False)),
                "can_discard": rl_enabled and bool(recording.get("can_discard", False)),
                "can_finish": rl_enabled and bool(recording.get("can_finish", False)),
                "human_input_ready": bool(recording.get("human_input_ready", False)),
                "can_switch": rl_enabled
                and recording.get("state") == "RECORDING"
                and (
                    rl_state == "HUMAN_CONTROL"
                    or rl_state in {"POLICY", "PAUSED"}
                    and bool(recording.get("human_input_ready", False))
                ),
                "can_reward": rl_enabled and recording.get("state") == "RECORDING",
                "can_stop": rl_enabled
                and recording.get("state") == "RECORDING"
                and rl_state
                in {"POLICY", "STOPPING", "HUMAN_CONTROL"},
            },
            "hardware": hardware,
            "viewer": {
                "enabled": self.config.rerun.enabled,
                "ready": rerun.get("state") == "OK",
                "state": rerun.get("state", "UNKNOWN"),
                "url": viewer_url,
            },
            "errors": errors,
        }

    def call_inference_stop(self) -> dict:
        """Stop the active remote rollout through its existing Trigger service."""

        if not self.inference_enabled:
            return {"ok": False, "message": "Remote inference is not enabled", "state": "DISABLED"}
        return self._call_trigger(
            self.inference_stop_client,
            _INFERENCE_STOP_SERVICE,
            self.snapshot()["inference"]["phase"],
        )

    def call_rl_service(self, action: str) -> dict:
        """Apply one validated RL operator command through its Trigger service.

        Args:
            action: One key from ``_RL_SERVICE_NAMES``.

        Returns:
            JSON-safe success, message and latest RL state values.
        """

        if self.operation_mode != "rl":
            return {"ok": False, "message": "RL mode is not active", "state": "DISABLED"}
        if action not in self.rl_service_clients:
            return {"ok": False, "message": f"Unknown RL action: {action}", "state": "UNKNOWN"}
        return self._call_trigger(
            self.rl_service_clients[action],
            _RL_SERVICE_NAMES[action],
            self.snapshot()["rl"]["state"],
        )

    def _call_trigger(self, client, service_name: str, fallback_state: str) -> dict:
        """Call one Trigger service and return a bounded Web-safe result."""

        if not client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"ROS service unavailable: {service_name}",
                "state": fallback_state,
            }
        future = client.call_async(Trigger.Request())
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"ROS service timed out: {service_name}",
                "state": fallback_state,
            }
        try:
            response = future.result()
        except Exception as exc:
            return {"ok": False, "message": str(exc), "state": fallback_state}
        return {
            "ok": bool(response.success),
            "message": response.message,
            "state": fallback_state,
        }

    def call_recording_service(self, action: str) -> dict:
        """Call one recorder Trigger service and wait without spinning this thread.

        Args:
            action: One of ``staging``, ``start``, ``save``, ``discard`` or
                ``finish``.

        Returns:
            JSON-safe ``ok``, ``message`` and latest recorder ``state`` values.
        """

        if action not in self.service_clients:
            return {"ok": False, "message": f"Unknown action: {action}", "state": "UNKNOWN"}
        client = self.service_clients[action]
        if not client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"ROS service unavailable: {_SERVICE_NAMES[action]}",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        future = client.call_async(Trigger.Request())
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"ROS service timed out: {_SERVICE_NAMES[action]}",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        try:
            response = future.result()
        except Exception as exc:
            return {"ok": False, "message": str(exc), "state": "UNKNOWN"}
        state = self.snapshot()["recording"].get("state", "UNKNOWN")
        return {"ok": bool(response.success), "message": response.message, "state": state}

    def call_evaluation_service(self, action: str) -> dict:
        """Call one success/failure labeling service for the active rollout.

        Args:
            action: ``success`` or ``failure``.

        Returns:
            JSON-safe operation result and latest recorder state.
        """

        client = self.evaluation_service_clients.get(action)
        if client is None:
            return {"ok": False, "message": f"Unknown evaluation action: {action}"}
        if not client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"ROS service unavailable: {_EVALUATION_SERVICE_NAMES[action]}",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        future = client.call_async(Trigger.Request())
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": f"Evaluation service timed out: {action}",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        try:
            response = future.result()
        except Exception as exc:
            return {"ok": False, "message": str(exc), "state": "UNKNOWN"}
        return {
            "ok": bool(response.success),
            "message": response.message,
            "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
        }

    def configure_recording_session(
        self,
        instruction: str,
        dataset_root: str,
        write_mode: str,
    ) -> dict:
        """Configure one recorder session before its schema is initialized.

        Args:
            instruction: Natural-language VLA task stored on every frame.
            dataset_root: Complete local LeRobot dataset directory.
            write_mode: ``resume`` or ``overwrite``.

        Returns:
            JSON-safe ``ok``, ``message`` and latest recorder ``state``.
        """

        client = self.session_parameter_client
        if not client.wait_for_service(timeout_sec=self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": "Recorder configuration service is unavailable",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter("session_instruction", value=instruction).to_parameter_msg(),
            Parameter("session_dataset_root", value=dataset_root).to_parameter_msg(),
            Parameter("session_write_mode", value=write_mode).to_parameter_msg(),
        ]
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(self.config.service_timeout_sec):
            return {
                "ok": False,
                "message": "Recorder configuration service timed out",
                "state": self.snapshot()["recording"].get("state", "UNKNOWN"),
            }
        try:
            result = future.result().result
        except Exception as exc:
            return {"ok": False, "message": str(exc), "state": "UNKNOWN"}
        state = self.snapshot()["recording"].get("state", "UNKNOWN")
        message = result.reason or (
            "Recording session configured" if result.successful else "Configuration rejected"
        )
        return {"ok": bool(result.successful), "message": message, "state": state}


def _coerce(value: str):
    """Convert diagnostic scalar strings into useful JSON primitive values."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _diagnostic_level(value: int | bytes) -> int:
    """Return one standard diagnostic level from Humble's byte representation.

    Args:
        value: ROS ``DiagnosticStatus.level`` as one byte or an integer.

    Returns:
        Integer diagnostic level, normally in ``[0, 3]``.
    """

    if isinstance(value, bytes):
        return value[0] if value else 0
    return int(value)


def _unknown_component(component_id: str) -> dict:
    """Return a stable placeholder before the first diagnostics message."""

    return {
        "id": component_id,
        "level": 1,
        "state": "UNKNOWN",
        "message": "Waiting for diagnostics",
    }


def _level_name(level: int) -> str:
    """Map standard diagnostic levels to an overall Web state string."""

    return {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}.get(int(level), "UNKNOWN")
