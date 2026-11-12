"""Verify manual RL outcomes, causal action labels and readable saved artifacts."""

import json
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from std_msgs.msg import Bool, Int32MultiArray, String
from xarm_ros2_teleop.recording.dataset_schema import build_features
from xarm_ros2_teleop.recording.lerobot_writer import LeRobotWriter
from xarm_ros2_teleop.recording.synchronizer import SynchronizationError
from xarm_ros2_teleop.rl.episode_recorder import RLEpisodeRecorder
from xarm_ros2_teleop.rl.human_intervention_node import HumanInterventionNode
from xarm_ros2_teleop.rl.intervention import InterventionController, InterventionState
from xarm_ros2_teleop.rl.recording import RLFrameHistory, rl_schema_fields


def _frame() -> dict:
    """Return a small RGB observation and canonical absolute state/action [8]."""

    return {
        "task": "insert the block",
        "observation.images.front": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
        "action": np.zeros(8, dtype=np.float32),
        **rl_schema_fields(),
    }


def _command(timestamp: float, source: str, value: float) -> dict:
    """Create an arbiter selection, with seven joint targets in radians."""

    action = [value] * 7 + [0.0]
    return {
        "timestamp": timestamp,
        "source": source,
        "selected_action": action,
        "policy_action": action if source == "policy" else [0.0] * 8,
        "policy_valid": source == "policy",
    }


def test_provenance_matches_accepted_command_instead_of_latest_candidate() -> None:
    """A newer unaccepted policy suggestion cannot relabel a prior human command."""

    history = RLFrameHistory(1.0)
    history.add(_command(100.0, "human", 0.1))
    history.add(_command(100.01, "policy", 0.2))
    frame = _frame()
    frame["action"][:7] = 0.1
    frame = history.annotate(frame, 100.03, (100.02, 100.02))

    assert frame["action.source"][0] == 1
    assert frame["action.intervention"][0] == 1
    assert frame["action.policy_valid"][0] == 0
    assert np.allclose(frame["action.selected"][:7], 0.1)
    frame["action"][:7] = 0.3
    with pytest.raises(SynchronizationError, match="matches"):
        history.annotate(frame, 100.03, (100.02, 100.02))


def test_rl_timestamps_preserve_subframe_precision_and_history_is_bounded() -> None:
    """Unix-scale time stays float64 while action/state tensors stay float32."""

    history = RLFrameHistory(1.0)
    now = 1_790_000_000.125
    history.add(_command(now - 2.0, "human", 0.1))
    history.add(_command(now, "policy", 0.0))
    frame = history.annotate(_frame(), now + 0.03, (now + 0.01, now + 0.02))
    assert len(history.samples) == 1
    assert frame["observation.timestamp_ros_s"][0] == now + 0.03
    assert build_features(frame)["observation.timestamp_ros_s"]["dtype"] == "float64"
    assert build_features(frame)["action"]["dtype"] == "float32"


def test_manual_rl_does_not_end_after_duration_or_episode_quota() -> None:
    """A long manual attempt stays open even after the teleop/eval limits."""

    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.rl_mode = True
    recorder.evaluation_mode = False
    recorder.state = "RECORDING"
    recorder.capture_started_at = time.monotonic() - 3600
    recorder.max_episode_sec = 30.0
    recorder._check_reset_timeout = lambda _now: None
    recorder._finish_evaluation_episode = lambda **_kwargs: pytest.fail("unexpected timeout")
    recorder.process_pending()
    assert recorder.state == "RECORDING"
    assert recorder._episode_limit_reached() is False


def test_start_requires_live_human_readiness_and_does_not_accept_stale_heartbeat() -> None:
    """ACT cannot start an episode until the already-running human device is ready."""

    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.system = SimpleNamespace(rl={})
    recorder.xarm_control_ok = True
    recorder.state = "IDLE"
    recorder.error_recovery_pending = False
    recorder.session_configured = recorder.writer_prepared = True
    recorder.inference_ready = True
    recorder.human_ready = False
    recorder.human_ready_received_at = 0.0
    assert not recorder._start_episode()[0]
    recorder.human_ready_callback(Bool(data=True))
    assert recorder.human_input_ready
    recorder.human_ready_received_at -= 1
    assert not recorder._start_episode()[0]
    recorder.human_ready_callback(Bool(data=True))
    recorder.writer_lock = threading.Lock()
    recorder.writer = SimpleNamespace(discard_episode=lambda: None)
    recorder.rl_history = SimpleNamespace(clear=lambda: None)
    recorder._begin_capture = lambda: None
    assert recorder._start_episode()[0]


def test_policy_fault_leaves_episode_open_for_manual_labeling() -> None:
    """A failed inference may be recovered with takeover and must not invent a task outcome."""

    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.require_remote_policy = True
    recorder.policy_connected = True
    recorder.state = "RECORDING"
    recorder.get_logger = lambda: SimpleNamespace(warn=lambda _detail: None)
    recorder._finish_rl_episode = lambda **_kwargs: pytest.fail("unexpected automatic label")
    recorder.inference_status_callback(String(data="EPISODE_FAILED: invalid policy response"))
    assert recorder.state == "RECORDING"
    assert recorder.inference_ready is False
    assert "invalid policy response" in recorder.last_error


def test_hardware_error_has_priority_over_late_human_success() -> None:
    """A crash label is latched once and cannot turn into success after stopping."""

    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.state = "RECORDING"
    recorder.system = SimpleNamespace(rl={"crash_reward": -0.5})
    labels = []

    def finish(**label) -> None:
        """Capture an end request and immediately stop the capture state."""

        labels.append(label)
        recorder.state = "ERROR_RESETTING"

    recorder._finish_rl_episode = finish
    recorder.xarm_error_callback(Int32MultiArray(data=[12, 0]))
    recorder.reward_event_callback(
        String(data=json.dumps({"event": "task_success", "reward": 1.0}))
    )
    assert len(labels) == 1
    assert labels[0]["category"] == "crash"
    assert labels[0]["reason"] == "xarm_error_12"
    assert labels[0]["reward"] == -0.5
    assert labels[0]["automatic_error_reset"] is True


def test_finishing_active_session_saves_failure_without_an_abort_category() -> None:
    """Closing a pending attempt is a normal failure followed by shared reset/finalize."""

    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.state = "RECORDING"
    recorder.system = SimpleNamespace(rl={"failure_reward": 0.0})
    labels = []
    recorder._finish_rl_episode = lambda **label: labels.append(label)
    recorder._finish_recording()
    assert labels[0]["category"] == "failure"
    assert labels[0]["terminated"] and not labels[0]["truncated"]
    assert labels[0]["finish_session"]


def test_idle_or_reset_arbiter_cannot_forward_commands() -> None:
    """End-of-episode hold commands cannot fight the reset coordinator."""

    node = SimpleNamespace(episode_active=False, reset_in_progress=False)
    assert HumanInterventionNode._selected_action(node) == (None, "hold")
    node.episode_active = True
    node.reset_in_progress = True
    assert HumanInterventionNode._selected_action(node) == (None, "hold")


def test_factr_loss_pauses_human_control_without_ending_episode() -> None:
    """Leader reconnect should preserve collected RL frames and allow policy resume."""

    controller = InterventionController()
    controller.state = InterventionState.HUMAN_CONTROL
    reasons = []
    node = SimpleNamespace(
        controller=controller,
        human_input=SimpleNamespace(connected=True, ready_since=1.0),
        episode_active=True,
        _pause_policy=reasons.append,
    )
    HumanInterventionNode._human_connected_callback(node, Bool(data=False))
    assert reasons
    assert node.episode_active is True


def test_rl_recorder_initializes_without_arming_motion() -> None:
    """The launched RL subclass exposes its episode gate while starting idle."""

    config = Path(__file__).parents[1] / "config" / "teleop_system.yaml"
    rclpy.init(args=["--ros-args", "-p", f"config_path:={config}", "-p", "operation_mode:=rl"])
    recorder = None
    try:
        recorder = RLEpisodeRecorder()
        assert recorder.state == "IDLE"
        assert recorder.teleop_active is False
        assert recorder.auto_start_pending is False
        assert recorder.keyboard is None
        assert recorder.dataset_root == Path("/home/gxy/datasets/rl")
        assert recorder.writer.dataset is None
        assert recorder._episode_limit_reached() is False
    finally:
        if recorder is not None:
            recorder.destroy_node()
        rclpy.shutdown()


def test_rl_label_and_parquet_are_readable_before_session_finishes(tmp_path: Path) -> None:
    """Mixed outcomes retain all rows, rewards and category-local indices."""

    parquet = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    writers = {
        category: LeRobotWriter(
            root=tmp_path / category,
            repo_id=f"local/rl_{category}",
            robot_type="xarm7_factr",
            fps=30,
            vcodec="h264",
            image_writer_threads=1,
            streaming_encoding=False,
        )
        for category in ("success", "failure", "crash")
    }
    recorder = RLEpisodeRecorder.__new__(RLEpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.writer_queue = Queue()
    recorder.writer = writers["success"]
    recorder.category_writers = writers
    recorder.config = SimpleNamespace(fps=30)
    recorder.frame_writer_error = ""
    recorder.sync_dropped_frames = 0
    recorder.writer_queue_dropped_frames = 0
    recorder._complete_episode_if_ready = lambda: None
    for writer in writers.values():
        writer.prepare(_frame())
    try:
        counts = {category: 0 for category in writers}
        for category in ("crash", "success", "failure", "crash", "success"):
            reward = {"crash": -0.5, "success": 1.0, "failure": 0.0}[category]
            attempt = counts[category]
            for index in range(3):
                frame = _frame()
                frame["action.source"][0] = 1 if index == 1 else 0
                frame["action.intervention"][0] = index == 1
                recorder.writer.add_frame(frame)
            recorder.captured_frames_in_episode = 3
            recorder.episode_category = category
            recorder.pending_rl_label = {
                "outcome": category,
                "reward": reward,
                "terminated": True,
                "truncated": False,
                "reason": "xarm_error_12",
            }
            recorder._write_episode(True)
            assert recorder.episode_write_error == ""
            counts[category] += 1
            assert {name: writer.episode_count for name, writer in writers.items()} == counts
            label_path = tmp_path / category / "rl" / f"episode_{attempt:06d}.json"
            label = json.loads(label_path.read_text())
            assert label["source_frame_counts"] == {"policy": 2, "human": 1, "hold": 0}
            assert label["terminal_frame_index"] == 2
            # Footers must be readable while the live writer is still ready to
            # record a subsequent episode; reopening must preserve earlier rows.
            shards = sorted((tmp_path / category / "data").rglob("*.parquet"))
            rows = [row for path in shards for row in parquet.read_table(path).to_pylist()]
            assert len(rows) == 3 * (attempt + 1)
            assert [row["reward"] for row in rows] == [0.0, 0.0, reward] * (attempt + 1)
    finally:
        for writer in writers.values():
            writer.finalize()
