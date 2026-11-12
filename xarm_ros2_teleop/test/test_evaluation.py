"""Pure tests for labeled real-robot policy-evaluation artifacts."""

import json
import threading
import time

import pytest
from std_msgs.msg import String
from xarm_ros2_teleop.evaluation.outcomes import EvaluationOutcomeWriter
from xarm_ros2_teleop.recording.episode_recorder import (
    EpisodeRecorder,
    decode_evaluation_key,
)


def test_evaluation_keys_require_explicit_success_or_failure() -> None:
    """Evaluation uses dedicated labels and never maps right-arrow to save."""

    assert decode_evaluation_key(b"s") == "start"
    assert decode_evaluation_key(b"Y") == "success"
    assert decode_evaluation_key(b"n") == "failure"
    assert decode_evaluation_key(b"\x1b[D") == "discard"
    assert decode_evaluation_key(b"\x1b") == "finish"
    assert decode_evaluation_key(b"\x1b[C") is None


def test_evaluation_outcomes_are_append_only_and_counted(tmp_path) -> None:
    """Committed episode labels survive restart as newline-delimited JSON."""

    writer = EvaluationOutcomeWriter(tmp_path / "dataset")
    writer.append({"episode_index": 0, "outcome": "success", "reward": 1.0})
    writer.append({"episode_index": 1, "outcome": "failure", "reward": 0.0})

    assert writer.counts() == (1, 1)
    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["episode_index"] for record in records] == [0, 1]
    assert [record["outcome"] for record in records] == ["success", "failure"]


def test_evaluation_label_carries_terminal_semantics_into_episode_save() -> None:
    """A human success creates provenance before invoking the save/reset path."""

    calls = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.evaluation_mode = True
    recorder.state = "RECORDING"
    recorder.capture_started_at = time.monotonic() - 2.0
    recorder.captured_frames_in_episode = 1
    recorder.discarded_attempts = 0
    recorder.evaluation_config = {"success_reward": 1.0, "failure_reward": 0.0}
    recorder.instruction = "insert the block"
    recorder.pending_outcome = None
    recorder.last_outcome = ""
    recorder.last_outcome_reason = ""
    recorder._finish_episode = lambda **kwargs: (calls.append(kwargs) or (True, "ok"))

    accepted, _message = recorder._finish_evaluation_episode(
        outcome="success",
        reason="operator_success",
        terminated=True,
        truncated=False,
        automatic_error_reset=False,
    )

    assert accepted is True
    assert recorder.pending_outcome["outcome"] == "success"
    assert recorder.pending_outcome["reward"] == 1.0
    assert recorder.pending_outcome["terminated"] is True
    assert recorder.pending_outcome["truncated"] is False
    assert recorder.pending_outcome["duration_sec"] >= 2.0
    assert calls == [
        {
            "save": True,
            "finish_session": False,
            "automatic_error_reset": False,
        }
    ]


def test_evaluation_timeout_is_a_truncated_failure() -> None:
    """The 30-second deadline labels and resets rather than leaving policy active."""

    outcomes = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.evaluation_mode = True
    recorder.state = "RECORDING"
    recorder.capture_started_at = time.monotonic() - 31.0
    recorder.max_episode_sec = 30.0
    recorder.reset_deadline = None
    recorder.factr_recovery_pending = False
    recorder.factr_connected = True
    recorder._finish_evaluation_episode = lambda **kwargs: outcomes.append(kwargs)
    recorder._check_reset_timeout = lambda _now: None

    recorder.process_pending()

    assert outcomes == [
        {
            "outcome": "failure",
            "reason": "time_limit",
            "terminated": False,
            "truncated": True,
            "automatic_error_reset": False,
        }
    ]


def test_policy_failure_before_first_frame_still_discards_and_resets() -> None:
    """A zero-frame safety failure cannot be saved but must run post-episode reset."""

    calls = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.evaluation_mode = True
    recorder.state = "RECORDING"
    recorder.capture_started_at = time.monotonic()
    recorder.captured_frames_in_episode = 0
    recorder.discarded_attempts = 0
    recorder.evaluation_config = {"failure_reward": 0.0}
    recorder.instruction = "insert the block"
    recorder.pending_outcome = None
    recorder.last_outcome = ""
    recorder.last_outcome_reason = ""
    recorder._finish_episode = lambda **kwargs: (calls.append(kwargs) or (True, "ok"))

    accepted, _message = recorder._finish_evaluation_episode(
        outcome="failure",
        reason="inference_invalid_policy_response",
        terminated=True,
        truncated=False,
        automatic_error_reset=False,
    )

    assert accepted is True
    assert recorder.discarded_attempts == 1
    assert calls == [
        {
            "save": False,
            "finish_session": False,
            "automatic_error_reset": False,
        }
    ]


def test_episode_failed_inference_status_requests_failure_save_and_reset() -> None:
    """An episode-local policy failure reaches the normal evaluation reset path."""

    outcomes = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.evaluation_mode = True
    recorder.state = "RECORDING"
    recorder.inference_ready = True
    recorder._finish_evaluation_episode = lambda **kwargs: outcomes.append(kwargs)

    message = String()
    message.data = "EPISODE_FAILED: invalid policy response: unsafe action"
    recorder.inference_status_callback(message)

    assert recorder.inference_ready is False
    assert outcomes == [
        {
            "outcome": "failure",
            "reason": "inference_invalid policy response: unsafe action",
            "terminated": True,
            "truncated": False,
            "automatic_error_reset": False,
        }
    ]


@pytest.mark.parametrize("episode_index", [None, True, 1.5, "2"])
def test_evaluation_outcome_requires_integer_episode_index(
    tmp_path, episode_index
) -> None:
    """A label without a concrete LeRobot episode cannot enter the sidecar."""

    writer = EvaluationOutcomeWriter(tmp_path / "dataset")
    with pytest.raises(ValueError, match="integer episode_index"):
        writer.append({"episode_index": episode_index, "outcome": "failure"})
