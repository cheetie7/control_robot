"""Structured inference state and deployment contract checks without hardware."""

import json
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from std_msgs.msg import String
from xarm_ros2_teleop.inference.contracts import validate_policy_metadata
from xarm_ros2_teleop.inference.remote_policy_node import RemotePolicyNode
from xarm_ros2_teleop.inference.status import InferenceStatus
from xarm_ros2_teleop.recording.episode_recorder import EpisodeRecorder
from xarm_ros2_teleop.rl.episode_recorder import RLEpisodeRecorder
from xarm_ros2_teleop.rl.recording import ACTION_SOURCES, rl_schema_fields, write_episode_label


def test_status_wire_separates_phase_and_reason_without_changing_mode_rules() -> None:
    """A policy error ends evaluation but leaves manual RL open for human labeling."""

    status = InferenceStatus("EPISODE_FAILED", "camera: timed out")
    assert InferenceStatus.from_wire(status.to_json()) == status
    assert InferenceStatus.from_wire(status.text) == status
    assert not InferenceStatus("STOPPED_SOMETHING_ELSE").failed
    sent = []
    RemotePolicyNode._publish_status(
        SimpleNamespace(status_pub=SimpleNamespace(publish=sent.append)), status.text
    )
    assert json.loads(sent[0].data)["reason"] == "camera: timed out"
    evaluation = SimpleNamespace(
        evaluation_mode=True, lock=threading.RLock(), state="RECORDING",
        _finish_evaluation_episode=lambda **outcome: sent.append(outcome),
    )
    EpisodeRecorder.inference_status_callback(evaluation, String(data=status.to_json()))
    assert sent[-1]["outcome"] == "failure"
    rl = SimpleNamespace(
        require_remote_policy=True, lock=threading.RLock(), state="RECORDING",
        policy_connected=True, get_logger=lambda: SimpleNamespace(warn=lambda _text: None),
    )
    RLEpisodeRecorder.inference_status_callback(rl, String(data=status.to_json()))
    assert rl.state == "RECORDING"
    assert not rl.inference_ready
    assert rl.last_error == "inference_camera: timed out"


def test_status_readiness_uses_exact_phases() -> None:
    """Similar prefixes cannot accidentally enable episode start or stop controls."""

    assert InferenceStatus("WAITING_FOR_EPISODE").reusable
    assert InferenceStatus("PAUSED_BY_HUMAN").reusable
    assert not InferenceStatus("WAITING_FOR_EPISODE_BROKEN").reusable
    assert not InferenceStatus("STOPPED").can_stop
    assert InferenceStatus("INFERENCING").can_stop
    with pytest.raises(ValueError, match="schema"):
        InferenceStatus.from_wire('{"schema_version":99,"phase":"STOPPED","reason":""}')


@pytest.mark.parametrize("field,value", [
    ("protocol_version", 99), ("state_dim", 16), ("action_format", "delta"),
    ("image_keys", ["right_camera", "left_camera", "top_camera"]),
    ("action_frequency_hz", 100), ("execution_horizon", 0),
])
def test_handshake_rejects_deployment_mismatches(field: str, value) -> None:
    """Camera order, radians, state width and timing must match before robot motion."""

    fixture = json.loads((Path(__file__).parent / "fixtures/robot_contract_v1.json").read_text())
    metadata = {**fixture["metadata"], field: value}
    with pytest.raises(ValueError):
        validate_policy_metadata(metadata, fixture["metadata"]["image_keys"], 30)


def test_legacy_checkpoint_timing_is_warned_not_guessed() -> None:
    """Existing weights remain deployable at an explicitly configured local rate."""

    fixture = json.loads((Path(__file__).parent / "fixtures/robot_contract_v1.json").read_text())
    metadata = {**fixture["metadata"], "action_frequency_hz": None}
    assert len(validate_policy_metadata(metadata, metadata["image_keys"], 30)) == 1


def test_incompatible_connection_is_closed_before_becoming_ready() -> None:
    """A completed socket is not a ready policy until its metadata is checked."""

    fixture = json.loads((Path(__file__).parent / "fixtures/robot_contract_v1.json").read_text())
    closed, errors = [], []
    client = SimpleNamespace(
        metadata={**fixture["metadata"], "action_frequency_hz": 100},
        close=lambda: closed.append(True),
    )
    future = Future()
    future.set_result(client)
    node = SimpleNamespace(
        client=None, connect_future=future,
        tcp_mode=False,
        config=SimpleNamespace(
            images=[SimpleNamespace(key=key) for key in fixture["metadata"]["image_keys"]],
            policy_frequency_hz=30,
        ),
        _stop=errors.append,
        _publish_status=lambda _state: pytest.fail("unexpected policy-ready state"),
    )
    RemotePolicyNode._finish_connection(node)
    assert closed == [True]
    assert node.client is None
    assert "frequency" in errors[0]


@pytest.mark.parametrize("changes", [
    {"reward": float("nan")}, {"terminated": True, "truncated": True},
    {"outcome": "unknown"},
])
def test_invalid_episode_labels_do_not_create_partial_files(tmp_path: Path, changes: dict) -> None:
    """Invalid reward or terminal meaning is rejected before a label directory appears."""

    label = {
        "outcome": "success", "reward": 1.0, "terminated": True, "truncated": False,
        **changes,
    }
    with pytest.raises(ValueError):
        write_episode_label(tmp_path, 0, label)
    assert not (tmp_path / "rl").exists()


def test_rl_recording_fixture_matches_numeric_schema_and_versioned_label(tmp_path: Path) -> None:
    """Source coding and timestamps are durable data contracts, not folder-name guesses."""

    fixture = json.loads((Path(__file__).parent / "fixtures/robot_contract_v1.json").read_text())
    recording = fixture["recording"]
    assert recording["action_source_encoding"] == ACTION_SOURCES
    fields = rl_schema_fields()
    assert fields["action.policy"].shape == (8,)
    assert fields["observation.timestamp_ros_s"].dtype.name == recording["timestamp_dtype"]
    write_episode_label(tmp_path, 0, {
        "outcome": "failure", "reward": 0.0, "terminated": False, "truncated": True,
    })
    label = json.loads((tmp_path / "rl/episode_000000.json").read_text())
    assert label["schema_version"] == 1
    assert label["action_format"] == fixture["metadata"]["action_format"]
    assert label["truncated"] and not label["terminated"]
