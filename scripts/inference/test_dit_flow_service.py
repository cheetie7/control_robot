"""Test real DiT preprocessing, EMA and full-horizon WebSocket replies without hardware."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import numpy as np
import pytest
import torch
from awbc_data import ImagePreprocessConfig
from awbc_models import (
    ConditionalDiTConfig,
    FlowMatchingPolicy,
    FlowMatchingPolicyConfig,
    ObservationEncoderConfig,
    VisualEncoderConfig,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xarm_ros2_teleop"))

from xarm_ros2_teleop.inference.contracts import validate_policy_metadata  # noqa: E402
from xarm_ros2_teleop.inference.dit_flow_policy import DiTFlowChunkPolicy  # noqa: E402
from xarm_ros2_teleop.inference.websocket_policy import PolicyClient, PolicyServer  # noqa: E402

CAMERAS = ["top_camera", "left_camera", "right_camera"]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    """Bound CPU threads for tiny test networks and restore the process setting afterward."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Save distinct raw/EMA weights with embedded statistics and saved execution length 8."""

    config = FlowMatchingPolicyConfig(
        vision=VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=3),
        observation=ObservationEncoderConfig(
            state_dim=8,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
        ),
        dit=ConditionalDiTConfig(
            action_dim=8,
            action_horizon=10,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            timestep_embedding_dim=32,
            timestep_hidden_dim=32,
        ),
        image_preprocess=ImagePreprocessConfig(size=(32, 48)),
        num_inference_steps=8,
        n_action_steps=8,
    )
    stats = {
        "action": {"min": [-1.0] * 7 + [0.0], "max": [1.0] * 8},
        "observation.state": {"min": [-2.0] * 8, "max": [2.0] * 8},
    }
    model = FlowMatchingPolicy(config, stats=stats)
    raw = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.dit.velocity_head.bias.fill_(0.125)
    path = tmp_path_factory.mktemp("dit_checkpoint") / "policy.pt"
    torch.save(
        {
            "policy_config": asdict(config),
            "normalization_stats": stats,
            "policy": raw,
            "ema": {"policy": model.state_dict()},
        },
        path,
    )
    return path


@pytest.fixture
def adapter(checkpoint):
    """Restore the actual small model through the production adapter, using CPU EMA weights."""

    return DiTFlowChunkPolicy(checkpoint, device="cpu")


def request():
    """Return raw RGB views in reversed insertion order to test checkpoint camera ordering."""

    return {
        "state": np.linspace(-0.5, 0.5, 8, dtype=np.float32),
        "images": {
            key: np.full((48, 64, 3), [i * 50, 100, 200], np.uint8)
            for i, key in reversed(list(enumerate(CAMERAS)))
        },
        "prompt": "insert the block",
    }


def test_full_ten_steps_ema_and_exact_preprocessing(adapter, checkpoint):
    """Compare native and wire actions, including last two steps and one inverse transform."""

    assert adapter.policy.config.n_action_steps == 8
    torch.testing.assert_close(
        adapter.policy.dit.velocity_head.bias,
        torch.full((8,), 0.125),
    )
    wire = request()
    raw = {
        "images": {f"observation.images.{key}": image for key, image in wire["images"].items()},
        "state": wire["state"],
    }
    prepared = adapter.policy.prepare_observation(raw)
    torch.manual_seed(71)
    expected = adapter.policy.predict_action_chunk(prepared)["action_chunk"][0].numpy()
    expected[:, 7] = np.clip(expected[:, 7], 0.0, 1.0)
    torch.manual_seed(71)
    with (
        patch.object(adapter.policy, "predict_action", side_effect=AssertionError("would slice")),
        patch.object(
            adapter.policy.action_normalizer,
            "denormalize",
            wraps=adapter.policy.action_normalizer.denormalize,
        ) as inverse,
        patch.object(
            adapter.policy, "prepare_observation", wraps=adapter.policy.prepare_observation
        ) as prepare,
    ):
        actual = adapter.infer(wire)
    assert inverse.call_count == 1 and prepare.call_count == 1
    assert list(prepare.call_args.args[0]["images"]) == list(adapter.policy.config.camera_names)
    assert actual.shape == (10, 8) and actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    assert adapter.metadata["execution_horizon"] == 10
    assert adapter.metadata["weights"] == "ema" and adapter.metadata["num_inference_steps"] == 8
    # Deployment must not rewrite the checkpoint's n_action_steps or any weights.
    assert torch.load(checkpoint, weights_only=False)["policy_config"]["n_action_steps"] == 8


def test_gripper_overshoot_is_logged_and_saturated_after_denormalization(adapter, caplog):
    """Bound physical gripper outputs while preserving all ten joint targets exactly."""

    physical = torch.arange(80, dtype=torch.float32).reshape(1, 10, 8) / 100
    physical[0, :, 7] = torch.tensor([-0.02, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.03, 0.6])
    expected = physical[0].numpy().copy()
    expected[:, 7] = np.clip(expected[:, 7], 0.0, 1.0)
    timings = dict.fromkeys(
        ["vision_ms", "observation_ms", "euler_ms", "denormalization_ms", "total_ms"], 0.0
    )
    with patch.object(
        adapter.policy,
        "predict_action_chunk",
        return_value={"action_chunk": physical, "timings": timings},
    ):
        actual = adapter.infer(request())
    np.testing.assert_array_equal(actual, expected)
    assert "raw min=-0.020000 max=1.030000" in caplog.text


def test_existing_client_receives_ten_step_chunk(adapter):
    """Validate the real handshake and NumPy binary transport with the unchanged SDK client."""

    server = PolicyServer(adapter, "127.0.0.1", 0, metadata=adapter.metadata).create_server()
    thread = Thread(target=server.serve_forever)
    thread.start()
    client = PolicyClient("127.0.0.1", server.socket.getsockname()[1], receive_timeout=10)
    try:
        assert validate_policy_metadata(client.metadata, CAMERAS, 30.0) == []
        actions = client.infer(request())
        assert actions.shape == (10, 8) and np.isfinite(actions).all()
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.parametrize("bad", ["state", "image"])
def test_bad_observation_fails_before_model_inference(adapter, bad):
    """Reject malformed state or RGB rather than generating a physical target from it."""

    wire = request()
    if bad == "state":
        wire["state"][0] = np.nan
    else:
        wire["images"]["left_camera"] = np.zeros((48, 64), np.uint8)
    with patch.object(adapter.policy, "predict_action_chunk") as predict:
        with pytest.raises(ValueError):
            adapter.infer(wire)
        predict.assert_not_called()


def test_cli_selects_dit_without_importing_old_adapter(checkpoint, monkeypatch):
    """Exercise CLI dispatch while making the old LeRobot adapter unavailable."""

    spec = importlib.util.spec_from_file_location(
        "serve_test", ROOT / "scripts/inference/serve_diffusion.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Importing the old adapter here would fail immediately.
    monkeypatch.setitem(sys.modules, "xarm_ros2_teleop.inference.diffusion_policy", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_diffusion.py",
            "--policy-type",
            "dit-flow",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ],
    )
    with patch.object(module, "PolicyServer") as server:
        module.main()
        policy = server.call_args.args[0]
        assert isinstance(policy, DiTFlowChunkPolicy)
        assert policy.metadata["execution_horizon"] == 10
        server.return_value.create_server.return_value.__enter__.return_value.serve_forever.assert_called_once()
