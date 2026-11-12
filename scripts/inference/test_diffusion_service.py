"""Run with the training environment: pytest scripts/inference/test_diffusion_service.py."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "xarm_ros2_teleop"))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from xarm_ros2_teleop.inference.contracts import validate_policy_metadata
from xarm_ros2_teleop.inference.diffusion_policy import DiffusionChunkPolicy
from xarm_ros2_teleop.inference.websocket_policy import PolicyClient, PolicyServer
from xarm_ros2_teleop.system.config import load_system_config

IMAGE_KEYS = ["top_camera", "left_camera", "right_camera"]


class EchoNormalizedState:
    """Check model input layouts and emit normalized state as a [1,10,8] chunk."""

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Assert real processor outputs, including RGB scaling, then echo the state."""

        assert batch["observation.state"].shape == (1, 1, 8)
        for key, pixel in zip(IMAGE_KEYS, (0, 128, 255), strict=True):
            image = batch[f"observation.images.{key}"]
            assert image.shape == (1, 3, 4, 6)
            torch.testing.assert_close(image, torch.full_like(image, (pixel / 255 - 0.5) / 0.25))
        return batch["observation.state"].expand(1, 10, 8).clone()


@pytest.fixture
def adapter() -> DiffusionChunkPolicy:
    """Build an adapter with actual LeRobot processors and a cheap deterministic model."""

    features = {"observation.state": PolicyFeature(FeatureType.STATE, (8,))}
    features.update({f"observation.images.{key}": PolicyFeature(FeatureType.VISUAL, (3, 4, 6))
                     for key in IMAGE_KEYS})
    config = DiffusionConfig(
        input_features=features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
        n_obs_steps=1, n_action_steps=10, horizon=16, device="cpu", crop_shape=None,
    )
    stats = {
        "observation.state": {"min": torch.zeros(8), "max": torch.full((8,), 2.0)},
        "action": {"min": torch.zeros(8), "max": torch.tensor([2.0] * 7 + [1.0])},
    }
    for key in IMAGE_KEYS:
        stats[f"observation.images.{key}"] = {
            "mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25),
        }
    policy = DiffusionChunkPolicy.__new__(DiffusionChunkPolicy)
    policy.policy = EchoNormalizedState()
    policy.preprocess, policy.postprocess = make_diffusion_pre_post_processors(config, stats)
    policy.image_features = config.image_features
    policy.lock = Lock()
    policy.metadata = {
        "protocol_version": 1, "state_dim": 8, "execution_horizon": 10,
        "action_format": "absolute_joint_rad_plus_gripper",
        "image_keys": IMAGE_KEYS, "action_frequency_hz": 30.0,
    }
    return policy


def observation(state: float = 0.5) -> dict:
    """Return a fresh RGB/state wire request with distinguishable camera contents."""

    return {
        "protocol_version": 1,
        "state": np.full(8, state, dtype=np.float32),
        "images": {key: np.full((4, 6, 3), pixel, dtype=np.uint8)
                   for key, pixel in zip(IMAGE_KEYS, (0, 128, 255), strict=True)},
        "prompt": "insert the block",
    }


def test_physical_action_chunk_and_fresh_observation(adapter: DiffusionChunkPolicy) -> None:
    """Check shape, inverse scaling and use of new state on the second request."""

    for state in (0.5, 1.0):
        actions = adapter.infer(observation(state))
        assert actions.shape == (10, 8) and actions.dtype == np.float32
        np.testing.assert_allclose(actions, np.tile([state] * 7 + [state / 2], (10, 1)), atol=1e-6)


def test_wrong_image_geometry_is_rejected(adapter: DiffusionChunkPolicy) -> None:
    """Reject mismatched camera resolution rather than silently changing training geometry."""

    request = observation()
    request["images"]["top_camera"] = np.zeros((6, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="top_camera"):
        adapter.infer(request)


def test_deployment_profile_loads_with_reset_and_matching_cameras() -> None:
    """Validate the merged deployment profile, including reset prerequisites and camera order."""

    path = Path(__file__).resolve().parents[2] / "xarm_ros2_teleop/config/inference_diffusion.yaml"
    config = load_system_config(path)
    assert [image.key for image in config.inference.images] == IMAGE_KEYS
    assert config.inference.host == "127.0.0.1" and config.inference.port == 8081
    assert config.inference.policy_frequency_hz == 30.0
    assert config.reset["enabled"] and config.evaluation["enabled"]
    assert not config.rl["enabled"]


def test_existing_client_accepts_handshake_and_chunks(adapter: DiffusionChunkPolicy) -> None:
    """Exercise binary MessagePack over a real local socket with the existing robot client."""

    server = PolicyServer(adapter, "127.0.0.1", 0, metadata=adapter.metadata).create_server()
    thread = Thread(target=server.serve_forever)
    thread.start()
    client = PolicyClient("127.0.0.1", server.socket.getsockname()[1])
    try:
        assert validate_policy_metadata(client.metadata, IMAGE_KEYS, 30.0) == []
        for state in (0.5, 1.0):
            actions = client.infer(observation(state))
            np.testing.assert_allclose(actions[:, 7], state / 2, atol=1e-6)
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=5)
