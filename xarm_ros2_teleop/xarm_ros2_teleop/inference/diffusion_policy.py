"""LeRobot AWBC Diffusion adapter; run in the training environment, without ROS."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from .contracts import ACTION_FORMAT, PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class DiffusionChunkPolicy:
    """Load one checkpoint and map RGB/state requests to physical joint actions.

    ``policy`` and saved processors live on ``device``. Inputs contain uint8 HWC
    images and state [8]; output is float32 [n_action_steps,8], in rad and gripper
    [0,1]. This adapter supports single-frame checkpoints (n_obs_steps=1).
    """

    def __init__(
        self, checkpoint: str | Path, device: str = "cuda", frequency_hz: float = 30.0,
    ) -> None:
        """Load local weights strictly and restore processors; frequency is training Hz."""

        checkpoint = Path(checkpoint).expanduser().resolve()
        config = DiffusionConfig.from_pretrained(checkpoint)
        if config.n_obs_steps != 1:
            raise ValueError("this synchronous adapter requires n_obs_steps=1")
        if config.robot_state_feature.shape != (8,) or config.action_feature.shape != (8,):
            raise ValueError("this robot requires 8-D joint/gripper state and actions")
        config.device = device
        self.policy = DiffusionPolicy.from_pretrained(checkpoint, config=config, strict=True)
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.image_features = config.image_features
        # The scheduler is mutable: serialize calls if another socket connects.
        self.lock = Lock()
        self.metadata = {
            "server": "control_robot_diffusion",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": "diffusion_awbc",
            "state_dim": 8,
            "action_dim": 8,
            "action_format": ACTION_FORMAT,
            "execution_horizon": config.n_action_steps,
            "image_keys": [key.removeprefix("observation.images.") for key in self.image_features],
            "action_frequency_hz": frequency_hz,
            "checkpoint": str(checkpoint),
            "num_inference_steps": self.policy.diffusion.num_inference_steps,
        }

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Convert one wire observation to absolute float32 [T,8] actions.

        Images must match checkpoint CHW dimensions after HWC conversion. Saved
        processors own normalization; the network owns its internal resize.
        No action queue or temporal history survives between requests.
        """

        with self.lock:
            started = time.perf_counter()
            state = np.asarray(observation["state"], dtype=np.float32)
            if state.shape != (8,):
                raise ValueError("state must have shape (8,)")
            batch = {"observation.state": torch.from_numpy(state.copy())}
            for feature_key, feature in self.image_features.items():
                image_key = feature_key.removeprefix("observation.images.")
                image = np.asarray(observation["images"][image_key])
                channels, height, width = feature.shape
                if image.dtype != np.uint8 or image.shape != (height, width, channels):
                    raise ValueError(f"{image_key} must be uint8 {(height, width, channels)}")
                # HWC uint8 -> CHW float [0,1], then the saved MEAN_STD transform.
                batch[feature_key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            batch = self.preprocess(batch)
            # Preprocessing adds B; direct chunk prediction also needs observation time S.
            batch["observation.state"] = batch["observation.state"].unsqueeze(1)
            normalized = self.policy.predict_action_chunk(batch)  # [1,10,8], already sliced
            actions = self.postprocess(normalized)[0].numpy()  # Saved MIN_MAX inverse, on CPU.
            logger.info(
                "chunk %s in %.1f ms", actions.shape, (time.perf_counter() - started) * 1000
            )
            return np.ascontiguousarray(actions, dtype=np.float32)
