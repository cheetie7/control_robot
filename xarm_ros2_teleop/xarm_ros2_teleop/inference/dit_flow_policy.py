"""Adapt a standalone DiT Flow Matching checkpoint to the robot's wire protocol."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from awbc_training.inference import load_policy

from .contracts import ACTION_FORMAT, PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class DiTFlowChunkPolicy:
    """Serve complete physical action chunks using checkpoint EMA weights.

    ``policy`` owns image/state preprocessing and action denormalization. Requests
    contain uint8 RGB HWC images and state [8]; replies contain float32 [10,8].
    The full prediction horizon is executed, regardless of saved n_action_steps=8.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        frequency_hz: float = 30.0,
    ) -> None:
        """Load a local .pt file strictly through the project's EMA inference loader."""

        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        logger.info("Loading DiT Flow EMA weights from %s on %s", checkpoint, device)
        self.policy = load_policy(checkpoint, use_ema=True, device=device)
        config = self.policy.config
        if config.observation.state_dim != 8 or config.dit.action_dim != 8:
            raise ValueError("this robot requires 8-D joint/gripper state and actions")
        self.camera_names = config.camera_names
        self.lock = Lock()
        self.metadata = {
            "server": "control_robot_dit_flow",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": "dit_flow_fm",
            "state_dim": 8,
            "action_dim": 8,
            "action_format": ACTION_FORMAT,
            "execution_horizon": config.dit.action_horizon,
            "image_keys": [key.removeprefix("observation.images.") for key in self.camera_names],
            "action_frequency_hz": frequency_hz,
            "checkpoint": str(checkpoint),
            "weights": "ema",
            "sampler": "euler",
            "num_inference_steps": config.num_inference_steps,
        }

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Map short camera names to training names and return unnormalized [T,8] targets.

        Reuse the project's RGB resize/ImageNet and MIN_MAX preprocessing exactly
        once. Complete chunks bypass predict_action(), which would slice to 8 steps.
        Saturate the physical gripper target to [0,1], logging any overshoot.
        Existing SDK action checks remain responsible for joint limits.
        """

        with self.lock:
            started = time.perf_counter()
            state = np.asarray(observation["state"], dtype=np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("state must be finite shape (8,)")
            images = {}
            for name in self.camera_names:
                key = name.removeprefix("observation.images.")
                image = np.asarray(observation["images"][key])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"{key} must be uint8 RGB with shape (H,W,3)")
                images[name] = image
            prepared = self.policy.prepare_observation({"images": images, "state": state})
            result = self.policy.predict_action_chunk(prepared)
            # The policy already denormalized the entire horizon. Only remove B.
            actions = result["action_chunk"][0].cpu().numpy()
            # Euler samples are unbounded; MIN_MAX inversion does not enforce limits.
            gripper = actions[:, 7]
            if np.any((gripper < 0.0) | (gripper > 1.0)):
                logger.warning(
                    "Clipping gripper predictions to [0,1]: raw min=%.6f max=%.6f",
                    float(gripper.min()),
                    float(gripper.max()),
                )
                actions[:, 7] = np.clip(gripper, 0.0, 1.0)
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings = result["timings"]
            logger.info(
                "chunk %s in %.1f ms; vision %.1f, observation %.1f, Euler %.1f, "
                "denormalization %.1f, model total %.1f ms",
                actions.shape,
                elapsed_ms,
                timings["vision_ms"],
                timings["observation_ms"],
                timings["euler_ms"],
                timings["denormalization_ms"],
                timings["total_ms"],
            )
            return np.ascontiguousarray(actions, dtype=np.float32)
