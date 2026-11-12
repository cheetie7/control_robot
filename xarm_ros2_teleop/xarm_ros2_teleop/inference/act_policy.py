"""Serve the supplied DETR-VAE ACT model with checkpoint-native preprocessing."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from torchvision.transforms.functional import normalize

from .contracts import ACTION_FORMAT, PROTOCOL_VERSION, STATE_DIM

logger = logging.getLogger(__name__)


class ACTChunkPolicy:
    """Map three RGB views and joint/gripper feedback to absolute ACT waypoints.

    The checkpoint supplies the network, camera order and four [8] statistics.
    The raw DETR-VAE runs in eval mode with the zero prior latent and returns
    all query slots. No training optimizer or temporal action cache is needed.
    """

    def __init__(
        self, checkpoint: str | Path, act_root: str | Path,
        device: str = "cuda", frequency_hz: float = 30.0,
    ) -> None:
        """Restore a format-v2 joint ACT BC checkpoint using the supplied ACT source."""

        checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        act_root = Path(act_root).expanduser().resolve(strict=True)
        if not (act_root / "detr/main.py").is_file():
            raise ValueError("--act-root must point to the directory containing detr/main.py")
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("action frequency must be positive and finite")
        self.device = torch.device(device)

        # The checkpoint stores float32 statistics as NumPy arrays alongside tensors.
        with torch.serialization.safe_globals([
            np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType,
        ]):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 2 or saved.get("strategy_id") != "act_bc":
            raise ValueError("expected a format-v2 act_bc checkpoint")
        config = saved["model_config"]
        horizon = int(config["num_queries"])
        if config["state_dim"] != STATE_DIM or horizon <= 0 or horizon != config["chunk_size"]:
            raise ValueError("expected 8-D ACT with matching positive query and chunk sizes")
        self.image_keys = tuple(saved["inference_image_keys"])
        camera_names = tuple(config["camera_names"])
        if (
            set(self.image_keys) != {"top_camera", "left_camera", "right_camera"}
            or len(self.image_keys) != 3
            or camera_names != tuple(f"observation.images.{key}" for key in self.image_keys)
            or camera_names != tuple(saved["training_camera_keys"])
        ):
            raise ValueError("checkpoint training and inference camera orders must agree")

        self.stats = {}
        for key in ("state_mean", "state_std", "action_mean", "action_std"):
            values = np.asarray(saved["normalization"][key], dtype=np.float32)
            if (
                values.shape != (STATE_DIM,) or not np.isfinite(values).all()
                or (key.endswith("_std") and np.any(values <= 0))
            ):
                raise ValueError(f"invalid ACT normalization statistic {key}")
            self.stats[key] = values

        # Use the supplied model builder directly. policy.py imports a training-only
        # loss and constructs an optimizer; the checkpoint wraps the model as "model".
        # Upstream imports both detr.main and its legacy top-level util package.
        sys.path[:0] = [str(act_root), str(act_root / "detr")]
        try:
            from detr.main import build_ACT_model, get_args_parser
        finally:
            del sys.path[:2]
        args = get_args_parser().parse_args([])
        vars(args).update(config)
        args.pretrained_backbone = False  # ResNet parameters are included in the checkpoint.
        wrapper = torch.nn.Module()
        wrapper.model = build_ACT_model(args)
        wrapper.load_state_dict(saved["model_state_dict"], strict=True)
        self.model = wrapper.model.to(self.device).eval()
        self.horizon = horizon
        self.lock = Lock()
        self.metadata = {
            "server": "control_robot_act",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": "act_bc",
            "state_dim": STATE_DIM,
            "action_dim": STATE_DIM,
            "action_format": ACTION_FORMAT,
            "execution_horizon": horizon,
            "image_keys": list(self.image_keys),
            "action_frequency_hz": frequency_hz,
            "checkpoint": str(checkpoint),
        }

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Consume finite state [8] and RGB uint8 HWC views; return float32 [T,8]."""

        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("state must be finite shape (8,)")
        images = []
        for key in self.image_keys:
            image = np.asarray(observation["images"][key])
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f"{key} must be uint8 RGB with shape (480,640,3)")
            images.append(image)

        with self.lock:
            started = time.perf_counter()
            qpos = (state - self.stats["state_mean"]) / self.stats["state_std"]
            qpos = torch.as_tensor(qpos, device=self.device).unsqueeze(0)
            # [K,H,W,C] -> [B,K,C,H,W]. Match the provided ACTPolicy RGB transform.
            rgb = torch.as_tensor(np.stack(images), device=self.device, dtype=torch.float32)
            rgb = rgb.permute(0, 3, 1, 2) / 255.0
            rgb = normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            predicted, _, _ = self.model(qpos, rgb.unsqueeze(0), None)
            if predicted.shape != (1, self.horizon, STATE_DIM):
                raise ValueError(f"unexpected ACT prediction shape {tuple(predicted.shape)}")
            actions = (
                predicted[0].cpu().numpy() * self.stats["action_std"] + self.stats["action_mean"]
            )
            if not np.isfinite(actions).all():
                raise ValueError("ACT returned non-finite actions")
            # Only saturate the gripper. The existing client validates physical joint targets.
            np.clip(actions[:, 7], 0.0, 1.0, out=actions[:, 7])
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("ACT chunk %s in %.1f ms", actions.shape, elapsed_ms)
            return np.ascontiguousarray(actions, dtype=np.float32)
