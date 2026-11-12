"""Record the current SDK camera views to separate MP4 files during inference."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np

from .sdk_cameras import SDKCamera, snapshot_images

VIDEO_FPS = 30.0
logger = logging.getLogger(__name__)


class SDKVideoRecorder:
    """Sample fresh RGB uint8 [H,W,3] views and write one MP4 per camera on one thread.

    Each output frame index represents the same sampling tick in every file.
    The sampler uses the SDK's latest frames so slow encoding cannot queue up
    frames or block the control loop.
    """

    def __init__(
        self, root: Path, cameras: dict[str, SDKCamera], timeout: float, skew: float
    ) -> None:
        """Store the destination, camera sources and freshness limits in seconds."""

        self.directory = root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.cameras = cameras
        self.timeout = timeout
        self.skew = skew
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.stopping = Event()
        self.thread: Thread | None = None
        self.error: Exception | None = None
        self.frames_written = 0
        self.frames_skipped = 0

    def prepare(self, images: dict[str, np.ndarray]) -> None:
        """Open one MP4 per view, using the shape of the initial RGB snapshot."""

        self.directory.mkdir(parents=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for key, rgb in images.items():
            height, width = rgb.shape[:2]
            path = self.directory / f"{key}.mp4"
            writer = cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, (width, height))
            self.writers[key] = writer
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")

    def start(self) -> None:
        """Begin continuous sampling after the runner completes initialization."""

        self.thread = Thread(target=self._record, name="sdk-video", daemon=True)
        self.thread.start()
        logger.info("Recording %d views at %.0f FPS to %s", len(self.writers), VIDEO_FPS, self.directory)

    def _record(self) -> None:
        """Write fresh groups on a monotonic 30 Hz clock, skipping missed ticks."""

        period = 1.0 / VIDEO_FPS
        deadline = time.monotonic()
        try:
            while not self.stopping.wait(max(0.0, deadline - time.monotonic())):
                try:
                    images = snapshot_images(self.cameras, self.timeout, self.skew)
                except RuntimeError:
                    # Skip the whole group so the three files retain matching frame indices.
                    self.frames_skipped += 1
                else:
                    for key, writer in self.writers.items():
                        writer.write(cv2.cvtColor(images[key], cv2.COLOR_RGB2BGR))
                    self.frames_written += 1
                deadline += period
                late = time.monotonic() - deadline
                if late >= 0:
                    missed = int(late / period) + 1
                    self.frames_skipped += missed
                    deadline += missed * period
        except Exception as exc:
            self.error = exc
            logger.exception("Video recording failed in %s", self.directory)

    def raise_if_failed(self) -> None:
        """Report an encoder failure to the runner at an action-chunk boundary."""

        if self.error is not None:
            raise RuntimeError(f"Video recording failed in {self.directory}") from self.error

    def stop(self) -> None:
        """Signal the recording thread to stop without waiting for disk I/O."""

        self.stopping.set()

    def close(self) -> None:
        """Join the sampler and finalize MP4 containers, including after interruption."""

        self.stop()
        if self.thread is not None:
            self.thread.join()
        for writer in self.writers.values():
            writer.release()
        if self.thread is not None:
            logger.info(
                "Video output: %s (%d frames per view, %d skipped ticks)",
                self.directory,
                self.frames_written,
                self.frames_skipped,
            )
