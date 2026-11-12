"""Direct SDK RGB cameras with current frames; no ROS imports or model inference."""

from __future__ import annotations

import time
from threading import Event, Thread
from typing import Any

import cv2
import numpy as np

from ..recording.message_converters import center_crop_resize


def orbbec_rgb(frame: Any) -> np.ndarray:
    """Decode an Orbbec RGB/BGR/MJPG/YUYV frame into RGB uint8 [H,W,3]."""

    height, width = frame.get_height(), frame.get_width()
    name = str(frame.get_format()).rsplit(".", 1)[-1]
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if name == "RGB":
        return data.reshape(height, width, 3).copy()
    if name == "BGR":
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
    if name == "MJPG":
        return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if name in ("YUYV", "YUY2", "UYVY"):
        conversion = cv2.COLOR_YUV2RGB_UYVY if name == "UYVY" else cv2.COLOR_YUV2RGB_YUY2
        return cv2.cvtColor(data.reshape(height, width, 2), conversion)
    raise ValueError(f"unsupported Orbbec color format: {name}")


class SDKCamera:
    """Own one SDK pipeline and a thread storing (monotonic arrival time, RGB frame).

    ``device`` is a shared YAML CameraConfig; ``image`` defines the training crop.
    Capture continues during inference so the next request never uses queued old frames.
    """

    def __init__(self, device: Any, image: Any) -> None:
        """Store configuration; hardware opens only when start() is called."""

        self.device = device
        self.image = image
        self.pipeline = None
        self.latest = None
        self.error = None
        self.ready = Event()
        self.stopping = Event()
        self.thread = None
        self.started = False

    def start(self) -> None:
        """Open the exact serial/profile in YAML and begin RGB capture."""

        color = self.device.color
        if self.device.vendor == "realsense":
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.device.serial_number)
            config.enable_stream(
                rs.stream.color, color.width, color.height, rs.format.rgb8, color.fps
            )
            self.pipeline.start(config)
        else:
            import pyorbbecsdk as ob

            self.context = ob.Context()
            device = self.context.query_devices().get_device_by_serial_number(
                self.device.serial_number
            )
            self.pipeline = ob.Pipeline(device)
            profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            # Select only the configured physical profile; never silently resize a different view.
            formats = (ob.OBFormat.RGB, ob.OBFormat.BGR, ob.OBFormat.MJPG, ob.OBFormat.YUYV)
            candidates = [
                profiles.get_stream_profile_by_index(i).as_video_stream_profile()
                for i in range(profiles.get_count())
            ]
            profile = next(
                (
                    p
                    for fmt in formats
                    for p in candidates
                    if p.get_format() == fmt
                    and p.get_width() == color.width
                    and p.get_height() == color.height
                    and p.get_fps() == color.fps
                ),
                None,
            )
            if profile is None:
                raise ValueError(
                    f"{self.device.name}: no RGB/BGR/MJPG/YUYV profile at "
                    f"{color.width}x{color.height}@{color.fps}"
                )
            config = ob.Config()
            config.enable_stream(profile)
            self.pipeline.start(config)
        self.started = True
        self.thread = Thread(target=self._capture, name=self.device.name, daemon=True)
        self.thread.start()

    def _capture(self) -> None:
        """Read RGB frames, apply the recording crop and publish current snapshots."""

        try:
            while not self.stopping.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                frame = frames.get_color_frame()
                if not frame:
                    continue
                arrived = time.monotonic()
                rgb = (
                    np.asanyarray(frame.get_data()).copy()
                    if self.device.vendor == "realsense"
                    else orbbec_rgb(frame)
                )
                if self.image.output_size is not None:
                    rgb = center_crop_resize(
                        rgb, source_size=self.image.source_size, output_size=self.image.output_size
                    )
                self.latest = (arrived, rgb)
                self.ready.set()
        except Exception as exc:
            if not self.stopping.is_set():
                self.error = exc
                self.ready.set()

    def close(self) -> None:
        """Stop acquisition and join its thread before releasing the camera."""

        self.stopping.set()
        if self.started:
            self.pipeline.stop()
        if self.thread is not None:
            self.thread.join(timeout=2)


def snapshot_images(cameras: dict[str, SDKCamera], timeout: float, skew: float) -> dict:
    """Return fresh RGB images, or raise for missing/stale/skewed frames (seconds)."""

    samples = {}
    for key, camera in cameras.items():
        if camera.error is not None:
            raise RuntimeError(f"camera {key} failed") from camera.error
        if camera.latest is None:
            raise RuntimeError(f"camera {key} has no frame")
        samples[key] = camera.latest
    times = [sample[0] for sample in samples.values()]
    if time.monotonic() - min(times) > timeout:
        raise RuntimeError("camera frames are stale")
    if max(times) - min(times) > skew:
        raise RuntimeError("camera arrival-time skew exceeds configured limit")
    return {key: sample[1] for key, sample in samples.items()}
