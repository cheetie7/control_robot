"""Convert stamped ROS sensor messages into fixed-shape NumPy values."""

from typing import Any

import numpy as np

_IMAGE_ENCODINGS = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def stamp_seconds(message: Any) -> float:
    """Return ``message.header.stamp`` as scalar ROS seconds."""

    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def image_array(message: Any) -> np.ndarray:
    """Decode a ROS Image into a contiguous ``(H,W,C)`` or ``(H,W)`` array.

    Args:
        message: ``sensor_msgs/Image`` using one supported integer or float encoding.

    Returns:
        NumPy image preserving the physical rows and columns while removing row
        padding. Color output remains in its declared ROS channel order.
    """

    if message.encoding not in _IMAGE_ENCODINGS:
        raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")
    dtype, channels = _IMAGE_ENCODINGS[message.encoding]
    item_size = np.dtype(dtype).itemsize
    row_items = int(message.step) // item_size
    expected_items = int(message.width) * channels
    if row_items < expected_items:
        raise ValueError("ROS image step is smaller than its declared width")
    array = np.frombuffer(message.data, dtype=dtype).reshape(int(message.height), row_items)
    array = array[:, :expected_items]
    shape = (int(message.height), int(message.width), channels)
    array = array.reshape(shape)
    if bool(message.is_bigendian) != (np.dtype(dtype).byteorder == ">") and item_size > 1:
        array = array.byteswap()
    return np.ascontiguousarray(array[..., 0] if channels == 1 else array)


def rgb_image(message: Any) -> np.ndarray:
    """Return one ROS color image as contiguous uint8 RGB ``(H,W,3)``."""

    image = image_array(message)
    if message.encoding in {"bgr8", "bgra8"}:
        image = image[..., [2, 1, 0]]
    elif message.encoding in {"rgba8", "bgra8"}:
        image = image[..., :3]
    elif image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.dtype != np.uint8 or image.shape[-1] != 3:
        raise ValueError("RGB recording streams must decode to uint8 shape (H,W,3)")
    return np.ascontiguousarray(image)


def depth_preview(
    message: Any,
    *,
    input_unit: str,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """Convert metric depth to a fixed-range uint8 RGB MP4 representation.

    Args:
        message: Single-channel ROS Image with shape ``(H,W)``.
        input_unit: ``mm`` or ``m`` describing the source numeric values.
        depth_min_m: Metres represented by video value 1.
        depth_max_m: Metres represented by video value 255.

    Returns:
        Grayscale RGB uint8 image with shape ``(H,W,3)``. Zero denotes invalid
        depth. This representation is intentionally visualization-oriented and
        is not a lossless metric-depth encoding.
    """

    depth = metric_depth(message, input_unit=input_unit)
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth - depth_min_m) / (depth_max_m - depth_min_m)
    scaled[valid] = np.clip(np.rint(normalized[valid] * 254.0 + 1.0), 1, 255).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=2)


def metric_depth(message: Any, *, input_unit: str) -> np.ndarray:
    """Return a single-channel float32 depth image measured in metres.

    Args:
        message: ROS Image containing scalar depth values with shape ``(H,W)``.
        input_unit: Physical source unit, either ``mm`` or ``m``.

    Returns:
        Contiguous float32 array with shape ``(H,W)`` in metres. Invalid zero
        and non-finite source values are preserved for the visualization sink.
    """

    if input_unit not in {"mm", "m"}:
        raise ValueError("Depth input_unit must be mm or m")
    depth = image_array(message).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth streams must be single-channel images")
    if input_unit == "mm":
        depth *= 0.001
    return np.ascontiguousarray(depth)


def center_crop_resize(
    image: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    nearest: bool = False,
) -> np.ndarray:
    """Center-crop one decoded stream frame to the output aspect and rescale it.

    Args:
        image: Decoded uint8 stream frame with shape ``(H, W, 3)``.
        source_size: Configured topic ``(width, height)``. A frame of any other
            size raises so a silently changed driver profile cannot mix
            geometries inside one dataset.
        output_size: Stored training-view ``(width, height)``. The largest
            aspect-preserving center crop is taken first, then the crop is
            rescaled; e.g. 1280x720 -> crop 960x720 -> 640x480 (uniform 2/3).
        nearest: Use nearest-neighbour resampling instead of area averaging so
            invalid zero pixels in depth previews are never blended into
            valid neighbouring pixels.

    Returns:
        Contiguous uint8 array with shape ``(output_h, output_w, 3)``.
    """

    import cv2

    source_w, source_h = int(source_size[0]), int(source_size[1])
    output_w, output_h = int(output_size[0]), int(output_size[1])
    height, width = image.shape[:2]
    if (width, height) != (source_w, source_h):
        raise ValueError(
            f"stream frame is {width}x{height} but source_size is "
            f"{source_w}x{source_h}; the camera profile changed"
        )
    # Largest center crop that keeps the output aspect ratio without upscaling.
    scale = min(source_w / output_w, source_h / output_h)
    crop_w = max(1, min(source_w, int(round(output_w * scale))))
    crop_h = max(1, min(source_h, int(round(output_h * scale))))
    x0 = (source_w - crop_w) // 2
    y0 = (source_h - crop_h) // 2
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return np.ascontiguousarray(
        cv2.resize(cropped, (output_w, output_h), interpolation=interpolation)
    )


def joint_values(message: Any, count: int) -> np.ndarray:
    """Return finite ``JointState.position`` values as float32 shape ``(count,)``."""

    if len(message.position) < count:
        raise ValueError(f"JointState requires at least {count} positions")
    values = np.asarray(message.position[:count], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("JointState positions must be finite")
    return values
