"""Build a fixed LeRobot feature schema from the first synchronized frame."""

import numpy as np


def build_features(frame: dict[str, np.ndarray | str]) -> dict[str, dict]:
    """Return LeRobot v3 feature definitions for one complete frame.

    Args:
        frame: Mapping containing visual arrays ``(H,W,3)``, float vectors
            ``(N,)`` and the special scalar string key ``task``.

    Returns:
        Feature mapping accepted by ``LeRobotDataset.create``. Visual inputs are
        declared as videos so LeRobot writes MP4 files when ``use_videos=True``.
    """

    features = {}
    for key, value in frame.items():
        if key in {"task", "timestamp"}:
            continue
        array = np.asarray(value)
        if key.startswith("observation.images."):
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Video feature '{key}' must have uint8 shape (H,W,3)")
            features[key] = {
                "dtype": "video",
                "shape": tuple(array.shape),
                "names": ["height", "width", "channel"],
            }
            continue
        if array.dtype not in (np.float32, np.float64) or array.ndim != 1:
            raise ValueError(f"Numeric feature '{key}' must have float32/float64 shape (N,)")
        features[key] = {
            "dtype": str(array.dtype),
            "shape": tuple(array.shape),
            "names": [f"value_{index}" for index in range(array.size)],
        }
    return features
