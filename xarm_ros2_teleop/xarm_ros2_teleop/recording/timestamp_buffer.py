"""Bounded timestamp buffers used for deterministic sensor synchronization."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimestampedSample:
    """Associate one decoded sensor value with a ROS timestamp in seconds.

    Args:
        timestamp: Finite ROS time expressed as scalar seconds.
        value: Decoded scalar, vector or image value with modality-specific shape.
    """

    timestamp: float
    value: Any


class TimestampBuffer:
    """Keep recent samples ordered by timestamp for nearest-frame queries."""

    def __init__(self, duration_sec: float) -> None:
        """Create an empty buffer covering ``duration_sec`` seconds."""

        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        self.duration_sec = float(duration_sec)
        self._samples: list[TimestampedSample] = []

    def add(self, timestamp: float, value: Any) -> TimestampedSample:
        """Insert one value and remove samples outside the configured window."""

        sample = TimestampedSample(float(timestamp), value)
        timestamps = [item.timestamp for item in self._samples]
        self._samples.insert(bisect_right(timestamps, sample.timestamp), sample)
        newest = self._samples[-1].timestamp
        cutoff = newest - self.duration_sec
        first_valid = bisect_right([item.timestamp for item in self._samples], cutoff)
        if first_valid:
            del self._samples[:first_valid]
        return sample

    def nearest(self, timestamp: float) -> TimestampedSample | None:
        """Return the closest sample, preferring the earlier sample on a tie."""

        if not self._samples:
            return None
        timestamps = [item.timestamp for item in self._samples]
        index = bisect_right(timestamps, float(timestamp))
        candidates = self._samples[max(0, index - 1) : min(len(self._samples), index + 1)]
        return min(candidates, key=lambda item: (abs(item.timestamp - timestamp), item.timestamp))

    def latest_before(self, timestamp: float) -> TimestampedSample | None:
        """Return the newest causal sample whose timestamp is not after ``timestamp``."""

        index = bisect_right([item.timestamp for item in self._samples], float(timestamp))
        return None if index == 0 else self._samples[index - 1]

    def newest(self) -> TimestampedSample | None:
        """Return the sample with the greatest timestamp, or ``None`` when empty."""

        return self._samples[-1] if self._samples else None

    def clear(self) -> None:
        """Remove all buffered values without changing the configured duration."""

        self._samples.clear()
