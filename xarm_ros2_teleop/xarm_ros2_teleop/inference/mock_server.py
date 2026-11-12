"""Safe hold-position policy server for end-to-end dry-run checks."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from .websocket_policy import PolicyServer


class HoldPositionPolicy:
    """Repeat the received physical state as an absolute action chunk."""

    def __init__(self, chunk_size: int, delay_sec: float) -> None:
        """Configure output length in steps and artificial inference delay in seconds."""

        self.chunk_size = int(chunk_size)
        self.delay_sec = float(delay_sec)

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Return float32 hold targets shape ``(chunk_size,8)``."""

        if self.delay_sec:
            time.sleep(self.delay_sec)
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"expected state shape (8,), got {state.shape}")
        return np.repeat(state[None], self.chunk_size, axis=0)


def main() -> None:
    """Run the local hold-position server from console arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.delay < 0:
        raise ValueError("chunk size must be positive and delay non-negative")
    PolicyServer(
        HoldPositionPolicy(args.chunk_size, args.delay), args.host, args.port
    ).serve_forever()


if __name__ == "__main__":
    main()
