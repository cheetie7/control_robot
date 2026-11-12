"""Serve a LeRobot Diffusion, DiT Flow or ACT checkpoint over WebSocket."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Import only the transport/adapter modules; this process needs no ROS installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "xarm_ros2_teleop"))

from xarm_ros2_teleop.inference.websocket_policy import PolicyServer


def main() -> None:
    """Parse checkpoint/device/endpoint options and serve synchronous action chunks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy-type", choices=("diffusion", "dit-flow", "act"), default="diffusion"
    )
    parser.add_argument("--act-root", type=Path, help="ACT source directory containing detr/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--frequency-hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.policy_type == "act" and args.act_root is None:
        parser.error("--policy-type act requires --act-root")
    # Library imports may have configured the root logger before this entry point.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    # Import only the selected framework.
    if args.policy_type == "dit-flow":
        from xarm_ros2_teleop.inference.dit_flow_policy import DiTFlowChunkPolicy

        policy = DiTFlowChunkPolicy(args.checkpoint, args.device, args.frequency_hz)
    elif args.policy_type == "act":
        from xarm_ros2_teleop.inference.act_policy import ACTChunkPolicy

        policy = ACTChunkPolicy(args.checkpoint, args.act_root, args.device, args.frequency_hz)
    else:
        from xarm_ros2_teleop.inference.diffusion_policy import DiffusionChunkPolicy

        policy = DiffusionChunkPolicy(args.checkpoint, args.device, args.frequency_hz)
    logging.info("Loaded %s", policy.metadata)
    transport = PolicyServer(policy, args.host, args.port, metadata=policy.metadata)
    with transport.create_server() as server:
        logging.info("Listening on ws://%s:%s; waiting for SDK client", args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
