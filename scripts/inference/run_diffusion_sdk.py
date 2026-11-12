"""Ubuntu 20.04 native SDK runner: fresh observation -> infer -> execute whole chunk."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xarm_ros2_teleop"))

from xarm_ros2_teleop.inference.contracts import (  # noqa: E402
    build_observation,
    interpolate_action_chunk,
    validate_action_chunk,
    validate_policy_metadata,
)
from xarm_ros2_teleop.inference.sdk_cameras import SDKCamera, snapshot_images  # noqa: E402
from xarm_ros2_teleop.inference.sdk_robot import SDKRobot  # noqa: E402
from xarm_ros2_teleop.inference.sdk_video import SDKVideoRecorder  # noqa: E402
from xarm_ros2_teleop.inference.websocket_policy import PolicyClient  # noqa: E402
from xarm_ros2_teleop.system.config import load_system_config  # noqa: E402


def execute_chunk(robot: SDKRobot, actions: np.ndarray, frequency: float, period: float) -> None:
    """Execute [T,8] over T/frequency seconds; interpolate on the local clock, no catch-up burst."""

    started = time.monotonic()
    while True:
        tick = time.monotonic()
        action = interpolate_action_chunk(actions, tick - started, frequency)
        if action is None:
            break
        robot.command(action)
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    # Even on a missed final deadline, retain the final target during the next inference.
    robot.command(actions[-1])


def main() -> None:
    """Load shared YAML and run bounded, synchronous chunks; --execute explicitly enables motion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "xarm_ros2_teleop/config/inference_diffusion.yaml"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--activate-gripper",
        action="store_true",
        help="Run the full-stroke activation cycle; requires --execute",
    )
    parser.add_argument("--skip-initial-move", action="store_true")
    parser.add_argument("--record-video", action="store_true", help="Record the camera views to MP4")
    parser.add_argument(
        "--video-dir", type=Path, default=Path("~/datasets/diffusion_eval/videos"),
        help="Parent directory for timestamped video runs",
    )
    args = parser.parse_args()
    if args.activate_gripper and not args.execute:
        parser.error("--activate-gripper requires --execute")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    system = load_system_config(args.config)
    config = system.inference
    logging.info("%s", "MOTION ARMED" if args.execute else "DRY RUN: feedback only")
    with ExitStack() as resources:
        client = PolicyClient(
            config.host, config.port, receive_timeout=config.inference_timeout_sec
        )
        resources.callback(client.close)
        for warning in validate_policy_metadata(
            client.metadata, [image.key for image in config.images], config.policy_frequency_hz
        ):
            logging.warning(warning)
        devices = {device.name: device for device in system.cameras if device.enabled}
        cameras = {}
        for image in config.images:
            camera = SDKCamera(devices[image.key], image)
            resources.callback(camera.close)
            camera.start()
            cameras[image.key] = camera
        for key, camera in cameras.items():
            if not camera.ready.wait(10):
                raise TimeoutError(f"camera {key} did not become ready")
        # Verify all cameras before any motion-related initialization.
        initial_images = snapshot_images(
            cameras, config.observation_timeout_sec, config.max_camera_skew_sec
        )
        recorder = None
        if args.record_video:
            recorder = SDKVideoRecorder(
                args.video_dir, cameras, config.observation_timeout_sec, config.max_camera_skew_sec
            )
            resources.callback(recorder.close)
            recorder.prepare(initial_images)
        robot = SDKRobot(system, execute=args.execute)
        resources.callback(robot.close)
        robot.connect(args.activate_gripper, args.skip_initial_move)
        if recorder is not None:
            # Signal recording to stop, then shut down the robot before waiting for encoding.
            resources.callback(recorder.stop)
            recorder.start()
        started = time.monotonic()
        for index in range(config.max_chunks):
            if time.monotonic() - started >= config.max_runtime_sec:
                break
            images = snapshot_images(
                cameras, config.observation_timeout_sec, config.max_camera_skew_sec
            )
            state = robot.state()
            observation = build_observation(state[:7], float(state[7]), images, config.prompt)
            infer_started = time.monotonic()
            actions = validate_action_chunk(
                client.infer(observation),
                state,
                np.asarray(config.joint_lower_rad),
                np.asarray(config.joint_upper_rad),
                config.max_joint_step_rad,
            )
            if recorder is not None:
                recorder.raise_if_failed()
            if time.monotonic() - started >= config.max_runtime_sec:
                break
            # Also validate against measured state now, in case the robot moved during inference.
            validate_action_chunk(
                actions,
                robot.state(),
                np.asarray(config.joint_lower_rad),
                np.asarray(config.joint_upper_rad),
                config.max_joint_step_rad,
            )
            logging.info(
                "chunk %d: %s, inference %.3fs, first target %s",
                index + 1,
                actions.shape,
                time.monotonic() - infer_started,
                actions[0],
            )
            execute_chunk(robot, actions, config.policy_frequency_hz, system.xarm["control_period"])
        if recorder is not None:
            recorder.raise_if_failed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; hardware connections closed.")
