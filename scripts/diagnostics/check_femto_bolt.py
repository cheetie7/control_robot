"""Standalone Femto Bolt startup test driven by the unified YAML.

Builds the exact same launch actions the main teleop launch uses for the
configured camera (default ``top_camera``), runs them alone, and waits for
the first color frame while logging every step to a file for later review.
"""

import argparse
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PACKAGE_SRC = Path(__file__).resolve().parents[2]
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

DEFAULT_LOG_PATH = Path.home() / ".config" / "control_robot" / "femto_bolt_test.log"


def main(args: list[str] | None = None) -> None:
    """Start one configured camera alone and report its first-frame latency.

    Args:
        args: Optional CLI argument sequence; ``None`` reads ``sys.argv``.

    Returns:
        ``None``. The launch service is shut down after the first frame or on
        timeout, and every diagnostic line appends to ``--log-file``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-config",
        type=Path,
        default=PACKAGE_SRC / "xarm_ros2_teleop" / "config" / "teleop_system.yaml",
    )
    parser.add_argument("--camera", default="top_camera", help="YAML 中的相机名")
    parser.add_argument("--timeout", type=float, default=90.0, help="等待首帧秒数")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"输出日志路径，默认 {DEFAULT_LOG_PATH}",
    )
    options = parser.parse_args(args)

    log_path = options.log_file.expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        """Print one line to stdout and append it to the test log file."""

        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    from xarm_ros2_teleop.camera.driver_actions import build_camera_actions
    from xarm_ros2_teleop.system.config import load_system_config

    system = load_system_config(options.system_config.expanduser())
    camera = next(
        (item for item in system.cameras if item.name == options.camera), None
    )
    if camera is None:
        available = [item.name for item in system.cameras]
        raise ValueError(
            f"相机 {options.camera!r} 不存在；可用: {available}"
        )
    camera = replace(camera, enabled=True, startup_delay_sec=0.0)

    log(
        f"=== 会话 {datetime.now():%Y-%m-%d %H:%M:%S} | camera={camera.name} "
        f"vendor={camera.vendor} serial={camera.serial_number} "
        f"color={camera.color.width}x{camera.color.height}@{camera.color.fps} ==="
    )
    log(f"topic: /{camera.topic_prefix}/color, 等待首帧上限 {options.timeout:.0f} s")

    import rclpy
    from launch import LaunchDescription, LaunchService
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    wait_node = Node("femto_bolt_startup_test")
    received = threading.Event()
    first_frame = {}

    def on_image(message: Image) -> None:
        """Record the first received frame and wake the waiting watchdog."""

        if received.is_set():
            return
        first_frame["width"] = message.width
        first_frame["height"] = message.height
        first_frame["encoding"] = message.encoding
        received.set()

    topic = f"/{camera.topic_prefix}/color"
    wait_node.create_subscription(
        Image, topic, on_image, qos_profile_sensor_data
    )
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(wait_node,), daemon=True
    )
    spin_thread.start()

    actions = build_camera_actions(camera)
    launch_service = LaunchService(argv=[])
    launch_service.include_launch_description(LaunchDescription(actions))
    log("相机驱动已单独启动（带 respawn），等待首帧…")

    # LaunchService.run() must own the main thread; a watchdog thread stops it
    # on the first frame or at the timeout, whichever comes first.
    def watchdog() -> None:
        """Stop the launch on the first frame or when the deadline expires."""

        deadline = time.monotonic() + options.timeout
        while not received.is_set() and time.monotonic() < deadline:
            time.sleep(0.2)
        launch_service.shutdown()

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()
    launch_service.run()

    if received.is_set():
        log(
            f"成功: 首帧 {first_frame['width']}x{first_frame['height']} "
            f"({first_frame['encoding']})"
        )
    else:
        log(
            f"失败: {options.timeout:.0f} s 内未收到 {topic} 首帧。"
            "请把上方驱动容器的报错输出一并发给我。"
        )
    wait_node.destroy_node()
    rclpy.shutdown()
    log(f"测试结束，驱动已停止。日志: {log_path}")


if __name__ == "__main__":
    main()
