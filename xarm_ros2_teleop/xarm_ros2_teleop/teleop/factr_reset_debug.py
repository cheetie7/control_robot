"""Standalone FACTR reset-servo debug tool; runs without a ROS master.

Opens the eight-servo Dynamixel bus in current mode and repeats the exact
reset control law used by the leader node (gravity compensation plus the
interpolated PD toward the calibration-derived home pose that corresponds to
``xarm.initial_joint_positions``), while printing per-joint target, measured
position, error, commanded torque and measured current for bench tuning.
Every line is appended to a log file so a session can be reviewed afterwards.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from ament_index_python.packages import get_package_share_directory
from factr_teleop.dynamixel.driver import DynamixelDriver

from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.teleop.controller import FACTRTeleop, interpolated_pd_torque
from xarm_ros2_teleop.teleop.factr_calibration_cli import _factr_config_path
from xarm_ros2_teleop.teleop.offset_calibration import (
    derive_factr_home,
    load_offset_calibration,
)

ARM_JOINT_COUNT = 7
DEFAULT_LOG_PATH = Path.home() / ".config" / "control_robot" / "factr_reset_debug.log"


def main(args: list[str] | None = None) -> None:
    """Drive FACTR to its home pose while streaming the servo state to stdout.

    Args:
        args: Optional CLI argument sequence; ``None`` reads ``sys.argv``.

    Returns:
        ``None``. Zero current is written and torque is disabled on any exit,
        including Ctrl+C. All output lines also append to ``--log-file``.
    """

    package_share = Path(get_package_share_directory("xarm_ros2_teleop"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-config",
        type=Path,
        default=package_share / "config" / "teleop_system.yaml",
    )
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 等于一直运行")
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--kp", type=float, default=None, help="覆盖 YAML kp")
    parser.add_argument("--kd", type=float, default=None, help="覆盖 YAML kd")
    parser.add_argument("--step", type=float, default=None, help="覆盖 YAML 插值步长")
    parser.add_argument(
        "--max-current-ma", type=float, default=None, help="覆盖 YAML 电流上限"
    )
    parser.add_argument(
        "--gravity-gain",
        type=float,
        default=None,
        help="覆盖重力增益（默认取 gravity_comp.gain）",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=None,
        help="覆盖规划速度上限 rad/s",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=None,
        help="覆盖规划加速度 rad/s²",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"输出日志路径，默认 {DEFAULT_LOG_PATH}",
    )
    options = parser.parse_args(args)
    if options.hz <= 0 or options.print_hz <= 0 or options.duration < 0:
        raise ValueError("hz/print_hz/duration 参数非法")

    log_path = options.log_file.expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        """Print one line to stdout and append it to the debug log file."""

        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    system = load_system_config(options.system_config.expanduser())
    factr_config_path = _factr_config_path(system.teleop["config_file"], package_share)
    factr_config = yaml.safe_load(factr_config_path.read_text(encoding="utf-8"))
    dynamixel = factr_config["dynamixel"]
    servo_types = list(dynamixel["servo_types"])
    joint_signs = np.asarray(dynamixel["joint_signs"], dtype=float)
    position_config = factr_config["controller"]["joint_position_control"]
    kp = float(options.kp or position_config["kp"])
    kd = float(options.kd or position_config["kd"])
    step = float(options.step or position_config["interpolation_step_size"])
    max_current_ma = float(
        options.max_current_ma or system.reset["factr_max_current_ma"]
    )
    if not 0 < max_current_ma <= 900:
        raise ValueError("电流上限必须在 (0, 900] mA 内")
    gravity_gain = float(
        options.gravity_gain or factr_config["controller"]["gravity_comp"]["gain"]
    )
    velocity = float(
        options.velocity or system.reset.get("factr_reset_velocity_rad_s", 1.2)
    )
    acceleration = float(
        options.acceleration
        or system.reset.get("factr_reset_acceleration_rad_s2", 3.0)
    )

    calibration = load_offset_calibration(
        system.teleop["offset_file"],
        dynamixel_port=dynamixel["dynamixel_port"],
        servo_types=servo_types,
        joint_signs=joint_signs,
        calibration_joint_positions=np.asarray(
            factr_config["arm_teleop"]["initialization"]["start_joints"], dtype=float
        )[:ARM_JOINT_COUNT],
        operating_mode=0,
        xarm_to_factr_signs=np.asarray(system.teleop["xarm_to_factr_signs"], dtype=float),
    )
    offsets = calibration.joint_offsets_rad
    xarm_initial = np.asarray(system.xarm["initial_joint_positions"], dtype=float)
    origin = derive_factr_home(
        calibration.factr_reference_joint_positions,
        xarm_initial,
        calibration.xarm_reference_joint_positions,
        np.asarray(system.teleop["xarm_to_factr_signs"], dtype=float),
    )
    gripper_target = float(system.reset.get("factr_gripper_open_position_rad", 0.0))

    urdf_path = package_share / "urdf" / factr_config["arm_teleop"]["leader_urdf"]
    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()
    gravity_controller = _gravity_controller(
        pin_model,
        pin_data,
        calibration.gravity_joint_position_offset_rad,
        gravity_gain,
    )

    log(
        f"=== 会话 {datetime.now():%Y-%m-%d %H:%M:%S} | kp={kp} kd={kd} "
        f"step={step} 上限={max_current_ma:.0f}mA 重力增益={gravity_gain} "
        f"速度={velocity}rad/s 加速度={acceleration}rad/s² hz={options.hz} ==="
    )
    log("xArm 初始位 rad: " + np.array2string(xarm_initial, precision=3))
    log("FACTR 目标 rad : " + np.array2string(origin, precision=3))
    log("请清空 FACTR 周围空间；Ctrl+C 停止并断力矩。")
    input("按 Enter 开始: ")
    log("开始伺服。")

    driver = DynamixelDriver(
        np.arange(len(servo_types)) + 1,
        servo_types,
        dynamixel["dynamixel_port"],
        baudrate=int(dynamixel.get("baudrate", 1000000)),
    )
    # Current mode with torque enabled; mirrors leader node startup exactly.
    driver.enter_current_control()
    period = 1.0 / options.hz
    print_period = 1.0 / options.print_hz
    started = time.monotonic()
    last_print = 0.0
    gripper_pos_prev = None
    cmd_target = None
    cmd_speed = 0.0
    try:
        while True:
            cycle_start = time.monotonic()
            if options.duration > 0 and cycle_start - started >= options.duration:
                break
            raw_positions, raw_velocities = driver.get_positions_and_velocities()
            arm_pos = (
                raw_positions[:ARM_JOINT_COUNT] - offsets[:ARM_JOINT_COUNT]
            ) * joint_signs[:ARM_JOINT_COUNT]
            arm_vel = raw_velocities[:ARM_JOINT_COUNT] * joint_signs[:ARM_JOINT_COUNT]
            gripper_pos = (
                raw_positions[ARM_JOINT_COUNT] - offsets[ARM_JOINT_COUNT]
            ) * joint_signs[ARM_JOINT_COUNT]
            gripper_vel = (
                0.0
                if gripper_pos_prev is None
                else (gripper_pos - gripper_pos_prev) / period
            )
            gripper_pos_prev = gripper_pos

            # Nearest whole-turn equivalent target: a multi-turn wrist may rest
            # branches away from home and must travel the short way. The
            # commanded target follows a trapezoidal profile toward it.
            goal = origin + 2.0 * np.pi * np.rint(
                (arm_pos - origin) / (2.0 * np.pi)
            )
            if cmd_target is None:
                cmd_target = arm_pos.copy()
                cmd_speed = 0.0
            cmd_speed = min(cmd_speed + acceleration * period, velocity)
            advance = cmd_speed * period
            cmd_target = cmd_target + np.clip(goal - cmd_target, -advance, advance)
            torque_arm = FACTRTeleop.gravity_compensation(
                gravity_controller, arm_pos, arm_vel
            ) + interpolated_pd_torque(
                arm_pos, arm_vel, cmd_target, kp=kp, kd=kd, step=step
            )
            torque_gripper = (
                -kp * (gripper_pos - gripper_target) - kd * gripper_vel
            )
            driver.set_torque(
                np.append(torque_arm, torque_gripper) * joint_signs,
                max_current_ma=max_current_ma,
            )

            now = time.monotonic()
            if now - last_print >= print_period:
                last_print = now
                currents = driver.latest_currents_ma
                error = goal - arm_pos
                for index in range(ARM_JOINT_COUNT):
                    current = (
                        "  n/a"
                        if currents is None
                        else f"{currents[index]:5.0f}mA"
                    )
                    log(
                        f"J{index + 1} 终点{origin[index]:+7.3f}"
                        f" 命令{cmd_target[index]:+7.3f}"
                        f" 当前{arm_pos[index]:+7.3f}"
                        f" 误差{error[index]:+7.3f}"
                        f" τ{torque_arm[index]:+6.3f}N·m 电流{current}"
                    )
                log(
                    f"夹爪 目标{gripper_target:+7.3f} 当前{gripper_pos:+7.3f}"
                    f" τ{torque_gripper:+6.3f}N·m",
                )
            time.sleep(max(0.0, cycle_start + period - time.monotonic()))
    except KeyboardInterrupt:
        log("收到 Ctrl+C。")
    finally:
        if not driver.emergency_disable():
            log("FACTR 断电失败！电机可能仍保持力矩，请立即切断 FACTR 电源。")
        driver.close()
        log(f"FACTR 已断力矩，调试结束。日志: {log_path}")


def _gravity_controller(pin_model, pin_data, gravity_offset, gravity_gain):
    """Build the stand-in consumed by ``FACTRTeleop.gravity_compensation``."""

    from types import SimpleNamespace

    return SimpleNamespace(
        pin_model=pin_model,
        pin_data=pin_data,
        num_arm_joints=ARM_JOINT_COUNT,
        gravity_joint_position_offset=np.asarray(gravity_offset, dtype=float),
        gravity_comp_modifier=gravity_gain,
        tau_g=np.zeros(ARM_JOINT_COUNT),
    )


if __name__ == "__main__":
    main()
