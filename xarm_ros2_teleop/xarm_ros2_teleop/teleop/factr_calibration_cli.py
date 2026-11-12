"""Interactive torque-free FACTR encoder calibration command."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from factr_teleop.dynamixel.driver import DynamixelDriver
from xarm.wrapper import XArmAPI

from xarm_ros2_teleop.system.config import load_system_config
from xarm_ros2_teleop.teleop.offset_calibration import (
    calculate_gravity_joint_offset,
    calculate_joint_offsets,
    logical_joint_positions,
    save_offset_calibration,
)

FACTR_OPERATING_MODE = 0
XARM_JOINT_COUNT = 7


def main(args=None) -> None:
    """Read two known FACTR poses and save one validated eight-motor offset YAML.

    Args:
        args: Optional one-dimensional CLI string sequence. ``None`` reads
            process arguments.

    Returns:
        Nothing. The command moves xArm to its configured reference, disables
            FACTR torque in current mode, captures the xArm-home match pose for
            the teleop frame, then captures the folded reference pose for the
            gravity/URDF frame, and writes ``teleop.offset_file`` only after
            both captures pass their placement tolerances and confirmation.
    """

    package_share = Path(get_package_share_directory("xarm_ros2_teleop"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-config",
        type=Path,
        default=package_share / "config" / "teleop_system.yaml",
        help="Unified teleoperation YAML path.",
    )
    parser.add_argument(
        "--max-error-rad",
        type=float,
        default=0.08,
        help="Maximum accepted arm-joint error after pi/2 offset selection.",
    )
    parser.add_argument(
        "--gravity-error-rad",
        type=float,
        default=0.06,
        help="Maximum accepted URDF-frame error at the folded gravity pose.",
    )
    parser.add_argument(
        "--joint-lock-sec",
        type=float,
        default=0.5,
        help="Per-joint within-tolerance dwell before that joint locks.",
    )
    parser.add_argument(
        "--xarm-tolerance-rad",
        type=float,
        default=0.03,
        help="Maximum accepted xArm joint error after calibration movement.",
    )
    options = parser.parse_args(args)
    tolerances = (
        options.max_error_rad,
        options.gravity_error_rad,
        options.joint_lock_sec,
        options.xarm_tolerance_rad,
    )
    if any(value <= 0 for value in tolerances):
        raise ValueError("calibration tolerances must be positive")

    system = load_system_config(options.system_config.expanduser())
    if not bool(system.xarm.get("enabled", False)):
        raise ValueError("FACTR calibration requires xarm.enabled=true")
    factr_config_path = _factr_config_path(system.teleop["config_file"], package_share)
    factr_config = yaml.safe_load(factr_config_path.read_text(encoding="utf-8"))
    dynamixel = factr_config["dynamixel"]
    start_joints = np.asarray(
        factr_config["arm_teleop"]["initialization"]["start_joints"], dtype=float
    )
    joint_signs = np.asarray(dynamixel["joint_signs"], dtype=float)
    if start_joints.shape != (8,) or not np.all(np.isfinite(start_joints)):
        raise ValueError("arm_teleop.initialization.start_joints must contain 8 finite radians")
    # URDF coordinates of the physical folded pose used by the second capture.
    model_reference = np.asarray(
        factr_config["controller"]["gravity_comp"]["model_reference_joint_positions"],
        dtype=float,
    )
    if model_reference.shape != (7,) or not np.all(np.isfinite(model_reference)):
        raise ValueError(
            "gravity_comp.model_reference_joint_positions must contain 7 finite radians"
        )

    xarm_target = np.asarray(system.xarm["initial_joint_positions"], dtype=float)
    if xarm_target.shape != (XARM_JOINT_COUNT,) or not np.all(np.isfinite(xarm_target)):
        raise ValueError("xarm.initial_joint_positions must contain 7 finite radians")
    print("xArm 将移动到标定姿态 rad:")
    print(np.array2string(xarm_target, precision=4))
    input("请清空机械臂周围空间，停止主 launch，按 Enter 开始移动；Ctrl+C 取消。")
    xarm_reference = _move_xarm_to_calibration_pose(
        system.xarm, xarm_target, options.xarm_tolerance_rad
    )

    driver = DynamixelDriver(
        np.arange(len(dynamixel["servo_types"])) + 1,
        dynamixel["servo_types"],
        dynamixel["dynamixel_port"],
        baudrate=int(dynamixel.get("baudrate", 1000000)),
    )
    try:
        driver.set_torque_mode(False)
        driver.set_operating_mode(FACTR_OPERATING_MODE)
        driver.verify_operating_mode(FACTR_OPERATING_MODE)
        print("FACTR torque 已关闭。xArm 已在标定姿态。请确认 FACTR 夹爪张开。")
        raw_positions = _sequential_capture_aligned_pose(
            driver,
            start_joints,
            joint_signs,
            options.max_error_rad,
            stage_label="第一段（匹配 xArm home）",
            joint_lock_sec=options.joint_lock_sec,
        )
        offsets = calculate_joint_offsets(raw_positions, start_joints, joint_signs)
        logical_positions = logical_joint_positions(raw_positions, offsets, joint_signs)
        errors = logical_positions - start_joints
        print(f"raw positions rad: {raw_positions.tolist()}")
        print(f"joint offsets rad: {offsets.tolist()}")
        print(f"logical pose rad:  {logical_positions.tolist()}")
        print(f"pose error rad:    {errors.tolist()}")
        max_arm_error = float(np.max(np.abs(errors[:7])))
        if max_arm_error > options.max_error_rad:
            raise RuntimeError(
                f"标定姿态最大关节误差 {max_arm_error:.3f} rad 超过 "
                f"{options.max_error_rad:.3f} rad，请调整 FACTR 后重试"
            )

        # Second capture: gravity-frame calibration at the known folded pose.
        # This replaces the old guessed reference with a physical measurement,
        # so RNEA positions come from the pose the operator actually held.
        print("第二段：重力模型标定。折叠参考姿态即旧版每次开机标定用的折叠位。")
        gravity_raw_positions = _sequential_capture_aligned_pose(
            driver,
            np.append(model_reference, 0.0),
            joint_signs,
            options.gravity_error_rad,
            stage_label="第二段（折叠参考位）",
            joint_lock_sec=options.joint_lock_sec,
        )
        gravity_joint_offset = calculate_gravity_joint_offset(
            gravity_raw_positions, offsets, joint_signs, model_reference
        )
        model_offsets = calculate_joint_offsets(
            gravity_raw_positions, np.append(model_reference, 0.0), joint_signs
        )
        model_position = logical_joint_positions(
            gravity_raw_positions, model_offsets, joint_signs
        )[:7]
        gravity_errors = model_position - model_reference
        print(f"gravity frame offset rad: {gravity_joint_offset.tolist()}")
        print(f"URDF pose at capture rad: {model_position.tolist()}")
        print(f"gravity pose error rad:   {gravity_errors.tolist()}")
        max_gravity_error = float(np.max(np.abs(gravity_errors)))
        if max_gravity_error > options.gravity_error_rad:
            raise RuntimeError(
                f"折叠位最大关节误差 {max_gravity_error:.3f} rad 超过 "
                f"{options.gravity_error_rad:.3f} rad，请重新摆放后重试"
            )

        if input("输入 yes 保存标定，其他输入取消: ").strip().lower() != "yes":
            print("已取消，未写入 offset 文件。")
            return
        save_offset_calibration(
            system.teleop["offset_file"],
            dynamixel_port=dynamixel["dynamixel_port"],
            servo_types=dynamixel["servo_types"],
            joint_signs=joint_signs,
            calibration_joint_positions=start_joints[:7],
            joint_offsets=offsets,
            raw_positions=raw_positions,
            logical_positions=logical_positions,
            gravity_joint_position_offset=gravity_joint_offset,
            operating_mode=FACTR_OPERATING_MODE,
            factr_reference=logical_positions[:7],
            xarm_reference=xarm_reference,
            xarm_to_factr_signs=np.asarray(
                system.teleop["xarm_to_factr_signs"], dtype=float
            ),
        )
        print(f"FACTR 标定已保存: {Path(system.teleop['offset_file']).expanduser()}")
    finally:
        try:
            driver.set_torque_mode(False)
        finally:
            driver.close()


def _sequential_capture_aligned_pose(
    driver: DynamixelDriver,
    target_pose: np.ndarray,
    joint_signs: np.ndarray,
    tolerance_rad: float,
    *,
    stage_label: str,
    joint_lock_sec: float,
) -> np.ndarray:
    """Lock each arm joint independently once it dwells inside the tolerance.

    Each Dynamixel reports its own axis angle, so joint ``i``'s offset depends
    only on the reading taken while joint ``i`` is aligned. Joints therefore
    lock one at a time and keep their captured value when the operator moves
    on, which is equivalent to a simultaneous whole-arm capture.

    Args:
        driver: Torque-disabled Dynamixel driver for the eight FACTR motors.
        target_pose: Desired logical pose, shape ``(8,)`` in rad; the eighth
            value is a gripper target handled by the final averaged sample.
        joint_signs: Raw-to-logical direction signs, shape ``(8,)``.
        tolerance_rad: Per-joint tolerance for both the ``*`` marker and the
            per-joint lock trigger.
        stage_label: Human-readable stage name printed once before the loop.
        joint_lock_sec: Per-joint within-tolerance dwell before that joint
            locks, so sweeping through tolerance never locks a joint.

    Returns:
        Raw encoder sample, shape ``(8,)`` in rad: per-joint locked readings
        for the seven arm joints, followed by a ten-sample gripper average.
        Ctrl+C aborts without capturing.
    """

    if not sys.stdin.isatty():
        input(f"{stage_label}: 请摆到目标姿态后按 Enter 读取编码器。")
        return _average_raw_positions(driver)
    print(f"{stage_label}: 目标关节角 rad {np.array2string(target_pose[:7], precision=3)}")
    print(
        f"容差 {tolerance_rad} rad：逐个关节对准，误差进入容差并保持 "
        f"{joint_lock_sec:.1f} s 后该关节自动锁定；全部锁定即完成。"
        "已锁定的关节可以不再管（各电机读数独立）。Ctrl+C 取消。"
    )
    locked_raw = np.full(7, np.nan)
    within_since = np.full(7, np.nan)
    while not np.all(np.isfinite(locked_raw)):
        raw_positions, _ = driver.get_positions_and_velocities()
        offsets = calculate_joint_offsets(raw_positions, target_pose, joint_signs)
        errors = (
            logical_joint_positions(raw_positions, offsets, joint_signs) - target_pose
        )[:7]
        now = time.monotonic()
        for index in range(7):
            if np.isfinite(locked_raw[index]):
                continue
            if abs(errors[index]) <= tolerance_rad:
                if not np.isfinite(within_since[index]):
                    within_since[index] = now
                elif now - within_since[index] >= joint_lock_sec:
                    locked_raw[index] = float(raw_positions[index])
                    print(f"\r\033[KJ{index + 1} 已锁定（误差 {errors[index]:+.4f} rad）")
            else:
                within_since[index] = np.nan
        cells = " ".join(
            f"J{index + 1} ✓"
            if np.isfinite(locked_raw[index])
            else f"J{index + 1} {errors[index]:+.3f}"
            f"{'*' if abs(errors[index]) > tolerance_rad else ''}"
            for index in range(7)
        )
        remaining = int(np.count_nonzero(np.isnan(locked_raw)))
        print(f"\r\033[K{cells} | 剩余 {remaining} 个关节", end="", flush=True)
        time.sleep(0.05)
    print("\r\033[K全部关节已锁定，读取夹爪样本…")
    gripper_raw = _average_raw_positions(driver)[7]
    return np.append(locked_raw, gripper_raw)


def _average_raw_positions(driver: DynamixelDriver) -> np.ndarray:
    """Return the mean of ten consecutive raw encoder samples, shape ``(8,)``."""

    samples = []
    for _ in range(10):
        raw_positions, _ = driver.get_positions_and_velocities()
        samples.append(np.asarray(raw_positions, dtype=float))
    return np.mean(samples, axis=0)


def _move_xarm_to_calibration_pose(
    xarm_config: dict, target: np.ndarray, tolerance_rad: float
) -> np.ndarray:
    """Move xArm to the absolute seven-joint calibration pose.

    Args:
        xarm_config: Unified YAML xArm mapping containing IP and motion limits.
        target: Absolute xArm joint angles, float array shape ``(7,)`` in rad.
        tolerance_rad: Maximum final per-joint position error in rad.

    Returns:
        Measured absolute xArm joint angles, float array shape ``(7,)`` in rad.
        SDK errors, limit violations or final pose errors raise before the
        FACTR serial device is opened or its offset file is changed.
    """

    arm = XArmAPI(str(xarm_config["ip"]), is_radian=True, do_not_open=True)
    try:
        arm.connect()
        if not arm.connected:
            raise ConnectionError(f"xArm 连接失败: {xarm_config['ip']}")
        status_code, error_warn = arm.get_err_warn_code()
        _require_sdk_success("读取 error/warn", status_code)
        if int(error_warn[0]) != 0:
            _require_sdk_success("clean_error", arm.clean_error())
            print(f"xArm error={int(error_warn[0])} 已清除，等待 1 秒。")
            time.sleep(1.0)
        _require_sdk_success("motion_enable", arm.motion_enable(True))
        _require_sdk_success("set_mode(0)", arm.set_mode(0))
        _require_sdk_success("set_state(0)", arm.set_state(0))
        limit_code, is_limited = arm.is_joint_limit(target.tolist(), is_radian=True)
        _require_sdk_success("检查关节限位", limit_code)
        if bool(is_limited):
            raise ValueError("xArm 标定姿态超出 SDK 关节限位")
        _require_sdk_success(
            "移动到标定姿态",
            arm.set_servo_angle(
                angle=target.tolist(),
                speed=float(xarm_config["initial_joint_speed_rad_s"]),
                mvacc=float(xarm_config["max_joint_acceleration_rad_s2"]),
                is_radian=True,
                wait=True,
            ),
        )
        state_code, joint_states = arm.get_joint_states(is_radian=True)
        _require_sdk_success("读取最终关节位置", state_code)
        actual = np.asarray(joint_states[0][:XARM_JOINT_COUNT], dtype=float)
        if actual.shape != (XARM_JOINT_COUNT,) or not np.all(np.isfinite(actual)):
            raise RuntimeError("xArm 最终关节位置无效")
        max_error = float(np.max(np.abs(actual - target)))
        if max_error > tolerance_rad:
            raise RuntimeError(
                f"xArm 标定姿态最大误差 {max_error:.3f} rad 超过 {tolerance_rad:.3f} rad"
            )
        print(f"xArm 已到达标定姿态，最大关节误差 {max_error:.4f} rad。")
        return actual
    finally:
        arm.disconnect()


def _require_sdk_success(operation: str, code: int) -> None:
    """Validate one xArm SDK return code.

    Args:
        operation: Human-readable hardware operation name.
        code: Integer SDK result where zero denotes success.

    Returns:
        Nothing. A nonzero code raises ``RuntimeError`` immediately.
    """

    if code != 0:
        raise RuntimeError(f"xArm {operation} 失败, SDK code={code}")


def _factr_config_path(config_file: str, package_share: Path) -> Path:
    """Resolve one FACTR driver YAML path.

    Args:
        config_file: Absolute path or package-config filename as a string.
        package_share: Installed xarm_ros2_teleop share directory.

    Returns:
        Absolute ``Path`` pointing to the FACTR driver YAML.
    """

    configured_path = Path(config_file).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return package_share / "config" / config_file


if __name__ == "__main__":
    main()
