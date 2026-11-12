"""Direct xArm mode-6 and USB-RS485 Robotiq control, independent of ROS."""

from __future__ import annotations

import logging
import time
from threading import Event, Thread
from typing import Any

import numpy as np


def sdk_ok(code: int, operation: str) -> None:
    """Raise on a nonzero xArm SDK return code; never hide command rejection."""

    if code != 0:
        raise RuntimeError(f"{operation} failed: SDK code {code}")


class SDKRobot:
    """Own hardware connections; dry-run allows feedback reads only.

    The arm receives absolute radian targets in mode 6 as in collection. Robotiq
    serial I/O runs at its configured period so it cannot block the arm clock.
    ``gripper_sample`` stores (monotonic time, measured gPO/255); ``target`` is gPR.
    """

    def __init__(self, system: Any, execute: bool) -> None:
        """Store configuration without connecting to hardware."""

        self.config = system
        self.execute = execute
        self.arm = None
        self.gripper = None
        self.armed = False
        self.target = None
        self.gripper_sample = None
        self.error = None
        self.ready = Event()
        self.stopping = Event()
        self.thread = None

    def connect(self, activate_gripper: bool = False, skip_initial_move: bool = False) -> None:
        """Connect for feedback and prepare the arm and open gripper for execution."""

        from pyrobotiqgripper import GRIPPER_MODE_RTU, RobotiqGripper
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(self.config.xarm["ip"], is_radian=True, do_not_open=True)
        self.arm.connect()
        self.gripper = RobotiqGripper(
            com_port=self.config.robotiq["com_port"],
            device_id=self.config.robotiq["device_id"],
            connection_type=GRIPPER_MODE_RTU,
        )
        if self.execute:
            if activate_gripper:
                runtime_timeout = self.gripper.timeOut
                self.gripper.timeOut = self.config.robotiq["activation_timeout_sec"]
                try:
                    self.gripper.activate(reset=True, start=True, refreshStatus=False)
                finally:
                    self.gripper.timeOut = runtime_timeout
            if not self.gripper.isActivated(refreshStatus=True):
                raise RuntimeError(
                    "gripper is inactive; activate explicitly with --activate-gripper"
                )
            sdk_ok(self.arm.motion_enable(True), "motion_enable")
            self.armed = True
            sdk_ok(self.arm.set_mode(0), "set_mode(0)")
            sdk_ok(self.arm.set_state(0), "set_state(0)")
            if not skip_initial_move:
                sdk_ok(
                    self.arm.set_servo_angle(
                        angle=self.config.xarm["initial_joint_positions"],
                        speed=self.config.xarm["initial_joint_speed_rad_s"],
                        is_radian=True,
                        wait=True,
                    ),
                    "initial pose",
                )
            # Keep the same online position planner and speed/acceleration limits as collection.
            sdk_ok(self.arm.set_mode(6), "set_mode(6)")
            sdk_ok(self.arm.set_state(0), "set_state(0)")
            self._open_gripper_for_start()
        self.thread = Thread(target=self._gripper_loop, name="robotiq", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError("no Robotiq feedback within 5 seconds")
        self.check_health()

    def _open_gripper_for_start(self) -> None:
        """Request gPR=0 and wait for measured gPO to reach the configured open tolerance."""

        options = self.config.robotiq
        threshold = self.config.reset.get("robotiq_open_threshold_bits", 5)
        self.gripper.move(
            0,
            speed=options["speed"],
            force=options["force"],
            wait=False,
            readStatus=False,
            refreshStatus=False,
            start=True,
        )
        deadline = time.monotonic() + 5.0
        while (position := self.gripper.position(refreshStatus=True)) > threshold:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Robotiq did not open before inference: gPO={position}")
            time.sleep(options["control_period"])
        logging.info("Initial gripper open confirmed: gPO=%d/255", position)

    def _gripper_loop(self) -> None:
        """Read physical feedback and send changed gPR targets at the configured serial rate."""

        previous = None
        options = self.config.robotiq
        try:
            while not self.stopping.is_set():
                position = self.gripper.position(refreshStatus=True)
                self.gripper_sample = (time.monotonic(), float(position) / 255.0)
                self.ready.set()
                target = self.target
                if (
                    self.armed
                    and target is not None
                    and (previous is None or abs(target - previous) >= options["minimal_motion"])
                ):
                    self.gripper.move(
                        target,
                        speed=options["speed"],
                        force=options["force"],
                        wait=False,
                        readStatus=False,
                        refreshStatus=False,
                        start=True,
                    )
                    previous = target
                self.stopping.wait(options["control_period"])
        except Exception as exc:
            self.error = exc
            self.ready.set()

    def check_health(self) -> None:
        """Reject disconnection, xArm faults or failed/stale gripper I/O before another command."""

        if self.error is not None:
            raise RuntimeError("Robotiq I/O failed") from self.error
        if (
            not self.arm.connected
            or self.arm.error_code
            or (self.armed and self.arm.state in (4, 5))
        ):
            raise RuntimeError("xArm is disconnected, stopped or in error")
        if self.gripper_sample is None or (
            time.monotonic() - self.gripper_sample[0]
            > self.config.inference.observation_timeout_sec
        ):
            raise RuntimeError("Robotiq feedback is stale")

    def state(self) -> np.ndarray:
        """Return measured float32 [7 joint rad, gPO/255], never the previous target."""

        self.check_health()
        code, joints = self.arm.get_servo_angle(is_radian=True)
        sdk_ok(code, "get_servo_angle")
        state = np.asarray(list(joints[:7]) + [self.gripper_sample[1]], dtype=np.float32)
        if not np.all(np.isfinite(state)) or not 0 <= state[-1] <= 1:
            raise RuntimeError("invalid measured joint/gripper feedback")
        return state

    def command(self, action: np.ndarray) -> None:
        """Send a prevalidated absolute [8] action, or perform health reads only in dry-run."""

        self.check_health()
        if not self.armed:
            return
        sdk_ok(
            self.arm.set_servo_angle(
                angle=action[:7].astype(float).tolist(),
                is_radian=True,
                wait=False,
                speed=self.config.xarm["max_joint_speed_rad_s"],
                mvacc=self.config.xarm["max_joint_acceleration_rad_s2"],
            ),
            "policy target",
        )
        self.target = int(np.clip(float(action[7]) * 255, 0, 255))

    def close(self) -> None:
        """Stop serial work, stop armed hardware, then disconnect; dry-run never sends stop."""

        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        serial_busy = self.thread is not None and self.thread.is_alive()
        try:
            try:
                if self.armed and self.arm.connected:
                    sdk_ok(self.arm.set_state(4), "stop arm")
            finally:
                if self.armed and self.gripper is not None and not serial_busy:
                    self.gripper.stop(refreshStatus=True, readStatus=True)
            if serial_busy:
                raise RuntimeError("Robotiq I/O did not stop; serial connection left untouched")
        finally:
            try:
                if self.gripper is not None and not serial_busy:
                    self.gripper.disconnect()
            finally:
                if self.arm is not None:
                    self.arm.disconnect()
