# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Based on:
# https://github.com/wuphilipp/gello_software/blob/main/gello/dynamixel/driver.py
# ---------------------------------------------------------------------------


from threading import Lock
from typing import Protocol, Sequence

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_LOBYTE,
)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_CURRENT = 126
LEN_PRESENT_CURRENT = 2
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3
CURRENT_BASED_POSITION_CONTROL_MODE = 5

POSITION_UNIT_RAD = np.pi / 2048.0
PROFILE_VELOCITY_UNIT_RAD_S = 0.229 * 2.0 * np.pi / 60.0
PROFILE_ACCELERATION_UNIT_RAD_S2 = 214.577 * 2.0 * np.pi / 3600.0

TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000 / 2.69,
    "XL330_M288_T": 1470 / 0.52,
    "XL330_M077_T": 1470 / 0.215,
}


class DynamixelCommunicationError(RuntimeError):
    """Report a recoverable serial transaction failure on the FACTR bus."""


class DynamixelDriverProtocol(Protocol):
    """Interface for current and current-limited-position Dynamixel control."""

    def set_current(self, currents: Sequence[float]):
        """Set the current for the Dynamixel servos.

        Args:
            currents (Sequence[float]): A list of currents in mA.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_positions(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""

    def reconnect_current_control(self) -> None:
        """Reopen the bus and restore zero-current operating mode."""

    def get_positions_and_velocities(
        self, tries: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return joint position and velocity arrays, each with shape ``(N,)``."""


class DynamixelDriver(DynamixelDriverProtocol):
    """Drive a one-dimensional Dynamixel chain in current or position mode.

    Joint positions and velocities are returned as arrays of shape ``(N,)`` in
    radians and rad/s. Normal teleoperation uses mode 0 current commands;
    warm-up/reset uses mode 5 Goal Position with a bounded Goal Current.
    """

    def __init__(
        self,
        ids: Sequence[int],
        servo_types: Sequence[str],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
    ):
        """Open one serial bus and configure synchronous read/write groups.

        Args:
            ids: Motor IDs as an integer sequence of length ``N``.
            servo_types: Servo model names as a string sequence of length ``N``.
            port: Linux serial-device path used for the physical bus.
            baudrate: Serial communication rate in bits per second.
        """

        self._ids = ids
        self._port = port
        self._baudrate = int(baudrate)
        self._positions = None
        self._currents_ma = None
        self._lock = Lock()

        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_CURRENT,
            LEN_PRESENT_CURRENT + LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
        )
        self._groupPositionSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")

        self.torque_to_current_map = np.array(
            [TORQUE_TO_CURRENT_MAPPING[servo] for servo in servo_types]
        )

        self._torque_enabled = False
        self._operating_mode = None
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

    @property
    def torque_enabled(self) -> bool:
        """Return whether motor torque/current output is enabled."""

        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        """Enable or disable output torque for every configured motor.

        Args:
            enable: ``True`` enables physical motor output; ``False`` disables it.

        Returns:
            ``None``. A communication error raises ``RuntimeError``.
        """

        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise DynamixelCommunicationError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )
        self._torque_enabled = enable

    def close(self):
        """Close the physical serial port and return ``None``."""

        self._portHandler.closePort()

    def reset_port(self):
        """Reopen the serial bus at the configured baudrate.

        Returns:
            ``None``. A failed reopen raises ``RuntimeError``.
        """

        self._portHandler.closePort()
        if not self._portHandler.openPort():
            raise DynamixelCommunicationError("Failed to reopen the port")
        if not self._portHandler.setBaudRate(self._baudrate):
            raise DynamixelCommunicationError(
                f"Failed to change the baudrate, {self._baudrate}"
            )

    def reconnect_current_control(self) -> None:
        """Reopen the serial bus and restore zero-current mode-0 control.

        Returns:
            ``None`` after torque-off, current-mode selection, zero current and
            torque-on all succeed. A communication failure raises so the ROS
            leader can retry without terminating the complete launch.
        """

        self.reset_port()
        self.enter_current_control()

    def emergency_disable(self, retries: int = 3) -> bool:
        """Best-effort zero-current write followed by torque disable.

        Args:
            retries: Number of attempts. A failed attempt reopens the serial
                port before the next one so a mid-transaction abort or a
                poisoned sync-write group cannot leave the motors energized.

        Returns:
            ``True`` when every motor accepted zero current and torque off;
            ``False`` when every attempt failed and the caller must tell the
            operator to cut the actuator power.
        """

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self.set_current(np.zeros(len(self._ids)))
                self.set_torque_mode(False)
                return True
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                last_error = exc
                if attempt < retries:
                    try:
                        self.reset_port()
                    except Exception as reset_error:  # noqa: BLE001
                        last_error = reset_error
        if last_error is not None:
            print(f"emergency_disable failed: {last_error}")
        return False

    def set_operating_mode(self, mode: int):
        """Write one Dynamixel operating-mode integer to every motor.

        Args:
            mode: Control-table operating-mode value shared by all motors.

        Returns:
            Nothing. Torque must already be disabled by the caller.
        """

        for dxl_id in self._ids:
            dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE, mode
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise DynamixelCommunicationError(
                    f"Failed to set operating mode for Dynamixel with ID {dxl_id}"
                )
        self._operating_mode = int(mode)

    def verify_operating_mode(self, expected_mode: int):
        """Raise ``RuntimeError`` unless all motors report ``expected_mode``.

        Args:
            expected_mode: Expected scalar control-table mode value.

        Returns:
            Nothing when every motor matches.
        """

        for dxl_id in self._ids:
            mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                self._portHandler, dxl_id, ADDR_OPERATING_MODE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                raise DynamixelCommunicationError(
                    f"Operating mode mismatch for Dynamixel ID {dxl_id}"
                )

    def get_positions_and_velocities(self, tries=10):
        """Read all motor positions and velocities with bounded retries.

        Args:
            tries: Remaining synchronous-read retries after communication errors.

        Returns:
            Tuple of float arrays ``(position_rad, velocity_rad_s)``, each with
            shape ``(N,)``.
        """

        _positions = np.zeros(len(self._ids), dtype=int)
        _velocities = np.zeros(len(self._ids), dtype=int)
        _currents_ma = np.zeros(len(self._ids), dtype=int)

        # perform the group sync read transaction
        dxl_comm_result = self._groupSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            if tries > 0:
                return self.get_positions_and_velocities(tries - 1)
            else:
                raise DynamixelCommunicationError(
                    f"Warning, communication failed: {dxl_comm_result}"
                )

        for i, dxl_id in enumerate(self._ids):
            if self._groupSyncRead.isAvailable(
                dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            ):
                current = self._groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
                )
                if current > 0x7FFF:
                    current -= 0x10000
                _currents_ma[i] = current
            else:
                raise DynamixelCommunicationError(
                    f"Failed to get current for Dynamixel with ID {dxl_id}"
                )

            # read velocity data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                velocity = self._groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                )
                # apply sign correction
                if velocity > 0x7FFFFFFF:
                    velocity -= 0x100000000
                _velocities[i] = velocity
            else:
                raise DynamixelCommunicationError(
                    f"Failed to get velocity for Dynamixel with ID {dxl_id}"
                )

            # read position data
            if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                position = self._groupSyncRead.getData(
                    dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                )
                # apply sign correction
                if position > 0x7FFFFFFF:
                    position -= 0x100000000
                _positions[i] = position
            else:
                raise DynamixelCommunicationError(
                    f"Failed to get position for Dynamixel with ID {dxl_id}"
                )

        self._positions = _positions
        self._velocities = _velocities
        self._currents_ma = _currents_ma.astype(float)

        # return positions and velocities in meaningful units
        positions_in_radians = _positions / 2048.0 * np.pi
        velocities_in_units = _velocities * 0.229 * 2 * np.pi / 60

        return positions_in_radians, velocities_in_units

    @property
    def latest_currents_ma(self) -> np.ndarray | None:
        """Return the latest measured motor currents, shape ``(N,)`` in mA.

        Returns:
            A copied float array after a successful state read, otherwise
            ``None`` before the first sample.
        """

        return None if self._currents_ma is None else self._currents_ma.copy()

    def set_current(self, currents: Sequence[float], max_current_ma: float = 900.0):
        """Write motor current commands after applying a symmetric safety limit.

        Args:
            currents: Current commands in milliamps, sequence length ``N``.
            max_current_ma: Positive symmetric current limit in milliamps. The
                physical driver ceiling remains 900 mA.

        Returns:
            ``None``. Invalid shape or communication state raises an exception.
        """

        if len(currents) != len(self._ids):
            raise ValueError("The length of currents must match the number of servos")
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set currents")
        if not 0 < max_current_ma <= 900:
            raise ValueError("max_current_ma must be in (0, 900]")

        currents = np.clip(currents, -max_current_ma, max_current_ma)
        self._sync_write_currents(currents)

    def _sync_write_currents(self, currents: Sequence[float]) -> None:
        """Write signed Goal Current values, shape ``(N,)`` in milliamps."""

        # Clear stale parameters first so a previously failed transaction
        # cannot poison every later sync write.
        self._groupSyncWrite.clearParam()
        try:
            for dxl_id, current in zip(self._ids, currents):
                current_value = int(current)

                param_goal_current = [
                    DXL_LOBYTE(current_value),
                    DXL_HIBYTE(current_value),
                ]

                if not self._groupSyncWrite.addParam(dxl_id, param_goal_current):
                    raise DynamixelCommunicationError(
                        f"Failed to set current for Dynamixel with ID {dxl_id}"
                    )
            dxl_comm_result = self._groupSyncWrite.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise DynamixelCommunicationError("Failed to syncwrite goal current")
        finally:
            self._groupSyncWrite.clearParam()

    def enter_current_control(self) -> None:
        """Enter mode 0 with zero Goal Current before torque is enabled.

        Returns:
            Nothing. The motors are ready for normal gravity/current commands.
        """

        self.set_torque_mode(False)
        self.set_operating_mode(CURRENT_CONTROL_MODE)
        self.verify_operating_mode(CURRENT_CONTROL_MODE)
        self._sync_write_currents(np.zeros(len(self._ids)))
        self.set_torque_mode(True)

    def enter_current_based_position_control(
        self,
        goal_positions_rad: Sequence[float],
        *,
        max_current_ma: float,
        profile_velocity_rad_s: float,
        profile_acceleration_rad_s2: float,
        position_p_gain: int,
        position_i_gain: int,
        position_d_gain: int,
    ) -> None:
        """Enter mode 5 and move toward one bounded multi-turn position target.

        Args:
            goal_positions_rad: Raw motor angles, sequence shape ``(N,)`` in rad.
            max_current_ma: Positive Goal Current limit in mA, at most 900.
            profile_velocity_rad_s: Positive internal profile speed in rad/s.
            profile_acceleration_rad_s2: Positive internal acceleration in rad/s².
            position_p_gain: Raw Position P Gain value in ``[0, 16383]``.
            position_i_gain: Raw Position I Gain value in ``[0, 16383]``.
            position_d_gain: Raw Position D Gain value in ``[0, 16383]``.

        Returns:
            Nothing. The goal is preloaded with torque disabled, then written
            again after torque enable so the default Drive Mode starts a new
            position profile.
        """

        goals = self._position_array(goal_positions_rad)
        if not 0 < max_current_ma <= 900:
            raise ValueError("max_current_ma must be in (0, 900]")
        velocity_value = self._profile_value(
            profile_velocity_rad_s,
            PROFILE_VELOCITY_UNIT_RAD_S,
            "profile_velocity_rad_s",
        )
        acceleration_value = self._profile_value(
            profile_acceleration_rad_s2,
            PROFILE_ACCELERATION_UNIT_RAD_S2,
            "profile_acceleration_rad_s2",
        )
        p_gain = self._gain_value(position_p_gain, "position_p_gain")
        i_gain = self._gain_value(position_i_gain, "position_i_gain")
        d_gain = self._gain_value(position_d_gain, "position_d_gain")
        self.set_torque_mode(False)
        self.set_operating_mode(CURRENT_BASED_POSITION_CONTROL_MODE)
        self.verify_operating_mode(CURRENT_BASED_POSITION_CONTROL_MODE)
        for dxl_id in self._ids:
            self._write_unsigned_2byte_register(
                dxl_id, ADDR_POSITION_P_GAIN, p_gain
            )
            self._write_unsigned_2byte_register(
                dxl_id, ADDR_POSITION_I_GAIN, i_gain
            )
            self._write_unsigned_2byte_register(
                dxl_id, ADDR_POSITION_D_GAIN, d_gain
            )
            self._write_unsigned_register(
                dxl_id, ADDR_PROFILE_VELOCITY, velocity_value
            )
            self._write_unsigned_register(
                dxl_id, ADDR_PROFILE_ACCELERATION, acceleration_value
            )
        self._sync_write_currents(np.full(len(self._ids), max_current_ma))
        # Preload a safe target before enabling output. With the default Drive
        # Mode, a Goal Position update made while torque is disabled does not
        # start a profile, so repeat it immediately after torque is enabled.
        self._sync_write_positions(goals)
        self.set_torque_mode(True)
        self._sync_write_positions(goals)

    def set_goal_positions(self, goal_positions_rad: Sequence[float]) -> None:
        """Write raw multi-turn Goal Position angles, shape ``(N,)`` in rad.

        Args:
            goal_positions_rad: Raw servo targets, sequence shape ``(N,)`` in rad.

        Returns:
            Nothing. The bus must already be in mode 5 with torque enabled.
        """

        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set goal positions")
        if self._operating_mode != CURRENT_BASED_POSITION_CONTROL_MODE:
            raise RuntimeError("Goal Position requires current-based position mode")
        self._sync_write_positions(self._position_array(goal_positions_rad))

    def _sync_write_positions(self, positions_rad: np.ndarray) -> None:
        """Encode raw radians, shape ``(N,)``, as signed 32-bit positions."""

        position_values = np.rint(positions_rad / POSITION_UNIT_RAD).astype(np.int64)
        if np.any(np.abs(position_values) > 1_048_575):
            raise ValueError("Goal Position exceeds current-based position range")
        for dxl_id, position_value in zip(self._ids, position_values):
            unsigned = int(position_value) & 0xFFFFFFFF
            encoded = [(unsigned >> shift) & 0xFF for shift in (0, 8, 16, 24)]
            if not self._groupPositionSyncWrite.addParam(dxl_id, encoded):
                raise RuntimeError(f"Failed to set goal position for Dynamixel ID {dxl_id}")
        dxl_comm_result = self._groupPositionSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")
        self._groupPositionSyncWrite.clearParam()

    def _write_unsigned_register(self, dxl_id: int, address: int, value: int) -> None:
        """Write one nonnegative 32-bit control-table value to one motor.

        Args:
            dxl_id: Integer bus ID of the target motor.
            address: Integer control-table address.
            value: Nonnegative integer register value.

        Returns:
            Nothing after a successful SDK transaction.
        """

        dxl_comm_result, dxl_error = self._packetHandler.write4ByteTxRx(
            self._portHandler, int(dxl_id), int(address), int(value)
        )
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            raise RuntimeError(
                f"Failed to write address {address} for Dynamixel ID {dxl_id}"
            )

    def _write_unsigned_2byte_register(
        self, dxl_id: int, address: int, value: int
    ) -> None:
        """Write one unsigned 16-bit RAM value to one Dynamixel.

        Args:
            dxl_id: Integer bus ID of the target motor.
            address: Integer control-table address.
            value: Integer register value in ``[0, 65535]``.

        Returns:
            Nothing after a successful SDK transaction.
        """

        dxl_comm_result, dxl_error = self._packetHandler.write2ByteTxRx(
            self._portHandler, int(dxl_id), int(address), int(value)
        )
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            raise RuntimeError(
                f"Failed to write address {address} for Dynamixel ID {dxl_id}"
            )

    def _position_array(self, positions_rad: Sequence[float]) -> np.ndarray:
        """Return one finite raw-position array with shape ``(N,)`` in radians.

        Args:
            positions_rad: Sequence containing one raw angle per motor.

        Returns:
            Float array shape ``(N,)`` in radians.
        """

        positions = np.asarray(positions_rad, dtype=float)
        if positions.shape != (len(self._ids),) or not np.all(np.isfinite(positions)):
            raise ValueError("Goal positions must contain one finite radian value per servo")
        return positions

    @staticmethod
    def _profile_value(value: float, unit: float, name: str) -> int:
        """Convert one positive SI profile value to a register integer.

        Args:
            value: Positive scalar velocity or acceleration in SI units.
            unit: SI value represented by one control-table count.
            name: Configuration field name used in validation errors.

        Returns:
            Positive integer register value.
        """

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        register_value = max(1, int(round(value / unit)))
        if register_value > 32767:
            raise ValueError(f"{name} exceeds the Dynamixel profile range")
        return register_value

    @staticmethod
    def _gain_value(value: int, name: str) -> int:
        """Validate one raw Dynamixel position gain register value.

        Args:
            value: Integer P, I or D gain register value.
            name: Configuration field name used in validation errors.

        Returns:
            The validated integer in ``[0, 16383]``.
        """

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 16383
        ):
            raise ValueError(f"{name} must be an integer in [0, 16383]")
        return value

    def set_torque(self, torques: Sequence[float], max_current_ma: float = 900.0):
        """Convert ``N`` motor torques to current with a scalar mA limit."""

        currents = self.torque_to_current_map * torques
        self.set_current(currents, max_current_ma=max_current_ma)


def main():
    """Run an interactive seven-motor hardware smoke test and return ``None``."""

    # Fixed values are intentionally limited to this manual hardware diagnostic.
    ids = [1, 2, 3, 4, 5, 6, 7]
    servo_types = ["XL330_M288_T"] * len(ids)
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISV6J-if00-port0"

    try:
        driver = DynamixelDriver(ids, servo_types, port=port, baudrate=57600)
    except FileNotFoundError:
        print(f"Port {port} not found. Please check the connection.")
        return

    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    try:
        while True:
            positions, _ = driver.get_positions_and_velocities()
            print(f"Current joint positions for IDs {ids}: {positions}")

            current_values = [0, 0, 0, 0, 0, 0, 0.0]
            driver.set_current(current_values)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()


if __name__ == "__main__":
    main()
