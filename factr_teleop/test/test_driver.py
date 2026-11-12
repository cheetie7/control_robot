"""Current and position-command tests for the FACTR Dynamixel driver."""

import numpy as np
from factr_teleop.dynamixel.driver import (
    ADDR_POSITION_D_GAIN,
    ADDR_POSITION_I_GAIN,
    ADDR_POSITION_P_GAIN,
    CURRENT_BASED_POSITION_CONTROL_MODE,
    DynamixelDriver,
)


class _FakeSyncWriter:
    """Record signed current register values written by the Dynamixel driver."""

    def __init__(self):
        """Create an empty integer current history."""

        self.currents = []

    def addParam(self, dxl_id, encoded):
        """Decode one little-endian signed 16-bit current and accept the write."""

        raw = int(encoded[0]) | (int(encoded[1]) << 8)
        self.currents.append(raw - 65536 if raw >= 32768 else raw)
        return True

    def txPacket(self):
        """Return the Dynamixel SDK success code."""

        return 0

    def clearParam(self):
        """Retain history while emulating a cleared SDK parameter buffer."""


class _FakePositionSyncWriter:
    """Record signed 32-bit Goal Position values written by the driver."""

    def __init__(self):
        """Create an empty integer position history."""

        self.positions = []

    def addParam(self, dxl_id, encoded):
        """Decode one little-endian signed 32-bit position and accept it."""

        raw = sum(int(byte) << shift for byte, shift in zip(encoded, (0, 8, 16, 24)))
        self.positions.append(raw - 2**32 if raw >= 2**31 else raw)
        return True

    def txPacket(self):
        """Return the Dynamixel SDK success code."""

        return 0

    def clearParam(self):
        """Retain history while emulating a cleared SDK parameter buffer."""


def test_reset_limit_preserves_normal_current_ceiling() -> None:
    """Reset may clamp at 150 mA while normal calls retain the 900 mA ceiling."""

    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1]
    driver._torque_enabled = True
    driver._groupSyncWrite = _FakeSyncWriter()
    driver.set_current([1000.0], max_current_ma=150.0)
    driver.set_current([1000.0])
    assert driver._groupSyncWrite.currents == [150, 900]


def test_mode_five_goal_positions_use_signed_raw_encoder_counts() -> None:
    """Mode-5 goals must preserve positive and negative multi-turn radians."""

    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1, 2]
    driver._torque_enabled = True
    driver._operating_mode = CURRENT_BASED_POSITION_CONTROL_MODE
    driver._groupPositionSyncWrite = _FakePositionSyncWriter()
    driver.set_goal_positions([np.pi / 2.0, -np.pi])
    assert driver._groupPositionSyncWrite.positions == [1024, -2048]


def test_position_profile_rejects_nonpositive_si_values() -> None:
    """A zero velocity profile must fail before any hardware transaction."""

    with np.testing.assert_raises(ValueError):
        DynamixelDriver._profile_value(0.0, 0.01, "velocity")


def test_mode_five_rewrites_goal_after_torque_enable() -> None:
    """The post-enable Goal Position update must trigger the hardware profile."""

    events = []
    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1]
    driver._position_array = lambda values: np.asarray(values, dtype=float)
    driver.set_torque_mode = lambda enabled: events.append(f"torque:{enabled}")
    driver.set_operating_mode = lambda mode: events.append(f"mode:{mode}")
    driver.verify_operating_mode = lambda mode: events.append(f"verify:{mode}")
    driver._write_unsigned_register = lambda *_args: None
    driver._write_unsigned_2byte_register = (
        lambda _dxl_id, address, value: events.append((address, value))
    )
    driver._sync_write_currents = lambda _values: events.append("current")
    driver._sync_write_positions = lambda _values: events.append("position")

    driver.enter_current_based_position_control(
        [0.5],
        max_current_ma=150.0,
        profile_velocity_rad_s=0.35,
        profile_acceleration_rad_s2=0.8,
        position_p_gain=800,
        position_i_gain=20,
        position_d_gain=400,
    )
    assert (ADDR_POSITION_P_GAIN, 800) in events
    assert (ADDR_POSITION_I_GAIN, 20) in events
    assert (ADDR_POSITION_D_GAIN, 400) in events
    assert events[-3:] == ["position", "torque:True", "position"]


def test_position_gain_rejects_noninteger_or_out_of_range_values() -> None:
    """Raw position-loop gain validation must match the 14-bit register range."""

    assert DynamixelDriver._gain_value(0, "gain") == 0
    assert DynamixelDriver._gain_value(16383, "gain") == 16383
    for invalid_value in (-1, 16384, 20.0, True):
        with np.testing.assert_raises(ValueError):
            DynamixelDriver._gain_value(invalid_value, "gain")


def test_emergency_disable_reopens_port_and_retries_until_success() -> None:
    """A poisoned bus is recovered by port reset so motors de-energize."""

    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1]
    driver._torque_enabled = True
    driver._groupSyncWrite = _FakeSyncWriter()

    attempts = []

    def flaky_set_current(values, max_current_ma=900.0):
        """Fail the first zero-current write, then succeed."""

        attempts.append(list(values))
        if len(attempts) == 1:
            raise RuntimeError("stale bus")

    driver.set_current = flaky_set_current
    driver.set_torque_mode = lambda enabled: None
    driver._baudrate = 1000000
    resets = []
    driver.reset_port = lambda: resets.append(True)

    assert driver.emergency_disable() is True
    assert resets == [True]
    assert attempts == [[0.0], [0.0]]

    def hopeless_set_current(values, max_current_ma=900.0):
        """Always fail so shutdown must report the failure."""

        raise RuntimeError("bus gone")

    driver.set_current = hopeless_set_current
    assert driver.emergency_disable(retries=2) is False


def test_reconnect_reopens_port_before_restoring_current_control() -> None:
    """Runtime recovery must reopen the port before enabling current control."""

    driver = DynamixelDriver.__new__(DynamixelDriver)
    events = []
    driver.reset_port = lambda: events.append("reset_port")
    driver.enter_current_control = lambda: events.append("current_control")

    driver.reconnect_current_control()

    assert events == ["reset_port", "current_control"]
