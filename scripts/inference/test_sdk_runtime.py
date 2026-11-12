"""No-hardware tests for SDK motion gating, training image geometry and synchronous execution."""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xarm_ros2_teleop"))

import run_diffusion_sdk as runner  # noqa: E402
from xarm_ros2_teleop.inference.sdk_cameras import orbbec_rgb, snapshot_images  # noqa: E402
from xarm_ros2_teleop.inference.sdk_robot import SDKRobot  # noqa: E402
from xarm_ros2_teleop.recording.message_converters import center_crop_resize  # noqa: E402
from xarm_ros2_teleop.system.config import load_system_config  # noqa: E402


class FakeHardware:
    """Record every SDK mutation while providing independent measured joint/gripper feedback."""

    connected = True
    error_code = 0
    state = 0
    timeOut = 5.0

    def __init__(self) -> None:
        """Create a command log and an event for background gripper commands."""

        self.writes = []
        self.moved = Event()
        self.disconnected = False
        self.gripper_position = 128

    def __getattr__(self, name):
        """Create recorders only for known mutating SDK methods."""

        if name not in {
            "activate",
            "motion_enable",
            "set_mode",
            "set_state",
            "set_servo_angle",
            "move",
            "stop",
        }:
            raise AttributeError(name)

        def record(*args, **kwargs):
            """Record a mutation and return a successful SDK status code."""

            self.writes.append((name, args, kwargs))
            if name == "activate":
                self.activation_timeout = self.timeOut
            if name == "move":
                if args[0] == 0:
                    self.gripper_position = 3
                self.moved.set()
            return 0

        return record

    def connect(self):
        """Simulate a feedback-only connection."""

    def disconnect(self):
        """Record resource cleanup without a hardware mutation."""

        self.disconnected = True

    def get_servo_angle(self, **kwargs):
        """Return seven distinct measured radians."""

        return 0, [0.1] * 7

    def position(self, **kwargs):
        """Return a measured gPO value distinct from policy target."""

        return self.gripper_position

    def isActivated(self, **kwargs):
        """Report an already-active gripper without an activation command."""

        return True


@pytest.fixture
def hardware(monkeypatch):
    """Substitute both vendor packages; real SDK libraries and hardware are unnecessary."""

    arm, gripper = FakeHardware(), FakeHardware()
    monkeypatch.setitem(sys.modules, "xarm.wrapper", SimpleNamespace(XArmAPI=lambda *a, **k: arm))
    monkeypatch.setitem(
        sys.modules,
        "pyrobotiqgripper",
        SimpleNamespace(GRIPPER_MODE_RTU="rtu", RobotiqGripper=lambda **k: gripper),
    )
    return arm, gripper


def make_robot(execute):
    """Load the actual deployment profile into a native SDK robot."""

    return SDKRobot(
        load_system_config(ROOT / "xarm_ros2_teleop/config/inference_diffusion.yaml"),
        execute=execute,
    )


def test_dry_run_never_enables_activates_moves_or_stops_hardware(hardware):
    """Exercise connect/read/action/close; all motion-related method logs must stay empty."""

    arm, gripper = hardware
    robot = make_robot(False)
    try:
        robot.connect()
        np.testing.assert_allclose(robot.state(), [0.1] * 7 + [128 / 255])
        robot.command(np.array([0.2] * 7 + [0.7], np.float32))
    finally:
        robot.close()
    assert arm.writes == [] and gripper.writes == []
    assert arm.disconnected and gripper.disconnected


def test_execute_uses_mode6_absolute_targets_and_raw_gripper_mapping(hardware):
    """Confirm an open start, then exercise mode 6, raw gPR mapping and state-4 stop."""

    arm, gripper = hardware
    robot = make_robot(True)
    try:
        robot.connect()
        assert gripper.writes[0][:2] == ("move", (0,))
        np.testing.assert_allclose(robot.state()[7], 3 / 255)
        gripper.moved.clear()
        robot.command(np.array([0.2] * 7 + [0.5], np.float32))
        assert gripper.moved.wait(1)
        moves = [entry for entry in gripper.writes if entry[0] == "move"]
        assert moves[-1][1] == (127,)
        assert [entry[1] for entry in arm.writes if entry[0] == "set_mode"] == [(0,), (6,)]
        target = [entry for entry in arm.writes if entry[0] == "set_servo_angle"][-1][2]
        np.testing.assert_allclose(target["angle"], [0.2] * 7)
        assert target["is_radian"] and target["speed"] == 1.2
        assert not any(entry[0] == "activate" for entry in gripper.writes)
    finally:
        robot.close()
    assert arm.writes[-1][:2] == ("set_state", (4,))


def test_explicit_activation_restores_serial_timeout(hardware):
    """Apply the configured cold-start timeout only during explicitly requested activation."""

    _, gripper = hardware
    robot = make_robot(True)
    try:
        robot.connect(activate_gripper=True, skip_initial_move=True)
        assert gripper.activation_timeout == 15.0
        assert gripper.timeOut == 5.0
    finally:
        robot.close()


def test_nonfinite_measured_joints_are_rejected(hardware, monkeypatch):
    """Reject unusable measured state, including the second read after model inference."""

    arm, _ = hardware
    robot = make_robot(False)
    try:
        robot.connect()
        monkeypatch.setattr(arm, "get_servo_angle", lambda **kwargs: (0, [np.nan] * 7))
        with pytest.raises(RuntimeError, match="feedback"):
            robot.state()
    finally:
        robot.close()


def test_orbbec_rgb_and_training_crop():
    """Convert BGR channel order and preserve training's 960x720 center crop."""

    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, 160:1120] = [1, 2, 3]
    frame = SimpleNamespace(
        get_height=lambda: 720,
        get_width=lambda: 1280,
        get_format=lambda: "OBFormat.BGR",
        get_data=lambda: image.tobytes(),
    )
    rgb = center_crop_resize(orbbec_rgb(frame), source_size=(1280, 720), output_size=(640, 480))
    assert rgb.shape == (480, 640, 3)
    np.testing.assert_array_equal(rgb, np.tile([3, 2, 1], (480, 640, 1)))


def test_snapshot_rejects_stale_camera():
    """Do not infer on a camera frame left over from an earlier request."""

    camera = SimpleNamespace(error=None, latest=(time.monotonic() - 1, np.zeros((2, 2, 3))))
    with pytest.raises(RuntimeError, match="stale"):
        snapshot_images({"top_camera": camera}, timeout=0.25, skew=0.08)


def test_chunk_keeps_30hz_duration_and_final_target(monkeypatch):
    """Use a deterministic monotonic clock to verify 10-step duration and final-target hold."""

    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        """Advance simulated time without sleeping."""

        clock.now += seconds

    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=advance))
    commands = []
    robot = SimpleNamespace(command=lambda action: commands.append(action.copy()))
    actions = np.tile(np.arange(10, dtype=np.float32)[:, None], (1, 8))
    runner.execute_chunk(robot, actions, frequency=30, period=0.01)
    assert 10 / 30 <= clock.now < 10 / 30 + 0.011
    np.testing.assert_array_equal(commands[-1], actions[-1])
    assert np.all(np.diff(np.array(commands)[:, 0]) >= 0)


@pytest.mark.parametrize("recording", ["off", "normal", "interrupt", "inference_error"])
def test_main_synchronous_inference_and_video_cleanup(monkeypatch, tmp_path, recording):
    """Verify synchronous chunks and playable, matched RGB videos on exit and failure."""

    events = []
    recorders = []
    clock = SimpleNamespace(now=0.0)
    system = make_robot(False).config
    system = replace(system, inference=replace(system.inference, max_chunks=2))
    camera_keys = [image.key for image in system.inference.images]
    video_root = tmp_path / "videos"
    recorder_class = runner.SDKVideoRecorder

    def make_recorder(*args):
        """Keep the real recorder available to synchronize the simulated inference call."""

        recorder = recorder_class(*args)
        recorders.append(recorder)
        return recorder

    def advance(seconds):
        """Advance the runner clock without changing the real camera freshness clock."""

        clock.now += seconds

    class Camera:
        """Provide a fresh RGB frame for each configured training view."""

        error = None

        def __init__(self, device, image):
            """Give each camera one distinguishable RGB view without opening hardware."""

            self.ready = Event()
            self.rgb = np.zeros((480, 640, 3), np.uint8)
            self.rgb[:, :, camera_keys.index(image.key)] = 240

        def start(self):
            """Mark the simulated camera ready."""

            self.ready.set()

        @property
        def latest(self):
            """Return a current, correctly shaped synthetic image."""

            return time.monotonic(), self.rgb

        def close(self):
            """Record cleanup of this camera."""

            events.append(("camera_close", clock.now))

    class Robot:
        """Record targets and provide measured state independently from commands."""

        def __init__(self, *args, **kwargs):
            """Construct a simulated robot without SDK imports."""

        def connect(self, *args):
            """Record initialization."""

            events.append(("connect", clock.now))

        def state(self):
            """Return a valid measured [8] state."""

            return np.zeros(8, np.float32)

        def command(self, action):
            """Record each command deadline."""

            events.append(("command", clock.now))

        def close(self):
            """Record robot cleanup."""

            events.append(("robot_close", clock.now))

    class Client:
        """Supply finite 10-step chunks over the same observation interface."""

        metadata = {
            "protocol_version": 1,
            "state_dim": 8,
            "action_format": "absolute_joint_rad_plus_gripper",
            "execution_horizon": 10,
            "action_frequency_hz": 30.0,
            "image_keys": ["top_camera", "left_camera", "right_camera"],
        }

        def __init__(self, *args, **kwargs):
            """Construct the simulated client."""

        def infer(self, observation):
            """Verify outgoing state/images and record when each new request occurs."""

            assert observation["state"].shape == (8,)
            assert list(observation["images"]) == self.metadata["image_keys"]
            events.append(("infer", clock.now))
            if recorders:
                recorder = recorders[0]
                target = recorder.frames_written + 3
                deadline = time.monotonic() + 3.0
                while recorder.frames_written < target and time.monotonic() < deadline:
                    recorder.raise_if_failed()
                    time.sleep(0.01)
                assert recorder.frames_written >= target
            if recording == "interrupt":
                raise KeyboardInterrupt
            if recording == "inference_error":
                raise RuntimeError("inference failed")
            return np.full((10, 8), 0.1, np.float32)

        def close(self):
            """Record the connection cleanup."""

            events.append(("client_close", clock.now))

    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=advance))
    monkeypatch.setattr(runner, "load_system_config", lambda path: system)
    monkeypatch.setattr(runner, "SDKCamera", Camera)
    monkeypatch.setattr(runner, "SDKRobot", Robot)
    monkeypatch.setattr(runner, "PolicyClient", Client)
    monkeypatch.setattr(runner, "SDKVideoRecorder", make_recorder)
    argv = ["run_diffusion_sdk.py"]
    if recording != "off":
        argv += ["--record-video", "--video-dir", str(video_root)]
    monkeypatch.setattr(sys, "argv", argv)
    expected_exit = {
        "interrupt": pytest.raises(KeyboardInterrupt),
        "inference_error": pytest.raises(RuntimeError, match="inference failed"),
    }.get(recording, nullcontext())
    with expected_exit:
        runner.main()
    queries = [index for index, event in enumerate(events) if event[0] == "infer"]
    if recording in {"interrupt", "inference_error"}:
        assert len(queries) == 1
        assert not any(event[0] == "command" for event in events)
    else:
        assert len(queries) == 2
        assert events[queries[1]][1] >= 10 / 30
        assert events[queries[1] - 1][0] == "command"
    assert events[-1][0] == "client_close"
    if recording == "off":
        assert not recorders
        return
    (run_dir,) = video_root.iterdir()
    assert {path.name for path in run_dir.iterdir()} == {f"{key}.mp4" for key in camera_keys}
    for index, key in enumerate(camera_keys):
        capture = cv2.VideoCapture(str(run_dir / f"{key}.mp4"))
        try:
            assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
            assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == recorders[0].frames_written
            ok, frame = capture.read()
            assert ok and frame.shape == (480, 640, 3)
            assert np.argmax(frame.mean(axis=(0, 1))) == 2 - index  # VideoCapture returns BGR.
        finally:
            capture.release()
