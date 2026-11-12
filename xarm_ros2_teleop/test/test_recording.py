"""Pure synchronization and schema tests for LeRobot episode recording."""

import threading
from datetime import datetime
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray
from xarm_ros2_teleop.recording.config import RecordingConfig
from xarm_ros2_teleop.recording.dataset_schema import build_features
from xarm_ros2_teleop.recording.episode_recorder import EpisodeRecorder, _operation_status
from xarm_ros2_teleop.recording.keyboard_controller import decode_key
from xarm_ros2_teleop.recording.lerobot_writer import LeRobotWriter, _feature_matches
from xarm_ros2_teleop.recording.message_converters import (
    center_crop_resize,
    depth_preview,
    metric_depth,
    rgb_image,
)
from xarm_ros2_teleop.recording.synchronizer import (
    FrameSynchronizer,
    SynchronizationError,
    XarmState,
)
from xarm_ros2_teleop.recording.timestamp_buffer import TimestampBuffer


def _config() -> RecordingConfig:
    """Create one anchor-only recorder configuration for array-level tests."""

    return RecordingConfig.from_mapping(
        {
            "enabled": True,
            "root": "/tmp/recording-test",
            "repo_id": "local/test",
            "robot_type": "xarm7_factr",
            "fps": 30,
            "instruction": "test task",
            "num_episodes": 1,
            "reset_joint_positions": [0.0] * 7,
            "anchor": {"key": "front_camera"},
            "streams": {
                "front_camera": {
                    "topic": "/camera/front/color",
                    "type": "rgb",
                }
            },
        }
    )


def _image(encoding: str, array: np.ndarray):
    """Create a minimal ROS Image-like object from a contiguous NumPy array."""

    channels = 1 if array.ndim == 2 else array.shape[2]
    return SimpleNamespace(
        encoding=encoding,
        height=array.shape[0],
        width=array.shape[1],
        step=array.shape[1] * channels * array.dtype.itemsize,
        data=array.tobytes(),
        is_bigendian=False,
    )


@pytest.mark.parametrize(
    ("write_done", "reset_done", "phase", "detail"),
    [
        (False, False, "SAVING_AND_RESETTING", "并行保存本条数据并复位"),
        (False, True, "SAVING", "正在保存本条数据"),
        (True, False, "RESETTING", "正在复位机械臂"),
    ],
)
def test_episode_completion_status_exposes_both_async_branches(
    write_done: bool,
    reset_done: bool,
    phase: str,
    detail: str,
) -> None:
    """Dashboard progress identifies whether writer, reset, or both are pending."""

    actual_phase, actual_detail = _operation_status(
        state="SAVING",
        completion_pending=True,
        write_done=write_done,
        reset_done=reset_done,
        last_error="",
        start_blocker="当前状态为 SAVING",
    )

    assert actual_phase == phase
    assert detail in actual_detail


def test_finalizing_status_explains_that_finish_is_asynchronous() -> None:
    """Accepted Finish requests explicitly report dataset finalization progress."""

    phase, detail = _operation_status(
        state="FINALIZING",
        completion_pending=False,
        write_done=True,
        reset_done=True,
        last_error="",
        start_blocker="当前状态为 FINALIZING",
    )

    assert phase == "FINALIZING"
    assert detail == "正在整理并关闭数据集"


def test_lerobot_writer_disables_process_based_video_encoding(tmp_path: Path) -> None:
    """Episode saves avoid unsafe process forks from the ROS recorder process."""

    class DatasetStub:
        """Capture the LeRobot parallel-encoding option passed by the wrapper."""

        def __init__(self) -> None:
            """Initialize the scalar call record used by this unit test."""

            self.parallel_encoding = None
            self.meta = SimpleNamespace(total_episodes=0, video_keys=[])

        def _wait_image_writer(self) -> None:
            """Match the LeRobot writer synchronization interface."""

        def save_episode(self, *, parallel_encoding: bool) -> None:
            """Store whether LeRobot was asked to spawn encoder processes."""

            self.parallel_encoding = parallel_encoding

    writer = LeRobotWriter(
        root=tmp_path,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
    )
    writer.dataset = DatasetStub()
    writer.frames_in_episode = 1

    writer.save_episode()

    assert writer.dataset.parallel_encoding is False
    assert writer.frames_in_episode == 0


def test_lerobot_writer_routes_buffered_episode_to_category(tmp_path: Path) -> None:
    """Outcome routing moves temporary frames and commits only the destination."""

    source = LeRobotWriter(
        root=tmp_path / "success",
        repo_id="local/task_success",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="libx264",
        image_writer_threads=1,
        streaming_encoding=False,
    )
    destination = LeRobotWriter(
        root=tmp_path / "failure",
        repo_id="local/task_failure",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="libx264",
        image_writer_threads=1,
        streaming_encoding=False,
    )

    def make_dataset(root: Path, episode_index: int):
        """Create the minimum LeRobot-like object needed by outcome routing."""

        dataset = SimpleNamespace()
        dataset.root = root
        dataset.meta = SimpleNamespace(
            total_episodes=episode_index,
            camera_keys=["front"],
            video_keys=["front"],
            image_keys=[],
        )
        dataset._wait_image_writer = lambda: None
        dataset._get_image_file_dir = (
            lambda index, key: root / "images" / key / f"episode-{index:06d}"
        )
        dataset.saved = []
        dataset.cleared = False

        def save_episode(*, episode_data: dict, parallel_encoding: bool) -> None:
            """Capture one explicit episode commit without encoding video."""

            dataset.saved.append((episode_data, parallel_encoding))
            dataset.meta.total_episodes += 1

        def clear_episode_buffer(*, delete_images: bool) -> None:
            """Mark the source buffer as cleared after a successful route."""

            dataset.cleared = bool(delete_images)

        dataset.save_episode = save_episode
        dataset.clear_episode_buffer = clear_episode_buffer
        return dataset

    source.dataset = make_dataset(source.root, episode_index=3)
    destination.dataset = make_dataset(destination.root, episode_index=1)
    source_dir = source.dataset._get_image_file_dir(3, "front")
    source_dir.mkdir(parents=True)
    frame_paths = []
    for frame_index in range(2):
        frame_path = source_dir / f"frame-{frame_index:06d}.png"
        frame_path.write_bytes(b"png")
        frame_paths.append(str(frame_path))
    source.dataset.episode_buffer = {
        "size": 2,
        "task": ["insert", "insert"],
        "episode_index": 3,
        "front": frame_paths,
    }
    source.frames_in_episode = 2

    source.save_episode_to(destination)

    saved_episode, parallel_encoding = destination.dataset.saved[0]
    destination_dir = destination.dataset._get_image_file_dir(1, "front")
    assert saved_episode["episode_index"] == 1
    assert all(Path(path).parent == destination_dir for path in saved_episode["front"])
    assert parallel_encoding is False
    assert source.dataset.cleared is True
    assert source.frames_in_episode == 0
    assert not source_dir.exists()
    assert destination_dir.is_dir()


def test_lerobot_writer_discard_removes_video_feature_frames(tmp_path: Path) -> None:
    """Discard removes temporary PNGs even though LeRobot omits video keys."""

    video_dir = tmp_path / "images" / "front" / "episode-000000"
    video_dir.mkdir(parents=True)
    (video_dir / "frame-000000.png").write_bytes(b"png")

    class DatasetStub:
        """Expose the private paths used by the pinned LeRobot recording API."""

        meta = SimpleNamespace(total_episodes=0, video_keys=["front"])

        def __init__(self) -> None:
            """Track whether the normal buffer clear path was also called."""

            self.cleared = False

        def clear_episode_buffer(self, *, delete_images: bool) -> None:
            """Record the requested LeRobot buffer cleanup operation."""

            self.cleared = delete_images

        def _wait_image_writer(self) -> None:
            """Match the LeRobot writer synchronization interface."""

        def _get_image_file_dir(self, episode_index: int, video_key: str) -> Path:
            """Return the exact temporary directory for one video feature."""

            assert episode_index == 0
            assert video_key == "front"
            return video_dir

    writer = LeRobotWriter(
        root=tmp_path,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
    )
    writer.dataset = DatasetStub()
    writer.frames_in_episode = 1

    writer.discard_episode()

    assert writer.dataset.cleared is True
    assert writer.frames_in_episode == 0
    assert not video_dir.exists()


def test_lerobot_writer_rejects_unpaired_video_tail(tmp_path: Path) -> None:
    """Save fails before Parquet commit when a camera has stale tail PNGs."""

    video_dir = tmp_path / "images" / "front" / "episode-000000"
    video_dir.mkdir(parents=True)
    for frame_index in range(3):
        (video_dir / f"frame-{frame_index:06d}.png").write_bytes(b"png")

    class DatasetStub:
        """Expose one video directory and record whether save was reached."""

        meta = SimpleNamespace(total_episodes=0, video_keys=["front"])

        def __init__(self) -> None:
            """Initialize the scalar save call record."""

            self.saved = False

        def _wait_image_writer(self) -> None:
            """Match the LeRobot writer synchronization interface."""

        def _get_image_file_dir(self, episode_index: int, video_key: str) -> Path:
            """Return the exact temporary directory for one video feature."""

            assert episode_index == 0
            assert video_key == "front"
            return video_dir

        def save_episode(self, *, parallel_encoding: bool) -> None:
            """Record an unexpected attempt to commit mismatched media."""

            self.saved = True

    writer = LeRobotWriter(
        root=tmp_path,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
    )
    writer.dataset = DatasetStub()
    writer.frames_in_episode = 2

    with pytest.raises(RuntimeError, match="temporary video frame mismatch"):
        writer.save_episode()

    assert writer.dataset.saved is False


def test_lerobot_writer_resume_removes_uncommitted_video_frames(
    tmp_path: Path, monkeypatch
) -> None:
    """Opening an existing dataset removes only its next episode's stale PNGs."""

    dataset_root = tmp_path / "session"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    stale_dir = dataset_root / "images" / "front" / "episode-000004"
    stale_dir.mkdir(parents=True)
    (stale_dir / "frame-000000.png").write_bytes(b"png")
    frame = {
        "observation.images.front": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }

    class DatasetStub:
        """Represent a compatible four-episode dataset being resumed."""

        fps = 30
        features = build_features(frame)
        meta = SimpleNamespace(total_episodes=4, video_keys=["front"])

        def _wait_image_writer(self) -> None:
            """Match the LeRobot writer synchronization interface."""

        def _get_image_file_dir(self, episode_index: int, video_key: str) -> Path:
            """Return the exact uncommitted temporary camera directory."""

            assert episode_index == 4
            assert video_key == "front"
            return stale_dir

    dataset = DatasetStub()

    class DatasetFactory:
        """Provide both constructor and create attributes expected by the wrapper."""

        create = staticmethod(lambda **_kwargs: None)

        def __new__(cls, *_args, **_kwargs):
            """Return the prepared resume stub instead of constructing a class."""

            return dataset

    monkeypatch.setattr(
        "xarm_ros2_teleop.recording.lerobot_writer.import_module",
        lambda _name: SimpleNamespace(LeRobotDataset=DatasetFactory),
    )
    writer = LeRobotWriter(
        root=dataset_root,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
    )

    writer.prepare(frame)

    assert writer.dataset is dataset
    assert not stale_dir.exists()


def test_overwrite_removes_only_the_exact_dataset_directory(tmp_path: Path) -> None:
    """Overwrite clears one explicit dataset root without touching its parent."""

    dataset_root = tmp_path / "session"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (dataset_root / "old.txt").write_text("old", encoding="utf-8")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    writer = LeRobotWriter(
        root=dataset_root,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
        write_mode="overwrite",
    )

    writer._remove_existing_root()

    assert not dataset_root.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_overwrite_rejects_a_non_dataset_parent_directory(tmp_path: Path) -> None:
    """Overwrite cannot recursively remove a broad folder of multiple datasets."""

    parent = tmp_path / "datasets"
    parent.mkdir()
    (parent / "another_run").mkdir()
    writer = LeRobotWriter(
        root=parent,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
        write_mode="overwrite",
    )

    with pytest.raises(ValueError, match="exact dataset folder"):
        writer._remove_existing_root()

    assert (parent / "another_run").is_dir()


def test_overwrite_rejects_a_filesystem_root() -> None:
    """A malformed Web path can never recursively remove the filesystem root."""

    writer = LeRobotWriter(
        root=Path("/"),
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
        write_mode="overwrite",
    )
    with pytest.raises(ValueError, match="Unsafe overwrite"):
        writer._remove_existing_root()


def test_session_settings_lock_before_writer_preparation(tmp_path: Path) -> None:
    """Task, exact path and mode may change once before dataset creation."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.session_configured = False
    recorder.writer_prepare_requested = False
    recorder.writer_prepared = False
    recorder.writer = SimpleNamespace(dataset=None)
    recorder.state = "IDLE"
    recorder.session_ready = False
    recorder.instruction = "default"
    recorder.dataset_root = tmp_path / "default"
    recorder.write_mode = "resume"
    created = []
    recorder._new_writer = lambda category=None: created.append(category) or SimpleNamespace(
        dataset=None
    )

    result = recorder._configure_session_parameters(
        [
            Parameter("session_instruction", value="insert the block"),
            Parameter("session_dataset_root", value=str(tmp_path / "target")),
            Parameter("session_write_mode", value="overwrite"),
        ]
    )

    assert result.successful is True
    assert recorder.instruction == "insert the block"
    assert recorder.dataset_root == tmp_path / "target"
    assert recorder.write_mode == "overwrite"
    assert recorder.session_configured is True
    assert created == ["success", "optimal", "failure", "crash"]
    rejected = recorder._configure_session_parameters(
        [Parameter("session_instruction", value="another task")]
    )
    assert rejected.successful is False


def test_episode_writer_job_runs_on_the_calling_thread() -> None:
    """Arrow save work executes on the main-loop caller rather than a ROS worker."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.episode_writer_jobs = Queue()
    calls = []
    recorder._write_episode = lambda save: calls.append(
        (threading.get_ident(), "episode", save)
    )
    recorder._prepare_writer = lambda frame: calls.append(
        (threading.get_ident(), "prepare", frame)
    )
    recorder._finalize_writer = lambda: calls.append(
        (threading.get_ident(), "finalize", None)
    )
    caller_thread = threading.get_ident()

    recorder.episode_writer_jobs.put_nowait(("prepare", "schema"))
    assert recorder.process_writer_job(timeout_sec=0.0) is True
    recorder.episode_writer_jobs.put_nowait(("episode", True))
    assert recorder.process_writer_job(timeout_sec=0.0) is True
    recorder.episode_writer_jobs.put_nowait(("finalize", None))
    assert recorder.process_writer_job(timeout_sec=0.0) is True
    assert recorder.process_writer_job(timeout_sec=0.0) is False

    assert calls == [
        (caller_thread, "prepare", "schema"),
        (caller_thread, "episode", True),
        (caller_thread, "finalize", None),
    ]


def test_timestamp_buffer_nearest_prefers_earlier_sample_on_tie() -> None:
    """Nearest-frame selection is deterministic when timestamps are equidistant."""

    buffer = TimestampBuffer(1.0)
    buffer.add(0.9, "past")
    buffer.add(1.1, "future")
    assert buffer.nearest(1.0).value == "past"
    assert buffer.latest_before(1.0).value == "past"


def test_color_and_depth_messages_become_rgb_video_frames() -> None:
    """BGR tactile images and millimetre depth become uint8 RGB video arrays."""

    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    rgb = rgb_image(_image("bgr8", bgr))
    assert rgb[0, 0].tolist() == [0, 0, 255]

    depth = np.array([[0, 1000, 2000]], dtype=np.uint16)
    preview = depth_preview(
        _image("16UC1", depth),
        input_unit="mm",
        depth_min_m=0.5,
        depth_max_m=2.0,
    )
    assert preview.shape == (1, 3, 3)
    assert preview[0, 0].tolist() == [0, 0, 0]
    assert preview[0, 2, 0] == 255
    np.testing.assert_allclose(
        metric_depth(_image("16UC1", depth), input_unit="mm"),
        [[0.0, 1.0, 2.0]],
    )


def test_training_view_center_crops_and_rescales_without_distortion() -> None:
    """A 1280x720 Femto Bolt frame becomes a 4:3 640x480 stored view."""

    # Sentinel colors in the two 160-px side margins must disappear; the kept
    # 960x720 center is constant-color so the uniform 2/3 rescale stays exact.
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :160] = (255, 0, 0)
    frame[:, -160:] = (0, 255, 0)
    frame[:, 160:-160] = (0, 0, 255)
    view = center_crop_resize(frame, source_size=(1280, 720), output_size=(640, 480))
    assert view.shape == (480, 640, 3)
    assert np.all(view == (0, 0, 255))


def test_training_view_rejects_frames_from_a_changed_camera_profile() -> None:
    """Frames whose size differs from source_size fail instead of corrupting data."""

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="camera profile changed"):
        center_crop_resize(frame, source_size=(1280, 720), output_size=(640, 480))


def test_synchronizer_separates_actual_state_from_causal_action() -> None:
    """State uses measured joints while action uses commands no later than the image."""

    config = _config()
    synchronizer = FrameSynchronizer(config)
    anchor_image = np.zeros((2, 3, 3), dtype=np.uint8)
    anchor = synchronizer.images["front_camera"].add(1.0, anchor_image)
    measured = np.arange(7, dtype=np.float32)
    synchronizer.xarm_state.add(
        1.01,
        XarmState(measured, np.ones(7, np.float32), np.full(7, 2, np.float32)),
    )
    synchronizer.robotiq_state.add(1.0, np.array([0.25], np.float32))
    synchronizer.xarm_action.add(0.99, np.full(7, 3, np.float32))
    synchronizer.xarm_action.add(1.01, np.full(7, 9, np.float32))
    synchronizer.robotiq_action.add(0.99, np.array([0.75], np.float32))
    frame = synchronizer.assemble(anchor, "test task")
    np.testing.assert_allclose(frame["observation.state"][:7], measured)
    np.testing.assert_allclose(frame["action"][:7], 3.0)
    assert frame["observation.state"][7] == pytest.approx(0.25)
    assert frame["action"][7] == pytest.approx(0.75)
    assert build_features(frame)["action"]["shape"] == (8,)


def test_synchronizer_uses_timestamp_nearest_secondary_frame() -> None:
    """Each anchor selects the closest buffered image instead of callback order."""

    config = RecordingConfig.from_mapping(
        {
            "enabled": True,
            "anchor": {"key": "front_camera"},
            "streams": {
                "front_camera": {
                    "topic": "/camera/front/color",
                    "type": "rgb",
                },
                "wrist_camera": {
                    "topic": "/camera/wrist/color",
                    "type": "rgb",
                },
            },
        }
    )
    synchronizer = FrameSynchronizer(config)
    anchor = synchronizer.images["front_camera"].add(
        1.0, np.zeros((1, 1, 3), dtype=np.uint8)
    )
    synchronizer.images["wrist_camera"].add(
        0.99, np.full((1, 1, 3), 1, dtype=np.uint8)
    )
    synchronizer.images["wrist_camera"].add(
        1.04, np.full((1, 1, 3), 2, dtype=np.uint8)
    )
    synchronizer.xarm_state.add(
        1.0,
        XarmState(np.zeros(7), np.zeros(7), np.zeros(7)),
    )
    synchronizer.robotiq_state.add(1.0, np.zeros(1))
    synchronizer.xarm_action.add(1.0, np.zeros(7))
    synchronizer.robotiq_action.add(1.0, np.zeros(1))

    frame = synchronizer.assemble(anchor, "test")

    assert frame["observation.images.wrist_camera"][0, 0, 0] == 1
    assert frame["observation.sync_skew_s"][1] == pytest.approx(-0.01)


def test_synchronizer_keeps_nearest_secondary_image_despite_large_skew() -> None:
    """Hardware owns camera sync, so software records nearest images without drops."""

    config = RecordingConfig.from_mapping(
        {
            "enabled": True,
            "anchor": {"key": "front_camera"},
            "streams": {
                "front_camera": {
                    "topic": "/camera/front/color",
                    "type": "rgb",
                },
                "wrist_camera": {
                    "topic": "/camera/wrist/color",
                    "type": "rgb",
                },
            },
        }
    )
    synchronizer = FrameSynchronizer(config)
    anchor = synchronizer.images["front_camera"].add(
        1.0, np.zeros((1, 1, 3), dtype=np.uint8)
    )
    synchronizer.images["wrist_camera"].add(
        0.90, np.zeros((1, 1, 3), dtype=np.uint8)
    )
    synchronizer.xarm_state.add(
        1.0,
        XarmState(np.zeros(7), np.zeros(7), np.zeros(7)),
    )
    synchronizer.robotiq_state.add(1.0, np.zeros(1))
    synchronizer.xarm_action.add(1.0, np.zeros(7))
    synchronizer.robotiq_action.add(1.0, np.zeros(1))

    frame = synchronizer.assemble(anchor, "test")

    assert frame["observation.sync_skew_s"][1] == pytest.approx(-0.10)


def test_synchronizer_rejects_future_only_action() -> None:
    """A command after the anchor must never leak backward into its action label."""

    config = _config()
    synchronizer = FrameSynchronizer(config)
    anchor = synchronizer.images["front_camera"].add(
        1.0, np.zeros((2, 3, 3), dtype=np.uint8)
    )
    synchronizer.xarm_state.add(
        1.0,
        XarmState(np.zeros(7), np.zeros(7), np.zeros(7)),
    )
    synchronizer.robotiq_state.add(1.0, np.zeros(1))
    synchronizer.xarm_action.add(1.01, np.zeros(7))
    synchronizer.robotiq_action.add(0.99, np.zeros(1))
    with pytest.raises(SynchronizationError, match="causal"):
        synchronizer.assemble(anchor, "test task")


def test_recording_config_requires_a_30_fps_rgb_anchor() -> None:
    """The fixed dataset clock must remain a 30 FPS enabled RGB stream."""

    config = {
        "enabled": True,
        "fps": 20,
        "anchor": {"key": "depth"},
        "streams": {"depth": {"topic": "/camera/depth", "type": "depth"}},
    }
    with pytest.raises(ValueError, match="fps must be 30"):
        RecordingConfig.from_mapping(config)
    config["fps"] = 30
    config["anchor"]["key"] = "front_camera"
    config["streams"] = {
        "front_camera": {"topic": "/camera/depth", "type": "depth"}
    }
    with pytest.raises(ValueError, match="anchor stream must be rgb"):
        RecordingConfig.from_mapping(config)


def test_training_view_sizes_must_be_configured_as_a_pair() -> None:
    """source_size without output_size would leave the stored shape undefined."""

    config = {
        "enabled": True,
        "anchor": {"key": "front_camera"},
        "streams": {
            "front_camera": {
                "topic": "/camera/front/color",
                "type": "rgb",
                "source_size": [1280, 720],
            }
        },
    }
    with pytest.raises(ValueError, match="configured together"):
        RecordingConfig.from_mapping(config)
    config["streams"]["front_camera"]["output_size"] = [640, 480]
    parsed = RecordingConfig.from_mapping(config)
    assert parsed.streams[0].source_size == (1280, 720)
    assert parsed.streams[0].output_size == (640, 480)


def test_instruction_and_dataset_folder_are_explicit_vla_metadata() -> None:
    """YAML instruction maps to LeRobot task and session folders are deterministic."""

    config = _config()
    assert config.instruction == "test task"
    assert config.task == "test task"
    assert config.dataset_root() == Path("/tmp/recording-test")

    timestamped = RecordingConfig.from_mapping(
        {
            "root": "/data/control_robot",
            "dataset_name": "pick_cube",
            "append_timestamp": True,
            "instruction": "pick up the cube",
        }
    )
    assert timestamped.dataset_root(datetime(2026, 7, 19, 16, 30, 5)) == Path(
        "/data/control_robot/pick_cube_20260719_163005"
    )


def test_legacy_task_may_not_conflict_with_instruction() -> None:
    """A stale task alias cannot silently label frames with another instruction."""

    with pytest.raises(ValueError, match="must not disagree"):
        RecordingConfig.from_mapping({"task": "pick", "instruction": "place"})


def test_visual_schema_is_mp4_video_and_requires_matching_storage() -> None:
    """Visual observations use LeRobot video features rather than PNG features."""

    frame = {
        "observation.images.front_camera": np.zeros((480, 640, 3), dtype=np.uint8)
    }
    expected = build_features(frame)["observation.images.front_camera"]
    assert expected["dtype"] == "video"
    assert _feature_matches(expected, {"dtype": "video", "shape": [480, 640, 3]})
    assert not _feature_matches(expected, {"dtype": "image", "shape": [480, 640, 3]})
    assert not _feature_matches(expected, {"dtype": "video", "shape": [720, 1280, 3]})


def test_keyboard_sequences_map_to_episode_operations() -> None:
    """Linux terminal arrow and scalar keys map to the requested workflow."""

    assert decode_key(b"g") == "staging"
    assert decode_key(b"G") == "staging"
    assert decode_key(b"s") == "start"
    assert decode_key(b"o") == "optimal"
    assert decode_key(b"f") == "failure"
    assert decode_key(b"\x1b[D") == "discard"
    assert decode_key(b"\x1bOD") == "discard"
    assert decode_key(b"\x1b[C") == "save"
    assert decode_key(b"\x1bOC") == "save"
    assert decode_key(b"\x1b") == "finish"
    assert decode_key(b"x") is None


def test_episode_reset_goal_uses_yaml_absolute_joint_positions() -> None:
    """Post-episode reset forwards the configured seven xArm radians unchanged."""

    positions = tuple(float(index) / 10.0 for index in range(7))
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.config = SimpleNamespace(reset_joint_positions=positions)
    goal = recorder._reset_goal()
    assert goal.trajectory.joint_names == [f"joint{index + 1}" for index in range(7)]
    assert tuple(goal.trajectory.points[0].positions) == positions


def test_start_episode_begins_capture_without_requesting_reset() -> None:
    """An IDLE recorder is already reset and must enter capture immediately."""

    started = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.session_ready = True
    recorder.state = "IDLE"
    recorder.writer_prepared = True
    recorder.session_configured = True
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False
    recorder.config = SimpleNamespace(num_episodes=3)
    recorder.writer = SimpleNamespace(
        episode_count=0,
        discard_episode=lambda: None,
    )
    recorder.synchronizer = SimpleNamespace(clear=lambda: None)
    recorder.writer_queue = Queue()
    recorder.captured_frames_in_episode = 0
    recorder.last_error = "old"
    recorder.auto_start_pending = True
    recorder._begin_capture = lambda: started.append(True)
    recorder._start_reset = lambda *_args, **_kwargs: pytest.fail(
        "start must not request reset"
    )

    success, message = recorder._start_episode()
    assert success is True
    assert message == "episode recording started"
    assert started == [True]
    assert recorder.auto_start_pending is False


def test_staging_enables_teleoperation_without_starting_capture() -> None:
    """IDLE may enable follower commands while remaining outside RECORDING."""

    active_states = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.session_ready = True
    recorder.state = "IDLE"
    recorder.writer_prepared = True
    recorder.session_configured = True
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False
    recorder.config = SimpleNamespace(num_episodes=3)
    recorder.writer = SimpleNamespace(episode_count=0, discard_episode=lambda: None)
    recorder.synchronizer = SimpleNamespace(clear=lambda: None)
    recorder.writer_queue = Queue()
    recorder.captured_frames_in_episode = 0
    recorder.last_error = "old"
    recorder.auto_start_pending = True
    recorder._set_teleop_active = active_states.append
    recorder._set_status = lambda state: setattr(recorder, "state", state)

    success, message = recorder._start_staging()

    assert success is True
    assert message == "staging started; teleoperation is active and not recording"
    assert recorder.state == "STAGING"
    assert active_states == [True]
    assert recorder.captured_frames_in_episode == 0
    assert recorder.auto_start_pending is False


def test_start_episode_from_staging_preserves_pose_and_begins_capture() -> None:
    """Start in STAGING must record in place without requesting another reset."""

    started = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.session_ready = True
    recorder.state = "STAGING"
    recorder.writer_prepared = True
    recorder.session_configured = True
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False
    recorder.config = SimpleNamespace(num_episodes=3)
    recorder.writer = SimpleNamespace(episode_count=0, discard_episode=lambda: None)
    recorder.synchronizer = SimpleNamespace(clear=lambda: None)
    recorder.writer_queue = Queue()
    recorder.captured_frames_in_episode = 0
    recorder.last_error = ""
    recorder.auto_start_pending = False
    recorder._begin_capture = lambda: started.append(True)
    recorder._start_reset = lambda *_args, **_kwargs: pytest.fail(
        "staged start must not request reset"
    )

    success, message = recorder._start_episode()

    assert success is True
    assert message == "episode recording started"
    assert started == [True]


def test_discard_in_staging_requests_reset_without_writing_an_episode() -> None:
    """Abandoning a staged pose must only close teleop and reset hardware."""

    active_states = []
    reset_requests = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.state = "STAGING"
    recorder.post_reset_state = "COMPLETE"
    recorder._set_teleop_active = active_states.append
    recorder._start_reset = lambda continuation: reset_requests.append(continuation) or True

    success, message = recorder._discard_or_reset()

    assert success is True
    assert message == "staging abandoned and robot reset started"
    assert recorder.post_reset_state == "IDLE"
    assert active_states == [False]
    assert reset_requests == ["STAGING_ABORT"]


def test_start_episode_requires_explicit_dataset_confirmation() -> None:
    """Neither Web nor the S key may bypass the manual dataset settings gate."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.session_ready = True
    recorder.state = "IDLE"
    recorder.session_configured = False
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False

    success, message = recorder._start_episode()

    assert success is False
    assert message == "confirm dataset settings before starting an episode"


def test_failed_episode_must_reset_before_starting_again() -> None:
    """FAILED cannot bypass the discard-and-reset transition back to IDLE."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.session_ready = True
    recorder.state = "FAILED"
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False

    success, message = recorder._start_episode()
    assert success is False
    assert message == "recorder is FAILED"


def test_frame_append_does_not_block_recorder_state_lock() -> None:
    """Slow first-frame dataset creation runs only on the writer thread."""

    entered = threading.Event()
    release = threading.Event()

    class BlockingWriter:
        """Hold one synthetic frame append until the test releases it."""

        frames_in_episode = 0

        def add_frame(self, _frame) -> None:
            """Block one scalar fake frame without touching image data."""

            entered.set()
            release.wait(timeout=1.0)
            self.frames_in_episode += 1

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.capture_generation = 1
    recorder.state = "RECORDING"
    recorder.writer_queue = Queue(maxsize=2)
    recorder.writer_stop = threading.Event()
    recorder.writer_prepared = False
    recorder.writer_queue_dropped_frames = 0
    recorder.captured_frames_in_episode = 0
    recorder.writer = BlockingWriter()
    recorder._fail_episode = lambda message: pytest.fail(message)

    worker = threading.Thread(target=recorder._frame_writer_loop)
    worker.start()
    recorder._enqueue_frame({"frame": 1}, generation=1)
    assert entered.wait(timeout=1.0)
    assert recorder.lock.acquire(timeout=0.1)
    recorder.lock.release()
    release.set()
    recorder.writer_queue.join()
    recorder.writer_stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert recorder.writer.frames_in_episode == 1


def test_full_writer_queue_keeps_the_newest_synchronized_rows() -> None:
    """Bounded writer backlog drops its oldest row instead of blocking sensors."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.writer_queue = Queue(maxsize=2)
    recorder.writer_queue_dropped_frames = 0
    recorder.captured_frames_in_episode = 0

    recorder._enqueue_frame({"id": 1}, generation=3)
    recorder._enqueue_frame({"id": 2}, generation=3)
    recorder._enqueue_frame({"id": 3}, generation=3)

    queued = [recorder.writer_queue.get_nowait() for _ in range(2)]
    assert [item[1]["id"] for item in queued] == [2, 3]
    assert recorder.writer_queue_dropped_frames == 1
    assert recorder.captured_frames_in_episode == 2
    for _ in queued:
        recorder.writer_queue.task_done()


def test_single_anchor_camera_gap_does_not_fail_episode() -> None:
    """Normal D435 USB/DDS jitter must keep the next physical frame as anchor."""

    anchors = []
    image_buffer = SimpleNamespace(
        add=lambda timestamp, _value: anchors.append(timestamp) or timestamp
    )
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.stream_by_key = {
        "front_camera": SimpleNamespace(kind="rgb", output_size=None)
    }
    recorder.config = SimpleNamespace(
        anchor_key="front_camera",
        fps=30,
        task="test",
    )
    frames = []
    recorder.synchronizer = SimpleNamespace(
        images={"front_camera": image_buffer},
        assemble=lambda anchor, _task: {"anchor": anchor},
    )
    recorder.capture_generation = 2
    recorder.instruction = "test"
    recorder._enqueue_frame = lambda frame, generation: frames.append((frame, generation))
    recorder.state = "RECORDING"
    recorder.last_anchor_timestamp = 1.0

    message = Image()
    message.header.stamp.sec = 1
    message.header.stamp.nanosec = 100_000_000
    message.height = 1
    message.width = 1
    message.encoding = "rgb8"
    message.step = 3
    message.data = bytes([1, 2, 3])
    recorder.image_callback("front_camera", message)

    assert anchors == [1.1]
    assert len(frames) == 1
    assert frames[0][1] == 2


def test_missing_camera_frame_is_dropped_without_stopping_teleop() -> None:
    """A genuinely missing secondary image drops one row without stopping capture."""

    warnings = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    image_buffer = SimpleNamespace(add=lambda timestamp, value: (timestamp, value))
    recorder.stream_by_key = {
        "front_camera": SimpleNamespace(kind="rgb", output_size=None)
    }
    recorder.capture_generation = 1
    recorder.instruction = "test"
    recorder.state = "RECORDING"
    recorder.last_anchor_timestamp = None
    recorder.config = SimpleNamespace(
        anchor_key="front_camera",
        fps=30,
        task="test",
    )
    recorder.synchronizer = SimpleNamespace(
        images={"front_camera": image_buffer},
        assemble=lambda _anchor, _task: (_ for _ in ()).throw(
            SynchronizationError("No sample available for wrist_camera")
        )
    )
    recorder._enqueue_frame = lambda *_args: pytest.fail("must drop")
    recorder.sync_dropped_frames = 0
    recorder.last_sync_warning = ""
    recorder.last_sync_warning_at = 0.0
    recorder.get_logger = lambda: SimpleNamespace(warn=warnings.append)
    recorder._fail_episode = lambda message: pytest.fail(message)

    message = Image()
    message.header.stamp.sec = 1
    message.height = 1
    message.width = 1
    message.encoding = "rgb8"
    message.step = 3
    message.data = bytes([1, 2, 3])
    recorder.image_callback("front_camera", message)

    assert recorder.state == "RECORDING"
    assert recorder.sync_dropped_frames == 1
    assert recorder.last_sync_warning == "No sample available for wrist_camera"
    assert len(warnings) == 1


def test_xarm_error_saves_active_episode_as_crash_before_reset() -> None:
    """A controller error routes non-empty teleoperation data to ``crash``."""

    finish_requests = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.captured_frames_in_episode = 1
    recorder.state = "RECORDING"
    recorder.error_recovery_pending = False
    recorder.writer = SimpleNamespace(episode_count=2)
    recorder.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    recorder._finish_episode = lambda **kwargs: (
        finish_requests.append(kwargs) is None,
        "accepted",
    )

    error = Int32MultiArray()
    error.data = [12, 0]
    recorder.xarm_error_callback(error)
    assert finish_requests == [
        {
            "save": True,
            "finish_session": False,
            "automatic_error_reset": True,
            "category": "crash",
        }
    ]


def test_crash_reset_completion_unblocks_teleoperation_writer() -> None:
    """Crash reset completion joins the pending teleoperation save transaction."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.error_recovery_pending = True
    recorder.error_reset_started = True
    recorder.factr_recovery_pending = False
    recorder.factr_connected = True
    recorder.evaluation_mode = False
    recorder.episode_completion_pending = True
    recorder.episode_reset_done = False
    recorder.system = SimpleNamespace(
        speech=SimpleNamespace(phrase=lambda event: event)
    )
    recorder._announce = lambda *_args: None
    completions = []
    recorder._complete_episode_if_ready = lambda: completions.append(True)

    recorder.reset_status_callback(SimpleNamespace(data="COMPLETE"))

    assert recorder.episode_reset_done is True
    assert recorder.error_recovery_pending is False
    assert completions == [True]


def test_factr_disconnect_discards_current_episode_and_requests_reset() -> None:
    """A runtime leader loss drops only the active episode, not the session."""

    discarded = []
    cleared = []
    resets = []
    states = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.lock = threading.RLock()
    recorder.writer_lock = threading.Lock()
    recorder.capture_generation = 1
    recorder.writer_queue = Queue()
    recorder.captured_frames_in_episode = 4
    recorder.state = "RECORDING"
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False
    recorder.reset_continuation = None
    recorder.post_reset_state = "IDLE"
    recorder.last_error = ""
    recorder.writer = SimpleNamespace(
        episode_count=3,
        discard_episode=lambda: discarded.append(True),
    )
    recorder.synchronizer = SimpleNamespace(clear=lambda: cleared.append(True))
    recorder.system = SimpleNamespace(
        speech=SimpleNamespace(phrase=lambda event, episode=None: event)
    )
    recorder.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    recorder._discard_queued_frames = lambda: None
    recorder._set_teleop_active = lambda active: setattr(
        recorder, "teleop_active", active
    )
    recorder._announce = lambda *_args: None
    recorder._set_status = lambda state: (
        states.append(state),
        setattr(recorder, "state", state),
    )
    recorder._start_factr_recovery_reset = lambda: resets.append(True)

    recorder.factr_connected_callback(Bool(data=False))

    assert discarded == [True]
    assert cleared == [True]
    assert resets == [True]
    assert states == ["FACTR_RECONNECTING"]
    assert recorder.factr_recovery_pending is True
    assert recorder.teleop_active is False
    assert recorder.captured_frames_in_episode == 0


def test_episode_unlocks_only_after_write_and_reset_are_both_done() -> None:
    """Finishing video encoding alone must not enable the next episode."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.episode_completion_pending = True
    recorder.episode_write_done = False
    recorder.episode_reset_done = True
    recorder.episode_write_error = ""
    recorder.episode_reset_error = ""
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False
    recorder.episode_finish_session = False
    recorder.episode_saved = True
    recorder.episode_number = 1
    recorder.writer = SimpleNamespace(episode_count=1)
    recorder.config = SimpleNamespace(num_episodes=2)
    recorder.system = SimpleNamespace(
        speech=SimpleNamespace(phrase=lambda event, episode: f"{event}:{episode}")
    )
    states = []
    recorder._announce = lambda *_args: None
    recorder._set_status = states.append

    recorder._complete_episode_if_ready()
    assert states == []
    assert recorder.episode_completion_pending is True

    recorder.episode_write_done = True
    recorder._complete_episode_if_ready()
    assert states == ["IDLE"]
    assert recorder.episode_completion_pending is False


def test_episode_waits_for_reset_even_when_write_fails() -> None:
    """A writer failure is reported only after the concurrent reset branch ends."""

    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.episode_completion_pending = True
    recorder.episode_write_done = True
    recorder.episode_reset_done = False
    recorder.episode_write_error = "video encoding failed"
    recorder.episode_reset_error = ""
    recorder.episode_finish_session = False
    recorder.last_error = ""
    recorder.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    states = []
    recorder._set_status = states.append

    recorder._complete_episode_if_ready()
    assert states == []

    recorder.episode_reset_done = True
    recorder._complete_episode_if_ready()
    assert states == ["FAILED"]
    assert recorder.last_error == "video encoding failed"


def test_episode_reset_timeout_waits_for_writer_before_failing() -> None:
    """A timed-out Action must cancel safely without racing the writer thread."""

    canceled = []
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.reset_deadline = 1.0
    recorder.reset_goal_handle = SimpleNamespace(
        cancel_goal_async=lambda: canceled.append(True)
    )
    recorder.reset_continuation = "EPISODE_COMPLETION"
    recorder.episode_completion_pending = True
    recorder.episode_write_done = False
    recorder.episode_reset_done = False
    recorder.episode_write_error = ""
    recorder.episode_reset_error = ""
    recorder.factr_connected = True
    recorder.factr_recovery_pending = False

    recorder._check_reset_timeout(2.0)
    assert canceled == [True]
    assert recorder.episode_reset_done is True
    assert recorder.episode_completion_pending is True
    assert recorder.episode_reset_error == "reset action timed out"


def test_prepare_creates_missing_dataset_parents(tmp_path: Path) -> None:
    """Writer preparation creates the full missing parent chain of the root."""

    from unittest.mock import MagicMock

    dataset_root = tmp_path / "datasets" / "new_task" / "run1"
    writer = LeRobotWriter(
        root=dataset_root,
        repo_id="local/test",
        robot_type="xarm7_factr",
        fps=30,
        vcodec="h264",
        image_writer_threads=1,
        streaming_encoding=False,
    )
    created = MagicMock()
    created.save_episode = MagicMock()
    monkeypatch_target = "xarm_ros2_teleop.recording.lerobot_writer.import_module"
    import pytest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            monkeypatch_target,
            lambda _name: MagicMock(create=MagicMock(return_value=created)),
        )
        example_frame = {"observation.state": np.zeros(8, dtype=np.float32)}
        writer.prepare(example_frame)

    assert dataset_root.parent.is_dir()
