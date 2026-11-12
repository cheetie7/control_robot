"""Small lifecycle wrapper around the official LeRobot v3 dataset writer."""

import copy
import shutil
from importlib import import_module
from pathlib import Path

from xarm_ros2_teleop.recording.control_contract import validate_contract, write_contract
from xarm_ros2_teleop.recording.dataset_schema import build_features

_LEROBOT_INDEX_FEATURES = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
}


def validate_overwrite_target(root: Path) -> None:
    """Validate that an overwrite target is one exact dataset directory.

    Args:
        root: User-selected local filesystem path for one LeRobot dataset.

    Returns:
        ``None`` when the missing, empty or existing LeRobot directory is safe.

    Raises:
        ValueError: If the path is broad, symbolic, not a directory, or is a
            non-empty directory without LeRobot's ``meta/info.json`` marker.
    """

    root = Path(root).expanduser()
    resolved = root.resolve(strict=False)
    home = Path.home().resolve()
    if resolved in {Path("/"), home} or len(resolved.parts) < 3:
        raise ValueError(f"Unsafe overwrite dataset path: {root}")
    if root.is_symlink():
        raise ValueError(f"Refusing to overwrite a symbolic link: {root}")
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"Dataset path is not a directory: {root}")
    if any(root.iterdir()) and not (root / "meta" / "info.json").is_file():
        raise ValueError(
            "Refusing to overwrite a non-LeRobot directory; choose the "
            f"exact dataset folder, not a parent directory: {root}"
        )


class LeRobotWriter:
    """Create, append, save and finalize a local LeRobot v3 dataset."""

    def __init__(
        self,
        *,
        root: Path,
        repo_id: str,
        robot_type: str,
        fps: int,
        vcodec: str,
        image_writer_threads: int,
        streaming_encoding: bool,
        write_mode: str = "resume",
        control_contract: dict | None = None,
    ) -> None:
        """Store scalar writer settings without importing LeRobot prematurely.

        Args:
            root: Complete local LeRobot dataset directory.
            repo_id: Logical LeRobot repository identifier.
            robot_type: Robot metadata string stored by LeRobot.
            fps: Dataset frame rate in frames per second.
            vcodec: Video codec passed to LeRobot.
            image_writer_threads: Number of temporary image writer workers.
            streaming_encoding: Whether LeRobot streams video while recording.
            write_mode: ``resume`` opens a compatible dataset; ``overwrite``
                removes the exact dataset directory before creating it.
        """

        self.root = Path(root)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = int(fps)
        self.vcodec = vcodec
        self.image_writer_threads = int(image_writer_threads)
        self.streaming_encoding = bool(streaming_encoding)
        if write_mode not in {"resume", "overwrite"}:
            raise ValueError("write_mode must be 'resume' or 'overwrite'")
        self.write_mode = write_mode
        self.control_contract = control_contract
        self.overwrite_applied = False
        self.dataset = None
        self.frames_in_episode = 0
        self.finalized = False

    @property
    def episode_count(self) -> int:
        """Return the number of episodes already committed to metadata."""

        return 0 if self.dataset is None else int(self.dataset.meta.total_episodes)

    def add_frame(self, frame: dict) -> None:
        """Create the fixed schema on first use and append one complete frame."""

        if self.finalized:
            raise RuntimeError("Cannot append to a finalized dataset writer")
        if self.dataset is None:
            self._open(frame)
        self.dataset.add_frame(dict(frame))
        self.frames_in_episode += 1

    def prepare(self, example_frame: dict) -> None:
        """Open the dataset schema without recording the example frame.

        Args:
            example_frame: One complete LeRobot-shaped mapping used only to
                infer image and numeric feature shapes.

        Returns:
            ``None``. Existing compatible datasets are opened; new datasets
            are created with zero buffered episode frames.
        """

        if self.finalized:
            raise RuntimeError("Cannot prepare a finalized dataset writer")
        if self.dataset is None:
            self._open(example_frame)

    def save_episode(self) -> None:
        """Commit the current non-empty episode to Parquet and MP4 shards."""

        if self.dataset is None or self.frames_in_episode == 0:
            raise RuntimeError("Cannot save an empty episode")
        self._validate_temporary_video_frames()
        # LeRobot defaults to a ProcessPoolExecutor when a dataset has more
        # than one video stream. Forking the recorder while ROS and image
        # workers are active can segfault inside Arrow/PyAV/FFmpeg. Encode the
        # cameras serially on the recorder's main thread; the reset node still
        # runs concurrently in its own process.
        self.dataset.save_episode(parallel_encoding=False)
        self.frames_in_episode = 0

    def save_episode_to(self, destination: "LeRobotWriter") -> None:
        """Move and commit the buffered episode into another prepared dataset.

        Args:
            destination: Compatible writer for the selected outcome category.
                The source keeps numeric rows in memory and temporary camera
                frames on disk until this method chooses their final dataset.

        Returns:
            ``None``. The destination gains one committed episode and the
            source is reset to an empty episode buffer.

        Raises:
            RuntimeError: If either writer is unprepared, the episode is
                empty, streaming encoding is active, or the destination has
                an unfinished episode.
        """

        if destination is self:
            self.save_episode()
            return
        if self.dataset is None or destination.dataset is None:
            raise RuntimeError("Both category writers must be prepared before saving")
        if self.frames_in_episode == 0:
            raise RuntimeError("Cannot save an empty episode")
        if destination.frames_in_episode:
            raise RuntimeError("Destination category contains an unfinished episode")
        if self.streaming_encoding or destination.streaming_encoding:
            raise RuntimeError("Outcome folders require non-streaming video encoding")

        self._validate_temporary_video_frames()
        episode_data = copy.deepcopy(self.dataset.episode_buffer)
        source_index = int(self.dataset.meta.total_episodes)
        destination_index = int(destination.dataset.meta.total_episodes)
        episode_data["episode_index"] = destination_index

        moved_directories: list[tuple[Path, Path]] = []
        try:
            for camera_key in self.dataset.meta.camera_keys:
                source_dir = self.dataset._get_image_file_dir(source_index, camera_key)
                destination_dir = destination.dataset._get_image_file_dir(
                    destination_index, camera_key
                )
                if destination_dir.exists():
                    raise RuntimeError(
                        f"Destination contains stale temporary frames: {destination_dir}"
                    )
                destination_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_dir), str(destination_dir))
                moved_directories.append((source_dir, destination_dir))
                episode_data[camera_key] = [
                    str(destination_dir / Path(frame_path).name)
                    for frame_path in episode_data[camera_key]
                ]

            destination.dataset.save_episode(
                episode_data=episode_data,
                parallel_encoding=False,
            )
        except Exception:
            # A failed commit leaves the original in-memory rows intact. Move
            # temporary frames back so the operator can still discard safely.
            for source_dir, destination_dir in reversed(moved_directories):
                if destination_dir.exists() and not source_dir.exists():
                    source_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination_dir), str(source_dir))
            raise

        self.dataset.clear_episode_buffer(delete_images=True)
        self.frames_in_episode = 0

    def discard_episode(self) -> None:
        """Discard buffered rows and temporary images for the current episode."""

        if self.dataset is not None:
            episode_index = int(self.dataset.meta.total_episodes)
            if self.frames_in_episode:
                self.dataset.clear_episode_buffer(delete_images=True)
            self._remove_temporary_video_frames(episode_index)
        self.frames_in_episode = 0

    def flush_for_readers(self) -> None:
        """Close committed shards and reopen append mode for external data readers.

        Returns:
            ``None``. All committed Parquet footers and metadata become
            readable before the episode label is published. LeRobot resume
            starts fresh shards, so later saves cannot overwrite prior rows.
            Call only between episodes, with no buffered frames.
        """

        if self.dataset is None or self.frames_in_episode:
            raise RuntimeError("Only a prepared writer with no pending frames can be flushed")
        self.dataset.finalize()
        if self.dataset.image_writer is not None:
            self.dataset.stop_image_writer()
        dataset_class = import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset
        self.dataset = dataset_class(
            self.repo_id,
            root=self.root,
            vcodec=self.vcodec,
            streaming_encoding=self.streaming_encoding,
        )
        self.dataset.start_image_writer(num_threads=self.image_writer_threads)

    def finalize(self) -> None:
        """Close video and Parquet writers so the v3 dataset remains readable."""

        if self.finalized:
            return
        if self.dataset is not None:
            if self.frames_in_episode:
                self.discard_episode()
            self.dataset.finalize()
        self.finalized = True

    def _open(self, first_frame: dict) -> None:
        """Create a new dataset or resume one whose fixed schema matches."""

        dataset_class = import_module("lerobot.datasets.lerobot_dataset").LeRobotDataset
        if self.write_mode == "resume":
            validate_contract(self.root, self.control_contract)
        features = build_features(first_frame)
        if (
            self.control_contract is not None
            and self.control_contract["action_format"] == "absolute_tcp_pose_xyzw_plus_gripper"
        ):
            features["action"]["names"] = self.control_contract["action_fields"]
            features["observation.state"]["names"] = self.control_contract["state_fields"]
        # Operator-supplied dataset paths may name parents that do not exist
        # yet (e.g. ~/datasets/new_task/run1); create the full chain so the
        # dashboard does not need a manual mkdir before applying settings.
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.write_mode == "overwrite" and not self.overwrite_applied:
            self._remove_existing_root()
            self.overwrite_applied = True
        info_path = self.root / "meta" / "info.json"
        if info_path.exists():
            dataset = dataset_class(
                self.repo_id,
                root=self.root,
                vcodec=self.vcodec,
                streaming_encoding=self.streaming_encoding,
            )
            existing_features = set(dataset.features) - _LEROBOT_INDEX_FEATURES
            expected_features = set(features)
            missing = expected_features - existing_features
            extra = existing_features - expected_features
            mismatched = {
                key
                for key in features.keys() & dataset.features.keys()
                if not _feature_matches(features[key], dataset.features[key])
            }
            if dataset.fps != self.fps or missing or extra or mismatched:
                raise ValueError(
                    f"Existing dataset schema mismatch: fps={dataset.fps}, "
                    f"missing={sorted(missing)}, extra={sorted(extra)}, "
                    f"feature={sorted(mismatched)}"
                )
            self.dataset = dataset
            write_contract(self.root, self.control_contract)
            # LeRobot 0.4.x clears only ``image`` feature directories when an
            # episode is discarded. Our cameras are ``video`` features, so an
            # interrupted process can also leave PNGs for the next index.
            self._remove_temporary_video_frames(int(dataset.meta.total_episodes))
            return
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Recording root is not an empty dataset: {self.root}")
        self.dataset = dataset_class.create(
            repo_id=self.repo_id,
            root=self.root,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=True,
            image_writer_threads=self.image_writer_threads,
            vcodec=self.vcodec,
            streaming_encoding=self.streaming_encoding,
        )
        write_contract(self.root, self.control_contract)

    def _remove_temporary_video_frames(self, episode_index: int) -> None:
        """Delete uncommitted PNG directories for one episode index.

        Args:
            episode_index: Zero-based index of the uncommitted episode whose
                temporary camera frames must be removed.

        Returns:
            ``None``. Only directories belonging to configured ``video``
            features at the exact episode index are removed.
        """

        if self.dataset is None or self.streaming_encoding:
            return
        self.dataset._wait_image_writer()
        for video_key in self.dataset.meta.video_keys:
            image_dir = self.dataset._get_image_file_dir(episode_index, video_key)
            if image_dir.is_dir():
                shutil.rmtree(image_dir)

    def _validate_temporary_video_frames(self) -> None:
        """Require one temporary PNG per buffered row and camera before save.

        Returns:
            ``None`` when every non-streaming video directory contains exactly
            ``frames_in_episode`` PNG files.

        Raises:
            RuntimeError: If stale or missing camera frames would make video,
                state and action lengths diverge.
        """

        if self.dataset is None or self.streaming_encoding:
            return
        self.dataset._wait_image_writer()
        episode_index = int(self.dataset.meta.total_episodes)
        mismatches = {}
        for video_key in self.dataset.meta.video_keys:
            image_dir = self.dataset._get_image_file_dir(episode_index, video_key)
            frame_count = sum(1 for path in image_dir.glob("*.png") if path.is_file())
            if frame_count != self.frames_in_episode:
                mismatches[video_key] = frame_count
        if mismatches:
            raise RuntimeError(
                f"episode {episode_index} temporary video frame mismatch: "
                f"expected {self.frames_in_episode}, found {mismatches}; "
                "discard this episode before retrying"
            )

    def _remove_existing_root(self) -> None:
        """Remove only an explicitly configured LeRobot dataset directory.

        The operation rejects filesystem roots, the user's home directory and
        symbolic links. A missing or empty directory is a valid fresh target;
        a non-empty directory must contain LeRobot's ``meta/info.json`` marker.
        """

        root = self.root.expanduser()
        validate_overwrite_target(root)
        if root.exists():
            shutil.rmtree(root)


def _feature_matches(expected: dict, existing: dict) -> bool:
    """Return whether one stored feature has the expected shape and dtype."""

    expected_dtype = str(expected.get("dtype"))
    existing_dtype = str(existing.get("dtype"))
    return expected_dtype == existing_dtype and tuple(expected.get("shape", ())) == tuple(
        existing.get("shape", ())
    )
