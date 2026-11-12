#!/usr/bin/env python3
"""Convert a local LeRobot v3.0 dataset into the legacy v2.1 layout (data tool)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V21_STATS = ("min", "max", "mean", "std", "count")


def parse_args() -> argparse.Namespace:
    """Parse source/output paths and readable-prefix recovery options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeRobot v3.0 dataset directory")
    parser.add_argument("output", type=Path, help="New LeRobot v2.1 dataset directory")
    parser.add_argument(
        "--allow-readable-prefix",
        action="store_true",
        help="convert the readable leading episode metadata files before a corrupt file",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object from ``path`` and return it as a dictionary."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one dictionary to UTF-8 JSON with stable, human-readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as newline-delimited UTF-8 JSON records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_episode_rows(
    source: Path, allow_readable_prefix: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load ordered v3 episode rows, optionally stopping at the first corrupt file.

    Args:
        source: Root directory of one LeRobot v3.0 dataset.
        allow_readable_prefix: Whether a nonempty leading set of readable metadata
            files may be returned when the next Parquet file is corrupt.

    Returns:
        A pair containing episode dictionaries and human-readable recovery warnings.
    """

    metadata_paths = sorted((source / "meta" / "episodes").glob("*/*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"no v3 episode metadata found under {source}")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in metadata_paths:
        try:
            rows.extend(pq.ParquetFile(path).read().to_pylist())
        except (OSError, pa.ArrowException) as error:
            if not allow_readable_prefix or not rows:
                raise RuntimeError(f"cannot read episode metadata {path}: {error}") from error
            warnings.append(f"stopped before unreadable episode metadata {path}: {error}")
            break

    rows.sort(key=lambda row: int(row["episode_index"]))
    expected_indices = list(range(len(rows)))
    actual_indices = [int(row["episode_index"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(
            f"episode indices must be contiguous from zero: got {actual_indices[:5]}..."
        )
    return rows, warnings


def load_tasks(source: Path) -> list[dict[str, Any]]:
    """Convert the v3 task Parquet table into v2.1 task JSON records."""

    table = pq.ParquetFile(source / "meta" / "tasks.parquet").read()
    task_column = "task" if "task" in table.column_names else "__index_level_0__"
    if task_column not in table.column_names or "task_index" not in table.column_names:
        raise ValueError(f"unsupported task schema: {table.column_names}")
    records = [
        {"task_index": int(task_index), "task": str(task)}
        for task_index, task in zip(
            table["task_index"].to_pylist(), table[task_column].to_pylist(), strict=True
        )
    ]
    return sorted(records, key=lambda row: row["task_index"])


def build_v21_info(
    source_info: dict[str, Any], episode_rows: list[dict[str, Any]], task_count: int
) -> dict[str, Any]:
    """Build v2.1 ``info.json`` metadata for the selected episodes."""

    info = dict(source_info)
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    episode_count = len(episode_rows)
    info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": episode_count,
            "total_frames": sum(int(row["length"]) for row in episode_rows),
            "total_tasks": task_count,
            "total_videos": episode_count * len(video_keys),
            "total_chunks": (episode_count - 1) // int(info["chunks_size"]) + 1,
            "splits": {"train": f"0:{episode_count}"},
            "data_path": V21_DATA_PATH,
            "video_path": V21_VIDEO_PATH if video_keys else None,
        }
    )
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    return info


def episode_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Unflatten one v3 episode's statistics into the v2.1 nested representation."""

    stats: dict[str, dict[str, Any]] = {}
    for column, value in row.items():
        if not column.startswith("stats/"):
            continue
        feature, metric = column.removeprefix("stats/").rsplit("/", 1)
        if metric in V21_STATS:
            stats.setdefault(feature, {})[metric] = value
    if not stats:
        raise ValueError(f"episode {row['episode_index']} has no statistics")
    return stats


def convert_data(
    source: Path, output: Path, episode_rows: list[dict[str, Any]], chunks_size: int
) -> None:
    """Split packed v3 Parquet data into one v2.1 Parquet file per episode.

    Each output table keeps the original Arrow schema and contains ``length`` rows.
    Vector columns such as state/action retain their original fixed-size-list shapes.
    """

    rows_by_file: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        rows_by_file.setdefault(key, []).append(row)

    for (chunk_index, file_index), rows in rows_by_file.items():
        source_path = source / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        try:
            table = pq.ParquetFile(source_path).read()
        except (OSError, pa.ArrowException) as error:
            message = f"cannot read referenced data file {source_path}: {error}"
            raise RuntimeError(message) from error

        for row in rows:
            episode_index = int(row["episode_index"])
            expected_length = int(row["length"])
            mask = pc.equal(table["episode_index"], pa.scalar(episode_index, pa.int64()))
            episode_table = table.filter(mask)
            if episode_table.num_rows != expected_length:
                raise ValueError(
                    f"episode {episode_index}: expected {expected_length} data rows, "
                    f"found {episode_table.num_rows}"
                )
            episode_chunk = episode_index // chunks_size
            output_path = output / V21_DATA_PATH.format(
                episode_chunk=episode_chunk, episode_index=episode_index
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # v3 embeds a Hugging Face ``List`` feature marker that datasets 3.x
            # (used by LeRobot 0.3.3) cannot parse. Arrow's fixed-size-list type
            # carries the same shape and is inferred as the v2.1 ``Sequence``.
            episode_table = episode_table.replace_schema_metadata(None)
            pq.write_table(episode_table, output_path)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one external media command and raise an error containing its stderr."""

    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(command)}):\n{result.stderr.strip()}")
    return result


def probe_video(path: Path) -> tuple[int, float, float]:
    """Return ``(packet_count, start_seconds, duration_seconds)`` for one MP4 video."""

    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets,start_time,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["nb_read_packets"]), float(stream["start_time"]), float(stream["duration"])


def split_video(
    source_path: Path, output_path: Path, start_s: float, frame_count: int, fps: int
) -> None:
    """Remux an exact CFR frame interval into an independent v2.1 episode MP4.

    Args:
        source_path: Packed v3 MP4 containing one or more episodes.
        output_path: Per-episode v2.1 MP4 to create.
        start_s: Episode start timestamp in the packed video, in seconds.
        frame_count: Number of video frames to copy, shape-equivalent to one episode.
        fps: Constant video frame rate in frames per second.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.9f}",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frame_count),
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    actual_frames, actual_start_s, actual_duration_s = probe_video(output_path)
    expected_duration_s = frame_count / fps
    if actual_frames != frame_count or abs(actual_start_s) > 1e-4:
        raise ValueError(
            f"invalid split {output_path}: frames={actual_frames}/{frame_count}, "
            f"start={actual_start_s:.6f}s"
        )
    if abs(actual_duration_s - expected_duration_s) > 1.0 / fps + 1e-4:
        raise ValueError(
            f"invalid duration {output_path}: {actual_duration_s:.6f}s, "
            f"expected {expected_duration_s:.6f}s"
        )


def convert_videos(
    source: Path,
    output: Path,
    source_info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Split packed videos using the authoritative episode data lengths.

    Returns:
        Mapping from episode index to video key and the number of unpaired tail
        frames omitted from that camera stream. An empty mapping means every
        source interval already matched its state/action row count.
    """

    fps = int(source_info["fps"])
    chunks_size = int(source_info["chunks_size"])
    video_keys = [
        key for key, feature in source_info["features"].items() if feature["dtype"] == "video"
    ]
    trimmed_tail_frames: dict[str, dict[str, int]] = {}
    for episode_number, row in enumerate(episode_rows, start=1):
        episode_index = int(row["episode_index"])
        expected_frame_count = int(row["length"])
        episode_chunk = episode_index // chunks_size
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            source_path = source / source_info["video_path"].format(
                video_key=video_key,
                chunk_index=int(row[f"{prefix}/chunk_index"]),
                file_index=int(row[f"{prefix}/file_index"]),
            )
            start_s = float(row[f"{prefix}/from_timestamp"])
            end_s = float(row[f"{prefix}/to_timestamp"])
            source_frame_count = round((end_s - start_s) * fps)
            if source_frame_count < expected_frame_count:
                raise ValueError(
                    f"episode {episode_index} {video_key}: video interval has "
                    f"{source_frame_count} frames but data requires {expected_frame_count}"
                )
            extra_frames = source_frame_count - expected_frame_count
            if extra_frames:
                trimmed_tail_frames.setdefault(str(episode_index), {})[video_key] = extra_frames
            output_path = output / V21_VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            split_video(source_path, output_path, start_s, expected_frame_count, fps)
        print(f"videos: episode {episode_number}/{len(episode_rows)}", flush=True)
    return trimmed_tail_frames


def summarize_video_tail_trims(
    trimmed_tail_frames: dict[str, dict[str, int]],
) -> dict[str, int | dict[str, int]]:
    """Collapse equal per-camera tail counts into one episode-level value."""

    summary: dict[str, int | dict[str, int]] = {}
    for episode_index, camera_counts in trimmed_tail_frames.items():
        unique_counts = set(camera_counts.values())
        summary[episode_index] = (
            next(iter(unique_counts)) if len(unique_counts) == 1 else camera_counts
        )
    return summary


def convert_dataset(source: Path, output: Path, allow_readable_prefix: bool = False) -> list[str]:
    """Convert one local v3.0 dataset into a new, non-destructive v2.1 directory.

    Args:
        source: Existing LeRobot v3.0 dataset root.
        output: New destination directory; it must not already exist.
        allow_readable_prefix: Permit conversion of leading metadata files before
            a corrupt later file. Referenced data/video files must still be valid.

    Returns:
        Warning strings describing any deliberately skipped corrupt suffix.
    """

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    source_info = read_json(source / "meta" / "info.json")
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected v3.0 source, got {source_info.get('codebase_version')!r}")

    episode_rows, warnings = load_episode_rows(source, allow_readable_prefix)
    tasks = load_tasks(source)
    info = build_v21_info(source_info, episode_rows, len(tasks))
    output.mkdir(parents=True)
    try:
        write_json(output / "meta" / "info.json", info)
        write_jsonl(output / "meta" / "tasks.jsonl", tasks)
        write_jsonl(
            output / "meta" / "episodes.jsonl",
            [
                {
                    "episode_index": int(row["episode_index"]),
                    "tasks": row["tasks"],
                    "length": int(row["length"]),
                }
                for row in episode_rows
            ],
        )
        write_jsonl(
            output / "meta" / "episodes_stats.jsonl",
            [
                {
                    "episode_index": int(row["episode_index"]),
                    "stats": episode_stats(row),
                }
                for row in episode_rows
            ],
        )
        convert_data(source, output, episode_rows, int(info["chunks_size"]))
        trimmed_tail_frames = convert_videos(source, output, source_info, episode_rows)
        trim_summary = summarize_video_tail_trims(trimmed_tail_frames)
        for episode_index, extra_frames in trim_summary.items():
            warnings.append(
                f"episode {episode_index}: omitted {extra_frames} unpaired video tail frames"
            )
        write_json(
            output / "recovery_report.json",
            {
                "source": str(source),
                "source_was_modified": False,
                "recovered_episode_count": len(episode_rows),
                "recovered_frame_count": sum(int(row["length"]) for row in episode_rows),
                "trimmed_stale_video_tail_frames": trim_summary,
            },
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return warnings


def main() -> int:
    """Run the command-line conversion and print a concise result summary."""

    args = parse_args()
    warnings = convert_dataset(args.source, args.output, args.allow_readable_prefix)
    info = read_json(args.output / "meta" / "info.json")
    print(
        f"converted {info['total_episodes']} episodes / {info['total_frames']} frames "
        f"to {args.output.resolve()}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
