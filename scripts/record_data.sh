#!/usr/bin/env bash
# Shared FACTR/VR recording entry point. VR motion requires --motion.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_SRC}/.." && pwd)"

# --------------------------- User settings ---------------------------
# These values can be edited here, provided as environment variables, or
# overridden with the command-line flags documented by --help.
DATASET_PATH="${DATASET_PATH:-}"
NUM_EPISODES="${NUM_EPISODES:-50}"
INSTRUCTION="${INSTRUCTION:-insert the block into the socket}"
APPEND_TIMESTAMP="${APPEND_TIMESTAMP:-false}"
BASE_CONFIG="${BASE_CONFIG:-}"
TELEOP_SOURCE="${TELEOP_SOURCE:-factr}"
VR_MOTION=false
START_RVIZ=false
RL_MODE=false
# --------------------------------------------------------------------

FORCE_BUILD=false
DRY_RUN=false
SKIP_LATENCY_CHECK=false
RUNTIME_CONFIG=""

usage() {
  # Print scalar CLI options; this function does not modify hardware or files.
  cat <<'EOF'
Usage: ./scripts/record_data.sh [options]

Options:
  --source factr|vr    Select joint leader or native TCP VR (default: factr)
  --motion             Explicitly enable VR/RL physical motion (default: preview)
  --rviz               Display mapped TCP target axes for the VR source
  --rl                 Use the shared RL panel and human/policy ownership switch
  --dataset PATH       Exact dataset directory (default: /home/gxy/datasets/name)
  --episodes N         Maximum episodes in this session (default: 50)
  --instruction TEXT   LeRobot task instruction stored with every frame
  --timestamp          Save to PATH_YYYYmmdd_HHMMSS instead of exact PATH
  --no-timestamp       Save to the exact dataset path (default)
  --config PATH        Base unified YAML configuration
  --build              Rebuild both ROS packages before launching
  --skip-latency-check Do not check/set the FACTR FTDI latency timer
  --dry-run            Validate and print the generated configuration only
  -h, --help           Show this help

The same settings may be supplied as DATASET_PATH, NUM_EPISODES, INSTRUCTION,
APPEND_TIMESTAMP and BASE_CONFIG environment variables.
EOF
}

cleanup() {
  # Remove only the temporary YAML created by this process.
  if [[ -n "${RUNTIME_CONFIG}" && -f "${RUNTIME_CONFIG}" ]]; then
    rm -f -- "${RUNTIME_CONFIG}"
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      TELEOP_SOURCE="${2:?--source requires factr or vr}"
      shift 2
      ;;
    --motion)
      VR_MOTION=true
      shift
      ;;
    --rviz)
      START_RVIZ=true
      shift
      ;;
    --rl)
      RL_MODE=true
      shift
      ;;
    --dataset)
      DATASET_PATH="${2:?--dataset requires a path}"
      shift 2
      ;;
    --episodes)
      NUM_EPISODES="${2:?--episodes requires a positive integer}"
      shift 2
      ;;
    --instruction)
      INSTRUCTION="${2:?--instruction requires text}"
      shift 2
      ;;
    --timestamp)
      APPEND_TIMESTAMP=true
      shift
      ;;
    --no-timestamp)
      APPEND_TIMESTAMP=false
      shift
      ;;
    --config)
      BASE_CONFIG="${2:?--config requires a path}"
      shift 2
      ;;
    --build)
      FORCE_BUILD=true
      shift
      ;;
    --skip-latency-check)
      SKIP_LATENCY_CHECK=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${TELEOP_SOURCE}" != factr && "${TELEOP_SOURCE}" != vr ]]; then
  echo "--source must be factr or vr" >&2
  exit 2
fi
if [[ "${VR_MOTION}" == true && "${TELEOP_SOURCE}" != vr && "${RL_MODE}" != true ]]; then
  echo "--motion applies to VR/RL; FACTR teleop keeps its existing behavior" >&2
  exit 2
fi
if [[ -z "${BASE_CONFIG}" ]]; then
  PROFILE_NAME=teleop_system
  [[ "${TELEOP_SOURCE}" == vr ]] && PROFILE_NAME=teleop_vr
  [[ "${TELEOP_SOURCE}" == vr && "${RL_MODE}" == true ]] && PROFILE_NAME=teleop_vr_rl
  BASE_CONFIG="${PROJECT_SRC}/xarm_ros2_teleop/config/${PROFILE_NAME}.yaml"
fi
if [[ -z "${DATASET_PATH}" ]]; then
  DATASET_PATH="${HOME}/datasets/name"
  [[ "${TELEOP_SOURCE}" == vr ]] && DATASET_PATH="${HOME}/datasets/vr"
  [[ "${TELEOP_SOURCE}" == vr && "${RL_MODE}" == true ]] && DATASET_PATH="${HOME}/datasets/vr_rl"
fi

if [[ "${DATASET_PATH}" != /* ]]; then
  echo "Dataset path must be absolute: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ ! "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Episode count must be a positive integer: ${NUM_EPISODES}" >&2
  exit 2
fi
if [[ -z "${INSTRUCTION//[[:space:]]/}" ]]; then
  echo "Instruction must not be empty." >&2
  exit 2
fi
if [[ "${APPEND_TIMESTAMP}" != true && "${APPEND_TIMESTAMP}" != false ]]; then
  echo "APPEND_TIMESTAMP must be true or false." >&2
  exit 2
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base configuration does not exist: ${BASE_CONFIG}" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run ${PROJECT_SRC}/scripts/bootstrap.sh once." >&2
  exit 1
fi

if [[ ! -x "${PROJECT_SRC}/.venv/bin/python" ]]; then
  echo "Environment missing; run scripts/bootstrap.sh first." >&2
  exit 1
fi

RUNTIME_CONFIG="$(mktemp --tmpdir control_robot_recording.XXXXXX.yaml)"
PYTHONPATH="${PROJECT_SRC}/xarm_ros2_teleop:${PYTHONPATH:-}" \
"${PROJECT_SRC}/.venv/bin/python" - \
  "${BASE_CONFIG}" \
  "${RUNTIME_CONFIG}" \
  "${DATASET_PATH}" \
  "${NUM_EPISODES}" \
  "${INSTRUCTION}" \
  "${APPEND_TIMESTAMP}" "${TELEOP_SOURCE}" "${VR_MOTION}" "${RL_MODE}" <<'PY'
"""Create one temporary recording YAML without changing the checked-in profile."""

import sys
from pathlib import Path

import yaml
from xarm_ros2_teleop.system.config import load_system_mapping
from xarm_ros2_teleop.vr.config import VRConfig

source_path, output_path, dataset_path, episode_count, instruction, append_timestamp, source, motion, rl = (
    sys.argv[1:]
)
config = load_system_mapping(source_path)
config["teleop"]["source"] = source
if rl == "true":
    config["rl"]["enabled"] = True
    config["rl"]["dataset_root"] = dataset_path
    config["inference"]["enabled"] = True
if source == "vr":
    config["recording"]["robot_type"] = "xarm7_vr_tcp"
    config["recording"]["repo_id"] = "local/xarm7_vr_tcp"
    if motion == "true":
        VRConfig.from_mapping(config["teleop"].get("vr", {})).require_workspace()

recording = config["recording"]
recording["enabled"] = True
recording["root"] = dataset_path
recording["dataset_name"] = ""
recording["append_timestamp"] = append_timestamp == "true"
recording["num_episodes"] = int(episode_count)
recording["instruction"] = instruction

with Path(output_path).open("w", encoding="utf-8") as output_file:
    yaml.safe_dump(config, output_file, sort_keys=False, allow_unicode=True)
PY

PYTHONPATH="${PROJECT_SRC}/xarm_ros2_teleop:${PYTHONPATH:-}" \
  "${PROJECT_SRC}/.venv/bin/python" - "${RUNTIME_CONFIG}" <<'PY'
"""Validate the generated YAML through the same parser used by ROS launch."""

import sys

from xarm_ros2_teleop.system.config import load_system_config

system = load_system_config(sys.argv[1])
print("Recording configuration validated:")
print(f"  dataset: {system.recording.dataset_root()}")
print(f"  episodes: {system.recording.num_episodes}")
print(f"  instruction: {system.recording.instruction}")
print(f"  streams: {', '.join(stream.key for stream in system.recording.streams)}")
print(f"  anchor: {system.recording.anchor_key}")
print(f"  tactile enabled: {system.tactile_enabled}")
PY

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run complete; no hardware was started."
  exit 0
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS Humble setup was not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

# ROS setup scripts may read unset variables, so relax nounset only while sourcing.
set +u
source /opt/ros/humble/setup.bash
set -u

if [[ "${FORCE_BUILD}" == true || ! -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
  echo "Building control_robot workspace..."
  (
    cd "${WORKSPACE_ROOT}"
    uv run --project "${PROJECT_SRC}" --no-sync \
      python "$(command -v colcon)" build \
      --base-paths "${PROJECT_SRC}/factr_teleop" "${PROJECT_SRC}/xarm_ros2_teleop" \
      --symlink-install \
      --packages-select factr_teleop xarm_ros2_teleop
  )
fi

set +u
source "${WORKSPACE_ROOT}/install/setup.bash"
set -u

if [[ "${TELEOP_SOURCE}" == factr && "${SKIP_LATENCY_CHECK}" != true ]]; then
  "${PROJECT_SRC}/scripts/diagnostics/check_factr_latency.sh" --set
fi

echo "Starting three-camera recording. Dashboard: http://127.0.0.1:8080"
echo "Press Ctrl-C here to stop the ROS launch after finalizing the dataset in the dashboard."
uv run --project "${PROJECT_SRC}" --no-sync \
  python "$(command -v ros2)" launch \
  xarm_ros2_teleop teleop.launch.py \
  config:="${RUNTIME_CONFIG}" \
  rl:="${RL_MODE}" \
  rl_motion:="${VR_MOTION}" \
  vr_motion:="${VR_MOTION}" \
  vr_rviz:="${START_RVIZ}"
