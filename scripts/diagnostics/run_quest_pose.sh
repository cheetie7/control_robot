#!/usr/bin/env bash
# Build when requested, then launch the Quest-to-ROS pose bridge over USB.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_SRC}/.." && pwd)"

FORCE_BUILD=false
START_RVIZ=false
DEVICE_SERIAL=""

usage() {
  # Print supported scalar options without changing the ROS workspace or device.
  cat <<'EOF'
Usage: ./scripts/diagnostics/run_quest_pose.sh [options]

Options:
  --build          Rebuild xarm_ros2_teleop before launching
  --device SERIAL  Select one Quest when multiple devices are connected
  --rviz           Open RViz with current axes and 3 seconds of position dots
  -h, --help       Show this help

The bridge waits up to 120 seconds for an authorized Quest, wakes it, launches the
installed pose app, and configures the USB connection automatically.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      FORCE_BUILD=true
      shift
      ;;
    --device)
      DEVICE_SERIAL="${2:?--device requires a serial number}"
      shift 2
      ;;
    --rviz)
      START_RVIZ=true
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

if ! command -v metavr >/dev/null 2>&1; then
  echo "metavr is required and must be available on PATH." >&2
  exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS Humble setup was not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run ${PROJECT_SRC}/scripts/bootstrap.sh once." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u

if [[ "${FORCE_BUILD}" == true || ! -x "${WORKSPACE_ROOT}/install/xarm_ros2_teleop/lib/xarm_ros2_teleop/quest_pose_node" ]]; then
  echo "Building xarm_ros2_teleop..."
  (
    cd "${WORKSPACE_ROOT}"
    uv run --project "${PROJECT_SRC}" --no-sync \
      python "$(command -v colcon)" build \
      --base-paths "${PROJECT_SRC}/xarm_ros2_teleop" \
      --symlink-install \
      --packages-select xarm_ros2_teleop
  )
fi

set +u
source "${WORKSPACE_ROOT}/install/setup.bash"
set -u

LAUNCH_ARGUMENTS=()
if [[ -n "${DEVICE_SERIAL}" ]]; then
  LAUNCH_ARGUMENTS+=("device_serial:=${DEVICE_SERIAL}")
fi
if [[ "${START_RVIZ}" == true ]]; then
  LAUNCH_ARGUMENTS+=("rviz:=true")
fi

echo "Starting Quest pose bridge. Power on the Quest and keep its USB cable connected."
exec uv run --project "${PROJECT_SRC}" --no-sync \
  python "$(command -v ros2)" launch \
  xarm_ros2_teleop quest_pose.launch.py \
  "${LAUNCH_ARGUMENTS[@]}"
