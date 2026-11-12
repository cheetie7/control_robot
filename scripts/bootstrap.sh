#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SRC="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_SRC/.." && pwd)"

# ROS Humble's setup.sh reads AMENT_TRACE_SETUP_FILES without a default value,
# which aborts under this script's `set -u`; relax nounset around the source.
set +u
source /opt/ros/humble/setup.bash
set -u

cd "$PROJECT_SRC"
if [[ ! -d .venv ]]; then
  uv venv --python /usr/bin/python3 --system-site-packages
fi
uv sync --frozen

# rosdep needs one-time `sudo rosdep init && rosdep update`; when its database
# is missing, keep building instead of aborting — dependencies come from uv
# and the system image anyway.
if [ -f /etc/ros/rosdep/sources.list.d/index.yaml ]; then
  rosdep install --from-paths "$PROJECT_SRC" --ignore-src --rosdistro humble -r -y
else
  echo "Warning: rosdep 未初始化，跳过 apt 依赖检查（运行 sudo rosdep init && rosdep update 可启用）。"
fi

cd "$WORKSPACE_ROOT"
uv run --project "$PROJECT_SRC" --frozen \
  python /usr/bin/colcon build --symlink-install \
  --base-paths "$PROJECT_SRC/factr_teleop" "$PROJECT_SRC/xarm_ros2_teleop"

echo "Build complete. Run: source $WORKSPACE_ROOT/install/setup.bash"
echo "Before FACTR teleoperation, run: $PROJECT_SRC/scripts/diagnostics/check_factr_latency.sh"
echo "Sensor hosts also need: $PROJECT_SRC/scripts/setup/install_sensor_udev.sh"
