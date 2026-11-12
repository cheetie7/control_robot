<div align="center">

# control_robot

**FACTR-to-xArm 7 teleoperation, multi-camera sensing, Xense tactile sensing, and LeRobot v3 data collection**

![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Environment](https://img.shields.io/badge/environment-uv-DE5FE9)
![Dataset](https://img.shields.io/badge/dataset-LeRobot_v3-FFD21E)

</div>

Ubuntu 20.04 原生 SDK 推理（无需 ROS/Docker）：[安装与启动](docs/SDK_DIFFUSION_UBUNTU20_ZH.md)。

新 DiT Flow checkpoint（EMA、8 次 Euler、执行完整 10 步）：[模型服务部署](docs/DIT_FLOW_DEPLOY_ZH.md)。

ROS 2 Humble 下的 AWBC Diffusion 推理：[配置与启动说明](docs/DIFFUSION_INFERENCE_ZH.md)。

三相机 ACT BC（`b16_s30000_seed0`）：[模型服务与原生 SDK 部署](docs/ACT_INFERENCE_ZH.md)。

`control_robot` uses a FACTR/GELLO leader arm to teleoperate an xArm 7, controls a
Robotiq gripper through a dedicated USB-RS485 adapter, collects RealSense D435,
Orbbec Femto Bolt, and Xense tactile data, and stores synchronized episodes as a
LeRobot v3 dataset.

The repository contains two ROS 2 packages:

- `factr_teleop`: the FACTR Dynamixel driver and upstream compatibility code.
- `xarm_ros2_teleop`: the main system, including teleoperation, actuator nodes,
  sensors, reset coordination, recording, monitoring, Rerun, and the Web Dashboard.

The complete system has one public launch entry point:

```bash
ros2 launch xarm_ros2_teleop teleop.launch.py \
  config:=config/teleop_system.yaml
```

> [!CAUTION]
> xArm initialization and episode reset use direct joint-space motion without
> collision planning. Clear the workspace, begin with conservative speed/current
> limits, and keep the xArm emergency stop and FACTR power switch accessible when
> testing a new installation or changed calibration.

## Contents

中文阅读入口：[文件地图（目录与逐文件职责）](docs/FILE_MAP_ZH.md)；[深入架构导读](docs/ONBOARDING.md)。

模型通信与数据边界：[真机与学习端接口](docs/ROBOT_LEARNING_CONTRACT.md)。

- [Quick Start](#quick-start)
- [Requirements](#1-requirements)
- [Repository Layout](#2-repository-layout)
- [Project Installation](#3-project-installation)
- [Hardware Setup](#4-hardware-setup)
- [System Configuration](#5-system-configuration)
- [Build](#6-build)
- [FACTR Calibration File](#7-factr-calibration-file)
- [Run the System](#8-run-the-system)
- [Teleoperation and Recording](#9-teleoperation-and-recording)
- [Dataset Output](#10-dataset-output)
- [ROS Interfaces](#11-main-ros-interfaces)
- [Troubleshooting](#13-troubleshooting)
- [Testing](#14-testing)

## Quick Start

After installing the project dependencies, configuring udev, building the workspace,
and placing the existing FACTR calibration file, run the following from the repository
root (the ROS workspace `src` directory):

```bash
# First run after a clean checkout or build-system change:
./scripts/record_data.sh --build

# Normal data-collection runs reuse the existing symlink build:
./scripts/record_data.sh
```

The wrapper defaults to three RGB cameras, no tactile/depth/infrared streams, 50
episodes, and the exact dataset directory `/home/gxy/datasets/name`. Frequently
changed values can be passed without editing YAML:

```bash
./scripts/record_data.sh \
  --dataset /home/gxy/datasets/block_insertion_001 \
  --episodes 100 \
  --instruction "insert the block into the socket"
```

Run `./scripts/record_data.sh --help` for timestamped paths, custom base configs and
validation-only mode. The defaults are also grouped at the top of the script for
direct editing. The equivalent manual launch sequence is retained below for debugging:

```bash
./scripts/diagnostics/check_factr_latency.sh --set

cd ..
source /opt/ros/humble/setup.zsh
source install/setup.zsh

uv run --project src --frozen \
  python "$(command -v ros2)" launch \
  xarm_ros2_teleop teleop.launch.py \
  config:=config/teleop_system.yaml
```

Open <http://127.0.0.1:8080> and wait for `warm up finished` before starting an
episode.

## 1. Requirements

### Software

| Component | Required version |
|---|---|
| Operating system | Ubuntu 22.04 |
| ROS | ROS 2 Humble |
| Python | System CPython 3.10 |
| Python environment | uv, locked by `uv.lock` |
| Dataset library | LeRobot 0.4.4 / v3 format |
| Web UI | FastAPI + Uvicorn, local access only |
| Visualization | Rerun SDK 0.26.2 |

Do not launch this project from Conda. The uv environment must be created from the
system Python 3.10 interpreter with system site packages enabled. This allows the
environment to import ROS 2 Humble modules such as `rclpy`, ROS messages, and
Pinocchio while keeping the non-ROS Python dependencies locked by uv.

### Hardware

| Hardware | Connection | Runtime identity |
|---|---|---|
| xArm 7 | Ethernet | `xarm.ip` |
| FACTR/GELLO | FTDI USB serial, Dynamixel at 1 Mbps | Persistent `/dev/serial/by-id/...` path |
| Robotiq 2F | FTDI USB-RS485, Modbus RTU | `/dev/robotiq_rs485`, device ID 9 by default |
| RealSense D435 | USB 3 | Unique serial number |
| Orbbec Femto Bolt | USB 3 | Unique serial number |
| Xense tactile sensor | USB | Unique serial number |

FACTR and Robotiq use different FTDI adapters and must not share or swap serial
devices. Distribute the cameras across USB 3 controllers when possible. A D435 that
falls back to USB 2 may use a reduced frame rate and cause recording synchronization
failures.

## 2. Repository Layout

```text
src/
├── pyproject.toml / uv.lock          # Locked Python 3.10 environment
├── scripts/                          # Setup, udev, FTDI latency, and SDK patch scripts
├── factr_teleop/                     # FACTR Dynamixel driver and compatibility package
├── xarm_ros2_teleop/
│   ├── config/teleop_system.yaml     # Single system configuration
│   ├── config/gello_grav_comp_demo.yaml
│   ├── launch/teleop.launch.py       # Single public launch file
│   └── xarm_ros2_teleop/             # Functional ROS nodes
└── docs/ONBOARDING.md                # Architecture and maintenance guide
```

See [`docs/ONBOARDING.md`](docs/ONBOARDING.md) for the internal architecture,
control logic, ROS data flow, and code modification guide.

Remote model inference is available through the same launch file. It is
motion-disabled by default and must pass an end-to-end dry-run before the
separate physical-motion flag is used. Motion-armed inference provides labeled,
30-second evaluation episodes with automatic reset and durable outcomes. See
[`docs/REMOTE_POLICY_INFERENCE.md`](docs/REMOTE_POLICY_INFERENCE.md).

Meta Quest controller poses can be exposed as floor-relative ROS 2
`PoseStamped` topics through the standalone USB bridge. See
[`docs/QUEST_POSE_BRIDGE.md`](docs/QUEST_POSE_BRIDGE.md).

The RL data console accepts that remote evaluator or an external policy,
records manually started episodes with driver-confirmed actions, FACTR
interventions and human rewards, and saves hardware failures as labeled crash
data. Learning remains in external programs. See
[`docs/ONLINE_RL_INTERVENTION.md`](docs/ONLINE_RL_INTERVENTION.md).

## 3. Project Installation

This guide assumes the repository is cloned as the `src/` directory of a ROS 2
workspace. Unless a command explicitly starts with `cd ..`, run it from the repository
root. No user-specific workspace path is required.

### 3.1 Install project system dependencies

Ubuntu 22.04, ROS 2 Humble, and system Python 3.10 are prerequisites. This repository
does not document ROS 2 installation itself. Install only the additional build tools,
ROS drivers, speech engine, and video tools required by the project:

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-diagnostic-updater \
  ros-humble-pinocchio \
  ros-humble-realsense2-camera \
  ros-humble-realsense2-description \
  ros-humble-orbbec-camera \
  ros-humble-orbbec-description \
  espeak-ng \
  ffmpeg
```

If Orbbec reports a missing `libdiagnostic_updater.so`, reinstall the ROS package,
source Humble again, and verify the dynamic link:

```bash
sudo apt install -y ros-humble-diagnostic-updater
source /opt/ros/humble/setup.zsh
ldd /opt/ros/humble/lib/liborbbec_camera.so | grep diagnostic_updater
```

### 3.2 Install uv and Python dependencies

Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

Create the virtual environment from the system CPython 3.10 interpreter and enable
access to system site packages:

```bash
uv venv --python /usr/bin/python3.10 --system-site-packages
uv sync --frozen
./scripts/setup/patch_pyrobotiqgripper.sh
```

`patch_pyrobotiqgripper.sh` adds a missing `Optional` import required by
`pyrobotiqgripper==3.2.8` on Python 3.10. Run the patch again after recreating
`.venv`. The xArm SDK, Xense SDK, pyRobotiqGripper, LeRobot, and Rerun dependencies
are managed by `pyproject.toml` and `uv.lock`; do not override them with ad-hoc
`pip install` commands.

#### Verify that uv uses the system Python

```bash
grep -E '^(home|version_info|include-system-site-packages)' .venv/pyvenv.cfg

uv run --frozen python -c '
import sys
print("executable:", sys.executable)
print("base_prefix:", sys.base_prefix)
print("venv_prefix:", sys.prefix)
print("version:", sys.version.split()[0])
'
```

Expected properties:

- `home = /usr/bin`
- `include-system-site-packages = true`
- Python version `3.10.x`
- `base_prefix` is `/usr`, while `sys.prefix` points to `.venv`

The repository's `scripts/bootstrap.sh` uses the same settings:

```bash
uv venv --python /usr/bin/python3 --system-site-packages
```

### 3.3 Install USB permission rules

Install the RealSense, Orbbec, and Xense rules:

```bash
./scripts/setup/install_sensor_udev.sh
```

FACTR and Robotiq serial access also requires membership in `dialout`:

```bash
sudo usermod -aG dialout "$USER"
```

Robotiq requires a stable device alias. Find the FTDI serial number first:

```bash
find /dev/serial/by-id -maxdepth 1 -type l -printf '%f -> %l\n'
udevadm info --query=property --name=/dev/ttyUSB0 | grep ID_SERIAL_SHORT
```

Replace the value below with the Robotiq adapter serial number, then install the
rule:

```bash
ROBOTIQ_ADAPTER_SERIAL="REPLACE_WITH_FTDI_SERIAL"

sudo tee /etc/udev/rules.d/99-robotiq-rs485.rules >/dev/null <<EOF
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", ATTRS{serial}=="${ROBOTIQ_ADAPTER_SERIAL}", SYMLINK+="robotiq_rs485", GROUP="dialout", MODE="0660"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=tty
sudo udevadm settle
```

Reconnect the devices and log in again so the group changes take effect. Verify the
result:

```bash
id -nG | tr ' ' '\n' | grep -E 'dialout|xense'
readlink -f /dev/robotiq_rs485
ls -l /dev/robotiq_rs485
```

## 4. Hardware Setup

### 4.1 xArm

Place the control computer on the same Ethernet subnet as the xArm and verify that
the configured address is reachable:

```bash
XARM_IP="192.168.1.XXX"
ping -c 3 "$XARM_IP"
```

Set the same address in `xarm_ros2_teleop/config/teleop_system.yaml`:

```yaml
xarm:
  ip: "192.168.1.XXX"
```

### 4.2 FACTR

FACTR requires both USB and external power. Locate its persistent serial path:

```bash
find /dev/serial/by-id -maxdepth 1 -type l -printf '%f -> %l\n'
```

Set the FACTR device in `xarm_ros2_teleop/config/gello_grav_comp_demo.yaml`:

```yaml
dynamixel:
  dynamixel_port: "/dev/serial/by-id/REPLACE_WITH_FACTR_DEVICE_NAME"
```

The FTDI latency timer may return to its default after a reboot or reconnect. Check
and set it before running the system:

```bash
./scripts/diagnostics/check_factr_latency.sh --set
```

### 4.3 Robotiq

Robotiq is controlled directly through the computer's USB-RS485 adapter, not through
the xArm tool port. Verify the alias and configure:

```yaml
robotiq:
  com_port: "/dev/robotiq_rs485"
  device_id: 9
```

### 4.4 Cameras

Inspect RealSense devices and USB modes:

```bash
rs-enumerate-devices | grep -E 'Name|Serial Number|Usb Type Descriptor'
```

Inspect the Orbbec driver and connected devices:

```bash
source /opt/ros/humble/setup.zsh
ros2 pkg prefix orbbec_camera
ros2 run orbbec_camera list_devices_node
```

Enter each serial number under `cameras.devices`. Camera `name`, `topic_prefix`, and
serial number values must be unique. A missing serial number is a hard failure; the
system never falls back to another camera.

Example configuration:

| Name | Model | Serial number | Public topic prefix |
|---|---|---|---|
| `left_camera` | D435 | `REPLACE_WITH_LEFT_D435_SERIAL` | `/camera/left` |
| `right_camera` | D435 | `REPLACE_WITH_RIGHT_D435_SERIAL` | `/camera/right` |
| `top_camera` | Femto Bolt | `REPLACE_WITH_FEMTO_BOLT_SERIAL` | `/camera/side` |

Camera output flags control generation inside the vendor driver. `false` means the
data is not produced, rather than produced and discarded later:

```yaml
color:
  enabled: true
depth:
  enabled: false
point_cloud:
  enabled: false
```

Point-cloud output requires color and depth to be enabled as well.

### 4.5 Xense tactile sensors

Scan Xense serial numbers through the locked uv environment:

```bash
uv run --frozen python -c \
  'from xensesdk import Sensor; print(Sensor.scanSerialNumber())'
```

Assign the two serial numbers to `left_tactile` and `right_tactile`. Tactile output
flags are real SDK selection flags: a disabled output is not calculated or generated.
The default recording configuration enables `rectify` and disables depth, force, and
marker outputs.

## 5. System Configuration

All runtime switches live in
[`xarm_ros2_teleop/config/teleop_system.yaml`](xarm_ros2_teleop/config/teleop_system.yaml):

| Section | Purpose |
|---|---|
| `teleop` | FACTR configuration, offset file, and FACTR-to-xArm signs |
| `xarm` | IP address, initial pose, speed, and acceleration limits |
| `robotiq` | RS485 port, device ID, speed, and gripping force |
| `reset` | Reset timeout, tolerances, FACTR mode-5 current, and gains |
| `cameras` | Camera serial numbers, public topics, and generated streams |
| `tactile` | Xense serial numbers and generated outputs |
| `warmup` | Unified readiness timeout, stability timing, and recording gate |
| `speech` | Local speech command and announcement phrases |
| `recording` | LeRobot path, episode count, instruction, synchronization, and video streams |
| `dashboard` | Web/Rerun addresses, ports, and ROS service timeout |

Review at least the following values before recording:

```yaml
recording:
  root: "~/datasets/control_robot"
  dataset_name: "lerobot_dataset"
  append_timestamp: true
  fps: 30
  instruction: "teleoperate the robot to complete the task"
  num_episodes: 1
```

`instruction` is stored in LeRobot's standard `task` field for every frame. Topics in
`recording.streams` must match outputs that the enabled hardware actually generates.
The stream selected by `anchor.key` (`top_camera` in the checked-in RGB profile) is
the 30 FPS master clock. Every other image and robot state is selected from a bounded
history by the nearest ROS timestamp. Camera skew is stored in
`observation.sync_skew_s` for diagnosis but does not drop a row; normal camera
synchronization is provided by the hardware. Actions retain a short numeric history
so the label is always the newest command at or before the image timestamp. A
bounded background queue keeps dataset I/O out of sensor callbacks.

## 6. Build

Always start colcon through the locked uv interpreter. This prevents ROS 2 build tools
from silently using a different Python or setuptools installation:

```bash
# Enter the ROS workspace root from the repository root.
cd ..
source /opt/ros/humble/setup.zsh

uv run \
  --project src \
  --frozen \
  python "$(command -v colcon)" build \
  --symlink-install \
  --packages-select factr_teleop xarm_ros2_teleop

source install/setup.zsh
```

For initial provisioning, `src/scripts/bootstrap.sh` combines uv sync, rosdep, and
colcon. The explicit commands above are easier to debug step by step. The lock file
currently pins `setuptools==79.0.1`; do not override it with an incompatible version
that reintroduces `option --editable not recognized` under Humble.

## 7. FACTR Calibration File

The FACTR associated with this system is already calibrated. Normal startup does not
require running `factr_calibrate`. Runtime loads:

```text
~/.config/control_robot/offset.yaml
```

Verify that the file exists:

```bash
test -f "$HOME/.config/control_robot/offset.yaml" \
  && echo "FACTR calibration ready" \
  || echo "FACTR calibration missing"
```

When moving the system to another computer, copy the existing `offset.yaml` to the
same user-relative location and make sure the current user owns it. Do not recalibrate
after a normal reboot, USB reconnect, dependency sync, or rebuild.

<details>
<summary><strong>When is recalibration required?</strong></summary>

Recalibrate only after replacing a FACTR servo, changing the mechanical assembly,
changing the serial mapping, or modifying `joint_signs` or `xarm_to_factr_signs`.
Stop the main launch before calibration.

```bash
# Enter the ROS workspace root from the repository root.
cd ..
source /opt/ros/humble/setup.zsh
source install/setup.zsh

uv run \
  --project src \
  --frozen \
  python "$(command -v ros2)" run \
  xarm_ros2_teleop factr_calibrate \
  --system-config src/xarm_ros2_teleop/config/teleop_system.yaml
```

The calibration command:

1. Clears a recoverable xArm error and moves xArm to `xarm.initial_joint_positions`.
2. Disables FACTR torque.
3. Prompts the operator to place FACTR at the matching physical pose with its gripper open.
4. Reads all eight servos and writes `~/.config/control_robot/offset.yaml` only after
   the operator enters `yes`.

The main launch never re-estimates offsets. Warm-up uses the stored calibration to
return FACTR to its corresponding servo pose.

</details>

## 8. Run the System

### 8.1 Test all five sensors without actuators

Use the dedicated sensor profile before the full robot launch. It starts the two D435
cameras, the Femto Bolt and both Xense sensors, plus System Monitor, Dashboard and
Rerun. It cannot construct FACTR, xArm, Robotiq, reset, recording, inference or RL
processes.

```bash
cd ..
source /opt/ros/humble/setup.zsh
source install/setup.zsh

uv run \
  --project src \
  --frozen \
  python "$(command -v ros2)" launch \
  xarm_ros2_teleop sensors.launch.py
```

Open <http://127.0.0.1:8080>. All five sensor cards should become `OK`, and Rerun
should display five image views. Stop the test with `Ctrl-C`. Hardware serial numbers,
streams and staggered delays are defined in `config/sensors_all.yaml`.

### 8.2 Start the complete robot system

Run the following after each reboot, starting from the repository root:

```bash
./scripts/diagnostics/check_factr_latency.sh --set

cd ..
source /opt/ros/humble/setup.zsh
source install/setup.zsh

uv run \
  --project src \
  --frozen \
  python "$(command -v ros2)" launch \
  xarm_ros2_teleop teleop.launch.py \
  config:=config/teleop_system.yaml
```

Startup sequence:

1. Validate the complete YAML, FACTR serial path, and offset file before opening hardware.
2. Connect FACTR, xArm, Robotiq, reset, monitor, and recorder nodes.
3. Start cameras and tactile sensors independently at their configured absolute delays.
4. Announce warm-up and wait for all required hardware to remain healthy.
5. Return xArm and FACTR to their mapped initial poses and open Robotiq.
6. Wait for the configured settling delay, announce `warm up finished`, and unlock recording.

The Dashboard opens automatically by default:

- Dashboard: <http://127.0.0.1:8080>
- Rerun Web Viewer: embedded in the Dashboard, using port `9090`

The same local Dashboard is used for every launch mode. In ordinary teleoperation it
shows recording controls; inference dry-run shows rollout state and a stop control;
motion-armed inference shows start, success, failure, discard, countdown, and result
controls. With `rl:=true` it shows manual episode start, classified outcome saves,
policy ownership, automatic single-button human takeover, rewards and pause controls.
The human device must already be online before an episode starts. Success, failure
and discard automatically use the shared reset workflow; no separate reset/abort
button is needed. RL has no
automatic episode duration limit. Rerun continues to show configured cameras,
robot state, policy candidates, and executed RL actions. The header always displays
`DRY RUN` or `MOTION ARMED`; this is informational only. Physical motion can only be
armed with the `inference_motion` or `rl_motion` launch argument, never from the Web UI.

Dashboard or Rerun failure does not stop an active recording. FACTR, xArm, Robotiq,
reset and recorder remain critical processes. Sensor drivers recover locally; their
`required` flag controls whether missing or stale output is reported as an error or a
degraded warning.

## 9. Teleoperation and Recording

FACTR publishes leader displacement relative to the calibrated origin. The xArm target
is:

```text
xarm_target = xarm.initial_joint_positions + factr_delta
```

Robot state and action are recorded separately:

- `observation.state`: seven measured xArm joints plus measured Robotiq position,
  shape `(8,)`.
- `action`: seven commanded xArm joints plus commanded Robotiq position, shape `(8,)`.

Control recording from the Dashboard buttons or terminal shortcuts:

| Operation | Shortcut | Behavior |
|---|---|---|
| Enter staging | `G` | Resume FACTR following without recording any dataset frames |
| Start episode | `S` | Start recording immediately from `IDLE` or from the current staged pose |
| Save episode | `→` | Stop capture and run video writing and robot reset concurrently |
| Discard / reset | `←` | Discard capture, or abandon staging, and reset the robot |
| Finish recording | `Esc` | Finish the current work and finalize the dataset |

After save or discard, the next episode remains locked until both LeRobot writing and
the xArm/FACTR/Robotiq reset have completed. xArm then holds mode 0 and Robotiq ignores
leader commands while FACTR state publication continues. From `IDLE`, Start restores
following and records immediately; Staging restores following without recording so the
operator can choose a comfortable initial pose. Either activation rebases the FACTR
origin at its current pose, and xArm ignores queued nonzero deltas until the zero-origin
command arrives. Start from `STAGING` keeps that existing origin and begins capture in
place without reset, a second rebase, or a motion jump. Discard from `STAGING` performs
only the normal coordinated reset and creates no empty episode. Wait for `IDLE` or
`COMPLETE` and the episode announcement before continuing. Do not press `Ctrl+C` while the UI
shows `SAVING` or `RESETTING`; interruption can leave only `meta/info.json` without
committed Parquet or MP4 files.

## 10. Dataset Output

With the default configuration, a run creates:

```text
~/datasets/control_robot/lerobot_dataset_YYYYmmdd_HHMMSS/
├── meta/       # Dataset, episode, and task metadata
├── data/       # State, action, and timestamp Parquet files
└── videos/     # MP4 files for observation.images.* features
```

The output path is controlled by:

```yaml
recording:
  root: "~/datasets/control_robot"
  dataset_name: "lerobot_dataset"
  append_timestamp: true
```

RGB camera and tactile `rectify` streams are stored as MP4. When depth recording is
enabled, the current implementation maps metric depth into a fixed-range 8-bit RGB
preview MP4. It is intended for visualization and is not a reversible raw-depth
encoding. Rerun still displays the original ROS depth in metres.

Quickly inspect completed datasets with:

```bash
DATASET_ROOT="$HOME/datasets/control_robot"
find "$DATASET_ROOT" -maxdepth 4 -type f | sort
du -sh "$DATASET_ROOT"/*
```

## 11. Main ROS Interfaces

| Interface | Type | Meaning |
|---|---|---|
| `/xarm/arm_pos` | `sensor_msgs/JointState` | FACTR-generated xArm leader displacement |
| `/xarm/absolute_target` | `sensor_msgs/JointState` | ACT/RL absolute xArm joint target in radians |
| `/xarm/gripper_pos` | `sensor_msgs/JointState` | FACTR gripper target |
| `/xarm/arm_joint_state` | `sensor_msgs/JointState` | Measured xArm state |
| `/xarm/commanded_joint_state` | `sensor_msgs/JointState` | Commanded xArm action |
| `/factr/arm_joint_state` | `sensor_msgs/JointState` | Measured FACTR state |
| `/robotiq/position_bits` | `std_msgs/Int32` | Measured gripper position, 0 open and 255 closed |
| `/teleop/reset` | `FollowJointTrajectory` action | Seven-joint absolute reset target |
| `/teleop/reset/status` | `std_msgs/String` | Current reset phase |
| `/system/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Hardware and software health |
| `/recording/status` | `std_msgs/String` | Recorder lifecycle state |

Cameras publish `/<topic_prefix>/color`, `depth`, and `point_cloud` when enabled.
Tactile sensors publish `/<topic_prefix>/<enabled_output>`. See
`docs/ONBOARDING.md` for the complete interface table.

## 12. Shutdown and Restart

- Use `Esc` to finish a recording session and wait for dataset finalization.
- Use one `Ctrl+C` to stop the complete launch after finalization.
- `KeyboardInterrupt`, exit code `-2`, and camera `Stop Sensor`/`Close Sensor` messages
  during SIGINT are usually shutdown side effects rather than runtime hardware errors.
- Do not press `Ctrl+C` repeatedly. A second interrupt can terminate video finalization.

## 13. Troubleshooting

### FACTR serial device is missing

Confirm that FACTR has power and USB, inspect `/dev/serial/by-id`, and update
`gello_grav_comp_demo.yaml`. The launch checks this path before starting xArm or
Robotiq.

### FACTR cannot finish warm-up or oscillates

Stop the launch and check FACTR power, serial identity, FTDI latency, and mechanical
obstruction. This FACTR is already calibrated, so do not treat a normal startup fault
as a reason to recalibrate. Recalibrate only after a hardware, assembly, or mapping
change. Do not hide hardware faults by increasing current. Reset mode-5 current and
P/I/D values are under `reset.factr_*`; normal teleoperation remains protected by the
driver's `±900 mA` limit.

### `/dev/robotiq_rs485` is missing

The alias is persistent and should not be recreated after every reboot. Check the FTDI
serial in the udev rule, confirm that the user belongs to `dialout`, reload the rules,
and reconnect the adapter.

### Orbbec cannot load `libdiagnostic_updater.so`

Install `ros-humble-diagnostic-updater`, source Humble again, and verify with `ldd`.
This is a system dynamic-library issue and must not be fixed with uv or pip.

### A sensor remains `STARTING` or becomes `STALE`

Each sensor starts independently at its absolute `startup_delay_sec`; one failed device
does not block later devices. Check its serial number, udev rules, USB 3 bandwidth and
power. Camera and tactile drivers retry transient process failures locally. System
Monitor decides health from every configured output topic, and teleop warm-up fails at
the global `hardware_timeout_sec` if a selected sensor never becomes healthy.

### Dashboard reports `ROS service timed out` or `unavailable`

Check whether the recorder still advertises its interfaces:

```bash
ros2 service list | grep '^/recording/'
ros2 topic echo --once /recording/status
```

The Dashboard waits for the configured `dashboard.service_timeout_sec`. A save command
may already have been accepted, so do not click repeatedly. Use the
`SAVING`/`RESETTING`/`STAGING`/`IDLE`/`COMPLETE` state and recorder logs as the source
of truth.

### No MP4 appears after save

An episode is committed only after both writing and reset complete. Interrupting during
`SAVING` or `RESETTING` may leave an incomplete metadata directory. Record again and
wait for the episode-finished announcement.

### Speech is unavailable

Verify `espeak-ng` and `speech.enabled`:

```bash
command -v espeak-ng
espeak-ng "warm up"
```

Speech failure does not stop teleoperation or recording.

### colcon reports `option --editable not recognized`

Run colcon through the locked uv environment and keep the repository's setuptools
version:

```bash
uv sync --frozen
uv run --frozen python -c 'import setuptools; print(setuptools.__version__)'
```

## 14. Testing

Run code-level tests from the repository root:

```bash
source /opt/ros/humble/setup.zsh
uv run --frozen pytest -q
uv run --frozen ruff check \
  xarm_ros2_teleop \
  factr_teleop/factr_teleop/dynamixel/driver.py
```

Run colcon tests from the ROS workspace root:

```bash
cd ..
source /opt/ros/humble/setup.zsh
source install/setup.zsh

uv run --project src --frozen python "$(command -v colcon)" test \
  --packages-select factr_teleop xarm_ros2_teleop
uv run --project src --frozen python "$(command -v colcon)" test-result --verbose
```

Software tests do not replace hardware acceptance. Increase risk gradually: FACTR
single-joint low-current testing, seven-joint reset, Robotiq, one sensor, multiple
sensors, one episode, and finally multiple episodes.

## 15. Developer Documentation

- [Architecture, control logic, and modification guide](docs/ONBOARDING.md)
- [Legacy code overview entry point](CODEBASE_OVERVIEW_ZH.md)
- [FACTR upstream documentation](factr_teleop/README.md)
- [Repository code-style requirements](AGENTS.md)
## VR TCP 遥操作

VR 与 FACTR 共用录制/复位，VR 默认不启动执行器：

```bash
bash scripts/record_data.sh --source vr --dry-run
bash scripts/record_data.sh --source vr --rviz
```

首次需要更新支持握持/扳机的 Quest APK。配置实际工作空间和 TCP 工具偏移后才可添加
`--motion`。新 VR action 为 TCP xyz+xyzw+gripper，不兼容现有关节 action ACT。
VR 也可使用共用 RL 操作台：`bash scripts/record_data.sh --source vr --rl --dry-run`。
真实运行还需配置工作空间、显式 `--motion`，并提供符合 TCP 契约的策略服务（8081）；
T 在策略与 VR 之间切换，Y/N 给奖励并保存复位，不接入旧的关节 ACT。
完整命令、公式和故障处理见 [VR 中文使用说明](docs/VR_TELEOP_ZH.md)。
