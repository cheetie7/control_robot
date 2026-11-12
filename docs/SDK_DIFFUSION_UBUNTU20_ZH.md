# Ubuntu 20.04 原生 SDK + AWBC Diffusion

保留系统 Python 3.8 和 Noetic；本入口不导入 ROS，不使用 Docker，不需要 colcon。
两套 uv 环境：硬件 SDK 用独立 Python 3.10，模型用独立 Python 3.12。
相机/夹爪后台采集不等于异步模型推理：模型请求始终串行，执行完整个动作块才取新观测。

## 1. 路径和基础工具

把修改后的 control_robot、lerobot-awbc 源码和完整 002000 放到电脑，保留各自 uv.lock，
不要复制其他电脑的 .venv。以下路径只是示例，每个终端按真实位置填写：

```bash
export CONTROL_ROBOT="$HOME/workspaces/control_robot"
export LEROBOT_AWBC="$HOME/workspaces/lerobot-awbc"
export DIFFUSION_CHECKPOINT="$HOME/checkpoints/002000"

sudo apt update
sudo apt install -y curl ca-certificates libusb-1.0-0 libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

需要 7897 代理时，在执行下载的终端设置 `http_proxy` 和 `https_proxy` 为
`http://127.0.0.1:7897`。不要再运行仓库根目录的 `uv sync` 或安装 `ros-humble-*`。

## 2. 硬件 SDK 环境

```bash
export SDK_PROJECT="$CONTROL_ROBOT/scripts/inference/sdk"
uv python install 3.10
uv sync --project "$SDK_PROJECT" --python 3.10 --frozen
uv run --project "$SDK_PROJECT" --no-sync python "$SDK_PROJECT/patch_gripper.py"

uv run --project "$SDK_PROJECT" --no-sync python -c \
  'import pyrealsense2, pyorbbecsdk, pyrobotiqgripper; from xarm.wrapper import XArmAPI; print("SDK imports OK")'
```

uv 下载自己的 Python，不更改 `/usr/bin/python3`，也不依赖 apt 提供 python3.10。
该环境有独立锁文件；已锁定 pyorbbecsdk2 2.1.2（导入名 pyorbbecsdk）和
pyrealsense2 2.59.1.11053。SDK、相机固件及机器上的动态库是否匹配仍需现场检查。

安装相机厂商的 USB 权限规则（已有相同规则可跳过），只在宿主机执行一次：

```bash
curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
  -o /tmp/99-realsense-libusb.rules
curl -fsSL https://raw.githubusercontent.com/orbbec/OrbbecSDK_v2/main/scripts/env_setup/99-obsensor-libusb.rules \
  -o /tmp/99-obsensor-libusb.rules
sudo install -m 0644 /tmp/99-realsense-libusb.rules /etc/udev/rules.d/
sudo install -m 0644 /tmp/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo usermod -aG dialout,video "$USER"
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新登录并插拔设备。夹爪应有 `/dev/robotiq_rs485`；若没有，按真实 FTDI 序列号
安装别名规则。参考项目 `setup_robotiq_serial.sh` 写死的是 `DAAEAB6P`，
必须先确认它就是这台机器的适配器，不能直接照用别人的序列号。

```bash
ls -l /dev/serial/by-id/
ls -l /dev/robotiq_rs485
```

## 3. 模型环境

在没有激活 Noetic/Conda 的新终端，重新设置第 1 节三个路径变量：

```bash
uv python install 3.12
uv sync --project "$LEROBOT_AWBC" --python 3.12 --frozen --extra diffusion
uv pip install --python "$LEROBOT_AWBC/.venv/bin/python" \
  'msgpack>=1,<2' 'websockets>=15,<16'

nvidia-smi
uv run --project "$LEROBOT_AWBC" --no-sync python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'
```

当前训练仓库的 Linux 锁文件选择 CUDA 12.8 版 PyTorch。容器和 ROS 都不参与；
驱动兼容性由上面的实际 CUDA 检查确认，失败时提供 nvidia-smi 输出再选合适的轮子，
不要为了安装 Python 环境盲目替换现有驱动。

## 4. 启动和停止

先启动模型服务：

```bash
uv run --project "$LEROBOT_AWBC" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/serve_diffusion.py" \
  --checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
```

等 `Listening` 后，在 SDK 终端运行（重新设置 CONTROL_ROBOT 和 SDK_PROJECT）：

```bash
# 默认 dry-run：真实图像和实测 state，但不使能、不移动、不激活或停止硬件。
uv run --project "$SDK_PROJECT" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/run_diffusion_sdk.py"

# 真机运行：完成相机和握手检查后，使能并移动到 YAML 初始位姿。
uv run --project "$SDK_PROJECT" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/run_diffusion_sdk.py" --execute
```

夹爪默认要求已经激活；冷启动需要时，显式加 `--activate-gripper`（同时必须有
`--execute`）。激活会运行固件的完整开合行程，不是普通的位置命令。
`--skip-initial-move` 可跳过初始关节运动，但不会取消动作跳变校验。

启动硬件端时加 `--record-video` 可连续录制三路 RGB，包括模型等待和动作执行期间。
先在 dry-run 验证视频，再按需加 `--execute`。

```bash
uv run --project "$SDK_PROJECT" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/run_diffusion_sdk.py" \
  --record-video --video-dir "$HOME/datasets/diffusion_eval/videos"
```

每次运行在 `--video-dir` 下创建一个时间戳目录，包含 `top_camera.mp4`、
`left_camera.mp4` 和 `right_camera.mp4`。默认父目录为 `~/datasets/diffusion_eval/videos`。
录制从硬件初始化完成后开始，使用与模型观测一致的画面，按 30 FPS 写入。
正常结束、Ctrl+C 或推理异常时关闭视频。相机画面过期或编码来不及时，录制线程
按组三路跳过，并在退出日志里统计跳过的帧数。跳帧时，固定帧率视频会比实际运行时间短。

默认读取 `xarm_ros2_teleop/config/inference_diffusion.yaml`。机器人 IP、三路相机序列号、
顶视原始 1280×720 和居中裁剪到 640×480、夹爪串口、驱动限速均继承采集配置。
需要自定义时通过 `--config /absolute/path/profile.yaml` 指定，勿套用参考 ACT 的初始位姿。

每轮获取当前图像和实测 state，生成 10 步动作，按 30 Hz 时间轴占用约 0.333 秒，
以约 100 Hz 线性插值下发。xArm 使用与采集代码一致的 mode 6 在线位置规划，
不是参考项目 mode 1 的直接 servo_j 控制。夹爪串口按 20 Hz 独立更新，
反馈为 gPO/255，命令为 int(g×255)，不使用参考项目的 0.8 标定。
等待推理期间机械臂保留最后的目标；超时、非法动作或硬件反馈异常会退出。
动作对推理前后两次实测状态都做跳变检查，防止长推理期间机器人位姿变化。

Ctrl+C、正常结束或异常退出时，执行模式向 xArm 发 state=4 停止并停止夹爪后断开；
dry-run 只断开连接。不会执行自动复位。最大时长和块数由 inference 配置限制。
本入口不启动 ROS Dashboard，也不实现 evaluation 的录制、标签和复位服务。

## 5. 离线测试

```bash
cd "$CONTROL_ROBOT"
uv run --project "$SDK_PROJECT" --no-sync python -m pytest \
  scripts/inference/test_sdk_runtime.py -q
```

测试模拟硬件，覆盖 dry-run 零命令、绝对动作与夹爪映射、停止和资源释放、图像裁剪、
过期图像拒绝、10 步时长及完整循环的串行请求顺序。现场硬件尚未验证。
