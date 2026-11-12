# 本机 AWBC Diffusion 同步真机推理

本页的真机启动部分适用于 ROS 2 Humble。Ubuntu 20.04 + Noetic 使用
[独立 SDK 安装与启动说明](SDK_DIFFUSION_UBUNTU20_ZH.md)，无需构建 ROS 包。

适用于 `002000`：三路 RGB、实测关节/夹爪 state [8]、`n_obs_steps=1`、
U-Net、`horizon=16`、`n_action_steps=10`。模型服务和 ROS 在同一台电脑的两个
Python 环境运行，通过 `127.0.0.1:8081` 通信。

## 1. 启动模型服务（LeRobot Python ≥3.12 环境）

将本仓库、`lerobot-awbc` 和完整 checkpoint 目录放到真机电脑。模型环境使用
训练时的 `lerobot-awbc` 代码，不能替换成 ROS 环境里的 `lerobot==0.4.4`。
checkpoint 目录保留权重、config、两个 processor JSON 和两个 processor safetensors。

下面的路径按真机电脑实际位置填写，三个变量均为绝对路径：

```bash
export CONTROL_ROBOT=/absolute/path/control_robot
export LEROBOT_AWBC=/absolute/path/lerobot-awbc
export DIFFUSION_CHECKPOINT=/absolute/path/002000

# 在已经能加载 checkpoint 的模型环境中补装通信依赖，只需一次。
uv pip install --python "$LEROBOT_AWBC/.venv/bin/python" \
  'msgpack>=1,<2' 'websockets>=15,<16'

uv run --project "$LEROBOT_AWBC" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/serve_diffusion.py" \
  --checkpoint "$DIFFUSION_CHECKPOINT" --device cuda
```

默认只监听本机 `127.0.0.1:8081`。看到 `Listening` 后再启动 ROS。
服务使用 `strict=True` 加载权重，恢复 checkpoint 保存的预处理/反归一化统计，
保持原有 DDPM 100 步采样。每次记录一块动作的总处理耗时，包括 CPU 输出传输。
`--device cpu` 可用于无 CUDA 电脑的离线验证；默认部署使用 CUDA。

## 2. 构建并启动 ROS（原有 Python 3.10 环境）

在原有 ROS 2 Humble/colcon 工作区中使用修改后的本仓库源码，然后重新构建：

```bash
source /opt/ros/humble/setup.zsh
# 在 colcon 工作区根目录执行。
colcon build --symlink-install --packages-select xarm_ros2_teleop
source install/setup.zsh
```

另开终端，在原有 ROS 环境设置实际源码路径，首先启动无硬件命令的 dry-run：

```bash
export CONTROL_ROBOT=/absolute/path/control_robot
source /opt/ros/humble/setup.zsh
source /absolute/path/colcon_workspace/install/setup.zsh

uv run --project "$CONTROL_ROBOT" --frozen \
  python "$(command -v ros2)" launch xarm_ros2_teleop teleop.launch.py \
  config:="$CONTROL_ROBOT/xarm_ros2_teleop/config/inference_diffusion.yaml" \
  inference:=true inference_motion:=false
```

YAML 用 `base_config` 继承同目录的 `teleop_system.yaml`，包括机器人 IP、相机序列号、
图像裁剪、初始位姿、驱动限速与动作校验。只覆盖模型端口、相机顺序、任务和模式。
干运行只启动观测链路；缺少机器人反馈时，现有节点使用初始关节位姿和打开的夹爪，
因此它验证通信与动作格式，不能验证模型在真实当前位姿上的表现。

确认握手、三路图像和返回动作正常后，用同一条 launch 命令将
`inference_motion:=false` 改为 `inference_motion:=true` 启动真机测评。
打开本机 Dashboard（8080 端口），`S` 开始，`Y/N` 保存成功/失败并复位，
左箭头丢弃并复位，`Esc` 结束。结果写入 `~/datasets/diffusion_eval`。
模型服务可以跨 episode 保持运行。现有真机启动流程会先移动到配置的初始位姿。

## 3. 数据与执行关键点

- `state` 是实测 7 个绝对关节角（rad）和归一化夹爪位置；不是上一块动作。
- 三视角顺序为 `top_camera / left_camera / right_camera`。ROS 端复用采集时的裁剪，
  发送 RGB uint8 HWC；服务转成 CHW float [0,1] 后调用保存的 MEAN_STD 处理器，
  网络内部再缩放到 240×320。模型服务不重复执行采集侧的裁剪。
- state 使用保存的 MIN_MAX 统计；动作使用保存的逆变换。返回绝对关节 rad 和
  夹爪 [0,1]，其中 0 开、1 闭。不要再次对夹爪取负或乘 255，ROS 驱动已负责映射。
- `predict_action_chunk()` 已将 16 步预测截为 10 步；服务只去掉 batch 维，
  从 [1,10,8] 返回 [10,8]。每次请求使用新观测，无 `select_action()` 缓存。
- 10 步按训练数据的 30 Hz 时间轴占用约 0.333 秒；ROS 以 100 Hz 插值执行。
  执行完才发下一次请求，等待推理时保持上一个目标，没有后台预取下一动作块。
- prompt 用于任务标记；这个 Diffusion 网络没有语言输入，更换 prompt 不会切换技能。
- AWBC 权重仅用于训练，不需要在线 advantage、奖励模型或 Robometer。

关键代码：`inference/diffusion_policy.py` 负责模型适配，
`scripts/inference/serve_diffusion.py` 负责启动，`inference/websocket_policy.py`
复用既有 MessagePack 协议。机器人状态机、动作校验和驱动保持原逻辑。

## 4. 验证

在模型环境运行轻量集成测试（不需要 ROS、机器人或 checkpoint）：

```bash
cd "$CONTROL_ROBOT"
uv run --project "$LEROBOT_AWBC" --no-sync python -m pytest \
  scripts/inference/test_diffusion_service.py -q
```

测试覆盖真实 LeRobot 归一化/反归一化、输入维度、相机尺寸、下一次请求使用新观测、
配置继承，以及现有客户端的真实本机 WebSocket 通信。

2026-10-29 本地验证：4 项测试通过；`002000` 严格加载成功，通过现有客户端完成
一次真实权重、DDPM 100 步、三路合成灰色图像的端到端请求，输出 float32 [10,8]、
数值有限且夹爪在 [0,1]。该次 MPS 请求约 6.6 秒，仅说明本机测试耗时；
CUDA 性能和真机任务效果尚未验证。没有连接或驱动机器人。
