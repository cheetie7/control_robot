# DiT Flow Matching 真机部署：执行完整 10 步

适用于 `dit-flow-policy` 保存的完整 `.pt`，本次验证文件为 `step_00020000.pt`。
继续使用 Ubuntu 20.04 原生 SDK，无需 ROS 或 Docker；硬件环境和相机/串口安装见
[SDK 说明](SDK_DIFFUSION_UBUNTU20_ZH.md)。本次仅增加模型适配和服务类型选择。

## 1. 模型环境

以下示例按真机电脑的目录填写，不修改 HOME。新的 .pt 是单个文件，不是旧模型目录。

```bash
export CONTROL_ROBOT=/home/gxy/xhr/control_robot
export DIT_FLOW_POLICY=/home/gxy/xhr/dit-flow-policy
export DIT_CHECKPOINT=/home/gxy/xhr/checkpoints/step_00020000.pt
```

建议复用之前已能运行 CUDA 的模型环境，只增加本地项目和通信依赖：

```bash
export MODEL_PYTHON=/home/gxy/xhr/lerobot-awbc/.venv/bin/python
uv pip install --python "$MODEL_PYTHON" -e "$DIT_FLOW_POLICY" \
  'msgpack>=1,<2' 'websockets>=15,<16'
```

新模型代码不需要导入 LeRobot；这里仅复用已有的 Python/PyTorch 环境。
若已经给新项目单独安装了可用的环境，将 MODEL_PYTHON 指向该环境的 Python 即可。
不要用旧 ROS 工程根目录的 uv sync 重建这个模型环境。

## 2. 模型服务（终端一）

```bash
"$MODEL_PYTHON" "$CONTROL_ROBOT/scripts/inference/serve_diffusion.py" \
  --policy-type dit-flow \
  --checkpoint "$DIT_CHECKPOINT" \
  --device cuda
```

等待启动日志中的 `Loaded` 和 `Listening on ws://127.0.0.1:8081`。
默认严格恢复 EMA 权重。checkpoint 自带网络配置、ResNet 权重和归一化统计；
保存的训练服务器路径不需要存在，也不需要重新下载 ImageNet 权重。
普通 FM checkpoint 不需要在线 Advantage、RoboMeter 或奖励模型。

握手应显示 `strategy_id=dit_flow_fm`、`execution_horizon=10`、
`num_inference_steps=8`、`weights=ema`、`sampler=euler`。
模型服务没有请求时会等待，不会自行控制硬件。

## 3. SDK 端（终端二）

```bash
export CONTROL_ROBOT=/home/gxy/xhr/control_robot
export SDK_PROJECT="$CONTROL_ROBOT/scripts/inference/sdk"

# 先做读取真实图像/状态的 dry-run；不发送硬件运动命令。
uv run --project "$SDK_PROJECT" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/run_diffusion_sdk.py"
```

确认日志返回 `(10, 8)`、动作校验通过后，结束 dry-run，再运行同一命令并添加
`--execute` 开启运动。初始位姿、夹爪激活和 Ctrl+C 停止行为与原 SDK 入口相同。
默认读取 `xarm_ros2_teleop/config/inference_diffusion.yaml`，继续继承现场硬件配置。
首轮只执行一个动作块时，在该 YAML 的 `inference` 下设置 `max_chunks: 1`。

## 4. 数据和时间轴

- SDK 图像 key 为 top_camera/left_camera/right_camera；适配器映射到模型的
  observation.images.* 名称，并按 checkpoint 的顺序送入 prepare_observation。
- SDK 保留训练采集时的顶视裁剪；模型预处理负责将 RGB 缩放到 240×320 并做
  ImageNet 归一化，state 按 checkpoint 的 MIN_MAX 统计归一化。不调用旧处理器。
- 调用 predict_action_chunk，返回已经反归一化的 [1,10,8]，只去掉 batch 维。
  不调用会截取前 8 步的 predict_action；不改 checkpoint 内的 n_action_steps=8。
- 完整执行 10 步绝对关节 rad/夹爪 [0,1]，按 30 Hz 时间轴持续约 0.333 秒，
  SDK 以约 100 Hz 插值下发。执行后才取新观测，不开启时间集成或异步预取。
- 8 次 Euler 是每个动作块的采样迭代次数，与 10 个机器人动作点不同。
- 动作只反归一化一次。Euler 输出没有范围约束，适配层将反归一化后的第 8 维
  夹爪目标限幅到 [0,1]，发生限幅时打印原始最小值/最大值。SDK 再乘 255 转成
  Robotiq 指令；关节目标保持原值，由现有 SDK 拒绝关节越界和大跳变。
- 服务打印整次请求、视觉编码、观测融合、Euler 和反归一化耗时；首次有预热开销。
  GPU 性能和任务成功率需要现场验证，合成图像的离线测试不能代表真机效果。

旧 LeRobot 模型继续使用默认的 `--policy-type diffusion`，其目录 checkpoint 和
加载行为不变。新服务入口按需导入两种模型，SDK 端不需要知道网络架构。

## 5. 测试

模型环境中安装 pytest，运行：

```bash
uv pip install --python "$MODEL_PYTHON" 'pytest>=8,<10'
cd "$CONTROL_ROBOT"
"$MODEL_PYTHON" -m pytest scripts/inference/test_dit_flow_service.py \
  scripts/inference/test_sdk_runtime.py -q
```

测试使用小型实际 FlowMatchingPolicy 和临时 checkpoint，不下载权重或连接硬件。
