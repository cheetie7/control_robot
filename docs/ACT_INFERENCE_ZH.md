# 在另一台电脑上运行 ACT BC（原生 SDK）

适用权重为 `b16_s30000_seed0/policy/policy_last.ckpt`。把更新后的
`control_robot/`、提供的 `act/` 源码和该权重文件复制到目标电脑。
权重自带模型配置、ResNet 参数和归一化统计，启动时无需训练数据或训练环境。
部署沿用 DiT Flow 和 Diffusion 的两进程方式，两个进程均无需 ROS 2。

## 1. 终端一启动模型服务

下面的路径按目标电脑填写。`MODEL_PYTHON` 指向之前运行 DiT Flow 或 Diffusion
模型服务的 Python，例如目标电脑的 `lerobot-awbc/.venv/bin/python`。
模型环境需要匹配版本的 PyTorch 和 TorchVision，以及 NumPy 2.x、
packaging、MessagePack 和 WebSockets。
本地 CPU 验证使用 PyTorch 2.11、TorchVision 0.26、NumPy 2.2。

```bash
export CONTROL_ROBOT=/absolute/path/control_robot
export ACT_ROOT=/absolute/path/act
export ACT_CHECKPOINT=/absolute/path/b16_s30000_seed0/policy/policy_last.ckpt
export MODEL_PYTHON=/absolute/path/model-env/bin/python

uv pip install --python "$MODEL_PYTHON" 'numpy>=2,<3' packaging \
  'msgpack>=1,<2' 'websockets>=15,<16'

"$MODEL_PYTHON" "$CONTROL_ROBOT/scripts/inference/serve_diffusion.py" \
  --policy-type act --act-root "$ACT_ROOT" \
  --checkpoint "$ACT_CHECKPOINT" --device cuda
```

无 CUDA 时可使用 `--device cpu` 做离线验证。启动后检查日志是否显示
`Listening on ws://127.0.0.1:8081`、`strategy_id=act_bc` 和
`execution_horizon=25`。模型进程等待客户端请求。

如果模型服务与机器人客户端分别运行在两台电脑上，在服务端用
`--host <模型机的局域网地址>` 监听该地址，并在机器人端
`inference_act.yaml` 的 `inference.host` 填写同一地址。两边端口须一致。
该 WebSocket 协议无认证，跨机器部署应使用受信任的局域网。

## 2. 终端二启动原生 SDK

在机器人电脑设置 `CONTROL_ROBOT`，核对继承的 `teleop_system.yaml` 中
机器人 IP、相机序列号、夹爪串口和初始位姿。

如果已经跑过 DiT Flow 或 Diffusion 的原生 SDK，直接复用
`scripts/inference/sdk/.venv`。首次配置参见 [SDK 说明的第 1–2 节](SDK_DIFFUSION_UBUNTU20_ZH.md)。

```bash
export CONTROL_ROBOT=/absolute/path/control_robot

uv run --project "$CONTROL_ROBOT/scripts/inference/sdk" --no-sync python \
  "$CONTROL_ROBOT/scripts/inference/run_diffusion_sdk.py" \
  --config "$CONTROL_ROBOT/xarm_ros2_teleop/config/inference_act.yaml"
```

默认只读反馈和图像。确认返回 `(25, 8)` 且动作校验通过后，结束 dry-run。
真机运行使用同一命令并加 `--execute`。

若 uv 尝试安装 `xensesdk`，检查 `--project` 是否指向
`$CONTROL_ROBOT/scripts/inference/sdk`。根项目的环境包含 Xense 触觉驱动，
原生 SDK 环境只安装本流程需要的硬件依赖。

## 3. 观测和动作

- 三路相机按 `top_camera / left_camera / right_camera` 排列。模型接收
  640×480 的 RGB，顶视相机沿用现有采集裁剪。模型端将图像除以 255 后做
  ImageNet 归一化。
- 使用权重中保存的 mean/std 归一化实测的 7 个关节角和夹爪位置。
  ACT 使用零潜变量，返回完整 25 步。反归一化后的动作是绝对关节弧度和
  夹爪位置，夹爪目标限幅到 [0,1]，由原有客户端校验关节限位和相邻步长。
- 25 步在 30 Hz 时间轴上占约 0.83 秒。客户端执行完整块后再取新观测。
  `prompt` 用于标记任务，ACT 网络只接收图像和机器人状态。
