# 真机 RL 操作台与数据契约

`control_robot` 负责执行、人工接管、复位和记录。外部程序决定 replay、训练算法和权重更新方式。

版本化通信、动作时间轴和数据字段见[两仓接口说明](ROBOT_LEARNING_CONTRACT.md)。公共录制底座仍由遥操、测评和 RL 共用。

本文后续命令和关节动作字段描述的是 FACTR 路线。VR 是并列的 TCP action 路线，
同样支持 RL 操作台；其启动命令、严格 TCP 握手和握持接管规则见
[VR TCP 遥操与 RL](VR_TELEOP_ZH.md#vr--tcp-策略的-rl-模式)。VR 不支持旧关节 ACT。

## 启动

先启动提供绝对关节动作的模型服务。当前配置连接 `127.0.0.1:8081`。

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
colcon build --symlink-install --packages-select xarm_ros2_teleop
source install/setup.zsh

uv run --project src --frozen \
  python "$(command -v ros2)" launch xarm_ros2_teleop teleop.launch.py \
  config:=/home/gxy/workspace/control_robot/src/xarm_ros2_teleop/config/teleop_system.yaml \
  rl:=true rl_motion:=true inference:=true
```

FACTR 随 RL 系统启动即持续运行、重力补偿和发布候选输入，不需要在每条中另行启用。默认要求连接正常、手臂/夹爪数据连续新鲜 0.3 秒后才允许开始录制；这要求通信稳定，不要求人手静止。网页显示“人工设备”是否就绪。

网页为 `http://127.0.0.1:8080`。输入任务指令和保存父目录，应用设置并等待初始化，再点击“开始本条”。默认父目录 `/home/gxy/datasets/rl`，建议每个任务使用自己的子目录。任务指令同步给模型请求。

`rl_motion:=false` 启动传感器与界面，但不启动执行器和真实轨迹采集器，开始本条按钮不可用。网页不能更改这个启动设置。

## 操作语义

| 操作 | 快捷键 | 行为 |
|---|---|---|
| 开始本条 | S | 先记录，再从新观测启动策略 |
| 切换控制权 | T | ACT → 人工：自动停稳、捕获两端当前起点并接入；人工 → ACT：从新观测恢复策略。本条持续录制 |
| 成功 | Y | 保存 success，默认奖励 +1，然后复位 |
| 失败 | N | 保存 failure，默认奖励 0，然后复位 |
| 暂停策略 | Esc | 保持当前位姿，本条继续开放 |
| 丢弃本条 | 左箭头 | 删除未提交数据，然后复位 |
| 结束会话 | Q | 未结束的当前条按失败保存，复位后关闭全部写入器 |

RL 没有时间和 episode 数上限，也不受 `inference.max_chunks` 或 `inference.max_runtime_sec` 限制。单次网络请求的 `inference_timeout_sec` 仍用于识别故障。策略推理失败只暂停策略，保留当前条，操作者可接管或标记失败。人工设备断连/输入过期同样暂停人工控制并保留数据，恢复后按 T 自动重新接入，之后再按 T 交回策略。软件暂停后同样从 T 接入人工；不需要第二个“确认接入”按钮。

面板主区只保留开始、切换、成功、失败、丢弃、结束会话六个按钮；次要区域保留 Esc 软件暂停。A、R、Enter 不再作为 RL 操作，也不再暴露单独的 abort/resume/reset 服务。失败数据想保留就按 N；不想保留就按左箭头，直接丢弃并复位。Q 不是进程急停，不想保存当前条时先丢弃再结束会话。

录制期间非零 xArm error 自动结束并保存 crash，记录 error code，进入已有自动复位流程；warning 不结束本条。只有保存和复位都结束后才允许下一条。复位期间仲裁节点停止转发目标，避免 hold 与复位指令冲突。

## 分类与保存完成标记

```text
<父目录>/
├── success/
├── optimal/      # 延续四分类目录，RL 默认操作不写入
├── failure/      # 人工失败、未完成条关闭会话
└── crash/
    ├── meta/
    ├── data/
    ├── videos/
    └── rl/episode_000000.json
```

每类是独立 LeRobot 数据集，episode index 独立递增。遥操的 success 配额保持原义，RL 没有配额，网页分别显示已保存的 success/failure/crash 数量。

每条完成后先关闭 Parquet footer 和 episode metadata，再重新打开追加模式；后续条使用新数据文件，先前条立即可读。最后原子发布 `rl/episode_NNNNNN.json`，外部程序可把这个文件作为可读取标记。视频按 LeRobot metadata 的路径及时间区间定位，不能假设每条对应一个 MP4。不要读取临时图片或 `.tmp` 文件。

第一帧完整同步数据产生前的结束事件无法构成有效 LeRobot episode；界面仍显示原因并复位，不生成空训练轨迹。进程关闭时仍未结束的缓冲条会被丢弃；需要保存时先使用 Y/N 或 Q。

## 帧字段

记录以配置的 anchor 相机为时钟，目标 30 FPS。图像 RGB uint8 `[H,W,3]`。状态与动作 `[8]`：七个绝对关节角（rad），最后一维 Robotiq `[0,1]`，0 打开、1 闭合。

| 字段 | 类型与形状 | 含义 |
|---|---|---|
| observation.state | float32 [8] | 测量关节角和夹爪位置 |
| action | float32 [8] | 驱动确认下发的绝对目标 |
| action.selected | float32 [8] | 与确认机械臂命令匹配的仲裁目标 |
| action.policy | float32 [8] | 仲裁时有效策略候选；无有效候选时补零 |
| action.policy_valid | float32 [1] | policy 是否有效；人工控制或过期时为 0 |
| action.source | float32 [1] | 已确认机械臂命令来源：policy=0、human=1、hold=2 |
| action.intervention | float32 [1] | 真正处于人工控制的帧为 1 |
| reward | float32 [1] | 普通帧 0，结束事件奖励写入最后已记录帧 |
| terminated | float32 [1] | 成功、人工失败或 crash 的最后一帧为 1 |
| truncated | float32 [1] | 保留的数据契约字段；当前人工失败/关闭会话使用 terminated=1、truncated=0，不再单独记录中止类别 |
| observation.timestamp_ros_s | float64 [1] | 原始 anchor 图像 ROS 时间 |
| action.command_timestamp_ros_s | float64 [2] | 机械臂/夹爪确认命令时间 |
| observation.sync_skew_s | float32 [相机数量+4] | 各传感器/命令相对 anchor 的时间偏差 |

LeRobot 原始 Parquet 会把 `[1]` 特征保存为标量；加载器按照 schema 恢复。ROS 绝对时间用 float64 保留毫秒精度。

`action` 由 `/xarm/commanded_joint_state`（SDK 成功后发布）和 `/robotiq/commanded_joint_state` 组成，必须早于图像时间且满足同步窗口。`action.source` 匹配因果仲裁记录中的机械臂关节目标，不使用界面最新来源猜测。夹爪会量化和过滤小变化，`action[7]` 与 `action.selected[7]` 可能不同，两路命令时间分别保存。SDK 确认下发不等于机器人到达目标；真实位姿仍是 `observation.state`。

`action.policy` 是控制时钟上的候选目标。ACT chunk 已按 30 Hz waypoint 时间轴插值到控制时钟，这个字段不是原始 `[25,8]` 模型输出 chunk。外部算法若需要原始 chunk，应接入模型输出或用保存观测重新计算。

同步缺失或候选无法匹配时跳过该帧。标准 LeRobot `timestamp` 和视频按已保存帧的 30 FPS 时间轴排列；实际间隔应使用 `observation.timestamp_ros_s`，跳帧后不能假设固定 1/30 秒。

## 结束标签和外部读取

每条 JSON 保存 outcome、reason、reward、reward_source、terminated/truncated、结束时间、duration、`terminal_frame_index`、source_frame_counts、丢帧数、策略服务 metadata 与地址。人工事件同时保留原始 reward_event。ROS 时间可对齐帧；monotonic 时间只能解释本机事件时序。

默认奖励由 `rl.success_reward=1`、`failure_reward=0`、`crash_reward=0` 配置。这里保存原始事件奖励，缩放、折扣和 shaping 由外部方法处理。

奖励在最后已记录帧出现一次，JSON 另存精确按钮/error 时间。该行是正常观测和命令，不是新加的吸收状态。外部算法必须明确最后一行、transition 和 chunk 的 reward/bootstrap 规则，避免奖励错位或重复。

外部策略可以不启用 WebSocket 推理节点，设置 YAML `inference.enabled=false` 并启动 `rl:=true rl_motion:=true inference:=false`。向 `rl.policy_action_topic` 发布 `JointState.position[8]` 绝对动作，同时订阅 `/rl/episode_active`、`/rl/policy_paused`，只在 active 且未暂停时输出新动作。恢复时获取新观测，不重放旧 chunk。`/rl/task` 提供当前指令，`/inference/policy_metadata` 可发布策略版本 JSON。

`/rl/command_event` 记录仲裁选择，`/rl/executed_action` 表示转发到 ROS 的目标，两者都不是硬件确认。训练动作以保存的 `action` 或驱动确认话题为准。

实现集中在 `rl/episode_recorder.py`（复用同步录制/保存/复位），`rl/recording.py`（字段和标签），`rl/human_intervention_node.py`（控制与接管）。这些模块不调用 learner 或 replay。

## 其他人工控制设备

`rl/human_input.py` 与 `rl/intervention.py` 不导入 FACTR SDK。其他设备只需通过适配节点持续发布配置的 `human_arm_topic`（JointState.position[7]，符号对齐的连续弧度坐标）、`human_gripper_topic`（position[0]，连续夹爪坐标）和 `human_connected_topic`（Bool）。夹爪方向/比例由 `gripper_delta_scale` 指定，控制层继续按切换时的相对位移映射，不假定设备与机械臂绝对姿态相同。

当前三个话题来自 FACTR；更换设备时可以关闭 `teleop.enabled` 并配置上述话题，不需要复制仲裁、数据保存或 episode 生命周期。关闭 FACTR 后复位也不等待 FACTR 对齐，但仍要求实际人工设备就绪。通用就绪信号是 `/rl/human_ready`（10 Hz）；录制器不接受长期残留的 latched True。

相关时间参数为 `human_input_timeout_sec=0.25`、`human_ready_stable_sec=0.3`、`stop_stable_sec=0.3`；自动接管还要求新鲜实测位置/速度、正常 control_ok 和停稳阈值，不是删除运动保护。
