# control_robot 文件地图

本文按当前代码解释职责，不调整目录、不改变运行行为。覆盖项目自己的主要源码、配置和脚本；第三方 SDK、Unity 依赖、缓存与生成文件按目录解释，不逐个列出。

## 1. 先理解两个仓库的分工

| 仓库 | 负责 | 不负责 |
| --- | --- | --- |
| `control_robot/src` | 真机控制、遥操、接管、模型请求、重置、录制、人工标签、操作台 | ACT/IQL/残差网络的梯度更新 |
| `RL` | 数据读取、模型训练、奖励标注、评估、模型服务、仿真在线训练调度 | xArm SDK、FACTR 串口、真机重置与 Web 操作台 |

```text
相机 + 机械臂实测状态
    → control_robot/inference：整理观测、请求模型
    → RL 模型服务：ACT 或 ACT+残差，返回绝对动作 chunk
    → control_robot：按控制时钟执行，必要时人工接管
    → 驱动确认的动作 + 实测状态 + 图像 + 标签 → 数据集
```

真机 `rl/` 目前是**采集与控制端**，不是强化学习训练器。数据端记录齐全，不等于已经接上外部学习器；学习器还需要按自身 replay/transition 格式读取这些数据。

## 2. 根目录：哪些需要看

这里的 Git 仓库根目录是 `/home/gxy/workspace/control_robot/src`，上一级才是 ROS 工作空间。

| 路径 | 意义 |
| --- | --- |
| `xarm_ros2_teleop/` | 主系统：绝大多数日常修改都在这里 |
| `factr_teleop/` | FACTR 底层驱动和保留的上游示例；主系统使用其中的 Dynamixel 驱动 |
| `scripts/` | 安装、诊断、启动、格式转换的辅助脚本 |
| `docs/` | 各功能的使用与设计说明 |
| `quest_pose_bridge/` | 独立的可选姿态输入工程，不是当前 ACT/FACTR 主链路的必读内容 |
| `README.md` | 安装和日常启动说明 |
| `CODEBASE_OVERVIEW_ZH.md` | 旧导览入口，指向 `docs/ONBOARDING.md` |
| `pyproject.toml`、`uv.lock`、`.python-version` | uv 环境、依赖锁定和 Python 版本；ROS 节点入口在各包的 `setup.py`，不在这里 |
| `AGENTS.md` | 给代码协作者的开发规范，不参与运行 |
| `.gitignore`、`.git/` | Git 忽略规则与版本历史 |
| `.venv/` | 安装好的 Python 环境，不是项目源码 |
| `log/`、`.pytest_cache/`、`.ruff_cache/` | 日志和检查缓存 |
| `.agents/`、`.codex/` 等隐藏工具目录 | 开发工具状态，不是机械臂业务模块；不要作为学习源码的起点 |
| 上一级的 `build/`、`install/`、`log/` | ROS 构建、安装和运行产物；应改 `src` 源码，不要直接改安装副本 |

## 3. 为什么有两层 xarm_ros2_teleop

```text
xarm_ros2_teleop/                  ← ROS 包：安装、配置、启动和测试
├── setup.py / setup.cfg / package.xml
├── config/
├── launch/
├── test/
└── xarm_ros2_teleop/              ← Python 包：实际功能实现
    ├── xarm/
    ├── recording/
    ├── inference/
    └── ...
```

外层是“怎样安装、启动这些模块”，内层是“模块怎样工作”，不是重复保存两份源码。

### 安装、启动和配置文件

| 文件 | 意义 |
| --- | --- |
| `setup.py` | 把 Python 节点函数注册成 ROS 可执行入口，安装配置等资源 |
| `setup.cfg` | ROS Python 可执行文件的安装位置 |
| `package.xml` | ROS 包名、依赖和构建元信息 |
| `resource/xarm_ros2_teleop` | ROS 查找本包所需的标记文件 |
| `launch/teleop.launch.py` | 全系统总入口：根据参数组织遥操、推理或 RL 采集模式 |
| `launch/sensors.launch.py` | 仅启动传感器，便于检查相机和触觉 |
| `launch/quest_pose.launch.py` | 可选姿态桥接入口 |
| `config/teleop_system.yaml` | 当前主配置：硬件、相机、录制、推理、RL、重置等参数 |
| `config/teleop_real_rgb.yaml` | 另一套完整 RGB 配置；不是自动继承主配置的补丁，各参数可能不同 |
| `config/sensors_all.yaml` | 传感器检查配置 |
| `config/gello_grav_comp_demo.yaml` | 主系统 FACTR 控制器的动力学、重力补偿等参数 |
| `urdf/gello_real_dynamic.urdf` | FACTR 机构与动力学描述 |
| `rviz/quest_pose.rviz` | 可选姿态显示布局 |

下面所有路径均相对于内层 `xarm_ros2_teleop/xarm_ros2_teleop/`。

## 4. 控制、硬件与启动状态

| 文件 | 意义 |
| --- | --- |
| `system/config.py` | 读取、检查统一 YAML；将硬件、传感器和工作模式参数转换成各节点配置 |
| `system/sensor_actions.py` | 汇总需要启动的传感器进程；不是实时健康检查器 |
| `system/session_config.py` | 启动预热、提示语音等会话配置 |
| `session/session_coordinator.py` | 启动时预热、就绪检查和放行；区别于每条 episode 结束后的重置 |
| `teleop/controller.py` | FACTR 控制基础：重力补偿、电流/力矩和控制约束 |
| `teleop/leader_node.py` | 当前主臂 ROS 节点：读 FACTR、发布候选动作、断线重连、重置后对齐 |
| `teleop/offset_calibration.py` | FACTR 与机械臂参考位置/偏置的计算和校准文件读写 |
| `teleop/factr_calibration_cli.py` | 人工执行偏置校准的命令入口 |
| `teleop/factr_reset_debug.py` | 单独排查 FACTR 重置/对齐问题的工具 |
| `xarm/xarm_controller.py` | xArm SDK、实测状态、绝对目标执行、安全检查、跟随超时与重置动作 |
| `robotiq/robotiq_controller.py` | 夹爪串口通信、激活、状态、目标执行和重置时张开 |
| `reset/mapping.py` | FACTR 与 xArm 参考坐标/关节映射计算，不直接操作硬件 |
| `reset/reset_coordinator.py` | 协调机械臂、主臂和夹爪的重置顺序、超时与错误恢复 |
| `speech/speech_announcer.py` | 异步语音播报，避免控制线程等待语音完成 |

## 5. 相机、触觉与监控

| 文件 | 意义 |
| --- | --- |
| `camera/driver_actions.py` | 启动 RealSense/Orbbec 驱动、话题映射、延迟与重启参数 |
| `tactile/driver_actions.py` | 组织触觉设备进程的启动 |
| `tactile/xense_publisher.py` | 从 Xense SDK 获取触觉图像/深度并发布 ROS 消息 |
| `monitor/health.py` | 根据消息时间判断新鲜度的纯逻辑 |
| `monitor/system_monitor.py` | 汇总硬件/传感器健康状态，发布操作台需要的诊断信息 |
| `visualization/rerun_visualizer.py` | 把图像、状态、动作送到 Rerun；不负责保存训练数据 |
| `vr/protocol.py`、`vr/quest_pose_node.py` | 可选姿态数据的协议检查与 ROS 桥接，独立于主机械臂控制链路 |

## 6. 录制：为什么有这么多文件

不是每个文件都各录一份数据，而是把“缓存、同步、编码、保存、episode 状态”分开。

| 文件 | 意义 |
| --- | --- |
| `recording/config.py` | 帧率、图像流、同步时间窗口、保存参数 |
| `recording/timestamp_buffer.py` | 保存短期带时间戳消息，查询某时刻最近或之前的消息 |
| `recording/message_converters.py` | ROS 消息转数组，处理图像裁剪/尺寸/深度单位等 |
| `recording/synchronizer.py` | 把相机、实测 state、已接受 action 拼成一帧；不能拿未来动作给过去图像配标签 |
| `recording/dataset_schema.py` | 定义 LeRobot 每个字段的名字、类型和形状 |
| `recording/lerobot_writer.py` | 创建/恢复数据集，保存/丢弃 episode，写视频和表格，按类别落盘 |
| `recording/keyboard_controller.py` | 把键盘操作转成录制命令，不直接控制模型训练 |
| `recording/episode_recorder.py` | 共享录制状态机：开始、结束、异步帧队列、保存、复位；承载遥操和推理测评流程 |
| `evaluation/outcomes.py` | 保存推理测评的成功/失败结果记录 |

这里的 `state` 是实测关节/夹爪状态，`action` 是驱动确认接受的控制目标，不能混为一谈。同步警告主要应看 `synchronizer.py` 的条件和上游驱动是否持续发布有效动作。

## 7. 模型推理与人工接管

| 文件 | 意义 |
| --- | --- |
| `inference/status.py` | 唯一 phase/reason 解码与精确判定；新消息为 JSON，兼容读取旧状态文字 |
| `inference/config.py` | 模型地址、相机、推理/动作频率和动作检查参数 |
| `inference/contracts.py` | 整理模型 observation，检查绝对动作 `[T,8]`，对 chunk 插值并转换成控制消息 |
| `inference/websocket_policy.py` | 机器人端模型客户端：发送观测、接收动作；不包含 ACT 网络 |
| `inference/remote_policy_node.py` | 串联相机/状态、远程请求、chunk 执行和 ROS 发布 |
| `inference/mock_server.py` | 固定保持动作的假服务，用来检查连接和协议，不是训练模型 |
| `rl/intervention.py` | 人工接管状态机、动作来源和奖励事件的数据定义 |
| `rl/human_input.py` | 与设备无关的输入缓存、新鲜度和持续就绪窗口 |
| `rl/human_intervention_node.py` | policy/human/hold 仲裁；单按钮自动停稳接入、暂停和人工奖励；复位仍由 recorder 协调 |
| `rl/episode_recorder.py` | **继承共享 recorder**，增加手动 RL episode 生命周期、crash 自动结束保存和 RL 标签 |
| `rl/recording.py` | 匹配动作来源/时间戳，定义附加字段和每条 episode 的 RL JSON 标签 |
| `rl/episode_services.py` | RL 开始、丢弃、结束服务名，供录制器和操作台共用 |

两个 `episode_recorder.py` 是“通用底座 + RL 扩展”，不是两个独立录制系统。`/rl/executed_action` 表示仲裁后转发的指令，驱动确认接受的 action 仍应看对应执行器输出。

当前 RL 启动后人工设备持续运行，就绪后才能开始；T 单次点击自动切换，无独立中止/恢复/Reset 按钮。当前 RL 采集由人结束，不因普通时长/数量自动结束；机械臂 error 则结束并保留 crash 数据。保存类别、奖励与具体按键以 [RL 数据采集说明](RL_DATA_COLLECTION.md) 为准。

## 8. 网页操作台

| 文件 | 意义 |
| --- | --- |
| `dashboard/config.py` | 网页/Rerun 端口、缓存等配置 |
| `dashboard/ros_gateway.py` | ROS 状态转网页状态；网页动作转 ROS 服务/命令 |
| `dashboard/server.py` | HTTP/WebSocket 服务和 ROS 执行器的组织 |
| `dashboard/static/index.html` | 按钮、文字、面板结构 |
| `dashboard/static/app.js` | 状态显示、按钮可用性、点击/快捷键、请求与重连 |
| `dashboard/static/styles.css` | 颜色、尺寸、排版 |

按钮灰色时，不一定是 HTML 问题：先看 `ros_gateway.py` 输出什么状态，再看 `app.js` 的启用条件。

## 9. factr_teleop：当前用什么，哪些只是上游示例

| 路径 | 意义 |
| --- | --- |
| `factr_teleop/dynamixel/driver.py`（内层 Python 包） | 当前主链路使用的 Dynamixel 串口/寄存器驱动 |
| `factr_teleop/dynamixel/python/` | 随上游携带的 SDK/示例代码，区别于 uv 安装环境中的依赖 |
| `factr_teleop/factr_teleop.py` | 保留的上游 FACTR 基础实现；当前系统控制基类在主包的 `teleop/controller.py` |
| `factr_teleop/factr_teleop_ros2.py` | 上游 ROS 接入实现，不是当前总系统的主臂节点 |
| `factr_teleop/factr_teleop_franka_zmq.py` | 上游 Franka/ZMQ 示例 |
| `factr_teleop/factr_teleop_grav_comp_demo.py` | 上游重力补偿演示入口 |
| `factr_teleop/configs/`、`factr_teleop/urdf/` | 上游示例配置与模型资源；当前主系统另有自己的安装配置/URDF |
| `setup.py`、`setup.cfg`、`package.xml`、`resource/` | ROS 安装和包发现元数据 |
| `test/` | 底层驱动和包检查 |
| `README.md`、图片资源 | 上游使用说明和说明图片 |

上表除标注外相对于外层 `factr_teleop/`。判断实际使用哪套实现，应沿 `launch/teleop.launch.py → setup.py → Python 节点` 看，不要仅凭同名文件判断。

## 10. scripts、文档和测试

| 文件 | 意义 |
| --- | --- |
| `scripts/bootstrap.sh` | 准备环境/依赖和构建 |
| `scripts/record_data.sh` | 日常录制的一键包装，整理参数后启动主系统；不是另一套 recorder |
| `scripts/diagnostics/check_factr_latency.sh` | 检查/设置 FACTR USB 串口延迟 |
| `scripts/diagnostics/check_femto_bolt.py` | 单独诊断 Femto 相机 |
| `scripts/setup/install_sensor_udev.sh`、`scripts/setup/*.rules` | 安装设备权限和 USB 电源规则 |
| `scripts/setup/patch_pyrobotiqgripper.sh` | 修补夹爪依赖包，不是主夹爪节点 |
| `scripts/data/convert_lerobot_v30_to_v21.py` | LeRobot 格式转换工具；不要和 RL 仓库的 v2.1→v3 准备流程混淆 |
| `scripts/diagnostics/run_quest_pose.sh` | 可选姿态工程启动包装 |
| `docs/ONBOARDING.md` | 更深入的架构、ROS 接口、运行流程和修改指南 |
| `docs/ROBOT_LEARNING_CONTRACT.md` | 两仓协议、数据与时间轴契约入口 |
| `docs/REMOTE_POLICY_INFERENCE.md` | 远程模型推理协议与真机测评 |
| `docs/RL_DATA_COLLECTION.md` | 当前真机 RL 采集、按键、字段和人工奖励 |
| `docs/ONLINE_RL_INTERVENTION.md` | 接管说明入口，衔接当前数据采集文档 |
| `docs/QUEST_POSE_BRIDGE.md` | 可选姿态桥接说明 |

`xarm_ros2_teleop/test/test_*.py` 按同名功能测试：`recording`/`rl_recording` 测录制，`remote_inference` 测推理契约，`dashboard` 测操作台，`reset` 测复位，`xarm_controller`/`robotiq_node` 测执行器，`factr_*` 测校准与预热，`camera_*`/`tactile_*`/`sensor_actions` 测传感器，`system_config` 测配置，`evaluation` 测结果记录，`rl_intervention` 测仲裁，`quest_pose_protocol` 测可选协议。它们不是运行时业务入口。

各目录中的 `__init__.py` 用于 Python 包初始化/导出，不是每个都代表一个额外功能模块。

## 11. 建议阅读顺序与复杂度热点

1. 先读主 YAML：知道当前启用了什么，而不是先看全部驱动。
2. 看 `launch/teleop.launch.py` 和 `setup.py`：理解启动了哪些节点。
3. 看自己关心的一条链路：遥操看 `leader_node.py → xarm_controller.py`；推理看 `remote_policy_node.py → contracts.py → xarm_controller.py`。
4. 看 `recording/` 的 schema、synchronizer、writer，最后再看大的 recorder 状态机。
5. RL 采集再加读 `rl/intervention.py → human_intervention_node.py → rl/episode_recorder.py`。

当前复杂度并非全是目录数量造成的：共享 `recording/episode_recorder.py` 已超过两千行，系统配置和机械臂驱动也较大。理解入口后再按功能定位，比从这些大文件第一行开始阅读更有效。是否进一步拆分，应另做局部重构，不在本次导读中改动行为。

训练侧对应导读：[RL 文件地图](../../../RL/docs/FILE_MAP_ZH.md)。
## VR 路线补充

新增 VR TCP 路线的输入、控制和数据契约见 [VR_TELEOP_ZH.md](VR_TELEOP_ZH.md)。
`vr/` 负责设备输入与相对映射；`xarm/tcp_control.py` 负责原生 TCP 执行；
录制仍是原来的共用 recorder。VR profile 继承现有三相机配置，不另复制相机串号。

