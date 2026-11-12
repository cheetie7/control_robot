# VR TCP 遥操作与录制

VR 与 FACTR 是并列的人工输入路线，选择 `teleop.source: vr|factr`。
两者共用录制器、三路相机、夹爪、面板、四类保存和复位。
VR 支持普通遥操和 RL 数据采集模式；两种模式均使用 TCP action。
RL 模式复用原有面板、人工奖励、同步录制和复位，不在本仓库训练模型。
旧关节 ACT 不兼容，不能因 action 同为 8D 就直接接入。

## 1. 首次使用和命令

### 构建 ROS 包

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
uv run --project src --no-sync \
  python "$(command -v colcon)" build --symlink-install \
  --base-paths src/factr_teleop src/xarm_ros2_teleop \
  --packages-select factr_teleop xarm_ros2_teleop
```

日常运行复用现有 uv 环境，不自动 sync。首次环境准备仍用
`scripts/bootstrap.sh`；首次运行前关闭旧控制程序，避免两个 SDK 连接同时下发命令。

### 更新 Quest APK

旧 APK 只有位姿，没有离合/扳机；仍能做独立位姿诊断，但不能驱动 VR 遥操作。
Unity 工程和 XR 包版本未升级。用本机 Unity 编辑器执行现有
`QuestPoseBridgeSetup.BuildAndroid`，输出 `quest_pose_bridge/Builds/QuestPoseBridge.apk`。
例如，将 `UNITY_EDITOR` 设置为本机实际 Unity 可执行文件的绝对路径后：

```bash
cd /home/gxy/workspace/control_robot/src
"$UNITY_EDITOR" -batchmode -quit -buildTarget Android \
  -projectPath "$PWD/quest_pose_bridge" \
  -executeMethod QuestPoseBridgeSetup.BuildAndroid \
  -logFile /tmp/quest-vr-build.log

metavr device list --format json
metavr app install quest_pose_bridge/Builds/QuestPoseBridge.apk
```

安装前确认列表中是要更新的头显；多设备时按 metavr 帮助显式选择目标。
构建方法会重新生成该工程的验证场景，先保存编辑器中的工作。
本次代码验证没有构建/安装 APK，也没有在头显上验证新输入。

### 不启动任何设备的配置检查

```bash
cd /home/gxy/workspace/control_robot/src
bash scripts/record_data.sh --source vr --dry-run
```

### VR 预览：不连接机械臂和夹爪

```bash
cd /home/gxy/workspace/control_robot/src
bash scripts/record_data.sh --source vr --rviz
```

桥会启动现有 Quest 应用并维护 USB 连接。RViz 使用 `xarm_base`，显示映射后
TCP 目标的坐标轴。预览使用**虚拟零位姿**，不是机器人实测位姿；主要检查移动方向、
旋转和重新握持时是否无跳变。相机和面板可以运行，但不创建录制器或执行器节点。

### VR 实机录制

先编辑 `xarm_ros2_teleop/config/teleop_vr.yaml`：

- 配置实际安全工作空间 `workspace_lower_m`、`workspace_upper_m`，均为机器人基座
  坐标下的 xyz 米；默认 null，**不允许直接开启实机运动**。
- 预览核对 `base_from_floor_rotation`。默认单位矩阵仅表示两个坐标轴方向一致，
  不代表已完成物理标定；该矩阵必须是右手系旋转，不能配置镜像变换。
- 在控制器中确认正确的 TCP 工具偏移/负载。软件读取并保存当前配置，不自动猜测
  夹爪长度、不修改 TCP offset，也不进行自动负载辨识。
- 当前要求 xArm7 固件 >= 2.0.0，以便导出 DH；world offset 必须为零，保证 action
  明确位于基座坐标系。读取失败会阻止录制，而不是写入虚构运动学参数。

完成这些核对、清空工作区并备好硬件急停后：

```bash
cd /home/gxy/workspace/control_robot/src
bash scripts/record_data.sh --source vr --motion \
  --dataset /home/gxy/datasets/vr_block_insertion \
  --instruction "insert the block into the socket"
```

面板 `http://127.0.0.1:8080`：应用数据集设置，等待预热/初始位复位与数据集初始化，
开始预定位或本条录制。每次进入可控制状态后，**先松开握持键，再按住**。
右手握持键是离合，右手扳机控制夹爪。开始/分类保存/丢弃/结束仍使用现有面板或键盘；
不把手柄面键额外映射为保存按钮。

FACTR 的公共入口不变：

```bash
bash scripts/record_data.sh --source factr --dataset /home/gxy/datasets/factr_block_insertion
```

直接使用 ROS launch 时，选择 VR YAML 并显式传 `vr_motion:=true` 才允许机械臂运动；
普通遥操缺省为预览。`vr_rviz:=true` 显示映射后的目标。
TCP 策略暂只通过下面的 RL 仲裁模式执行，不开放绕过仲裁的直接推理链路。

### VR + TCP 策略的 RL 模式

配置入口为 `xarm_ros2_teleop/config/teleop_vr_rl.yaml`，继承 `teleop_vr.yaml`，
因此工作空间、手柄和坐标映射只需在公共 VR 配置维护。模型服务默认 `127.0.0.1:8081`。
先做不启动设备的配置检查：

```bash
cd /home/gxy/workspace/control_robot/src
bash scripts/record_data.sh --source vr --rl --dry-run
```

确认工作空间、TCP 工具偏移和新 TCP 策略已就绪后才开启物理运动：

```bash
cd /home/gxy/workspace/control_robot/src
bash scripts/record_data.sh --source vr --rl --motion --build \
  --dataset /home/gxy/datasets/vr_rl_block_insertion \
  --instruction "insert the block into the socket"
```

脚本只使用现有 uv 环境的 `--no-sync`，不安装依赖。构建后也可直接启动：

```bash
cd /home/gxy/workspace/control_robot
source /opt/ros/humble/setup.zsh
source install/setup.zsh
uv run --project src --no-sync python "$(command -v ros2)" launch \
  xarm_ros2_teleop teleop.launch.py \
  config:=/home/gxy/workspace/control_robot/src/xarm_ros2_teleop/config/teleop_vr_rl.yaml \
  rl:=true rl_motion:=true
```

不传 `--motion` / `rl_motion:=true` 时，不创建执行器和真实录制器，只检查连接/界面。
该模式不是可录制的仿真；缺少真实反馈时不会伪造机器人 state。

操作仍只有 S 开始、T 切换、Y 成功 +1、N 失败 0、左箭头丢弃复位、Q 结束会话；
Esc 是次要的软件暂停。没有自动 episode 超时或配额：

- 开始前 VR 输入需持续就绪；开始后策略持有控制权，VR 保持在线但不发送执行命令。
- T 接管：丢弃策略 chunk，驱动建立实测 TCP 保持；关节速度小于 0.03 rad/s
  持续 0.3 s 且反馈新鲜、健康后，启用 VR。必须在启用后松开再握持，不能预按握持接入。
- 松开握持：TCP 保持、夹爪保持上一目标；本条继续。再握持重新捕获起点。
- T 回策略：关闭 VR、丢弃人工目标和旧 chunk，从新鲜图像与**关节 state**重新推理。
- 人工输入失联：暂停并保留本条；恢复后 T 重新申请接管，再松开/握持。
- 人工 Y/N 或机械臂 error：沿用保存和自动复位；error 数据进入 crash。

### TCP 策略服务契约

本轮没有修改 RL 仓库，也没有提供 TCP ACT 训练/服务实现。后续模型端必须在 WebSocket
握手中明确给出下列字段，缺失或关节格式会在运动前被拒绝：

```json
{
  "protocol_version": 1,
  "state_dim": 8,
  "state_format": "absolute_joint_rad_plus_gripper",
  "action_dim": 8,
  "action_format": "absolute_tcp_pose_xyzw_plus_gripper",
  "action_frame": "xarm_base",
  "position_unit": "m",
  "quaternion_order": "xyzw",
  "image_keys": ["top_camera", "left_camera", "right_camera"],
  "execution_horizon": 25,
  "action_frequency_hz": 30.0
}
```

`image_keys` 必须替换为实际 `inference.images` 的有序 key，示例不是额外相机配置。
请求仍为 RGB 图像、joint+gripper `state[8]`、prompt；返回 `actions[H,8]`，
每行是绝对基座 xyz 米、单位 xyzw 四元数和夹爪闭合度。返回动作校验有限值、工作空间、
单位四元数和夹爪范围。动作按 30 Hz 时间轴采样到 100 Hz；xyz/夹爪线性插值，
旋转用最短路径 SLERP，末 waypoint 保持一个采样间隔（25/30 秒一个 chunk）。
驱动仍执行公共 TCP 速度/加速度/姿态速率限制，不绕过硬件碰撞保护。

VR-RL 中只有 `HumanInterventionNode` 向 `/xarm/tcp_target`、`/xarm/tcp_following`
和夹爪发布。VR 发布候选 `/control/vr_action`，策略发布配置的候选 topic；均为
包含 `timestamp`（ROS 秒）、`action_format` 和 `action[8]` 的 JSON String，
VR 另有 `following`。过期、错误控制来源和切换前的候选不允许恢复运动。

录制的 `action` 来自 SDK 实际接受的 TCP 命令，`action.selected` 保存原始选中目标，
`action.policy` 保存该时刻有效策略候选；限速时三者可能不同。TCP SDK 确认事件
`/xarm/tcp_command_event` 将实际命令与原选择时间关联，避免按四元数值相等猜来源。
保持命令标记 hold；缺少因果确认则跳帧，不伪造成功 action。
共享 RL 标签记录 TCP `action_format`，普通 FACTR 标签继续记录关节格式。

## 2. 具体控制公式

输入桥先把 Unity 位姿转换为 `quest_floor` FLU：前、左、上，米和 xyzw 四元数。
在每次握持接入时记录手柄起点 $(p_{v0},R_{v0})$ 和实测 TCP $(p_{r0},R_{r0})$：

$$
p_r^*=p_{r0}+kC(p_v-p_{v0}),\qquad
R_r^*=(CR_vR_{v0}^{T}C^{T})R_{r0}.
$$

$C$ 是配置中的固定轴旋转，$k$ 默认 1。平移始终按固定空间方向映射，
不随接入时手柄朝向变化；计算基于起点，不累积逐帧增量。
四元数内部归一化并保持符号连续，不直接相减欧拉角。

夹爪采用相对扳机：

$$
g^*=\operatorname{clip}(g_0+u-u_0,0,1).
$$

$g_0$ 是接入时实测闭合度，$u_0$ 是当时扳机值，避免重新握持导致夹爪突然打开。
松开后保持上次目标。想扩大扳机行程时，可松开握持、调整扳机位置后重新握持。

VR 节点不建立 SDK 连接，只输出目标；现有 xArm 节点独占 SDK，使用 mode 7 和
`set_position_aa(wait=False, relative=False, is_tool_coord=False)`。仅 SDK 边界转成
毫米/轴角弧度。100 Hz 控制循环只处理最新目标，不排队回放手柄历史。

默认 TCP 平移速度上限 0.10 m/s、加速度 0.20 m/s²；姿态命令按旋转最短路径限制
到 0.5 rad/s。超出工作空间会退出跟随，不能靠持续握持自动恢复；不是静默裁剪轨迹。
这些约束不替代碰撞检测、关节限位和现场风险验证。

## 3. 停止、恢复和复位

- 松开握持：驱动用新鲜实测 TCP 提交保持目标，覆盖旧远端目标；不是仅停发消息。
- 跟踪/按键无效、0.25 s 无新输入、原点变化、重连：保持本条，停止跟随。
- 位姿有效要求位置与旋转均处于跟踪状态；不是只检查设备是否连接。
- 自动重连只恢复设备通信。连续有效 0.3 s 后，还必须松开、再握持建立新起点。
  驱动自身也监控 ROS 命令超时，恢复报文不能直接复活旧目标。
- 预热、episode/staging、复位和 xArm 健康门控都有效时才允许物理跟随。
- TCP 反馈或 SDK 提交失败：锁定故障并请求停止；不发布成功 action。按面板的
  保存/丢弃流程结束并复位后才可重新接入。若无法确认停止，状态明确标记。
- 机械臂 error：按既有流程保存 crash、共享复位；失败则不解锁下一条。
- 复位期间使用关节模式执行原来的抬升/初始位流程；VR 无 FACTR 对齐要求。
  复位结束清空旧起点和目标，不能回到上次推理/遥操的末态。

软件暂停不是硬件急停。突然失联最多存在配置的超时检测窗口；夹爪保持上次提交目标，
不因短暂丢失手柄而自动张开。

## 4. 数据到底保存什么

| 字段 | 形状与含义 |
|---|---|
| observation.state | float32[8]，实测 joint1..7 弧度 + 实测夹爪闭合度 |
| action | float32[8]，SDK 接收成功的绝对 xyz 米 + xyzw + 夹爪闭合目标 |
| observation.teleop_status | float32[2]，tracking_valid、following |
| 其他观测 | 原有图像、关节速度/力矩、同步偏差 |

action 是**控制目标**，SDK 成功不意味着机器人已到达该目标。姿态限速后的命令才作为
标签，不能保存限速前的手柄目标。夹爪继续取实际夹爪驱动提交的命令；两个执行器
分别按相机时间戳做因果对齐，不假设它们同时完成。

30 FPS 的帧仍由相机锚定；不选时间戳晚于图像的命令。暂停期间能建立有效保持命令
就继续记录，并用 teleop_status 标记；无法可靠同步时仍跳过并报告原因，不伪造 action。

四个分类仍是 `success / optimal / failure / crash`，目标计数只算 success。
每个分类的 `meta/control_contract.json` 记录字段、坐标、单位和机器人配置。
DH 参数按控制器原始格式保留，明确绑定 SDK/固件版本；不能擅自把它当通用标准 DH
矩阵或 FACTR 的 URDF。之后做离线 FK 时需核对厂家模型及工具变换，本轮不提供 FK 求解器。

TCP 和关节 action 恰好都是 8D，但语义不同；续录校验会拒绝混用。新 VR 数据不能
直接套用原来的关节 ACT 权重或训练配置，需要后续专门适配 TCP action 的姿态处理和部署。

## 5. 脚本和模块位置

- `scripts/record_data.sh`：共用采集入口；`scripts/bootstrap.sh`：首次环境准备。
- `scripts/diagnostics/`：FACTR 延迟、Femto 相机、Quest 独立位姿/RViz 检查。
- `scripts/setup/`：夹爪依赖补丁、udev 安装及规则。
- `scripts/data/`：LeRobot 格式转换。
- `vr/protocol.py` 与 Quest 工程：同帧位姿/按键、协议版本和坐标原点世代。
- `vr/mapping.py`、`vr/pose_math.py`：不依赖硬件的离合与位姿数学。
- `vr/teleop_node.py`：门控、输入健康、预览、TCP/夹爪目标。
- `xarm/tcp_control.py`：借用现有驱动 SDK 连接执行 TCP、保持和故障锁定。
- `recording/control_contract.py`：数据语义与运动学续录校验。

单独查看原始手柄仍用 `bash scripts/diagnostics/run_quest_pose.sh --rviz`。
不要与公共录制入口同时启动同一个桥，避免重复绑定端口。

自动化测试不操作硬件。实机验收顺序：APK 输入 → 预览方向 → 低速平移 → 姿态/夹爪 →
松手/失联/重新握持 → 复位 → 短录制检查 Parquet 与契约。
