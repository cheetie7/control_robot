# 真机与学习端接口

两个仓库的边界、MessagePack v1、绝对动作与时间轴、真机 RL 字段及 source 编码见：

[RL 仓库接口 v1 文档](../../../RL/docs/robot_contract_v1.md)。

上述共享样例针对 FACTR/关节策略。VR 路线新增了独立 TCP action 契约，仍使用相同
MessagePack 传输和关节 state，但不能用旧关节 checkpoint：详见
[VR TCP 策略握手与 RL](VR_TELEOP_ZH.md#tcp-策略服务契约)。
本轮没有修改 RL 仓库的训练/服务接口，后续 TCP 模型服务需显式适配。

本仓库的实现位置：

- `inference/contracts.py`：输入、动作与服务握手检查。
- `inference/tcp_contracts.py`：TCP action 校验与 xyz/夹爪线性插值、四元数 SLERP。
- `inference/websocket_policy.py`：通信编码和客户端。
- `inference/status.py`：结构化 phase/reason，以及旧消息的统一读取。
- `rl/recording.py`：动作来源、精确时间戳、版本化 episode 标签。
- `xarm_ros2_teleop/test/fixtures/robot_contract_v1.json`：双方共同使用的契约样例。
- `xarm_ros2_teleop/test/test_inference_status.py`：状态、握手和记录格式测试。

上面的源码路径相对于 `xarm_ros2_teleop/xarm_ros2_teleop/`。样例中真机动作是 `[25,8]`，30 Hz 动作点由 100 Hz 控制循环插值执行；不改变驱动安全约束。

这次只统一接口与状态表达，未拆分公共录制器，也未新增真机 learner。旧 checkpoint 缺少频率会明确警告，不会被修改；新版服务已知频率与本地配置不一致时拒绝执行。
