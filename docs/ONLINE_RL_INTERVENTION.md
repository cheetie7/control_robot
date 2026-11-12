# 真机 RL 操作与数据记录

完整操作与数据契约见 [RL_DATA_COLLECTION.md](RL_DATA_COLLECTION.md)。

`control_robot` 负责策略执行、FACTR 接管、复位和数据记录。RL 模式现在具有手动开始、奖励结束、分类保存和 error 自动结束；模型训练由外部程序负责。
