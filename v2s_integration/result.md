# KML ROLL + Video2Scene：原生 Agent RL 联调结果

2026-09-23，`native_v10` 已成功完成两步原生 ROLL 训练，耗时约351秒，退出码0。

| 核验项 | 结果 |
| --- | --- |
| 模型 / 算法 | Qwen2.5-VL-7B / GRPO + LoRA |
| 后端 | FSDP2训练GPU0，vLLM推理GPU1，2×A800 |
| 轨迹 / 工具执行 | 8条完整轨迹，10次工具执行，1次工具错误 |
| 终局奖励 | 0.242761–0.948434；初始场景0.615980 |
| token / 图像检查 | 8/8通过；采样token保真、图像不计动作loss、vLLM实际展开输入与训练输入一致 |
| 最后一步梯度范数 | 0.121659，有限且非零 |
| LoRA参数 | 56/56个B张量非零且有限；最大绝对值0.0000200133 |
| checkpoint | 24个文件，约19.14GB，含adapter |
| 原代码保护 | manifest中的74个原harness文件SHA256全部一致 |
| 资源回收 | 结束后两张GPU均2MiB |

闭环为：参考图/当前渲染 → ROLL采样工具动作 → 原harness副本执行Blender编辑 → 渲染反馈/奖励 → ROLL构造轨迹并更新策略。未调用单进程替代训练脚本。

这是受控材质任务上的训练链路验证。奖励使用模型可见图像的RGB误差；两步训练和单次样本奖励不能证明策略提升或泛化。尚未验证checkpoint恢复。

远端证据根目录：`/home/xuboshen/zgw/ROLL/v2s_integration`。
主要证据：`native_v10_verification.json`、`native_v10.log`、`native_run/tensorboard`、`native_episodes/*/training_contract.json`。

Adapter：`/home/xuboshen/zgw/ROLL/v2s_integration/native_run/actor_train-0-G0/checkpoint-1/adapter_model.safetensors`。

接口与实现见同目录README.md；继续任务前读state.md。旧版本v1–v9的失败和零梯度结果均为历史证据。
