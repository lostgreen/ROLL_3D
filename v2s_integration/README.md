# Video2Scene Agent / ROLL 接入

当前只保留一套场景重建 Env 和 ROLL Manager，采集与训练共享 messages、工具 schema、模型模板、视觉 collator 和执行器。训练使用 ROLL 原生 AgenticPipeline / EnvironmentWorker / RolloutScheduler / ActorWorker；runtime 只执行工具。

```text
agent/       模型消息、图像选择、native JSON/XML 调用解析
envs/        ReconstructionEnv：任务 reset/step/close、观察与结束条件
runtime/     工具注册、schema 校验、执行；vendor/ 保留 28 个必要源码文件
rollout/     同一 ReconstructionManager；原生推理采集入口
training/    共享环境配置、AgenticPipeline 启动、逐 decision 训练打包
evaluation/  显式 reward 插件、离线统计
ops/         后端探针、混元安装、四组资源调度
configs/     task / batch / train 配置样例
prompts/     无预算提示的 system / task，保留版本和 hash
tests/       当前接口与训练张量契约
docs/        架构与最新验证状态
legacy/      历史材质 smoke、诊断及未使用的源码，退出当前入口和测试
```

从仓库根目录运行统一入口：

```bash
python -m v2s_integration.run_rollout collect --help
python -m v2s_integration.run_rollout train --config v2s_integration/configs/train.example.yaml --config-only
python -m v2s_integration.run_rollout batch --help
python -m v2s_integration.run_rollout probe --help
python -m v2s_integration.run_rollout summarize --help
```

训练样例需要填写模型、task、reward_function 与运行目录；`--config-only` 只构建配置，不加载模型或验证完整 ROLL dataclass。实际 train 使用当前同一场景重建环境，不再调用历史材质任务。

训练目标显式设为 `reasoning_and_action`：当前采样响应含 reasoning、工具调用和采样到的 EOS；prompt、历史响应、视觉占位/展开及 padding 不进入 loss。训练直接保存原生 PolicyProxy 输出，每个 decision 一行，拒绝 prompt/response/position 漂移和训练阶段截断。支持原生 `step_reinforce` 与 `grpo`，两者的奖励归属见 [架构说明](docs/architecture.md)。

采集允许无 reward；训练必须提供 `module:function` 形式的评分函数。函数接收 `env` 与 `turn`，返回有限 float；奖励记录在文件及 ROLL scores 中，不加入模型提示词。未内置或宣称已有有效的场景质量 reward。

当前验证：63 个 CPU/接口测试通过，模型、RPC transport、Blender IO 使用替身；测试执行了原生 ROLL 采样后处理函数。未运行 GPU 训练、引擎视觉展开对齐或参数更新。复现：`python -m pytest -q v2s_integration`，需要 torch/tensordict/omegaconf 与已有图像/schema/test 依赖。

历史 experiment 的 prompt/config/result 没有改写，归档源码不能作为当前启动入口。旧 behavior_study/roll_adapter import 兼容已删除；使用新的功能路径。源码来源记录保留在 [runtime/source_manifest.json](runtime/source_manifest.json)，该文件描述整理前快照，不是当前部署 hash。

详见 [当前架构与参考](docs/architecture.md) 和 [10.14 ROLL 原始框架讲解](../weekly_report/1014/doc/roll_format_walkthrough.md)。
