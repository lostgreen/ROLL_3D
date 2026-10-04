# Video2Scene Agent：与原生 ROLL 对齐的结构

更新于 2026-10-04。本次把场景重建采集与训练统一到一个 Env/Manager，移除旧材质任务的活动入口，按功能组织代码。

![当前 Agent 架构](../../weekly_report/1014/doc/images/14_v2s_unified.png)

## 职责与借鉴

| 层 | 当前实现 | 参考与取舍 |
| --- | --- | --- |
| 模型输入/动作 | [agent/messages.py](../agent/messages.py)、[agent/actions.py](../agent/actions.py) | VAGEN 区分环境 observation 与模型 messages；这里把消息策略集中，使用模型官方 template 接收 tools |
| 任务环境 | [envs/reconstruction.py](../envs/reconstruction.py) | Env 只负责任务状态、图片、reward 和终止，不自行调用模型 |
| 工具运行库 | [runtime/session.py](../runtime/session.py)、runtime/vendor | 复用原 Video2Scene providers；移出未使用的 run_agent/本地模型循环，保留 MCP/Blender 生命周期和执行器 |
| ROLL 适配 | [rollout/manager.py](../rollout/manager.py) | 继承原生 VLTrajEnvManager，保留 collator/PolicyProxy/Router/rollout loop；统一采集与训练 |
| 训练打包 | [training/packing.py](../training/packing.py) | 借鉴原生 StepEnvManager 的逐 decision 行，保留多模态位置和特征，不重建最终对话来猜 mask |
| 优化/调度 | [training/launch.py](../training/launch.py) | 原生 AgenticPipeline / EnvironmentWorker / RolloutScheduler / ActorWorker；DART 的慢环境经验作为参考，未另建异步框架 |
| 评分与分析 | [evaluation/rewards.py](../evaluation/rewards.py)、evaluation/summarize.py | MetaSpatial 将 Blender/约束/视觉评分分开；此处独立 reward 插件接口，具体质量函数待接入 |

外部参考：[VAGEN](https://github.com/mll-lab-nu/VAGEN)、[MetaSpatial](https://github.com/PzySeere/MetaSpatial)、[DART-GUI](https://github.com/Computer-use-agents/dart-gui)。本次借鉴职责划分，并未移植它们的 veRL trainer 或模型循环。实际底座仍是仓库中的 ROLL。

## 两个入口，同一任务

```text
collect -> ROLL inference Cluster / Router / PolicyProxy
        -> ReconstructionManager.begin_episode / make_decision / step

train   -> AgenticConfig -> AgenticPipeline -> EnvironmentWorker
        -> ReconstructionManager.run_rollout_loop（继承 ROLL 原生循环）
        -> make_decision / step / formulate_rollouts

共同路径：
  ReconstructionEnv.reset
  -> agent/messages -> apply_chat_template(messages, tools) -> VL collator
  -> 原生模型采样 -> agent/actions -> runtime/session -> tool providers
  -> observation / reward / terminated / truncated -> 下一轮 decision
```

注册名称都是 `video2scene_reconstruction`，manager 都是 `v2s_integration.rollout.manager.ReconstructionManager`，配置都由 [training/config.py](../training/config.py) 的 environment_config 构造。采集/训练的差异只在 worker、采样、评分启用和是否送入训练队列。

当前解析器接受模型原生 `<tool_call>JSON</tool_call>` 或 Qwen function/parameter 封装，拒绝裸 JSON、裸 bash、多调用和截断调用。schema 参数类型、允许工具和实际执行沿用同一 runtime。

## 每次 decision 的训练数据

历史裁剪与图像选择会让相邻 prompt 不具有单调增长关系，因此不能拿最终 episode 的字符串重建所有响应位置。当前保存实际采样返回的 DataProto：

```text
row t: [实际训练侧 prompt C_t] [实际 sampled response Y_t] [右侧 padding]
loss:  [             0      ] [               1       ] [      0      ]
```

保留 `input_ids/attention_mask/position_ids/prompt_mask/response_mask/infer_logprobs` 和该请求的 multi_modal_inputs；历史 assistant 响应只属于 C_t，本行只优化 Y_t。核对 prompt IDs、sampled IDs、响应位置及视觉 prompt positions。长度超过 sequence_length 会报错，不在训练阶段截断。

显式目标是 `reasoning_and_action`，当前未实现 action_only。EOS 只在模型确实采样到时进入 mask。完整响应中的 reasoning 和调用属于同一个采样序列，不人为拆成两个 assistant 消息。

每行还有 ROLL 原生的 `env_ids/group_ids/tags/step/step_scores/episode_scores/messages_list`；原生循环补充 traj_id/traj_group_id。step_reinforce 使用每步 reward 和原生折扣 return；grpo 将 episode 总 reward 放到每个 decision 的 scores 上，使前面的调用也得到 episode 回报。不同长度 episode 在 group 统计中的权重需要真实实验核对。

当前保留原生总训练 loss；独立动作 token loss/格式学习指标尚未接入。引擎 request ID/policy version/内部视觉展开验证也未完成，不能将 CPU 张量契约测试解释成完整训推概率一致性证明。

## 评分与旧代码

采集无评分时 reward=0；训练构造环境必须提供 reward_function=`module:function`。插件接收 `env` 和 `turn`，可读取公开参考、渲染或终局 scene.blend，返回有限 float。终局评分发生在保存/关闭之后，应读取产物；中间状态评分发生在该步结束时。reward 不加入模型可见 messages。

这次没有把旧 RGB MSE 当作新场景质量函数。样例里的 `your_project.rewards:scene_quality` 是待填写接口，不是已实现的评分模型。

legacy/material_smoke 保存旧 Env/Manager、入口、诊断、patch 和历史结果；legacy/runtime_unused 保存未调用的模型循环、metrics 和旧集成。当前 CLI 与 pytest 排除这些目录。runtime/vendor 从原来 74 个 tracked 文件缩减到 28 个必要文件，源码内容保留。

## 验证边界

63 个本地检查通过，覆盖共享配置、原生 JSON/XML 封装、schema、工具能力边界、无预算提示、外部耗尽、上下文裁剪，以及原生采样后处理下的 2D/3 通道/4 通道 positions、响应 token、logprobs、padding 和输入专用 token 排除。RPC transport 与模型/Blender 使用替身；未执行训练或参数更新。

下一次真实验收需要有效的质量 reward、完整 ROLL dataclass/worker 初始化、引擎视觉输入对齐、有限 loss/梯度和参数更新。当前代码组织及共用训练/rollout 接口已完成；训练效果尚无新实验结论。
