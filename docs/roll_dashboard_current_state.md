# ROLL Agent-RL 数据看板现状诊断

## 1. 诊断对象和结论

诊断对象是当前仓库的 Video2Scene 原生 ROLL 运行产物：

```text
/home/xuboshen/zgw/ROLL/v2s_integration/native_run
/home/xuboshen/zgw/ROLL/v2s_integration/native_episodes
```

当前已经有一套“实验证据集合”，但还没有一个能被不同 Agent-RL 环境复用的统一看板。现有产物能回答“这次任务有没有跑完、梯度是否存在、checkpoint 是否生成”，还不能稳定回答：

- 数据是否真的有训练价值；
- reward 是否在 group 内有区分度；
- 哪些错误来自模型动作，哪些来自环境基础设施；
- rollout 使用的 token 是否和训练 loss 对齐；
- 训练后的策略是否比训练前更好；
- 不同环境、任务和版本之间能否公平比较。

因此需要把看板分成三层：

```text
运行健康性（Operational）
        ↓
数据与训练契约（Data / Training Contract）
        ↓
学习效果与科学有效性（Learning / Evaluation）
```

## 2. 当前已有的数据源

| 数据源 | 当前路径 | 已能观察的内容 | 局限 |
|---|---|---|---|
| TensorBoard | `native_run/tensorboard/` | step、loss、梯度、吞吐等训练标量 | 缺少 episode 级 drill-down 和 reward 分解 |
| rollout dump | `native_run/rollouts/` | ROLL 合并后的 batch/轨迹 | 需要统一 schema 才能跨环境读取 |
| episode events | `native_episodes/*/events.json` | action、tool、error、每步 score | 当前是 Video2Scene 专用结构 |
| training contract | `native_episodes/*/training_contract.json` | token/mask/image 对齐检查 | 只有接入适配器写入，非 ROLL 通用契约 |
| result.json | `native_episodes/*/result.json` | terminal reward、mse、steps | 没有模型版本、数据版本、group 信息 |
| pipeline log | `native_run/logs/`、`native_v10.log` | 初始化、错误、训练阶段 | 原始日志不适合直接作为看板数据 |
| checkpoint | `native_run/.../checkpoint-*` | adapter 和训练状态 | 缺少 checkpoint 与评测结果的绑定 |

## 3. 现状优点

### 3.1 已有可审计的 episode 记录

每条 episode 能保存：

```text
initial.png
reference.png
step_*.png
events.json
result.json
scene.blend
training_contract.json
```

这使得“模型动作 → 环境状态 → reward”的链路可以离线复查。

### 3.2 已有训练输入完整性检查

当前 `training_contract.json` 检查：

- 实际 sampled token 数；
- response mask 覆盖数；
- prompt/response 精确匹配；
- image token 数和 image token in loss；
- multimodal feature 是否存在；
- position id 形状。

这是其他 Agent-RL 环境也应复用的好模式。

### 3.3 已有训练产物证据

v10 记录已经检查：

```text
finite nonzero gradient
nonzero LoRA tensors
checkpoint exists
GPU resources released
```

这比只看 `loss` 是否下降可靠。

## 4. 现状缺口

### 4.1 缺少统一的样本主键

现在同时存在：

```text
traj_id
traj_group_id
env_id
episode directory name
sample_uuid
```

但缺少一个统一的 run-level lineage：

```text
run_id → global_step → group_id → episode_id → sample_id → artifact paths
```

没有这条 lineage，TensorBoard 上的一个异常点不能直接跳到对应 action 和图片。

### 4.2 缺少“训练资格”视图

现有看板通常展示 reward 和 loss，却不展示：

```text
response_tokens
loss_tokens
image_tokens_in_loss
invalid/aborted responses
mask drift
exact token match
```

对于多模态 Agent-RL，这些字段比平均 loss 更接近“这批数据是否真的被正确训练”。

### 4.3 缺少 group 级 GRPO 诊断

GRPO 需要组内相对差异，但当前指标没有统一展示：

```text
group_reward_mean
 group_reward_std
group_reward_min/max
 valid_group_rate
 zero_variance_group_rate
```

如果 group 内 reward 全相同，训练可能有梯度但没有有效排序信号。看板必须把这种情况直接标出来。

### 4.4 reward 没有分解

当前 Video2Scene reward 来自终局 RGB MSE，但通用 Agent-RL 还需要区分：

```text
terminal_task_reward
format_reward
tool_success_reward
safety/constraint_penalty
time_penalty
truncation_penalty
```

只显示总 reward 会掩盖“模型动作错误”和“环境执行失败”。

### 4.5 缺少错误归因

需要把失败拆成至少四类：

```text
model_invalid_action
parse_error
tool_execution_error
environment_timeout
```

当前 `events.json` 有部分 error 信息，但没有统一枚举、错误率分母和 group/step 维度。

### 4.6 缺少训练前后评估

当前 v10 是训练机制 smoke test，不是学习效果实验。看板还缺：

- fixed validation prompts；
- held-out environment/task split；
- pretrain checkpoint 与 post-train checkpoint 对比；
- success rate、平均步数、工具成功率；
- reward 置信区间和按任务分桶结果。

### 4.7 历史运行混在同一目录

`native_run` 和 `native_episodes` 包含多个历史尝试。若看板按目录全量扫描，会把旧失败与 v10 混合，造成错误结论。每次运行必须有独立 `run_id` 和 manifest。

## 5. 现状看板应如何解释

建议把状态分成三种，而不是把“训练成功”设成一个布尔值：

| 状态 | 代表含义 |
|---|---|
| `pipeline_healthy` | 进程、GPU、Ray、rollout、checkpoint 链路正常 |
| `data_trainable` | token/mask/group/reward 契约通过，样本有有效训练信号 |
| `learning_validated` | 经过固定评测，训练后策略优于训练前或 baseline |

当前 Video2Scene v10 可以标记为：

```text
pipeline_healthy: true
data_trainable: true
learning_validated: false
```

## 6. 当前看板的最小改造建议

在不改 ROLL trainer 的情况下，先增加一个统一的 episode/step JSONL 导出器，所有环境都写以下最小字段：

```json
{
  "run_id": "...",
  "global_step": 0,
  "traj_group_id": "...",
  "traj_id": "...",
  "env_name": "...",
  "task_id": "...",
  "turn": 0,
  "prompt_tokens": 1234,
  "response_tokens": 87,
  "loss_tokens": 80,
  "reward_total": 0.7,
  "reward_components": {"task": 0.7},
  "action_status": "valid",
  "tool_status": "success",
  "terminated": false,
  "truncated": false,
  "training_contract": "pass",
  "artifact_uri": "..."
}
```

TensorBoard 继续保存高频标量，JSONL 负责 episode 级分析，二者通过 `run_id/global_step/traj_id` 关联。
