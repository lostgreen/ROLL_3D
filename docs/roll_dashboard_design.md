# 通用 Agent-RL 看板设计

## 1. 设计目标

看板服务于所有使用 ROLL 做 Agent-RL 的环境，包括数学、代码、网页、游戏、机器人和三维场景。它需要同时服务三类人：

```text
训练工程师：进程和资源是否正常？
算法研究者：这批 rollout 是否提供了有效 GRPO 信号？
任务研究者：策略是否真正学会了目标任务？
```

设计原则：

1. ROLL 负责产生统一的 batch 和训练指标，环境适配器负责补充 action、tool、reward component 和 artifact。
2. 高频指标进入 TensorBoard；episode/turn 明细进入 JSONL 或 Parquet；图片、视频、场景文件只保存 URI。
3. 所有数据必须能沿 `run_id → step → group → trajectory → turn` 回溯。
4. 把运行健康、数据可训练性和学习有效性分开判断。

## 2. 统一数据契约

### 2.1 Run manifest

每次实验根目录必须有 `run_manifest.json`：

```json
{
  "schema_version": "agent_rl.v1",
  "run_id": "20260924-v2s-001",
  "framework": "ROLL",
  "roll_commit": "192b1a0",
  "algorithm": "grpo",
  "model": "Qwen2.5-VL-7B",
  "env_name": "Video2Scene",
  "reward_version": "rgb_mse_v1",
  "config_uri": "native_config.yaml",
  "dataset_uri": "...",
  "start_time": "...",
  "seed": 42
}
```

### 2.2 Episode record

`episodes.jsonl` 每行一条 episode：

```json
{
  "schema_version": "agent_rl.v1",
  "run_id": "...",
  "global_step": 0,
  "group_id": "0",
  "traj_group_id": "Video2Scene_0_0_42",
  "traj_id": "Video2Scene_0_0_42_3",
  "env_id": 3,
  "env_name": "Video2Scene",
  "task_id": "material_edit_smoke",
  "seed": 42,
  "split": "train",
  "turn_count": 2,
  "episode_reward": 0.70,
  "reward_components": {
    "task": 0.70,
    "format": 0.0,
    "tool": 0.0,
    "penalty": 0.0
  },
  "terminated": true,
  "truncated": false,
  "valid_for_training": true,
  "artifact_uri": "native_episodes/..."
}
```

### 2.3 Turn record

`turns.jsonl` 每行一个环境交互 turn：

```json
{
  "run_id": "...",
  "traj_id": "...",
  "turn": 1,
  "observation_types": ["text", "image"],
  "prompt_tokens": 1820,
  "response_tokens": 91,
  "loss_tokens": 91,
  "image_tokens": 576,
  "image_tokens_in_loss": 0,
  "action_status": "valid",
  "tool_name": "execute_blender_code",
  "tool_status": "success",
  "parse_error": null,
  "environment_error": null,
  "reward": 0.0,
  "response_mask_valid": true,
  "artifact_uri": ".../step_1.png"
}
```

### 2.4 Group record

`groups.jsonl` 每行一个 GRPO group：

```json
{
  "run_id": "...",
  "global_step": 0,
  "traj_group_id": "...",
  "group_size_expected": 4,
  "group_size_actual": 4,
  "valid_trajectory_count": 4,
  "reward_mean": 0.54,
  "reward_std": 0.21,
  "reward_min": 0.20,
  "reward_max": 0.88,
  "zero_variance": false,
  "advantage_mean": 0.0,
  "advantage_std": 1.0
}
```

### 2.5 Training batch contract

每个 `DataProto` batch 生成一个摘要，不保存完整 tensor 到看板：

```json
{
  "run_id": "...",
  "global_step": 0,
  "batch_size": 4,
  "sequence_length": 4096,
  "response_tokens": 740,
  "loss_tokens": 732,
  "prompt_tokens": 6540,
  "image_tokens": 2304,
  "image_tokens_in_loss": 0,
  "mask_drift_count": 0,
  "empty_response_count": 0,
  "aborted_response_count": 0,
  "contract_pass": true,
  "loss_mask_keys": ["response_mask"]
}
```

## 3. 看板页面

### 页面 A：Run Overview

回答“这次 run 是否健康”：

```text
run status / elapsed time / current global step
GPU utilization / memory / Ray workers
rollout throughput / train throughput
checkpoint latest / checkpoint size
pipeline_healthy / data_trainable / learning_validated
```

建议图表：

- global step 时间线；
- rollout、train、checkpoint 延迟堆叠图；
- GPU 显存和利用率；
- worker error rate；
- 最近一次失败 fingerprint。

### 页面 B：Rollout Quality

回答“模型在环境中实际做了什么”：

- episode reward 分布；
- 每步 reward 曲线；
- 平均 episode 长度；
- terminated / truncated 比例；
- action valid rate；
- tool success rate；
- parse error、timeout、environment error 堆叠图；
- 按 `env_name/task_id/split` 分桶。

点击任意点后跳转：

```text
run_id → global_step → traj_group_id → traj_id → artifact_uri
```

### 页面 C：GRPO Signal

回答“这批数据是否能提供相对学习信号”：

- group reward mean/std/min/max；
- zero-variance group rate；
- group size expected/actual；
- valid trajectory rate；
- advantage mean/std/clip rate；
- reward normalization 前后分布；
- 各环境 tag 的 group 数量。

推荐警报：

```text
zero_variance_group_rate > 30%
valid_group_rate < 80%
advantage_std < 0.05
advantage_clip_rate > 20%
```

### 页面 D：Training Contract

回答“进入 loss 的数据是否正确”：

- response token 数；
- loss token 数；
- response mask 覆盖率；
- image token in loss；
- exact token match rate；
- prompt/response drift count；
- empty/aborted response；
- sequence truncation rate；
- old/ref log-prob 缺失率。

推荐警报：

```text
contract_pass_rate < 99%
image_tokens_in_loss > 0
mask_drift_count > 0
empty_response_rate > 1%
sequence_truncation_rate > 5%
```

### 页面 E：Learning and Evaluation

回答“训练是否有效”：

- pre vs post checkpoint 的固定评测；
- train / validation / held-out split；
- task success rate；
- terminal reward；
- 平均动作数；
- tool success rate；
- reward confidence interval；
- 按任务难度和环境类型分桶。

只有这个页面通过固定评测后，才允许把状态设为 `learning_validated=true`。

### 页面 F：Trajectory Explorer

提供按条件过滤的明细表：

```text
run_id / global_step / env / task / group / trajectory / reward / status
```

详情页展示：

```text
prompt 文本
每轮 action
每轮 reward
tool result
图片/视频/场景 artifact
response_mask 摘要
training_contract
```

不要把模型完整 token tensor 直接塞进网页；展示 token 数、mask 统计和必要的短片段，原始数据通过 artifact URI 下载或查看。

## 4. 通用指标命名

采用三段式：

```text
<domain>/<object>/<metric>
```

例如：

```text
rollout/episode/reward_mean
rollout/action/valid_rate
rollout/tool/success_rate
rollout/group/reward_std
train/data/loss_token_count
train/data/image_token_in_loss
train/grpo/advantage_std
train/actor/grad_norm
train/actor/pg_loss
eval/task/success_rate
system/gpu/memory_used
system/throughput/rollout_samples_per_sec
```

所有指标必须带以下维度：

```text
run_id, global_step, env_name, task_id, split
```

group/trajectory 级数据不建议直接作为 TensorBoard scalar，而应进入 JSONL/Parquet，并由看板按需聚合。

## 5. 当前 ROLL 字段到通用看板的映射

| 看板字段 | 当前来源 |
|---|---|
| `episode_reward` | `non_tensor_batch["episode_scores"]` |
| `step_reward` | `non_tensor_batch["step_scores"]` |
| `traj_group_id` | `non_tensor_batch["traj_group_id"]` |
| `traj_id` | `non_tensor_batch["traj_id"]` |
| `response_tokens` | `response_mask.sum()` |
| `loss_tokens` | `response_mask.sum()` 或训练 batch mask 统计 |
| `image_tokens_in_loss` | `((input_ids == image_id) & response_mask).sum()` |
| `advantage` | `agentic_compute_advantage()` 输出 |
| `grad_norm` | actor train metrics |
| `pg_loss` | actor train metrics |
| `tool_status` | 环境 adapter 的 step/event 记录 |
| `artifact_uri` | 环境 adapter 写入的 episode 路径 |

这说明通用看板不需要修改 ROLL 的核心训练算法，只需要约定环境 adapter 输出事件，并增加一个 batch summary exporter。

## 6. 实施路线

### Phase 1：无侵入导出

在 `EnvironmentManager.formulate_rollouts()` 完成后写出：

```text
run_manifest.json
episodes.jsonl
turns.jsonl
groups.jsonl
batch_summary.jsonl
```

现有 TensorBoard 继续运行，不改变训练逻辑。

### Phase 2：统一错误和 reward schema

约定枚举：

```text
action_status: valid | parse_error | empty | aborted
 tool_status: success | tool_error | timeout | skipped
episode_status: terminated | truncated | aborted | failed
```

reward 统一拆为：

```text
reward_components.task
reward_components.format
reward_components.tool
reward_components.penalty
```

环境没有某类 reward 时填 0，并记录 `reward_version`。

### Phase 3：仪表盘聚合

先用 TensorBoard 展示 step 级指标，再用 Streamlit/Grafana 等读取 JSONL/Parquet 做 episode drill-down。两类页面通过 `run_id/global_step/traj_id` 关联。

### Phase 4：评测门禁

每个实验提交前自动检查：

```text
pipeline_healthy
contract_pass_rate >= 0.99
group_size_actual == group_size_expected
zero_variance_group_rate <= threshold
held-out evaluation exists
pre/post comparison exists
```

## 7. 推荐的最小验收标准

一个新的 Agent-RL 环境接入 ROLL 后，至少应能在看板中回答：

1. 这条 rollout 属于哪个 run、step、group 和 task？
2. 模型生成了多少 response token，其中多少进入 loss？
3. 图像、prompt、padding 是否被正确 mask？
4. tool/action 是成功、解析失败还是环境失败？
5. group 内 reward 是否有方差？
6. advantage 是否有效、是否被大量 clip？
7. actor 是否出现有限且非零梯度？
8. checkpoint 对应哪一次评测？
9. post-training 是否优于 pre-training 和 baseline？

如果第 1 到第 7 项没有数据，这个 run 只能标记为 `pipeline_healthy`，不能标记为 `data_trainable`；如果第 8 到第 9 项缺失，不能标记为 `learning_validated`。
