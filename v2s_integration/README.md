# Video2Scene 与 ROLL 接入

当前场景复刻采集入口是 `python v2s_integration/run_rollout.py`，支持 `collect / study / probe / summarize`。四组工具、完整建模、128K 配置及目录职责见 [场景复刻适配层说明](behavior_study/README.md)。`harness/` 提供原工具执行器；`behavior_study/` 是其与 ROLL 的推理及环境接口适配。

下面记录的是较早的材质编辑训练 smoke，与当前场景复刻行为采集是两个入口。`launch_pipeline.py` 仍用于该训练 smoke；运行状态、部署路径和新调试记录保留在 Git 忽略的本地产物中。

# Legacy material-edit training smoke

This directory contains the scripts deployed into `/home/xuboshen/zgw/ROLL/v2s_integration`.
Latest accepted result: **native_v10 completed two native training iterations with nonzero finite gradients and saved LoRA weights**. See `result.md` and `state.md` for evidence and limits.

The ROLL checkout is pinned at `192b1a01ea61c113b2deb543f7b115783038dff8`.
The original KML harness and its Python environments are not edited. `bootstrap.py`
copies only the required harness source trees and records a SHA-256 manifest.

## What is exercised

- Real `RunSession` / `ToolExecutor`, bundled MCP server, and a headless Blender process.
- A controlled material-edit task: match an observed reference image by editing a cube and sphere.
- Reset, reference/current images, explicit JSON tool calls, rendered feedback, finish, budget, final artifact, and terminal reward.
- Diagnostic reward `exp(-30 * observed_RGB_MSE)` only. This is NOT held-out scene quality and NOT evidence of reconstruction/generalization improvement.
- `launch_pipeline.py`: native ROLL AgenticPipeline, VLTrajEnvManager subclass, FSDP2 LoRA trainer and vLLM inference; two requested optimizer iterations.
- `debug_single_process.py`: separate diagnostic fallback using actual ROLL multimodal collator, DataProto, GRPO reward normalization/advantages and loss aggregation with HF+LoRA. It is NOT the native distributed pipeline.

## Entrypoints on KML

Run from the isolated integration directory, with `PYTHONDONTWRITEBYTECODE=1`.

```bash
/home/xuboshen/zgw/tools/blender/blender -b --threads 4 --python-exit-code 1 \
  --python prepare_task.py -- /home/xuboshen/zgw/ROLL/v2s_integration/task
/home/xuboshen/Anaconda/envs/video2scene/bin/python v2s_env.py
venv/bin/python launch_pipeline.py --config-only
timeout 1200 venv/bin/python launch_pipeline.py
venv/bin/python audit.py
```

Use the KML detached-job helper for actual execution. Do not run GPU work through
the short-command bridge. Preserve per-attempt log and PID paths; aggregate errors
instead of loading raw logs. Repeated native runs currently use the same
`native_run` directory; archive/rename it before a fresh scientific experiment.

The component fallback creates `component_run` with `exist_ok=False`; explicitly
choose a new output directory in a retry rather than overwriting an earlier run.

## Acceptance and limits

The deterministic environment test is a contract test, not an agent rollout.
Learned-agent rollouts must be separately inspected for parsed tool actions,
actual edits, reward variation, and infrastructure failures.

The native manager writes `training_contract.json` per episode and refuses
training if generated tokens change during reconstruction, images are absent,
or image placeholders enter the loss mask. Check finite gradients and a real
parameter update/checkpoint before calling RL smoke successful. A saved checkpoint
does not by itself prove policy improvement or successful checkpoint resume.

The manager records every actual inference prompt and sampled response. Loss
spans use their lengths, with exact prefix/response equality checks against the
final trajectory. This excludes chat-template EOS tokens that were never sampled
when a response hits its token limit. Per-turn equality failures remain fatal.

## Integration interfaces

ROLL owns policy inference and optimization; the harness owns tool execution.
The environment does not call the existing `run_agent()` loop internally.

| Interface | Current use |
| --- | --- |
| gem registration + `reset(seed)` | Return reference/current images and task instructions, with a fresh Blender session |
| `step(action)` | Parse one tool call, invoke RunSession.executor, render feedback, return observation/reward/terminated/truncated/info |
| `close()` | Stop the episode's MCP client and Blender process |
| `VLTrajEnvManager` | Format multimodal turns, call policy generation, construct training trajectories |
| `DataProto` | Carry tokens, attention/response masks, multimodal features, positions and scores |
| `AgenticConfig` / `AgenticPipeline` | Configure grouped rollouts, GRPO, optimization and checkpoints |
| train/infer strategy configuration | FSDP2 LoRA trainer and vLLM inference on separate GPUs |

At the pinned commit, the alternative ProxyEnvManager/AgentRunner route does not
preserve the visual inputs required here. It needs further multimodal integration
before it can safely reuse the entire existing agent loop.

Read `state.md` for current progress; earlier failed logs are stale once a newer
attempt supersedes the same failure.

## Upstream integration fixes in the new ROLL clone

- `broadcast_diagnostic.patch`: surface early receiver RPC exceptions before the
  FSDP2 sender blocks in a collective. This exposed a formerly hidden error.
- `vllm_rpc_types.patch`: send dtype strings and shape lists to vLLM V1 RPC;
  its serializer rejects `torch.dtype`. The existing receiver resolves dtype
  strings, so no insecure pickle fallback is needed. Apply with `git apply --recount`.

The isolated venv was pinned to vLLM 0.10.2 from ROLL's torch2.8 requirements.
The source conda environment's custom vLLM installation remains unchanged.

## Token and action compatibility

The adapter is currently specific to Qwen ChatML. It restores actual sampled
assistant token IDs in both training input and vLLM input, recomputes multimodal
positions, and excludes template-only EOS from policy loss. Generated message
boundary tokens and empty/aborted generations are rejected. The new-clone
`vllm_prompt_contract.patch` compares actual vLLM expanded prompts against the
training-side IDs for requests carrying the V2S contract marker.

The material smoke requests plain JSON supported by the unchanged harness parser,
with at most two short existing-material assignments. A neutral-grey syntax
example is provided; target RGB values are not supplied. This is a constrained
protocol smoke, not a general scene-editing benchmark.

`check_token_contract.py` tests masks, context/response drift rejection, token
restoration and reserved message-boundary rejection. These synthetic tests do
not alone prove multimodal engine parity; native runs also enforce the vLLM
expanded-input assertion.
