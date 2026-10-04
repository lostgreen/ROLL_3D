# Video2Scene 与 ROLL 接入

> 历史说明，已退出当前 CLI。这里的路径、命令和验证只描述整理前的 smoke；当前实现见 [主 README](../../README.md)。复现历史运行应使用记录的 commit，而不是直接执行本归档。

统一入口为 `python v2s_integration/run_rollout.py`，支持 `collect / batch / probe / summarize / train-smoke`；`study` 是 `batch` 的兼容别名。四组工具、完整建模、128K 配置及目录职责见 [ROLL 接入层](roll_adapter/README.md)。`harness/` 提供工具执行器；`roll_adapter/` 连接它与 ROLL，旧 `behavior_study/` 只保留 import 兼容层。

`collect/batch` 调用当前多工具环境；`train-smoke` 调用较早的材质编辑训练，等价于 `launch_pipeline.py`，尚未复用当前环境的训练打包。命令与目录已经统一，历史协议与当前协议的训练能力没有混同。ROLL 原始架构、官方自定义 Agent 案例与接入契约见 [10.14 图文讲解](../weekly_report/1014/doc/roll_format_walkthrough.md)。

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
spans use their lengths and check sampled response equality against the final
trajectory. Prompt equality is recorded but not enforced by the legacy packer;
the current Qwen3.5 audit found mismatching prefixes. Consequently this is not a
full train/inference parity guarantee. Template-only EOS tokens that were never
sampled are excluded from policy loss.

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

The current ProxyEnvManager/AgentRunner route constructs text token IDs and 1D
positions, without the visual collator path used here. It needs multimodal
integration before it can reuse the existing visual agent loop. The KML checkout
HEAD verified on 2026-10-04 is `118c7af36e7bad20e57a84d7a340b4f4cef2b769`;
the earlier pin above describes the historical smoke only.

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
