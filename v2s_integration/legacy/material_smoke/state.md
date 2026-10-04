# ROLL + Video2Scene KML handoff — 2026-09-23

## Goal / completion
User authorized isolated ROLL clone + real harness Agent RL debugging on KML, preserving original code. Native integration smoke is COMPLETE: v10 ran two training iterations, nonzero finite gradients, nonzero LoRA weights and checkpoint. No experiment remains running. This proves mechanics, not policy improvement/generalization. No goal tool or automation was created.

## Current evidence (v10 ONLY)
- Native AgenticPipeline exit0, `DONE native_pipeline`, 350.7469s. Qwen2.5-VL-7B, GRPO group4, two iterations on2xA800.
- 8 completed episodes,10 tool executions,1 tool error. Terminal rewards .242761–.948434, initial reward .615980. No held-out evaluation.
- 8/8 exact sampled-response and full-context contracts; image tokens excluded from loss. Actual vLLM expanded prompt equality asserted against training-side IDs for every V2S generation.
- 7 synthetic regression cases passed before loading model (masks, drift rejection, token restoration, reserved-boundary rejection). They do not alone validate visual features; native engine assertion provides additional input evidence.
- Final step1 grad_norm .12165915966033936 finite; pg_loss -1.4901161193847656e-08 finite; system/samples8. Near-zero normalized GRPO scalar loss is not zero gradient.
- Checkpoint24 files /19,143,872,981 bytes. Adapter contains56 LoRA B tensors, all56 nonzero and finite, max_abs2.001333996304311e-05 (default B initialization zero).
- Source SHA256 audit: all74 manifest-tracked original harness files unchanged. Original conda envs unmodified.
- After exit both GPUs2MiB,0% utilization. Own Ray training workers released.

## Paths and exact version
- Remote official ROLL: `/home/xuboshen/zgw/ROLL`, commit `192b1a01ea61c113b2deb543f7b115783038dff8`.
- Remote integration: `/home/xuboshen/zgw/ROLL/v2s_integration`.
- Local authored source/docs: `research/experiments/roll_integration_20260922/`.
- Original source: `/m2v_intern/xuboshen/projects/video2scene_segmented_texture_20260922/code`; snapshot of src/headless/blender-mcp/metrics/examples plus source_manifest.json.
- Model: `/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct`.
- Blender: `/home/xuboshen/zgw/tools/blender/blender`.
- Adapter: `/home/xuboshen/zgw/ROLL/v2s_integration/native_run/actor_train-0-G0/checkpoint-1/adapter_model.safetensors`.
- Evidence: native_v10.log/pid, native_v10_verification.json, verify_v10.log/pid, native_status.json; native_episodes/*/{events.json,result.json,training_contract.json,scene.blend}; native_run/{tensorboard,rollouts}; requirements_snapshot.txt.

## Architecture / changed files
ROLL owns generation and optimization. The harness executes tools; no nested run_agent call.
- v2s_env.py: gem reset/step/close, copied RunSession/ToolExecutor/MCP/Blender. Reference/current PIL images. Plain JSON function calls,3 actions, at most two short material assignments suggested. Syntax example is neutral grey, not target RGB. Terminal exp(-30 observed RGB MSE). Controlled diagnostic task, not held-out reward or OS sandbox.
- v2s_manager.py + token_history.py: subclass VLTrajEnvManager; keep actual sampled assistant IDs in both expanded training input and unexpanded vLLM input; recompute multimodal position IDs; exact context/response checks; mask only sampled tokens. Pinned Qwen ChatML support. Reserved message-start tokens and empty/aborted responses fail closed; graceful context-budget termination remains future work.
- launch_pipeline.py: native AgenticConfig/AgenticPipeline; group4,2 iterations,3x512 tokens,seq4096; FSDP2 LoRA q/v rank8 alpha16 onGPU0, vLLM onGPU1; save_steps1,no eval; timeout900 used externally. Private Ray with SIGTERM/finally cleanup.
- check_token_contract.py:7 regression checks. verify_native.py: compact attempt-specific source/episode/gradient/adapter audit. audit.py: older broader report.
- bootstrap.py/prepare_task.py: isolated source snapshot and synthetic material task.
- debug_single_process.py: uploaded but NEVER RUN; not used for success evidence.
- New-clone patches: broadcast_diagnostic.patch (surface receiver RPC failures before NCCL sender blocks), vllm_rpc_types.patch (dtype strings/shape lists), vllm_prompt_contract.patch (actual vLLM expanded-input equality). No insecure pickle fallback.

## Environment decisions
Isolated venv --system-site-packages based on `/home/xuboshen/Anaconda/envs/roll-t28`: torch2.8/cu128,transformers4.57.3,Ray2.48. New venv only: TransferQueue0.1.6,opentelemetry-exporter-otlp,official vllm0.10.2 --no-deps. Original custom vllm0.12dev unchanged. Dependencies frozen to requirements_snapshot.txt. ProxyEnvManager at pinned ROLL revision drops needed visual inputs, so use VLTrajEnvManager.

## A800 development machine / B300 training WebShell split — 2026-09-27
- The KML development machine is **2x NVIDIA A800-SXM4-80GB (compute capability 8.0)**. It is for source checkout, downloads, packaging, and A800-side checks. Do not use its environment as evidence of B300 compatibility.
- Its `roll-qwen35` environment currently has `torch==2.5.1+cu121`, `transformers==5.2.0`, `accelerate==1.14.0`, and `roll==0.3.0`; `pip show` reports no vLLM there. Keep this as the A800-side/download environment and use the B300 venv below for the actual B300 model run.
- The training WebShell is **8x NVIDIA B300 SXM6 (compute capability 10.3, 275040 MiB each)**. Its isolated runtime is `/home/xuboshen/zgw/ROLL/v2s_integration/b300_qwen35_venv`.
- B300 runtime check passed with `torch==2.8.0+cu128`, CUDA 12.8, `transformers==5.2.0`, `accelerate==1.14.0`, and a CUDA tensor smoke (`1.0`). Qwen3.5-9B and Qwen3.5-27B are complete at `/m2v_intern/xuboshen/models/Qwen3.5-9B` (4 safetensors shards, about 19G) and `/m2v_intern/xuboshen/models/Qwen3.5-27B` (11 shards, about 52G); both have config and safetensors index files.
- The B300 venv originally imported `roll` from `/home/xuboshen/fy/roll-caption` and `vllm` from `/home/xuboshen/fy/src/vllm-0.12.0-src`; this was the environment-mixing bug. Current `import roll` resolves to `/home/xuboshen/zgw/ROLL/roll/__init__.py` after `pip install --no-deps --no-build-isolation -e /home/xuboshen/zgw/ROLL`.
- Installed the missing B300 runtime dependencies `TransferQueue==0.1.6` and `opentelemetry-exporter-otlp`; `launch_pipeline.py --config-only` now completes with `DONE config validated` in the B300 venv. Validation log: `/tmp/roll_b300_config_20260927.log`.
- B300 model smoke passed with the isolated venv and local files only: `LOAD_OK device=cuda:0 ids=(1, 14)` followed by a short greeting. The same smoke passed for Qwen3.5-9B. The vLLM import remains the pre-existing B300 shared build (`0.12.1.dev0+g4fd9d6a85.d20260516`); do not replace it with the A800-side package without a separate compatibility check.
- The model-load smoke log is `/tmp/qwen35_b300_smoke_20260927/model.log` on the WebShell. The editable-install log is `/tmp/roll_b300_install.log`.

## A800 development-machine asset inventory — 2026-09-27
- The only confirmed scene-level Video2Scene GT pool on the development machine is `/m2v_intern/xuboshen/projects/video2scene_segmented_texture_20260922/scene_pilot_v1`: one table-top scene with four GT instances (`patterned_jug`, `ribbed_vase`, `table`, `woven_basket`), a GT `.blend`, per-instance/surface `.npy`, six rendered views and a public/private camera split.
- The single-object pool contains one `ribbed_vase` reference and one textured Hunyuan GLB under `.../single_object`; it is an asset-generation result, not a multi-scene benchmark.
- Local expansion models are present: Hunyuan3D-2 (22G), Hunyuan3D-2.1 (7.9G), Hunyuan3D-2mini (3.6G), SAM2.1 (857M), rembg (168M), DiffusionGS (2.2G), and InfiniSplat (3.3G). These are generation/segmentation/representation tools, not labeled scene data by themselves.
- `/home/xuboshen/data` has large video pools (`VideoRetrieval`, `video_shards`, `mybenchvideo`, `VisInt` and variants), but they are video retrieval/QA or caption data; no scene mesh, camera, or object-GT contract was confirmed there.
- The harness exposes provider hooks for PolyHaven, Sketchfab, Hyper3D and Hunyuan3D. Only the local Hunyuan path is confirmed usable without another asset-service setup; external providers still need access and license checks.

## Stale evidence / resolved failures
All native_v1–v9 failures/results are historical; do not mix with v10. v5 sender hang/v6 dtype serialization resolved by RPC patch. v7/v8 trajectory checks exposed template-only EOS and actual decode/re-encode drift; fixed with token preservation. v9 completed with all actions invalid, identical rewards and zero gradient: explicitly NOT effective RL success. Plain JSON and short material-edit instructions resolved protocol failure in v10. BasePipeline saves only when global_step>0, so two iterations were required for checkpoint.
Shared native_run and native_episodes include older attempts. v10 verification filters by PID-file start time. Future experiments need fresh run directories; do not overwrite this accepted checkpoint/evidence.

## Next actions (future work; no active run)
1. Preserve v10 outputs; choose fresh output/episode directories before another run.
2. Test checkpoint reload and train/infer parity; successful save is not proof of successful resume.
3. Expand to a small held-out task split and fixed pre/post evaluation before claiming learning improvement; keep reward/geometry tests explicit.
4. Extend action budget/context termination and other chat templates only with exact-token tests; consider independent reward views for real scene tasks.

## Access / operational constraints
Use approved absolute kwai-kml helper; serialized browser bridge, no raw logs/model outputs/tokens/URLs. Long work via start-job; inspect compact job-status or verification JSON. Finalize upload ZIP before serve_upload_file (server caches bytes). No active upload servers or monitoring agent jobs. Original local working-tree edits outside this research directory belong to the user and were untouched.
