# Current Refactor

- Goal: one reconstruction Env and native ROLL Manager for collection/training,
  organized by function; archive material smoke and unused agent loops.
- Current evidence: VLTrajEnvManager provides the visual collator and native
  rollout lifecycle; StepEnvManager supports one training row per decision.
  PolicyProxy returns prompt/response IDs, masks, positions and logprobs together.
- Decision: keep these native outputs instead of rebuilding an episode string.
  Preserve model templates/tools and the existing no-budget prompts.
- Training target: sampled reasoning and action tokens, explicitly configured.
  A configured reward function is required for training; collection allows none.
- Completed: functional directories; one ReconstructionEnv/Manager; native
  AgenticPipeline launcher and exact decision packing; 28 retained vendor files
  (previously 74). Old smoke/model loops archived, old CLI/import aliases removed.
- Latest check: 63 CPU/interface tests passed using Torch 2.8.0 and TensorDict
  0.9.1 in /tmp/v2s_contract_env. Tests execute native ROLL postprocess_generate;
  transport/base lifecycle/model/Blender IO use substitutes. No current failure.
- Config-only builds the current manager/target/reward configuration; it does
  not validate the complete native dataclass or initialize any worker.
- Final checks: current CLI help/config build and document build pass; 43
  retained vendor/historical file hashes match HEAD. ROLL core is unchanged.
- Files/artifacts: README.md, docs/architecture.md, weekly_report/1014/doc/
  v2s_architecture_and_references.html, images/14_v2s_unified.png.
- Next: implement a valid scene-quality reward, verify engine visual expansion
  and worker initialization, then inspect real loss/gradient/parameter updates.
  Action-only masking and action loss/engine-version metadata remain unimplemented.
- Constraints: preserve the user's .gitignore and recorded experiments; no remote
  deployment, model sampling, GPU training, or optimizer update in this refactor.
