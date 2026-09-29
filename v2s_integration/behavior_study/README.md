# 场景复刻 rollout 适配层

这个 Python 包连接 ROLL 推理和已有 Video2Scene harness。它不复制 Blender、混元或资产检索工具的实现，也不是新的虚拟环境。

## 为什么仍需要适配层

`harness/src/harness/agent.py::run_agent()` 自带模型调用和对话循环；直接运行它可以做 harness 实验。当前采集要求使用 ROLL 的 `Cluster / PolicyProxy / VLTrajEnvManager`、真实多模态 token 计数和逐请求采样 seed，因此由 ROLL 驱动循环，并复用 harness 的 `RunSession / ToolExecutor / providers`。不能在 ROLL 的一步里再调用整个 `run_agent()`，否则会嵌套两套循环。

四组工具权限、输入任务和预算是配置；ROLL 的 token、图像和生命周期接口需要代码适配。`behavior_study` 保留为内部模块名，统一命令入口是上一级 `run_rollout.py`。

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 原工具环境 | `../harness/` | Blender MCP、混元、资产检索、执行器与工具预算 |
| 环境适配 | `runtime.py`, `env.py`, `actions.py` | 复用工具注册、schema 校验、reset/step/close 与错误反馈 |
| 模型输入 | `inputs.py`, `manager.py`, `seeded_proxy.py` | 对话、图像、真实 token 预算、采样 seed、ROLL 推理 |
| 采集调度 | `demo.py`, `run_study.py`, `resources.py` | 单组采集、四组并行、GPU 快照与进程收尾 |
| 后端与分析 | `isolated_hunyuan.py`, `backend_probe.py`, `summarize.py` | 独立依赖、真实工具探针、离线行为统计 |
| 任务设计 | `system.txt`, `task.txt`, `tool_guidance.py` | 模型实际收到的任务描述、系统提示与调用格式 |

## 从一个入口运行

从仓库根目录，使用已经安装好 ROLL 与工具依赖的 Python：

```sh
python v2s_integration/run_rollout.py --help
python v2s_integration/run_rollout.py collect --help
python v2s_integration/run_rollout.py study --help
```

单组参数化采集（`task.json` 基于 `task.example.json`，修改其中 `group`）：

```sh
python v2s_integration/run_rollout.py collect \
  --repo /absolute/path/to/ROLL_3D \
  --model /absolute/path/to/model \
  --task /absolute/path/to/task.json \
  --output /absolute/path/to/new_run \
  --gpu 0 --episodes 8 --seed-start 42 --steps 40 \
  --context 131072 --output-tokens 8192
```

四组实验使用 `study.example.json` 的本地副本。填入主机名、模型、Python、Blender、CUDA 库、资产索引和混元路径；`source_task` JSON 至少提供 `initial_blend` 和 `reference_image`。该入口先验证真实后端链路，再做短上下文预检，然后四组并行：

```sh
python v2s_integration/run_rollout.py study \
  --config /absolute/path/to/study.local.json \
  --output /absolute/path/to/new_study --detach
python v2s_integration/run_rollout.py summarize \
  --root /absolute/path/to/new_study/groups \
  --output /absolute/path/to/summary.json
```

当前调度仍要求单机八张 B300 全部可见：GPU0–3 分别运行四组，GPU4 供混元生成。每组内部 episode 串行，不是八条并发；没有宣称组内批量调度已接入。`expected_host` 可配置以核验部署目标。`episodes / seed_start / steps / context / output_tokens` 均由配置传入。单组入口不自动创建 headless 渲染补丁；无 EGL 的节点应使用四组入口准备的运行副本，或设置 `V2S_HARNESS_ROOT` 指向已验证的副本。

## 四组工具和模型输入

- `blender`：完整 `execute_blender_code`，可以创建、删除、建模和修改材质，不限于改色。
- `hunyuan`：图像生成资产、导入、变换、查询和渲染；隔离后端当前为 shape-only。
- `assets`：检索、检查、导入、变换、查询和渲染。
- `mixed`：三类能力一起开放，模型自行选择。

各组共享参考图裁剪、场景查询、变换、渲染和结束工具。模型输入包含真实 reference、工具 schema、近期场景和工具图以及历史调用/反馈。文本历史先完整保留，超过实际 token 预算后按完整交互对移除；输出预留独立预算。原生 Qwen 工具格式与显式 JSON 均由 parser 校验。

初始 blend 提供公开相机和灯光，reset 清空其他几何体；不要传入隐藏答案。单视角任务不会被复制成假多视角。任意 Blender Python 不是安全沙箱，相机/灯光保护只检查指定属性；错误执行可能留下部分场景修改。

## 证据和环境

每条 episode 保存 manifest、schema、实际请求、生成记录、逐步工具结果、最终结果与场景文件。原始日志和轨迹放在独立 output 目录；本地 `artifacts/`、部署记录、venv 与运维脚本不提交 Git。虚拟环境使用安装路径运行，不应随意搬动；`setup_hunyuan.py` 仅用于显式创建独立混元环境，复用已有 PyTorch，代理和 pip 镜像通过环境变量设置。

`run_study.py` 在每次输出目录创建原 harness/ROLL 的运行副本，记录 CPU 渲染和采样兼容补丁的哈希；不热改已运行的实验版本。历史的单节点 Demo 和端口修复脚本属于部署记录，不是公共入口。

本模块只采集行为，没有优化器更新或有效的场景质量奖励。`finish`、工具成功和运行结束不代表场景重建成功。128K 配置能加载、短请求能执行，也不等价于测满128K。

## 本地验证

```sh
PYTHONPATH=v2s_integration python -m pytest -q \
  v2s_integration/behavior_study v2s_integration/test_action_response.py
```

测试覆盖原工具注册、参数和动作协议、采样 seed、上下文记录、协调器配置与统计，外部 Blender/模型调用使用测试替身。真实后端应另外执行 `run_rollout.py probe --task TASK --output OUTPUT`；这不是模型能力测试。
