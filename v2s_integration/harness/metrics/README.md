# Video2Scene metrics v0.1

独立、离线评估已经冻结的图像、点云和 mesh。所有实现、配置、示例和测试放在本目录，方便单独迭代。当前入口不连接 Blender，不修改场景，不改变旧 harness 的历史计分。

包名为 **`v2s_metrics`**，避免与 `src/harness/metrics.py` 的旧 `import metrics` 冲突。Python >= 3.10。

## 已实现

| 模块 | 指标/能力 | 约定 |
|---|---|---|
| `image.py` | MSE↓、PSNR↑、单尺度 SSIM↑、可选 LPIPS↓ | 同尺寸 RGB；无隐式 resize、曝光拟合或对齐 |
| `views.py` | observed / feedback / hidden 分组，mean / median / std / worst / 最差比例均值 | 缺图/失败保留在分母，报告有效覆盖率；不混合 split |
| `geometry.py` | Chamfer↓、accuracy↓、completeness↓、多个距离阈值的 P/R/F↑ | 世界坐标米制，无 ICP；mesh 按表面积采样 |
| `instances.py` | 一对一匹配、物体 P/R/F1、缺失/多余列表、实例宏平均表面 F、世界/中心对齐形状指标 | 匹配不按 GT 数量调聚类，不强制匹配超距/类别不符对象 |
| `instances.py` | 中心距离↓、AABB IoU↑、AABB 尺寸相对误差↓、可选对称性旋转误差↓ | 明确是 AABB，不冒充 OBB；语义旋转需显式提供 |
| `evaluate.py` / CLI | manifest、参数覆盖、依赖版本、输入 SHA256、严格 JSON、失败状态 | 输入只读；输出原子写入，禁止覆盖输入 |

这是指标计算包，不是完整 benchmark renderer。VLM judge、MEt3R、DCD、voxel IoU、深度/法线、工具成本及训练 reward 暂未实现，后续可作为独立模块接入。LPIPS 需要另行安装依赖和权重；离线测试覆盖缺依赖和后端失败。

## 安装与运行

使用项目中已有的 Python 环境，或自行创建环境；默认依赖不含大模型：

```bash
python -m pip install -r metrics/requirements.txt
# 需要 GLB/OBJ/PLY/STL mesh 输入时：
python -m pip install 'trimesh>=4.0'
# 需要 LPIPS 时，按机器选择兼容的 torch/torchvision，再安装 lpips：
python -m pip install 'lpips>=0.1.4'
```

从仓库根目录运行：

```bash
python metrics/evaluate.py \
  --manifest /path/to/episode/manifest.json \
  --config metrics/config.default.json \
  --output /path/to/episode/metrics-v0.1.json
```

配置优先级：Python 默认值 < manifest 的 `config` < CLI `--config`。不认识的参数会报错；避免拼错参数却继续使用默认值。

退出码：`0` 全部请求项完成计算；`2` 已写报告但有缺图、缺依赖、无效输入等不完整项；`1` manifest/参数/输出错误。**完成计算不代表重建质量合格**：空预测点云会被成功评估为失败结果，F=0；阈值达标率需另行定义。

不依赖模型/Blender的可运行示例：

```bash
python metrics/examples/make_demo.py /tmp/video2scene-metrics-demo
python metrics/evaluate.py \
  --manifest /tmp/video2scene-metrics-demo/manifest.json \
  --output /tmp/video2scene-metrics-demo/result.json
```

示例故意把 hidden 图像横移 4 像素：observed MSE=0、hidden MSE≈0.023228；相同点云 CD=0、Object F1=1。这是合成验收样例，不能当作 Agent 的实验成绩。

## 输入协议

完整模板见 `examples/manifest.json`。按需省略 `views`、`geometry` 或 `instances` 整块，至少请求一块。路径相对 **manifest 所在目录** 解析，可使用绝对路径。

```json
{
  "schema_version": 1,
  "protocol": {
    "units": "m",
    "alignment": "none",
    "render_config_id": "fixed-eval-v1"
  },
  "config": {"distance_thresholds_m": [0.02, 0.05, 0.1]},
  "views": [
    {
      "id": "test-01",
      "split": "hidden",
      "camera_id": "camera-01",
      "prediction": "pred/test-01.png",
      "reference": "gt/test-01.png"
    }
  ],
  "geometry": {"prediction": "pred/surface.npy", "reference": "gt/surface.npy"},
  "instances": {
    "prediction": [{"id": "chair-p", "category": "chair", "path": "pred/chair.npy"}],
    "reference": [{"id": "chair-g", "category": "chair", "path": "gt/chair.npy"}]
  }
}
```

- `camera_id` / `render_config_id` 是调用方对配对条件的声明，**本包不能从 PNG 验证相机/灯光真的一致**。预测和 GT 必须共享固定相机、投影、分辨率、色彩管理、背景及评分光照。旧 `recon_visual_compare` 的独立自动取景尚未在本次改动修正，不能直接将其旧分数等同固定相机协议。
- 实拍 RGB 的光照协议应明确；灰模检查光照与原视频不是同一个 photometric 任务。
- `hidden` 的保密、相机基线和划分需要上游保证。用于动作反馈或训练 reward 的帧使用 `feedback`。
- `.npy` 必须为有限的 `(N,3)` 世界坐标点，`allow_pickle=False`。不会再次重采样或假装点云已经表面积均匀；采样来源由调用方负责。空预测用 `(0,3)`；空 GT 属于无效输入。
- `.glb/.gltf/.obj/.ply/.stl` 作为三角 mesh 读取，应用场景节点变换后按面积采样。PLY 点云请显式转换为 `.npy`，不把顶点当表面。单位转换、Y/Z-up 转换、全局合法配准需上游完成；本包不会猜测。
- 实例点集必须已经处于世界坐标。实例划分来自稳定资产/语义实例，不能把每个 Blender 子 mesh 自动当物体。空实例请从预测列表删除，表现为缺失；不可用一个空文件代替实例。
- 可选 `rotation` 是对象语义局部坐标到世界坐标的 3×3 正交矩阵。GT `symmetries` 是局部对称旋转矩阵列表；默认总是包含 identity。该元数据**不会再次变换点云**。不提供旋转时角度指标为 `not_applicable`。

## 数值定义与限制

**图像。** uint8 RGB 转 [0,1]；浮点数组须已经在 [0,1]。文件允许 RGB/L，透明图像必须先按统一背景显式合成。MSE 为所有 RGB 通道均值，PSNR 为 `-10 log10(MSE)`，上限默认 100dB，相同图像记录 `capped=true`，不向 JSON 写 Infinity。SSIM 为单尺度、uniform window=7、sample covariance、data_range=1，窗口可配置；不是 MS-SSIM。图像太小会单独标记 SSIM 不可用。

**LPIPS。** 配置 `image_metrics` 加入 `"lpips"`，使用 AlexNet、version=0.1、CPU eval、[-1,1] 输入；本协议限制宽高 >=64。默认 `lpips_allow_download=false`，要求 torchvision 的 AlexNet backbone 已缓存；校准权重使用 lpips 包内文件。缺依赖/权重返回 null 和原因，不悄悄下载。只有显式改为 true 才允许模型初始化下载。报告记录库版本；正式实验还应锁定环境及模型权重。

**视角聚合。** 最差 20% 取 `ceil(0.2*N_valid)`；MSE/LPIPS 取最大值一侧，PSNR/SSIM 取最小值一侧。std 是总体标准差，不表示越小质量越好。失败的视角不造出伪分数：输出 `n_expected/n_valid/coverage/failure_counts`，并明确均值为 valid-only。只要不完整就不能仅凭均值排名；主结果应另加成功率/达标率。尚未实现跨场景平均和置信区间。

**表面。** `CD = 0.5*(mean_p min_q ||p-q||₂ + mean_q min_p ||q-p||₂)`，单位米，非平方。accuracy 为预测到 GT，completeness 为 GT 到预测。旧 harness CD 使用双向和，因此旧值不能无版本转换直接混报。F@τ 使用严格 `<τ`，阈值默认 2/5/10cm，仅是起始配置，需按数据噪声校准。空预测 F=0、CD=null；部分扫描必须由上游提供评价区域，不能把未扫描背面当错误。

**实例匹配。** 先按类别（两边有类别时）与最大中心距离门控，使用有 dummy unmatched 的 Hungarian。优先最大化合法匹配数，再最小化 `distance/max_distance + 0.25*(1-AABB_IoU)`。中心是点集包围盒中心，默认门限 1m，需要按场景校准。未知类别允许几何匹配，并在报告标出此策略；推荐提供可靠类别。分类门控不意味着包能够识别资产语义。

Object P/R/F1 表示该门控下的实例存在性，**不要求匹配后的形状质量达标**；应结合对象表面 F 判断重建质量。世界与中心平移后的 shape F 均按 GT 实例宏平均，缺失计 0；额外实例通过 detection precision 惩罚。`centered_shape` 仅去中心，不缩放、不旋转，所以不能把它叫尺度/旋转不变形状指标。AABB 尺寸误差受旋转影响，是诊断，不能冒充语义局部尺度误差。

## Python API

```bash
PYTHONPATH=metrics python -m v2s_metrics --manifest /path/manifest.json --output /path/result.json
```

```python
from v2s_metrics import MetricConfig, compare_images, compare_points, Instance, compare_instances

cfg = MetricConfig(distance_thresholds_m=(0.02, 0.05))
image_scores = compare_images("pred.png", "gt.png", cfg)
geometry_scores = compare_points(pred_world_points, gt_world_points, cfg)
objects = compare_instances(
    [Instance("pred-chair", pred_world_points, category="chair")],
    [Instance("gt-chair", gt_world_points, category="chair")], cfg,
)
```

各标量包含 `value/status/reason/direction/unit/version`。空值不是零；不要在看板里用 `value or 0`。原始值用严格 JSON 输出，per-view/per-object 保留证据，聚合不丢失失败数量。

## 验证与后续

```bash
python -m pytest -q metrics/tests
```

测试覆盖解析值、相同图像、尺寸/透明/范围协议、多视角失败分母与最差方向、严格几何阈值、空预测、面积加权采样与细分稳定性、GLB 变换、缺失/多余实例、最大合法匹配、形状/位置分离、旋转对称、参数与严格 JSON、CLI 输入保护。GLB 测试在未安装 trimesh 时显式跳过。


后续推荐顺序：固定独立 Blender 评分 renderer → 接入冻结 episode artifact → 物体语义实例/lineage → 人工校准的 VLM judge → 成本曲线与更昂贵的多视角指标。旧计分入口暂时保留，避免新旧协议混写同一个字段。

## 对象形状与多视图诊断（独立 v0.2）

新增 `object_quality.py`：逐对象匹配后同时计算世界坐标CD/F、等比缩放到单位立方体的形状CD/F、软位置匹配、缺失/多余列表和尺寸/位置诊断。默认仅做24个upright yaw候选的近似朝向对齐，**不声称任意三维旋转不变或已知语义朝向**。统一缩放保留长宽高比例；world分不做任何对齐。类别来自输入声明，匹配计数不是经过独立验证的语义存在率。归一化形状距离单位为unit_box，不能与米制world CD混报。

```bash
python metrics/evaluate_objects.py --prediction pred/instances.json \
  --reference gt/instances.json --output matched-object-quality.json
```

每个JSON是 `{id,path,category?}` 的列表，path相对JSON；可选rotation必须是可信语义帧。保留旧 `instances.py` 的v0.1字段和含义，不悄悄改动历史分数。

新增 `multiview.py`：GT固定裁剪框的对象外观评分；校准相机和光轴depth下的双向重投影诊断。CV相机约定为right/down/forward，零深度无效；要报告有效覆盖和遮挡剔除，内部一致性高不证明接近GT。crop不对预测单独居中或缩放。

实验运行脚本和结果保存在本地 research/，不属于公开指标 API。


## 单物体与工具证据更新（2026-09-21）

`compare_object_sets()` 新增 `object_set_world`：各距离阈值按 `2 * sum(matched_surface_F) / (n_prediction + n_reference)` 计算质量加权F1，同时惩罚漏物体和多余物体。原 `macro_fscore_missing_as_zero` 不变。两边均空时F1为null；它仍依赖输入实例分组，不是独立语义识别。

五个复杂单物体实验和协议在 `research/experiments/metrics/object5/`，文档在 `research/deliverables/20260921/object5/`。这些是隔离研究脚本，不改变通用metrics CLI的输入合同。GS无有效表面时不计算World/Shape，不用Gaussian中心冒充表面。Depth reprojection仅作诊断。

Hunyuan生成输入的图像blob/有效参数在provider入口记录；轨迹看板依据generation_id关联。输入图、工具返回图和观察者场景图分别标注，缺少旧证据时不猜测补全。

### Blender 曲线表面适配（2026-09-22）

`v2s_metrics/blender_surface.py` 提供可选的 Blender evaluated surface 适配：带厚度CURVE、SURFACE、FONT、META与MESH都先按实际评估几何提取三角面，再按面积采样。不能仅筛MESH，否则图片里的编织结构可能被评为“空物体”。未知几何类型应报告不支持，不能静默记空/0分。当前适配不保证集合隐藏规则、未实现实例及体积/透明表面的统一物理语义。

本轮独立重评分脚本位于 `research/experiments/metrics/object5/export_surface_v2.py` 和 `rescore_surfaces.py`。原quality.json保留；修正版为quality_surface_v2.json。冻结scene.blend与RGB逐字节校验不变，不重跑Agent或VLM。
