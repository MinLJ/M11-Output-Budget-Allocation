# Faster R-CNN 臂交付件

本目录是 M11 跨检测器实验里 **Faster R-CNN 这一臂**的完整交付: 检测器身份、导出协议、
本臂特征构造、分配器资产、25 个条件的原始结果与统计口径。参照臂 (RT-DETRv2-R18VD)
不在这里, 它有自己的交付件。

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.14.5 |
| PyTorch | 2.12.0+cu130 |
| torchvision | 0.27.0+cu130 |
| GPU | 单卡 CUDA (开发机: RTX 4060 Laptop 8 GB) |
| 其他 | pycocotools 2.0.11 (评价)、pandas / pyarrow / joblib (数据与资产) |

检测器训练用 fp16 autocast + GradScaler; **导出与评价一律 fp32**: 官方前向与逐条复刻
必须同精度, 后面的逐位一致性检查才成立。见 `configs/export_config.json` 的
`export_precision`。

---

## 2. 检测器身份 (冻结)

| 项 | 值 |
|---|---|
| detector_id | `FasterRCNN-resnet18-FPN-Road8` |
| 架构 | Faster R-CNN (ResNet-18 + FPN): torchvision `FasterRCNN` 直接构造, 骨干 `resnet_fpn_backbone('resnet18')` (torchvision 0.27.0+cu130) |
| 骨干 | resnet18-FPN, ImageNet-pretrained init (IMAGENET1K_V1), trainable_layers=5 |
| 骨干深度 | 与参照臂 RT-DETRv2-R18VD 对齐, 使**唯一的变量是检测范式**, 不是骨干容量 |
| checkpoint | `best.pth`, epoch 15, sha256 `fa6c83a7b662ccdb…` |
| 类别 | 8 类 Road8, 背景为第 0 类 |

选 `best.pth` 而不是 `last.pth`: 它在 EARLY_STOP 上按最早严格最优轮次选出, 与冻结的选轮
规则一致。训练脚本自己报的 `val_f1_50 = 0.073` 是**贪心近似的悲观值**, 不代表检测器
实际水平; 真实水平看 `results/coco_ap_results.csv`。

---

## 3. 导出协议 (冻结)

主用候选流是 **`PRE_NMS_TOPN100`** (primary candidate stream for the M11 comparison):

- **导出点**: after detection-head classification and box regression, after clipping and background removal, BEFORE the final RoI score threshold / degenerate-box filter / NMS / top-N。
- **分数空间**: softmax over the complete 9-way class space, then the 8 Road8 columns; not renormalised to eight classes。
- **框**: the decoded box of that record's own class (pred_boxes[:, class])。
- **坐标空间**: original image xyxy, via the same transform.postprocess inverse path as the official output。
- **排序**: score descending → source_id ascending → Road8 class id ascending → original (roi, class) order。
- **top_n**: 100。

登记备查流 `POST_NMS_TOPN300` 是官方 `postprocess_detections` 的最终输出,
**逐位等于** `model(images)` 的返回; 它只作对照, 不参与 M11 比较。

**RPN 内部管线未改**: RPN 自己仍有 1000 个 proposal 的上限, 但那是 RPN 的内部预算,
不是输出候选预算 —— 输出候选数是 8000 =
1000 个 RoI × 8 类。§03 要求的真实候选数:

| 指标 | 值 |
|---|---|
| pre-NMS 每图候选数 | min 8000 / median 8000 / max 8000 |
| N_i < 100 的图 | 0 |
| N_i < 50 的图 | 0 |

候选充足性不成问题, §03 的短缺分支不会被触发。

---

## 4. `source_id` 的含义 (重要)

`source_id = "roi{五位数}"` 是 **RoI 在 box_head 输入批次里的行号**, 是稳定的行索引,
不是对象 id。一个 RoI 最多产生 8 条类别记录, 它们**共享同一个 `source_id`**。

- join 只用 `(image_id, source_id)`; 该键在 **native sidecar 内**唯一, 在候选表内**不**唯一
  (上一段说的多条类别记录共享同一个 `source_id` 就是指这个) —— 所以 join 时绝不能顺手去重。
- **绝不允许**按 `source_id` 去重、合并或重排 —— 同一 RoI 的不同类别记录有各自的框。
- native 向量对同一 RoI 的多个类别记录是共享的, 这是设计, 不是重复。

原生特征 schema: `frcnn_r18fpn_road8_roihead1024_logits9_v1`, 存 **10 项**:
`native_vector` 1024 维 (box_head 输出) + `class_signal` **9 列** (pre-softmax, 含背景)。

存 9 列而不是 8 列, 是因为少了背景列 softmax 的分母就不完整, `score = softmax(logits)[class]`
就无法**正向**重建并逐条校验 —— 而用 score 的逆 sigmoid 冒充 logits 是合同明令禁止的。
特征构造时只用 1..8 列, 与参照臂 8 列 block 的宽度对齐 (见 `code/features_frcnn.py`)。

---

## 4b. 输入规模

| 切片 | 图像 | 主用候选流记录 |
|---|---|---|
| TRAIN10K | 10000 | 1000000 |
| DEV2K | 2000 | 200000 |

TRAIN10K 按冻结角色切成 FIT / CALIBRATION / EARLY_STOP; DEV2K 只用于评价,
**从不参与调参**, 按冻结的 50 个 40 图组使用。两个切片都不在做完导出后改动。

---

## 5. 特征与资产

| 项 | 值 |
|---|---|
| 输入维度 | 90 (58 候选 + 12 前缀 + 20 图像上下文), 与参照臂同配方 |
| PCA | 32 维, **仅用 FIT 拟合** |
| scaler | **仅用 FIT 拟合** |
| 输出维度 | 10 = 10 个 IoU 阈值 |
| seed | -1, 730101, 730102, 730103 —— **三个独立训练**, 不是集成 |
| asset_id | `56b293f30948a1cb1ad50f913f9c0b13…` |

PCA 与 scaler 是**本分支用本检测器的 native 向量重新拟合**的, 不复用参照臂的。理由是
两个检测器的 RoI 特征分布不同, 复用会让"跨检测器"退化成"同一套预处理下的再训练"。

---

## 6. 复现顺序

```bash
# 0) 前置: 检测器已训练 (frcnn_road8/outputs/.../best.pth)
cd "$M11_EXPERIMENT_ROOT/frcnn_road8"
./env/Scripts/python.exe -B src/train_frcnn.py            # 若已有 best.pth 可跳过

# 1) 导出候选 + native (两流一次前向, 每图逐位校验; DEV 与 TRAIN 都要)
./env/Scripts/python.exe -B src/export_frcnn.py \
    --checkpoint outputs/frcnn_r18fpn_road8/best.pth --split DEV2K   --out outputs/frcnn_r18fpn_road8/export_dev
./env/Scripts/python.exe -B src/export_frcnn.py \
    --checkpoint outputs/frcnn_r18fpn_road8/best.pth --split TRAIN10K --out outputs/frcnn_r18fpn_road8/export_train

# 2) 转成 M11 输入格式
# frcnn_m11 不带自己的 venv; 步骤 2-5 统一用 frcnn_road8 的解释器
cd "$M11_EXPERIMENT_ROOT/frcnn_m11"
../frcnn_road8/env/Scripts/python.exe -B lc_frcnn/prepare_inputs.py

# 3) 合同要求的候选/native 小样本 QA 与 §04 小样本闭环
../frcnn_road8/env/Scripts/python.exe -B lc_frcnn/qa_small_sample.py
../frcnn_road8/env/Scripts/python.exe -B lc_frcnn/qa_closed_loop.py

# 4) adapter 配置 (从步骤 0/1 的真实产物生成, 不手填; 不依赖步骤 5 的流水线产物)
../frcnn_road8/env/Scripts/python.exe -B lc_frcnn/make_adapter_config.py \
    --declaration ../frcnn_road8/outputs/frcnn_r18fpn_road8/detector_declaration.json \
    --export-config ../frcnn_road8/outputs/frcnn_r18fpn_road8/export_dev/export_config.json

# 5) 全流程 (标签/特征/训练/校准/资产/分配/评价/bootstrap/交付打包)
#    步骤 5 末尾的交付打包会读取步骤 4 生成的 adapter_config/ 三件套 —— 干净树上必须先完成步骤 4
../frcnn_road8/env/Scripts/python.exe -B lc_frcnn/run_pipeline.py --stage all
```

`--stage` 可单独重跑任一阶段; 各阶段幂等, 已有产物会跳过 (删掉对应文件即可强制重算)。

**依赖说明**: `delivery` 阶段 (即 `--stage all` 的最后一步) 会把步骤 4 生成的
`adapter_config/` 三件套 (`faster_rcnn_adapter.json` / `capabilities.json` /
`lc_alloc/adapters/faster_rcnn.py`) 拷进交付包。干净树上若未做步骤 4 直接跑
`run_pipeline.py --stage delivery`, 打包会报 `FileNotFoundError: required deliverable missing`
—— 先执行步骤 4 即可 (它只读步骤 0/1 的产物, 不依赖流水线)。

---

## 7. 结果口径 (§06)

**核心汇总 = K10/15/20 等权平均**, 不是单个预算, 也不是 K5/K50 端点:

表内数字统一 **×100**: AP 是 0–1 比例的百分点, Coverage/QUALITY 是逐图匹配计数/计分均值
(原值本来就可 >1, 如首行 Coverage=3.0745) 的 ×100。同列两数相减得到的是**百分点或计数均值差**,
不是相对增幅:

| method | Coverage (K10/15/20 均值, 原值×100) | QUALITY (原值×100) | AP (%) |
|---|---|---|---|
| M11 | 382.33 | 229.76 | 3.56 |
| S_ADAPT | 266.52 | 153.87 | 1.59 |
| S_FIXED | 290.67 | 167.41 | 1.95 |

`results/` 下的 CSV / parquet 里只有 **AP/AR/recall (含逐类 `coverage_recall`)** 存 **0–1** 原值;
**Coverage/QUALITY 存逐图匹配计数/计分均值的原值** (`coverage_total`/`quality_total` 为计数总量,
可 >1, 如 `main_results.csv` 首行 Coverage=3.0745、QUALITY=1.8822), 不是 0–1 比例;
只有这张展示表对它们做了 ×100。
引用时要说清是百分点还是相对增幅: 例如 AP 从 0.0204 到 0.0283 是 **+0.79 个百分点**,
不是 +38.7 个百分点 (+38.7% 是相对增幅)。

配对 bootstrap: 以**完整 40 图组**为单位配对 M11 − S_ADAPT 的逐图差值, 5000 次重采样,
固定 seed=530002, 所有行**共享同一组重采样索引**; 区间是 2.5% / 97.5% 分位数。三 seed
汇总行 (`seed_summary=three_seeds_averaged_per_group`) 先把三个 seed 在每组上取均值
(不把 seed 当新增图像) 再重采样; §06 还要求**保留每个 seed 的点估计和区间**, 因此另有一批
逐 seed 行 (`seed_summary=per_seed`, `training_seed` 列 = 730101/730102/730103), 每个 seed
只在**自己的 50 组配对差值**上单独重采样 —— 三个区间端点**不平均**。两种行的 `seed` 列都是
重采样种子 530002。逐预算行与 K10/15/20 等权汇总行 (budget_summary=`K10_15_20_equal_weight`)
同在 `results/bootstrap_results.csv` (§08 点名文件), K10/15/20 等权行 (三 seed 汇总 + 逐 seed)
另存一份于 `results/bootstrap_core_summary.csv` (两处同一批数字)。

- 逐 seed 在全 DEV 上算 AP, **不平均各 40 图组的 AP**。
- 评价分数**始终是原 detector score**, 不重排、不改分。
- 有 GT 而 AP = −1 → 记 0.0; 没有有效 GT → 标 missing, 两者不混。
- 每条件 = (method, seed, budget), 共 25 行。方法集合: M11, S_ADAPT, S_FIXED。
  其中 `S_FIXED`/`S_ADAPT` 的 seed=−1 是确定性基线, 其余都是 M11 的三个独立 seed。

主结果表 `results/main_results.csv` = 25 行 × 预算 [10, 15, 20, 30, 40]。

---

## 7b. NMS 协议与耗时边界 (§07)

主比较是 **`PFX_EXACT`**: 固定候选 → 原序前缀分配 → 评价。它的最终记录数**精确等于**
组预算 (每图 rank ≤ K_i, 每组 sum(K_i) = 40 × budget), 没有任何后置删除。

另附参考协议 **`PFX_THEN_NMS`**: 固定候选 → 前缀分配 → 对选集追加 NMS → 评价。
仅 NMS 前精确; NMS 后条数**另报**, 不改动主结果表。§08 规定的那张计数表是
`results/post_nms_counts.csv` (逐 method/seed/budget/image: `n_before` / `n_after` /
`removed_count` / NMS 后保留的 record IDs), 同内容的逐组汇总见
`results/pfx_then_nms_by_group.csv` (每组的图数、记录总数、前后条数); 逐条件的全体均值/最小值/最大值
与空输出图数见 `results/pfx_then_nms_by_condition.csv` (同批数值亦在
`results/pfx_then_nms_summary.json` 的 conditions 中)。不变量 `removed = n_before − n_after` 与
`n_after ≤ n_before` 已逐行核验, **不回填槽位**。

两条必须守住的边界:

- **PFX_THEN_NMS 不能证明 NMS_THEN_PFX 也有效。** 后置 NMS 删掉重复记录后, 不会给其他图像
  补回原先失去的预算; 相对次序保留不等于已排除重复候选的作用。这是对处理顺序的逻辑解释,
  **不是已完成的因果实验**。本交付件不含 `NMS_THEN_PFX` 的任何数字。
- **IoU 不套用 YOLO 参考里的 0.70。** §07 明确说 0.70 只沿用 YOLO 既有参考, 不自动作为
  Faster R-CNN 的 NMS 参数。这里用的是**本检测器冻结声明里的那个阈值**, 从实际实现记录,
  没有通过 DEV 质量搜索。

**耗时: 未测。** 本交付件不提供任何 detector-only latency 或端到端加速声明。导出的墙钟时间
包含解码、resize、后处理与磁盘写入, **不能**冒充检测器延迟; 也没有做 warm-up 次数、
GPU 同步点、分项 median/p95 的计时设计。需要效率结论时另立实验实测。

---

## 8. 边界 (本交付件**不**声称的事)

- 不含参照臂 (RT-DETRv2) 的结果; 跨检测器结论要看两臂的合表。
- 不含原始大数据集与虚拟环境; 候选/native 见 `references/README_CN.md`。
- `manifest.csv` 里的 SHA256 只用来**识别文件**, 不替代科学正确性验证。
- 检测器只在 Road8 的 9k 图训练集上训练: 骨干以 ImageNet `IMAGENET1K_V1` 权重初始化,
  RPN 与检测头为随机初始化, **没有**用任何 COCO 预训练检测器权重微调 —— 这正是本臂要测
  的东西: 换一个检测范式、换一套在 Road8 上自训练的参数, M11 的选择策略还成不成立。
- **未核实项单列**: 本臂没有跑 `NMS_THEN_PFX`, 没有做阈值搜索, 没有测运行时,
  也没有与 COCO 预训练权重的外部基准对照。这些是**未做**, 不是"做了但没差别"。

### 命名纪律 (§09)

- **适配器代码** (`code/adapter.py`, `code/features_frcnn.py`, 导出链) 是**适配**, 不是训练出来的;
- **分配器 (梯度训练) / PCA / scaler** 是在本检测器的 **FIT 切片**上**重新拟合**出来的;
  **温度**是在本检测器的 **CALIBRATION 切片**上按 §05 逐 seed 拟合的 (拟合集见
  `assets/temperatures.json` 的 `fitted_on`), **不在** FIT 切片上拟合。
- 这两者分开命名, **不统称"重新训练 adapter"**。`assets/asset.json` 把两边的身份分别登记。

### 可采用的表述 (§09)

> 本分支采用检测器专用候选与原生特征适配, 并重新拟合分配器。结果按该固定候选流内
> 相对共同基线的差异报告。

**在全部结果核验完成前, 不补入"有效""领先""验证通过"等结果词。** 本交付件只报差异与区间,
不下结论。已有开发结果不继承主线 TEST 确认身份, 也不能推出 detector-agnostic 或 SOTA。
