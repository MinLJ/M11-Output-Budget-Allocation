# member_yolov8_delivery —— M11 跨检测器验证（YOLOv8n 分支）交付说明

> 交付方：YOLO 分支 · 交付日期 2026-09-29
> 依据：《M11 跨检测器实验指导与交付规范 v1.0》§08 一次性交付格式、表 2（Y1–Y4）、表 7（文件最小内容）
> 本包为**机器可读产物优先**：Word 报告由本包结果生成，不是本包的替代品。

---

## 0. 一句话身份

在 **BDD100K 派生 TRAIN10K/DEV2K** 上，对**冻结的 YOLOv8n**（ultralytics 8.4.77）的
**固定 NMS 前 Top100 候选协议**，比较 M11 输出预算分配与两个基线（S_ADAPT / S_FIXED）。

| 身份字段 | 值 |
|---|---|
| `dataset_id` | `bdd100k_road8_dev2k`（TRAIN 10000 / DEV 2000） |
| `detector_id` | `yolov8n_ultralytics_8.4.77_f59b3d83` |
| `candidate_asset_id` | `yolov8n_bdd100k_road8_top100_v1` |
| `protocol_id` | `PFX_EXACT`（主比较）· `PFX_THEN_NMS`（参考） |
| `native_schema_id` | `yolov8n_detect_head_raw84_road8_scores_v1` |
| `asset_id` | `yolo6cb829be3caf51a0` |
| `allocation_weight_id` | `yolo_class_weights_fit_v1` |
| `evaluation_weight_id` | `yolo_class_weights_fit_v1` |

> 两个 weight_id 本次**同源**（同一 `class_weights.json`），但按规范分列保存于
> `configs/allocation_weights.json` 与 `configs/evaluation_weights.json`，以便日后改用共同权重重评价时替换其一。

---

## 1. 环境

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 11 Home China 10.0.26200 |
| Python | 3.14.6 |
| PyTorch | 2.9.1+cu128（CUDA 可用） |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| ultralytics | 8.4.77 |
| torchvision | （随 torch 2.9.1） |
| pycocotools | 已安装，用于标准 AP/AR |
| scikit-learn / SciPy / NumPy | 1.9.0 / 1.18.0 / 2.5.0 |
| pandas / PyArrow / joblib / Pillow | 3.0.3 / 25.0.1 / 1.5.3 / 12.2.0 |
| 共享方法包 | `LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1/lc_alloc`（脚本内以绝对路径注入 `sys.path`） |

**依赖共享包**：`lc_alloc` 提供前缀标签、精确 DP、训练/温度校准、评价与 bootstrap。
`code/m11_yolo/` 只增加**检测器适配层**（候选/native 导出 + 特征 gather）与规范 NMS。

---

## 2. 目录结构

```
member_yolov8_delivery/
  README_CN.md          # 本文件
  manifest.csv          # 全部 63 个文件的 相对路径 / 大小 / SHA256 / 用途
  code/                 # 01..08 管线脚本 + m11_yolo 适配层
  configs/              # 检测器、角色、分组、权重、评价口径
  assets/               # PCA、scaler、3 模型、温度、训练历史、asset.json
  results/              # 全部原始结果表与选择记录
  references/           # 候选/native 全量文件（data/）+ 校验索引
```

---

## 3. 复现顺序（严格按此顺序）

```bash
# 环境变量：先让共享方法包可见（脚本内已内置该路径，亦可由外部 PYTHONPATH 提供）
#   M11_EXPERIMENT_ROOT\material from memberA\LC_ALLOC_M11_HANDOFF_v1\handoff_LC_ALLOC_M11_v1

# 1) 导出候选 + native（NMS 前稠密 anchor 流）
python code/01_export.py --split TRAIN --batch 64 --out work/TRAIN
python code/01_export.py --split DEV   --batch 64 --out work/DEV

# 2) GT 转换 + 角色/组划分
python code/02_prepare.py --out work/prep

# 3) 训练 3 个独立种子（PCA / scaler / MLP / 温度）
python code/03_train.py --train-candidates work/TRAIN/candidates.parquet \
  --train-native work/TRAIN/native.npz --train-gt work/prep/TRAIN_gt.json \
  --roles work/prep/roles.csv --out work/assets

# 4) 分配 + 评价 + bootstrap（必须先于 05）
python code/04_allocate_eval.py --dev-candidates work/DEV/candidates.parquet \
  --dev-native work/DEV/native.npz --dev-gt work/prep/DEV_gt.json \
  --groups work/prep/dev_groups.csv --assets work/assets --out work/eval

# 5) 标准 AP/AR（读取 04 生成的 allocations，不重新求解）
python code/05_ap_ar.py --dev-candidates work/DEV/candidates.parquet \
  --dev-gt work/prep/DEV_gt.json --allocations work/eval/allocations.parquet --out work/eval

# 6) NMS 后条数（后处理补记录，不加载训练器、不改 K_i）
python code/06_post_nms_counts.py --dev-candidates work/DEV/candidates.parquet \
  --allocations work/eval/allocations.parquet --groups work/prep/dev_groups.csv --out work/eval

# 7) 逐槽预测效用重放（k=6..50）
python code/07_predicted_utility.py --dev-candidates work/DEV/candidates.parquet \
  --dev-native work/DEV/native.npz --assets work/assets --out work/eval

# 8) 组装本交付包
python code/08_build_delivery.py
```

> **顺序说明（规范 Y4）**：`05_ap_ar.py` **依赖** `04` 产出的 `allocations.parquet`，
> 因此必须排在 `04` 之后。此前报告附录 A 曾把 05 排在 04 之前，已修正。

---

## 4. 方法身份：本分支实际固定的配置

### 4.1 检测器冻结（不复用于任何其他分支）

- 预处理：RGB → `LetterBox((640,640), auto=False, stride=32)` → `/255.0` float32 → CUDA；**无** mean/std、**无**增广、**无** TTA
- 架构/库：CSPDarknet backbone(C2f) + PAN-FPN 颈 + Detect 头；ultralytics 8.4.77（pip wheel，未记录 git commit）——详见 `configs/yolo_adapter.json`
- 候选导出点：**NMS 前稠密 anchor 流**（8400 锚点 × 8 Road8 类 = 67200 假设），正面积框过滤，无置信度阈值、无 NMS、无 objectness
- 排序：分数降序 → 锚点升序 → Road8 类升序，取 Top-100
- 分数语义：anchor 在其 Road8 类上的 **post-sigmoid 逐类分数**（不做 objectness 乘法）
- `source_id` = 锚点索引 0..8399；record↔native 为**一对八**，仅按 `(image_id, source_id)` 关联
- `native_vector` = 该锚点 84 维原始 Detect 头行（4 框 xywh 信箱像素 + 80 类 sigmoid 分数）

### 4.2 训练配方（与主线 M11 同配方，**权重全部重训**）

| 项 | 值 |
|---|---|
| 网络 | `Linear(90,128) → LayerNorm(128) → GELU → Linear(128,64) → GELU → Linear(64,10)` |
| 损失 | BCEWithLogits（自然 logits，未加权） |
| 优化器 | AdamW，lr=1e-3，weight_decay=1e-4 |
| batch / epochs / patience | 4096 / 最多 30 / 5 |
| 模型选择 | EARLY_STOP 上最早严格最优 |
| PCA | 32 维，**仅 FIT 拟合**（FIT native 向量按 `(image_id, source_id)` 去重） |
| scaler | StandardScaler，**仅 FIT 拟合** |
| 温度 | CALIBRATION 上每种子一个共享正温度，bounds [0.25, 4.0] |
| 种子 | 830101 / 830102 / 830103（**3 个独立训练，非集成**）；确定性基线 seed = −1 |
| 随机性来源 | PCA/训练/温度/bootstramp 的种子均在结果文件中记录 |

### 4.3 数据与划分（本分支**重新生成**，见 §6 边界）

| 项 | 值 |
|---|---|
| 角色划分 | `split_hash("YOLO_BDD_ROLE_V1\|", image_id)` → FIT 8000 / EARLY_STOP 1000 / CALIBRATION 1000 |
| 分组 | `split_hash("YOLO_BDD_GROUP_V1\|", image_id)` → 50 组 × 40 图 |
| 类别权重 | 按本分支 **FIT** 计数重算（`compute_class_weights`），与主线数值不同属预期 |

---

## 5. 结果文件与口径

### 5.1 主结果矩阵（`results/main_results.csv`）

- **25 行** = 5 方法身份 × 5 预算：`S_FIXED(seed=-1)`、`S_ADAPT(seed=-1)`、`M11(seed=830101/830102/830103)`
- 均值由原始行重建；**不把三种子均值伪造成第四个模型**
- 字段：`dataset_id, detector_id, candidate_asset_id, protocol_id, method, seed, budget, image_count, total_records, Coverage_per_image, QUALITY_per_image, coverage_total, quality_total, coverage_recall, valid_gt_total, AP, AP50, AP75, AR100`
- 校验：`total_records == image_count × budget`（组内精确预算闭合）

### 5.2 指标定义（写入 `configs/evaluator_config.json`）

```
Coverage_i(K)      = m_i^{0.50}(K)                     # 只用 IoU = 0.50 单阈值
QUALITY_i(K)       = mean_tau sum_c w_c * m_{i,c}^tau(K)   # 十阈值 0.50:0.05:0.95
delta_{i,k}^tau    = m_i^tau(k) - m_i^tau(k-1),  k = 6..50  # 训练标签
```

- `m` 为**同类**最大基数匹配，候选与 GT 均不重复使用；增广路径可重配已有匹配
- **空 GT 图像保留在评价集合中**
- K_i 为整数，5 ≤ K_i ≤ 50；每组 40 图；平均预算 10/15/20/30/40，组内精确求和，组间不借预算
- 每条 record 占一个槽位，**不按 source_id 合并记录**

### 5.3 标准 COCO 评价

`iouType="bbox"`、`useCats=1`、IoU=0.50:0.05:0.95（10 档）、101 recall 点、`maxDets=[1,10,100]`。
使用同一 GT/预测类别 ID 空间、完整 DEV 2000 图清单与**原 detector score**。
AP/AR **逐 seed 在完整 DEV 上计算**，不平均各 40 图组的 AP。

**`ap_ar_prefix.csv` / `ap_ar_reference_nms.csv` 字段映射**（规范 §08：保留既有文件名并提供字段映射）：

| 字段 | 含义 |
|---|---|
| `action_space` | `frozen_prefix_no_nms`（A，主比较）；`frozen_prefix_plus_nms0.70`（B，参考） |
| `method` | `M11` / `S_ADAPT` / `S_FIXED` |
| `seed` | `-1`（确定性基线）或 `830101/830102/830103` |
| `budget` | 每图平均预算（10/15/20/30/40） |
| `mean_K` | DEV 上实际 K_i 的均值（PFX_EXACT 下恒等于 budget） |
| `AP` / `AP50` / `AP75` / `AR100` | pycocotools COCOeval 统计（0–1） |
| `detections` | 提交给 COCOeval 的检测记录总数 |

### 5.4 单位（规范 §06）

原始 CSV 中 AP/AR/recall 存为 **0–1**；换算百分点时取 ×100 之差。
例：预算 10 的 AP `0.0283 − 0.0204 = 0.0079`，即 **+0.79 个百分点**；`+38.7%` 是**相对增幅**，两者不可混用。

### 5.5 统计

配对单位 = 40 图组；重采样 5,000 次，种子 530002，配对、共享抽样索引。
`results/bootstrap_results.csv` 同时含：
- 逐种子行（`seed_summary = seed_8301xx`）——区间端点**不跨种子平均**
- 三种子汇总行（`seed_summary = mean_of_3_seeds`）——先对每组求三种子差值均值，再 bootstrap

**核心汇总（规范 §06）**：`results/bootstrap_core_summary.csv` 给出 `K10/15/20 等权平均` 的
配对 bootstrap —— 对每组先按预算 10/15/20 等权平均 M11−S_ADAPT 差值（三种子汇总行先求
三种子差值均值再跨预算平均），再以 50 组为单位重采样。逐预算区间见 `bootstrap_results.csv`。

> **Coverage/QUALITY 的区间不能证明 AP/AR 显著**；多个预算/种子也不构成整体错误率控制。
> 三 seed 仅是本次有限训练重复，不是总体分布保证。

### 5.6 NMS 参考（`PFX_THEN_NMS`，**独立动作空间**）

- 实现：`code/m11_yolo/nms.py`，**class-aware**，torchvision.ops.nms，IoU = **0.70**（沿用既有参考，未做阈值搜索）
- 并列规则：高分优先；分数完全相等时取较小输入索引
- 不变式：`removed = n_before − n_after` 且 `n_after ≤ n_before`；**不回填槽位**
- DEV 全体：`n_before` 合计 1,150,000（= 精确预算总和），`n_after` 合计 264,586，删除 885,414，**空输出图 0**
- 每图 NMS 前后条数、存活 record IDs 见 `results/post_nms_counts.csv`
- 参考 NMS 的 AP/AR（`ap_ar_reference_nms.csv`）中 **M11 只取 seed 830101**（`05_ap_ar.py` 的参考循环仅跑该 seed），S_ADAPT/S_FIXED 取 seed −1；这是「处理顺序对照」参考，不重复 3 seed —— 主比较 `ap_ar_prefix.csv` 仍是 3 seed 全量

### 5.7 逐槽效用（`results/predicted_utility.parquet`）

每 seed / 每图 / k=6..50：10 维原始 logits、温度后概率、`delta_hat = w_class × mean_tau(prob)`、`Uhat` 累计曲线。
**Uhat 以 `Uhat_i(5) = 0` 为锚**；k=1..5 是必选常数前缀，**不补造模型预测**。
溯源（asset_id、模型 SHA、weight_id、生成时间）见同目录 `predicted_utility_manifest.json`。

> 校验：本文件可**逐位重放** `allocations.parquet` 的 K_i（对 1 个完整图组 × 3 种子 × 5 预算 = 15 个条件验证通过，预算精确闭合）。

---

## 6. 边界与声明（**请勿超出**）

### 6.1 可采用的表述（规范 §09）

> 在 BDD100K 派生开发集的**固定 NMS 前 Top100 候选协议**下，重新拟合的 YOLOv8n 分配器取得了
> 高于固定配额和分数分配的**已报告汇总指标**。追加相同 NMS 后，相对优势仍保留；
> **该参考不隔离重复候选对初始配额的贡献**。

### 6.2 本包**不**声称

1. 任何 SOTA 性能；
2. 「更高效 / 更快 / 实时」——**未测干净推理延迟**（导出墙钟含 PIL 解码 / LetterBox / 磁盘写，不是 detector-only latency）；
3. 「适用于所有检测器」——只有 YOLOv8n 与 RT-DETR 两个分支的证据；用 `applicable across multiple detector families`，**禁用 `detector-agnostic`**；
4. 「覆盖率与精确率无取舍」——`AP 提高不能证明所有运行阈值上的 precision 均不下降`；
5. 「优势并非重复候选造成」——`冻结前缀追加相同 NMS 后相对优势仍保留；本实验未隔离重复候选对初始配额决策的影响`；
6. 「AP 低是预期的」——`本表测量特定前缀协议下的 AP，尚未量化低绝对值的全部来源`；
7. 「只替换了检测器」——角色划分、分组、类别权重均按本分支重新生成（§4.3），**检测器间绝对差距不应全部归因于架构**；
8. 原 90 维特征或原模型可直接跨检测器迁移——YOLO 的 84 维 native_vector **不是** RT-DETR 查询嵌入；
9. 三种子为集成——是**独立训练**，逐种子报告。

### 6.3 明确未做（规范允许，不应通过文字「补成已完成」）

- **`NMS_THEN_PFX`（先 NMS 再分配）**：未做。这是新的动作空间，候选/标签/上下文与可行集合都已改变，**不能由现有表推出**。
- **完整计时矩阵**：未做。如需效率主张，须另用相同机器/输入/batch/精度/同步规则实测三段路径并保存计时样本。
- **NMS 阈值搜索**：未做，且不应做（规范 §07：不通过 DEV 质量搜索）。
- **重训分配器 / 重跑检测器**：未做，也不因文字修正而重算已有主结果。
- **RT-DETR 分支的 AP/AR**：不在本包范围（本包只覆盖 YOLO 分支）。

---

## 7. 大文件与外部依赖

候选/native 与 GT 全量文件（8 个文件，合计约 634 MB）**不随本 GitHub 上传**，其大小/SHA256
登记在 `references/candidate_native_index.csv` 与 `references/README_CN.md`；离线交付包
`member_yolo_delivery` 中它们位于 `references/data/`，`source_path` 保留原始出处供溯源。

```
references/data/DEV/candidates.parquet     20.5 MB   DEV 候选（重放 04–07 评价必需）
references/data/DEV/native.npz             81.6 MB   DEV native
references/data/DEV/export_log.json         <1 KB    DEV 导出日志
references/data/TRAIN/candidates.parquet   96.8 MB   TRAIN 候选（重跑 01–03 训练）
references/data/TRAIN/native.npz          408.6 MB   TRAIN native
references/data/TRAIN/export_log.json       <1 KB    TRAIN 导出日志
references/data/prep/DEV_gt.json            4.4 MB   DEV GT
references/data/prep/TRAIN_gt.json         22.0 MB   TRAIN GT
```

> 重新生成方式：`code/01_export.py`（候选/native）+ `code/02_prepare.py`（GT）。包内文件与索引 SHA 一致即可复核，无需依赖本机路径。

**未打包项**（规范 §08）：原始 BDD100K 图片数据集、虚拟环境、令牌、未经授权的 TEST/RESERVE 资产。

---

## 8. 接收方自查清单

| 检查项 | 依据 |
|---|---|
| `manifest.csv` 的 SHA256 与实际文件一致 | 63 个文件逐个校验（manifest 自身除外） |
| `results/main_results.csv` 为 25 行，`total_records = image_count × budget` | 组预算闭合 |
| `results/class_results.csv` 为 200 行（25 条件 × 8 类） | 逐类原始量对账 |
| `results/per_image_results.csv` 50,000 行；`per_group_results.csv` 1,250 行 | 规范 §08 最低矩阵 |
| `results/selected_records.parquet` 1,150,000 行，可由 `allocations` + 候选 rank 严格重建 | 选集可追溯 |
| `results/post_nms_counts.csv` 满足 `removed = before − after`、`after ≤ before` | 规范 §07 |
| `results/predicted_utility.parquet` 可重放 `allocations` 的 K_i | 规范 §05 |
| GT 不进入选择/选模路径（`code/04` 的分配路径只读候选与原分数） | 规范 §08 执行边界 |
| `code/05_ap_ar.py` 排在 `code/04` 之后 | 规范 Y4 |

---

## 9. 历史文件说明

本交付目录之外的 `memberB_yolo_validation_report.md`、`lc_p12_yolo_infer.py`、`lc_p12_yolo_validate.py`
是**早期手写的 Coverage 跨检测器尝试（2026-09-26）**，已被本 M11 分支取代，文件中已加 **SUPERSEDED** 标记。
其中的 `detector-agnostic` 措辞属旧口径，**不得带入论文正文**。
