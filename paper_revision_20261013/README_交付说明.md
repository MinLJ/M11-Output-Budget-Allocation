# M11 论文修订交付包 · 说明

**日期：** 2026-10-13
**对应任务：** 按老师 19 条修改意见修订 M11 论文并完整打包，交老师评审后再投稿。

---

## 一、包里有什么

```
M11_paper_revision_20261013/
├── 01_manuscript/
│   ├── M11_manuscript_v2_revised.docx   ← 给老师评审的主文件（含图、公式、表格）
│   ├── M11_manuscript_v2_revised.pdf    ← 同一稿的 PDF 渲染（版式核对用）
│   ├── M11_manuscript_v2_revised.md     ← Markdown 源（唯一事实源，改动都从这里生成）
│   └── figures/fig1–fig4.png            ← 从原稿 PDF 中裁出的四张图（400 dpi）
├── 02_response/
│   ├── Response_to_Comments.docx        ← 逐条回复信（英文，按 19 条组织）+ 第16条清单勾选 + 待确认事项
│   ├── Response_to_Comments.pdf         ← 同上 PDF
│   └── Response_to_Comments.md          ← 回复信 Markdown 源
└── 03_reproducibility/
    ├── recompute_seed_spread.py         ← 从冻结的 per-seed 结果 CSV 复算 Mean ± STD
    ├── seed_spread_report.md            ← 复算结果（论文里所有 ± 数字的来源）
    └── seed_spread.csv                  ← 同上，机器可读版
```

**发给老师时建议同时附：** `01_manuscript/M11_manuscript_v2_revised.docx` + `02_response/Response_to_Comments.md`（后者可转 docx 或直接贴邮件）。`03_reproducibility/` 是备查材料。

## 二、修订做了什么（对应 19 条）

详版见回复信；要点：

| # | 意见 | 落实位置 |
|---|---|---|
| 1 | 改标题（二选一） | 采用第 2 个：*Learning Prefix Marginal Utility for Resource-Aware Output Budget Allocation in Intelligent Vehicle Perception* |
| 2 | 参考文献加到约 30 条 | 16 → 33 条，逐条核对出处后才加入；3 条页码按 DOI 记录更正；2 条原有 CVPR 2026 引用核对属实 |
| 3 | 缺贡献段 | 引言新增第 12 段，三条贡献映射到 III-A–III-C / III-B / III-D+IV-D |
| 4 | M00 在 Coverage 上胜过 M11 | IV-D 明说 Coverage 只算 IoU.50、QUALITY 均十阈值；M00 直接优化上报指标（+0.2313），M11 赢 QUALITY/AP/AR100；M00 列为并列可选配置 |
| 5 | 求解器才是主要贡献 | V 节开篇新增"Solver and target"段：求解器 +0.5616、监督目标 +0.1867（约 3 倍）；贪心解在同预测下比 Score 基线还低 0.4681 |
| 6 | pre-NMS AP 太低没解释 | IV-A 指标段后照抄老师给的两句说明；III/IV 表题加"pre-NMS"标注 |
| 7 | 开销没说明值不值 | IV-E 补三句：17.8×、摊销 3.519 ms/图、≈44 ms 分配器时间/多覆盖 1 个目标；并明确"没有做下游端到端测量" |
| 8 | 缺自适应 proposal 数量综述 | II 节新增约 150 词段落（"Proposal-count decisions are the closest existing problem"），一句话划清 produce vs. keep 的界限 |
| 9 | TEST 标注暴露只在脚注 | 脚注删除，改为 Appendix A（触发、范围、控制、可核验 artefact 四项俱全，并点名两个真实 artifact） |
| 10 | LC 未展开 | IV-A 首次出现处展开为 LargeClean-v1（LC-v1），含 68,074 = 70,000 − 1,926 的口径；"Frozen LC" 改写为"LC-v1-trained allocator transferred unchanged" |
| 11 | M11 无含义 | III-C 定义描述性名称 class-weighted multi-threshold prefix-increment utility + M11 两比特命名（M00/M10/M01） |
| 12 | 六个公式没被引用 | (1)–(8) 全部在正文中被显式引用 |
| 13a | Fig 1/3 图注太长 | 两图图注均为两句；Fig 1、Fig 3 均标注"Schematic; values are prescribed for illustration." |
| 13b | 多次仿真应报 Mean ± STD | Table I(b)/II/IV/V 补 ± （三个独立训练模型的标准差）；表题写明口径；确定性基线无 spread |
| 14 | 摘要没说难点 | 摘要补三句：监督信号只能从检测器自身输出推断、共享预算导致跨图耦合 |
| 15 | 目标期刊 | 记录（T-ITS → T-IV → Access → Neurocomputing）；本稿为评审稿，未套具体期刊模板 |
| 16 | 12 项 resubmit 清单 | 回复信内逐项勾选 |
| 17 | 作者块 | 标题页照录老师给的名单与单位 |
| 18 | 致谢 | 新增 Acknowledgment，照录老师给的原文（NSFC 42671235） |
| 19 | 第三人称 | 全文转换完毕；复查无 "we/our/us" 单词出现 |

## 三、待老师/组长确认（4 项）

1. **作者块里的 `*` / `**`**：按通讯作者标记理解，加了脚注"*Yang Luo and Yi Chen are the corresponding authors"和联系邮箱 `Yi.Chen02@xjtlu.edu.cn`（这是包内唯一一处"补充"而非"照录"的联系方式）。若含义不同请指出。
2. **参考文献核验结论**：已全部核完。原稿两条 CVPR 2026 引用（PaQ-DETR [21]、WhisperNet [28]）核对属实、页码无误；Focal Loss、RAN、When2com 三条页码按 DOI 登记记录更正；Efficient DETR 无同行评审出处，保留为 arXiv 预印本引用。老师建议的两条候选**按描述均查无此文献**：最接近的 Guan et al.（ICASSP 2019，自适应 proposal 数量）已核实并加入（[23]）；Cui et al. 只有 arXiv 预印本（2207.05252），因无发表出处**未加入**，如需可补。**未核验通过的引用都不在稿子里。**
3. **YOLO 臂的 per-seed 数据**：仓库里 `experiments/yolov8` 已被删除（commit 693b9d5，2026-10-10，且为目录链接、无可恢复内容），Table III 目前只能给三模型均值并如实加注说明。若能找回该臂的 per-seed 表，可补齐 ± 。
4. **Appendix A 的 artifact 标识**：如需在正文中写入 LC-v1 release manifest 的准确文件名与 P0 decision SHA，请在投稿前补上。

## 四、数据与可复算性

- 本包**未重训任何模型、未重跑任何分配、未重算任何指标**。稿中数字全部来自原稿或各臂冻结的 per-seed 结果记录。
- 所有 ± 数字可由 `03_reproducibility/recompute_seed_spread.py` 从冻结 CSV 原样复算（`python -B recompute_seed_spread.py`），输出即 `seed_spread_report.md`。
- 数据来源（只读）：`experiments/baselines_ablation/research_LC_ALLOC_ABLATION_P0`、`.../research_LC_ALLOC_P1A`、`experiments/coco/...ROUTE_A/ROUTE_B`、Faster R-CNN 交付包 `member_faster_rcnn_delivery/results/main_results.csv`。

## 五、边界声明

- 未改动 `LC_ALLOC_M11_HANDOFF_v1/`、`frcnn_m11/delivery_out/`、`member_faster_rcnn_delivery/` 内任何文件（全部只读使用）。
- 未改动任何实验目录；本包为新建目录，独立于仓库。
- 公式在 docx 中为 OMML 原生公式（pandoc 转换），可在 Word 中直接编辑。
- 若需与期刊模板对齐（IEEE 双栏），投稿前需重排；文字内容不受影响。
