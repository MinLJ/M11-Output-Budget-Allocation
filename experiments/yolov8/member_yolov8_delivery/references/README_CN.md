# references —— 候选 / native / GT 全量文件（不随 GitHub 上传）

候选 / native / GT 全量约 **634 MB**（含一个 408.6 MB 的 `TRAIN/native.npz`），超出
GitHub 单文件 100 MB 上限，**不随本仓库上传**。它们的身份（相对路径、大小、SHA256）
登记在 `candidate_native_index.csv`；离线交付包 `member_yolo_delivery` 中这些文件位于
`references/data/`，`source_path` 的机器根已归一为 `M11_EXPERIMENT_ROOT`。

| split | file | rel_path | bytes | sha256 |
|---|---|---|---|---|
| DEV | candidates.parquet | data/DEV/candidates.parquet | 20503388 | afa5e3ccea71a30d01546b3827077f793ba75e6886ec9e379322e0224a450ffb |
| DEV | native.npz | data/DEV/native.npz | 81639450 | c749f174d4e7b670d818faa4d5899c9edd578a1f2a9b860cb1170c645dcd60a0 |
| DEV | export_log.json | data/DEV/export_log.json | 670 | 0baa3546456af9d6a09871913451ff8c09e8501b0159bc03b85b506d57df2fa6 |
| TRAIN | candidates.parquet | data/TRAIN/candidates.parquet | 96792192 | 3008723a255fe0484cfd0c4a9d800870c17f601c2402514b772dc7aae2f85518 |
| TRAIN | native.npz | data/TRAIN/native.npz | 408564722 | d769cbad2919fe4806aba4502c15c8f1efc9844f1ed0e231a229ee9f2de2cee8 |
| TRAIN | export_log.json | data/TRAIN/export_log.json | 675 | b43ff15452ccd363182978e5a9c8c554fa0728b39e35b8867abcf4e69a52745b |
| prep | DEV_gt.json | data/prep/DEV_gt.json | 4398487 | d5eeec2d39128b37fbcf71f65b1cd784a556f969017b41306f209dc174988e07 |
| prep | TRAIN_gt.json | data/prep/TRAIN_gt.json | 21967584 | cb300bbae7c8017ece066edde4c65fd51499ba7ed119f2477f0295e6fedc96a6 |

重新生成：候选/native 由 `code/01_export.py` 从 BDD100K 原始图片导出（参数见
`configs/yolo_adapter.json`）；GT 由 `code/02_prepare.py` 从 RT-DETR release 的
`DEV2K_ROAD8_GT.json` / `TRAIN10K_ROAD8_GT.json` 转换。文件与索引 SHA 一致即可复核，
无需依赖本机路径。

按规范 §08 不打包的仅限：原始 BDD100K 图片数据集、虚拟环境、令牌、未经授权的
TEST/RESERVE 资产。
