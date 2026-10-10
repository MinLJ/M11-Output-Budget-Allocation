# 候选与原生特征的引用说明

原始候选表与 native 分片体积大, 按 §08 不打包进交付件, 改为**受控路径 + 校验清单**:

- 主用候选流 DEV2K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_dev.parquet` — bytes=11119018, sha256=`4f71d2f7346cd1e15c1dfc13bd524d65d0553635f7e814baf4768d08b60c98e1`
- 登记备查流 DEV2K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_dev_post_nms.parquet` — bytes=28039250, sha256=`790b6d57900de1ab0bcc59732ddf02101b4562ed41e7dd6f0a5ee77236d03cc4`
- 主用候选流 TRAIN10K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_train.parquet` — bytes=44431997, sha256=`9d131335ac1e8080e42008a38ef358c27d6a98738255a5c8eb248abc1b27332b`
- 登记备查流 TRAIN10K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_train_post_nms.parquet` — bytes=134877218, sha256=`f28fc869867b6272d4f625689b52e08d6a8f97151038531f0d71fb25eea958a5`
- 合并 native DEV2K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\native_dev.npz` — bytes=587726194, sha256=`e7322923cc85c17819ee3d6292ea34a5cd4d82618a781c6a1f669167f844941f`
- 合并 native TRAIN10K: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\native_train.npz` — bytes=2937116098, sha256=`b89d8b9c3246ac3ebc92b1f7e5603a7701ab06892d46f7bca25ced25157ca99e`
- native 分片索引: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\native_index_dev.json` / `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\native_index_train.json`
- 候选校验索引: `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_index_dev.json` / `M11_EXPERIMENT_ROOT\frcnn_m11\prepared\candidates_index_train.json`

候选文件的 `bytes` 与 `sha256` 同时登记在候选校验索引 `candidates_index_*.json` 里;
native 索引里每个分片也都带 `bytes` 与 `sha256` —— 拿到受控路径后可直接重算比对。
native 索引里分片 `path` 是**相对路径**, 基准根目录 = `M11_EXPERIMENT_ROOT\frcnn_road8\outputs` 
(即把该根拼接在 `path` 前; 打包时已对两索引的 24 个分片逐片做文件存在性核对,
例: `frcnn_r18fpn_road8/export_dev/native_state/state_0000.npz` 实际位于 `M11_EXPERIMENT_ROOT\frcnn_road8\outputs\frcnn_r18fpn_road8\export_dev\native_state\state_0000.npz`)。

## native 结构小表 (合并 npz; 成员名 / shape / dtype 按实际文件头读取)

| 成员 (.npz 内) | DEV2K | TRAIN10K | dtype | 说明 |
|---|---|---|---|---|
| `image_ids` | (560739,) (560,739 行) | (2810117,) (2,810,117 行) | `<U17` | 逐行 image_id (字符串; 与候选表 / GT 的 image_id 对齐; 一行 = 一个 RoI) |
| `source_ids` | (560739,) (560,739 行) | (2810117,) (2,810,117 行) | `<U8` | `'roi{index:05d}'` RoI 行索引; 候选表侧同一 RoI 最多 8 条 (≤8) 类别记录共享此键, 不得去重 |
| `native_vectors` | (560739, 1024) (560,739 行) | (2810117, 1024) (2,810,117 行) | `float16` | box_head 输出的 1024 维原生向量 (每个 RoI 一行); 存储 float16, 读取端按需升精度 |
| `class_signals` | (560739, 9) (560,739 行) | (2810117, 9) (2,810,117 行) | `float32` | pre-softmax 类信号, **9 列含 background**: 列 0 = background, 列 1..8 = Road8 8 类; 特征只用 1..8 列, 9 列全存以便正向重建 score |
| `native_schema_id` | 标量 () | 标量 () | `<U41` | 标量字符串, 全部行同一值; 实际值 = `frcnn_r18fpn_road8_roihead1024_logits9_v1` |

按图取 native (只拿到 zip 的人照此即可闭环): join 只用 `(image_id, source_id)`, 不按 rank、
不按数组行号。该键在 native 内逐行唯一 (每个 RoI 恰 1 行); 候选表侧同一 RoI 最多 8 条类别
记录共享同一个 `source_id` —— join 时**不得去重、不得合并**: 多条候选记录引用同一 native 行
是设计, 不是重复。
