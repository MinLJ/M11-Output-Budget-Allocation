# COCO2017 Road8 export runner interface contract

状态：`INTERFACE_FROZEN_IMPLEMENTATION_NOT_EXECUTED`

版本：`COCO2017_ROAD8_EXPORT_RUNNER_INTERFACE_V1`

本文件冻结未来 COCO2017 detector-only export 的入口、输出与 fail-closed 规则。它不是 runner 实现，也不授权本任务运行 detector。未来实际 export 必须先物化 runner 和 QA validator、记录各自源码 bundle SHA256，并在第一次 model load/forward 前完成 preflight。

## 1. 唯一允许的入口

```python
export_coco_split(
    *,
    split: Literal["TRAIN2017", "VAL2017"],
    image_manifest_path: Path,
    image_manifest_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    export_config_path: Path,
    export_config_sha256: str,
    candidate_schema_path: Path,
    candidate_schema_sha256: str,
    candidate_asset_contract_path: Path,
    output_root: Path,
) -> ExportResult
```

`ExportResult` 只包含 candidate shard、native shard、native index、asset manifest、SHA ledger、QA result 和日志路径。入口不得接受 annotation、GT、evaluation result、Road8 target 或任何可由标注派生的参数。导出模块不得 import evaluator 或 matching 模块。

允许的 image manifest 是 execution gate 冻结的 image-only manifest；每行只提供 dataset/split identity、COCO numeric image ID、canonical image ID、relative image path、dimensions 和 exact image-byte SHA256。`image_id` 在全部导出记录中等于 `canonical_image_id`。

## 2. Preflight：任何 model load/forward 之前

未来 runner 必须按下列顺序 fail closed：

1. 对 runner source bundle、QA validator source bundle和实际 runtime environment 生成身份记录；不得用本接口文档 SHA 代替实现 SHA。
2. 验证 output root 不存在或为空；禁止覆盖、追加或复用先前 partial/canonical output。
3. binary hash exact schema、export config、identity contract、split manifest、checkpoint 和 model config，并与合同值逐字节比较。
4. 验证 split allowlist、manifest count、canonical ID 格式、COCO ID 唯一、relative path 安全且 image SHA 与 manifest 一致。
5. 验证 repo commit、tracked model/config source clean 状态和 checkpoint bytes。随后才允许加载 checkpoint；加载后、forward 前必须验证 `ema.module` state-key 合同与 strict load。
6. 验证 Road8 taxonomy/mapping、preprocessing、dtype、execution mode、query/layer/embedding shape 和禁止操作配置没有被 override。
7. 从 identity contract 重算该 split 的 CandidateAssetID，并与预期值一致。

本 freeze 只对 checkpoint 做了存在性、byte length 和原始 binary SHA 审计；没有加载 checkpoint，也没有验证内部 tensor/state-key。未来 runner 必须执行上述 load-time 检查。

## 3. Canonical per-image contract

每张图只允许一次 canonical forward，使用冻结 config：batch size 1、RGB 640×640、FP32、`deploy().eval()`、no-grad/inference mode、无 autocast、无 TF32。

单次 forward 同时产生：

- 300 个 L3 query rows，`query_index=0..299`；
- L3 Road8 logits、同一 forward 中由 `torch.sigmoid(float32_logits)` 物化的 Road8 scores、query embedding、L3 final normalized boxes；
- 300×8=2,400 个 query-class hypotheses；
- 按冻结四键全序排序的 Top300 candidate records。

candidate record 是预算记录，不是 unique query。同一 query 的不同 Road8 class records 必须保留。native join 只能使用 `(canonical_image_id, query_index)`；按 road8 rank、candidate row position、array position或 reshape 隐式连接均禁止。

## 4. Determinism 与 identity

科学身份由 dataset/split manifest、detector/checkpoint、config、schema、排序和 ID 算法共同定义。相同输入、相同 checkpoint、相同 config、相同 schema 和满足合同的相同环境必须给出相同：

- CandidateAssetID；
- candidate_record_id；
- canonical IDs、query indices、Road8 IDs、source order 与 ranks；
- tensor shapes、field dtypes 和排序结果。

Parquet/NPZ 容器 bytes 可能随已记录的兼容库版本或文件 metadata 改变，因此每次真实运行必须记录实际文件 SHA；不得把科学 identity 的确定性误写成跨序列化器 byte-for-byte 文件恒等。未来 release 若要求 byte-reproducible serialization，应另行冻结 writer 版本和 metadata policy。

## 5. Output transaction

所有输出先写到同一 `output_root` 下的新建 staging 子目录：

```text
staging/
  candidates/
  native_state/
  native_state_index.parquet
  asset_manifest.json
  sha256_ledger.csv
  qa_result.json
  runtime_environment.json
  logs/
```

要求：

- 不得写入输入 dataset、release、P1/P1A/P1B 或冻结 gate 目录；
- shard 写完后必须关闭、binary hash，并从磁盘 readback；
- aggregate QA 全部 PASS 后，才允许原子发布 completion marker 与可消费 asset manifest；
- partial shards 不得进入 canonical/consumer path；
- 失败可保留隔离的诊断 staging，但不得写 `COCO2017_CANDIDATE_ASSET_FROZEN`；
- 已存在的 canonical output 目录不得覆盖。

实际 `asset_manifest.json` 必须记录：本接口合同 SHA、runner source-bundle SHA、QA validator source-bundle SHA、runtime environment identity、schema/config/identity-contract SHA、split manifest SHA、CandidateAssetID、所有 shard/index/ledger SHA 和行数。annotation/GT path 或 SHA 不得进入 runner input、asset manifest 的 detector-input binding 或 CandidateAssetID。

## 6. Required hard failures

| Code | Condition |
|---|---|
| `FORBIDDEN_SCIENTIFIC_INPUT` | 提供或发现 annotation/GT/evaluator input |
| `OUTPUT_NOT_EMPTY` | 输出目标已存在或非空 |
| `IMAGE_MANIFEST_MISMATCH` | manifest path/SHA/count/ID/path/image bytes不符 |
| `DETECTOR_ASSET_MISMATCH` | repo commit/config/checkpoint bytes或 SHA 不符 |
| `CHECKPOINT_STATE_CONTRACT_FAIL` | 非 `ema.module`、strict load失败或 state shape不符 |
| `EXPORT_CONFIG_MISMATCH` | config bytes/SHA或冻结字段不符 |
| `SCHEMA_MISMATCH` | schema bytes/SHA不符 |
| `CLASS_MAPPING_INCOMPATIBLE` | Road8/COCO/detector ID映射不符 |
| `QUERY_SHAPE_MISMATCH` | query、layer、class或embedding shape不符 |
| `NONFINITE_NATIVE_STATE` | logits/scores/embedding/box出现 NaN/Inf |
| `SCORE_RECONSTRUCTION_FAILURE` | stored score与 logits/candidate对应关系不符 |
| `CANDIDATE_ORDERING_MISMATCH` | 四键排序、source order或rank不符 |
| `CANDIDATE_NATIVE_JOIN_FAIL` | join不为100%或native key不唯一 |
| `NONDETERMINISTIC_IDENTITY` | CandidateAssetID/record ID重算不符 |
| `SERIALIZED_READBACK_FAIL` | shard/index/manifest无法按冻结 schema readback |
| `QA_AGGREGATE_FAIL` | 任一 QA 检查失败 |

任何 hard failure 都必须停止，不得补零、换 checkpoint、降级字段、放宽 tolerance、跳过图像或重新排序。

## 7. 本任务边界

本文件使 runner **接口规范**可冻结；它不宣称存在已审计的 COCO executable runner。未来 export 任务必须在 forward 前补充并绑定实际 runner/validator source SHA。`READY_FOR_COCO_EXPORT` 在本任务中仅表示合同物化完成，可进入受控实现/执行阶段；不表示 detector forward 已经授权或资产已经生成。
