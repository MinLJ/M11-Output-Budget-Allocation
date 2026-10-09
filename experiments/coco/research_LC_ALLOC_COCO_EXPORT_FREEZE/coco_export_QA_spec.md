# COCO2017 Road8 export QA specification

状态：`QA_CONTRACT_FROZEN_NOT_EXECUTED`

版本：`COCO2017_ROAD8_EXPORT_QA_V1`

本文件定义未来 candidate/native export 完成后的必检项。任何一项失败都使该 split 的 export fail closed；不得发布 completion marker 或宣称 candidate asset frozen。

## 1. Preflight checklist

- [ ] schema、export config、identity contract 的 exact bytes SHA 与旁车一致；
- [ ] split 只为 `TRAIN2017` 或 `VAL2017`，image-only manifest SHA 与冻结 registry一致；
- [ ] checkpoint/config/repo commit 与冻结合同一致；
- [ ] runner source-bundle SHA、QA validator SHA、环境身份已记录；
- [ ] runner 参数及 import graph 不包含 annotation、GT、matching 或 evaluator path；
- [ ] output root 新建且为空，staging 与 canonical consumer path 隔离；
- [ ] CandidateAssetID 从六个 frozen strings 重算一致。

## 2. Image coverage and identity

对 split manifest 全量检查：

| Check | Required result |
|---|---:|
| manifest images processed | 100% |
| missing images | 0 |
| extra images | 0 |
| duplicate canonical image IDs | 0 |
| duplicate COCO image IDs | 0 |
| image byte SHA mismatch | 0 |
| identity mismatch among bundle/candidate/native/index | 0 |

`canonical_image_id` 必须为 `coco2017:<SPLIT>:<12-digit COCO id>`；compatibility field `image_id` 必须逐行严格等于它。不得以文件名代替 identity。

## 3. Cardinality and shape

每图必须满足：

- detector native query rows = 300；
- native `query_index` 是无重复、无缺失的整数集合 `0..299`；
- L3 Road8 logits shape `[300,8]`，dtype float32；
- L3 Road8 scores shape `[300,8]`，dtype float32；
- query embedding shape `[300,256]`，storage dtype float16；
- L3 predicted boxes shape `[300,4]`，dtype float32；
- raw hypotheses = `300×8=2400`；
- stored candidates = 300；
- stored `road8_rank` 是无重复、无缺失的 `1..300`；
- M11 consumption view 是且仅是 `road8_rank<=100`，exactly 100 records/image。

aggregate row counts 必须等于 image count 乘上述 per-image cardinality。候选表只存 Top300；raw 2,400 hypotheses 不要求另存一份表，但重建计数与 source-order domain 必须通过。

## 4. Candidate identity and class mapping

- [ ] CandidateAssetID 与当前 split 的 identity contract一致；
- [ ] 每个 `candidate_record_id` 按 frozen length-prefixed formula重算一致；
- [ ] `candidate_record_id` 在 split 内全局唯一；
- [ ] `source_order = query_index*8 + (road8_class_id-1)`；
- [ ] source order domain为0..2399；
- [ ] Road8 ID/name、COCO category ID、detector class index严格匹配冻结8元映射；
- [ ] 共享同一 query_index 的不同 class records保留，不做 query dedup；
- [ ] 原 candidate ID、class、query和rank readback无变化。

## 5. Candidate ↔ native join

唯一 join key：`(canonical_image_id, query_index)`。

必须检查：

- native key uniqueness = 100%；
- candidate rows with exactly one native match = 100%；
- join failure = 0；
- orphan native rows = 0；
- candidate/native identity and asset fields exact equality；
- 任一按 rank、source order、row position 或 reshape 推断 native row 的实现均失败。

## 6. Numerical reconstruction

所有 score、bbox、logits、stored scores 和 embeddings 必须 finite：`NaN=0`、`Inf=0`。

### Score

1. 未来 exporter在同一 forward 中计算 `l3_road8_scores = torch.sigmoid(l3_road8_logits)`，float32；
2. candidate `score` 必须与 joined native row、对应 Road8 class位置的 stored float32 score做 exact numeric equality（candidate column只是其float64无损展宽）；
3. QA validator再在冻结 runtime/device 上用 `torch.sigmoid(float32_logits)` 重建 score，要求 `atol=1e-7`、`rtol=1e-6`；
4. 不允许 softmax、calibration、threshold或第二次 sigmoid。

exact stored-score equality 与 logit reconstruction tolerance 是两项不同检查，不得混称 bitwise logit→sigmoid parity。

### Boxes

- native box是 normalized L3 `cx,cy,w,h` float32；
- candidate original-image xyxy按冻结 float32转换得到，之后无损展宽为float64；
- scalar xyxy必须 exact 等于 `bbox_xyxy`各元素；
- scalar/array cxcywh必须由 exported xyxy按冻结公式一致派生；
- candidate rows sharing a query的 boxes必须一致；
- 不得 clip、epsilon repair、round或修改坐标。

## 7. Sorting and determinism

从全部2,400 hypotheses重构 total order：

1. score descending；
2. query_index ascending；
3. M11 Road8 class ID ascending；
4. source_order ascending。

未来 validator必须确认 stored Top300 exactly等于此顺序的前300，rank 1..300连续。排序键的 NaN、unstable sort、隐式 dataframe index或 filename顺序不得参与结果。

determinism QA 至少在固定 manifest identity抽取的样本上重复完整单图 forward/export；身份字段、native/candidate tensor值、order和record IDs必须在冻结环境内一致。抽样规则和样本数须在未来 run config中预先固定，不能按结果挑样本。

## 8. Serialized readback and release integrity

- [ ] 所有 Parquet/NPZ/index/manifest 可从 staging readback；
- [ ] required columns/arrays存在，dtype/shape/nullability符合 schema；
- [ ] NPZ不含 object/pickle arrays，读取使用 `allow_pickle=False`；
- [ ] shard边界不改变每图300行 block或 join；
- [ ] 每个文件 binary SHA 已进入 ledger；
- [ ] ledger对 staging readback文件 mismatch=0；
- [ ] asset manifest记录 schema/config/interface/runner/validator/environment SHA；
- [ ] annotation/GT/evaluation asset path或SHA没有进入detector-only identity chain；
- [ ] consumer sampled read验证Top100 view及native join可用。

## 9. Aggregate decision

未来 `qa_result.json` 至少包含：

```json
{
  "qa_contract": "COCO2017_ROAD8_EXPORT_QA_V1",
  "split": "TRAIN2017_or_VAL2017",
  "all_checks_pass": true,
  "image_coverage_fraction": 1.0,
  "missing_images": 0,
  "extra_images": 0,
  "candidate_record_id_duplicates": 0,
  "candidate_native_join_failures": 0,
  "nan_count": 0,
  "inf_count": 0,
  "sorting_mismatches": 0,
  "sha_mismatches": 0
}
```

只有 `all_checks_pass=true` 且以上 hard counts全为0，才允许原子发布 asset manifest/completion marker。否则状态为 `COCO_EXPORT_QA_FAIL`，保留隔离诊断并停止；不得生成可消费 asset、训练 allocator或开始评价。

## 10. 本任务非执行声明

本 QA 合同在当前任务仅被静态冻结；没有读取 annotation、没有 detector forward、没有生成 candidate/native、没有 matching 或 AP/AR/Coverage/QUALITY 评价。
