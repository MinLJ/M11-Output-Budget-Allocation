# YOLOv8 arm

Cross-detector arm of M11: the output-budget allocator after a **YOLOv8n**
(ultralytics 8.4.77) detector on Road8. The RT-DETRv2-R18VD reference arm is
separate; cross-detector statements require the combined two-arm table.

The delivered package is [`member_yolov8_delivery/`](member_yolov8_delivery/);
its [`README_CN.md`](member_yolov8_delivery/README_CN.md) is the authoritative
package description (environment, frozen detector identity and export protocol,
feature construction, reproduction order, result conventions). This file
connects the published files to the experiment's paper items and registers what
is withheld.

> Naming: the historical delivery package is `member_yolo_delivery`; the repo
> directory is `member_yolov8_delivery` to align with `experiments/yolov8/` and
> `experiments/faster_rcnn/member_faster_rcnn_delivery/`.

## Published contents

| Path | Contents | Connection |
|---|---|---|
| `member_yolov8_delivery/manifest.csv` | bytes/SHA256/purpose for all uploaded package files | Package identity registry |
| `.../code/` (15 files) | Candidate/native export (01), GT/role/group prep (02), training (03), allocation+eval+bootstrap (04), AP/AR (05), NMS counts (06), utility replay (07), delivery assembly (08/09), candidate audit, `m11_yolo/` adapter layer | The arm's only variable: detector paradigm (YOLOv8n vs RT-DETR) |
| `.../configs/` | Detector freeze (`yolo_adapter.json`), evaluator, weights, DEV grouping (`groups.csv`) | Frozen configuration |
| `.../assets/*.json`, `*_history_*.csv` | Asset registry, class weights, per-seed temperatures and training histories | Allocator identity; binaries withheld and registered |
| `.../results/main_results.csv` | 25 conditions = (method, seed, budget in {10,15,20,30,40}); core = K10/15/20 equal-weight | Cross-detector comparison item |
| `.../results/class_results.csv`, `per_group_results.csv` | per-class and per-group tables | Class trade-offs and grouping discussion |
| `.../results/bootstrap_results.csv`, `bootstrap_core_summary.csv` | paired group bootstrap, 5000 resamples, seed 530002; per-seed intervals retained separately | Comparison intervals |
| `.../results/ap_ar_prefix.csv`, `ap_ar_reference_nms.csv`, `ap_ar_per_class.csv` | COCO AP/AR on prefix protocol (action space A) and NMS reference (space B) | Detector quality reference |
| `.../results/post_nms_group_summary.csv`, `post_nms_overall.csv` | NMS-protocol summaries | NMS-protocol separation |
| `.../references/README_CN.md`, `candidate_native_index.csv` | bytes/SHA256 identities of the withheld candidate/native/GT caches | Unreleased dependency registry |
| [`PUBLICATION_EDITS.json`](PUBLICATION_EDITS.json) | historical vs release SHA for every path-normalized copy | Publication provenance |

## Withheld from this batch

- Allocator checkpoints `assets/models/marginal_mlp_seed_{830101,830102,830103}.pt`,
  `assets/pca32.joblib`, `assets/scaler.joblib`, `configs/roles.csv` and
  `configs/roles_raw.csv` (private split identity), and the large result tables
  `predicted_utility.parquet`, `selected_records.parquet`, `allocations.parquet`,
  `post_nms_counts.csv`, `per_image_results.csv`: registered with existing sizes
  and SHA256 in [deferred assets](../../manifests/deferred_assets.json); no public
  download is established.
- Candidate/native/GT caches (about 634 MB, see `member_yolov8_delivery/references/`):
  withheld; identities retained in `candidate_native_index.csv`.
- External, not part of this submission: the frozen M11 handoff package, the
  Road8/BDD100K image set, the detector checkpoint, the dependency environment.

## Interpretation limits

- Seeds 830101/830102/830103 are **three independent trainings**; metrics are
  computed per seed and then averaged (per group), never a prediction ensemble
  (`assets/asset.json` records this).
- All results are DEV2K development results on 50 frozen 40-image groups.
- The primary protocol is `PFX_EXACT`; `PFX_THEN_NMS` is a reference protocol
  reported separately. `NMS_THEN_PFX` was **not** run.
- `PRE_NMS_TOPN100` is the candidate stream (8400 anchors x 8 Road8 classes, top-100).
- Coverage = single IoU 0.50 max-cardinality matching; QUALITY = class-weighted
  10-threshold mean; both are distinct from COCO AP/AR.
- Reported values are differences against common baselines within the fixed
  candidate stream; no "effective"/"leading"/"detector-agnostic" wording.
