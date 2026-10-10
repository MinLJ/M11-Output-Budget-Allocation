# Faster R-CNN arm

Cross-detector arm of M11: the output-budget allocator after a **Faster R-CNN**
(ResNet-18 + FPN) detector trained on Road8. The RT-DETRv2-R18VD reference arm
is separate; cross-detector statements require the combined two-arm table.

The delivered package is [`member_faster_rcnn_delivery/`](member_faster_rcnn_delivery/);
its [`README_CN.md`](member_faster_rcnn_delivery/README_CN.md) is the authoritative
package description (environment, frozen detector identity and export protocol,
feature construction, reproduction order, result conventions). This file
connects the published files to the experiment's paper items and registers what
is withheld.

## Published contents

| Path | Contents | Connection |
|---|---|---|
| `member_faster_rcnn_delivery/manifest.csv` | bytes/SHA256/purpose for all 51 package files | Package identity registry |
| `.../code/` (16 files) | Detector training, two-stream export, feature adapters, pipeline, evaluation | The arm's only variable: detector paradigm, backbone depth aligned with the reference |
| `.../configs/` | Detector declaration, export protocol, evaluator definitions, feature schema, adapter config, DEV grouping (`groups_dev.csv`) | Frozen configuration |
| `.../assets/*.json` | Asset registry, class weights, per-seed temperatures | Allocator identity; binaries withheld and registered |
| `.../results/main_results.csv` | 25 conditions = (method, seed, budget in {10,15,20,30,40}); core = K10/15/20 equal-weight | Cross-detector comparison item |
| `.../results/class_results.csv`, `per_group_results.csv` | per-class and per-group tables | Class trade-offs and grouping discussion |
| `.../results/bootstrap_results.csv`, `bootstrap_core_summary.csv` | paired group bootstrap, 5000 resamples, seed 530002; per-seed intervals retained separately | Comparison intervals |
| `.../results/coco_ap_results.csv` | COCO AP/AR on the released original GT | Detector quality reference |
| `.../results/pfx_then_nms_by_condition.csv`, `pfx_then_nms_by_group.csv`, `pfx_then_nms_summary.json` | PFX_THEN_NMS reference-protocol summaries | NMS-protocol separation |
| `.../references/README_CN.md` | bytes/SHA256 identities of the withheld candidate/native caches | Unreleased dependency registry |
| [`PUBLICATION_EDITS.json`](PUBLICATION_EDITS.json) | historical vs release SHA for every path-normalized copy | Publication provenance |

## Path normalization

Twelve files (8 in `code/`, `configs/evaluator.json`,
`results/pfx_then_nms_summary.json`, `README_CN.md`, `references/README_CN.md`)
were path-normalized for publication: historical machine roots are replaced by
the `M11_EXPERIMENT_ROOT` placeholder (the convention described in
[release scope](../../common/RELEASE_SCOPE.md)); `import os` was added where
needed. No numeric value, unit or result-table byte changed: every published CSV
is byte-identical to its package hash in `manifest.csv`, and
`PUBLICATION_EDITS.json` retains both the historical package SHA and the
release-copy SHA per file.

## Withheld from this batch

- Allocator checkpoints `assets/models/marginal_mlp_seed_{730101,730102,730103}.pt`,
  `assets/pca32.joblib`, `assets/scaler.joblib`, `configs/roles.csv` (private
  split identity), and the large result tables `predicted_utility.parquet`,
  `selected_record_ids.parquet`, `allocations.parquet`, `post_nms_counts.csv`,
  `per_image_results.csv`: registered with existing sizes and SHA256 in
  [deferred assets](../../manifests/deferred_assets.json); no public download is
  established.
- Internal QA audit logs (`results/qa_closed_loop.md`, `results/qa_small_sample.md`):
  withheld; their identities remain in `manifest.csv`.
- External, not part of this submission: the frozen M11 handoff package, the
  Road8 image set, the detector checkpoint, the dependency environment, and the
  candidate/native caches (see `references/README_CN.md`).

## Interpretation limits

- Seeds 730101/730102/730103 are **three independent trainings**; metrics are
  computed per seed and then averaged (per group), never a prediction ensemble
  (`assets/asset.json` records this).
- All results are DEV2K development results on 50 frozen 40-image groups; they do
  not inherit the main method's confirmation identity.
- The primary protocol is `PFX_EXACT`; `PFX_THEN_NMS` is a reference protocol
  reported separately with its own before/after counts. `NMS_THEN_PFX` was not run.
- `PRE_NMS_TOPN100` is the primary candidate stream; `POST_NMS_TOPN300` is a
  registration-only reference. The two streams are not interchangeable and their
  AP/AR levels differ; neither replaces the other.
- NMS uses the arm's frozen declared threshold; the YOLO-reference IoU 0.70 is
  not adopted as this detector's parameter.
- Coverage/QUALITY are computed on the package-normalized GT; AP/AR on the
  released original GT. The two inputs are not interchangeable.
- Not done (not "shown to be equal"): `NMS_THEN_PFX`, NMS-threshold search,
  runtime/latency measurement, and an external COCO-pretrained baseline.
- Adapter code is adaptation; the allocator, PCA and scaler were refit on this
  detector's FIT slice and the temperatures on the CALIBRATION slice — this is
  not "retraining the adapter".
- Reported values are differences against common baselines within the fixed
  candidate stream; no "effective"/"leading"/"validated" wording before all
  result verification is complete.
