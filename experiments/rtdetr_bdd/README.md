# RT-DETR / BDD reference materials

`research_LC_ALLOC_P1/` contains the actual P1 implementation, configuration,
feature schema, FIT class weights, three temperature files, historical training
summary/history and saved DEV main/class/bootstrap result tables. The source
files preserve their historical internal imports.

## Recorded protocol

RT-DETRv2-R18VD is fixed. Its candidate stream expands each of 300 queries into
eight Road8 class hypotheses, stores score-ranked Top300 and consumes Top100.
Native L3 signals are eight logits and a 256D query representation. The feature
schema contains 58 candidate, 12 preceding-prefix and 20 Top100 image-context
columns. PCA32 and scaling are fitted on FIT only.

TRAIN10K allocator roles are FIT8000, EARLY_STOP1000 and CALIBRATION1000,
deterministically split at the image level. Seeds 530101/530102/530103 use the
same MLP and AdamW recipe. Historical best epochs are 17/14/24, with temperatures
stored separately. Detector-training provenance has limitations recorded in the
project protocol; allocator roles should not be called detector training roles.

DEV2000 has 50 complete 40-image groups. Results in this directory are
development results, not the BDD confirmation tables. Table I(b)'s extended
baseline comparison is in `../baselines_ablation/research_LC_ALLOC_BENCH_P0/`.

## Entry points and dependencies

Read the actual CLI declarations in these files:

| Entry | Required inputs | Creates |
|---|---|---|
| `scripts/prepare_train.py` | explicit project/release roots, TRAIN candidates/native/GT | labels, PCA/scaler and prepared arrays |
| `scripts/train_models.py` | project root and prepared roles/arrays | MLP snapshots, history and temperatures |
| `scripts/predict_allocate.py` | project/release roots, R0 groups and frozen selections | predictions and exact-prefix selections |
| `scripts/evaluate_p1.py` | fixed selections, GT, R0 evaluator and COCO package location | full-split quality and detection results |
| `scripts/summarize_results.py` | saved main/class/group Parquet, TRAIN labels, runtime summary and model metadata | new bootstrap indices, comparisons, summaries and figures |

The first upload does not include the original TRAIN/DEV candidate/native,
role/group manifests, prepared arrays, label parquet, PCA/scaler/MLP binaries,
R0 evaluator/selection bundle, `table_model.npz`, evaluation-stage Parquet
sources or bootstrap indices. Their model/data identities
are registered where available in `../../manifests/deferred_assets.json`.
These stages were not invoked during upload; no incomplete CLI command is
presented as a validated runnable reproduction.

The recorded stack includes Python 3.10.20, torch 2.11.0+cu128, numpy 2.2.6,
pandas 2.3.3, pyarrow 24.0.0, scikit-learn 1.7.2 and scipy 1.15.3. Exact historical
pycocotools version and original shell details remain incomplete. Do not replace
that record with a current machine's package list.
Entry points also import joblib, matplotlib and pycocotools; their exact
historical versions are not all established.

The follow-up includes P1's original `scripts/profile_runtime.py` and saved
runtime summary. The separate `research_LC_ALLOC_EFF_AUDIT/` directory contains
the historical full-pipeline profile source, configuration, environment record
and saved runtime table. These are different timing boundaries, not a rerun.
