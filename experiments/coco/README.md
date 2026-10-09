# RT-DETR / COCO2017 Road8

This directory preserves three related historical roots:

- `research_LC_ALLOC_COCO_ROUTE_B_SELECTION/`: frozen LC allocator selection code.
- `research_LC_ALLOC_COCO_ROUTE_B_EVALUATION/`: independent fixed-selection
  evaluator and saved full-val results.
- `research_LC_ALLOC_COCO_ROUTE_A_ADAPTATION/`: target-domain preprocessing,
  allocator training, selection/evaluation code and saved fitted results.

These correspond to paper Table II and its discussion. The original method
identifiers remain in the result files: `S_FIXED`, `S_ADAPT`, `CAL_TEMP_ALLOC`,
`M11` in Route B, and `M11-COCO` in Route A. Read the raw method columns
rather than infer an identity from a filename.

## Shared evaluation contract

Both routes use all 5,000 val2017 images, including empty Road8-GT images,
125 groups of 40 and the same canonical candidate asset:
`5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707`.
The detector, original-score order, Top100 records, 5–50 bounds, exact five
budgets and LC class weights are shared. The evaluation is Road8, not COCO80.
Coverage/QUALITY use valid noncrowd same-class maximum matching; COCO bbox
AP/AR retain their crowd/ignore/area rules and original detector scores.

Frozen LC uses seeds 530101–530103 and the complete LC PCA/scaler/model/temperature
set. COCO-fitted uses 630101–630103, target-domain PCA/scaler and temperature,
and the **same LC class weights**. Its train2017 roles are
94629/11829/11829 FIT/EARLY_STOP/CALIBRATION images. All seeds are evaluated
independently. Existing 95% intervals use 125 paired groups, 5,000 resamples and
seed 530002; AP/AR are full-split point estimates, not average group AP.

Adaptation followed observation of the degraded frozen-transfer results.
COCO-fitted restores much of that degradation but on the core budgets (10, 15,
20) retains a QUALITY versus coverage/AP/AR trade-off against Score. val2017 was detector development/selection
data. Neither route should be described as detector-untouched confirmation.
Route B did not save complete per-slot prediction curves; ranking recovery is
not established by an unavailable prediction-error analysis.

## Actual entry points

Route A preparation/training are in `scripts/prepare_route_a.py` and
`scripts/train_route_a.py`, using `scripts/route_a_common.py`. Selection and
evaluation use `working/routea_select_val.py`, `routea_evaluate_val.py` and
their separate `routea_pipeline_common.py`. Route B selection and evaluation
are `working/route_b_selection_runner.py` and `working/evaluate_route_b.py`.
They are historical stage implementations, not a new consolidated launcher.

Saved `adapted_main_results.csv`, class/COCO JSONs and existing bootstrap files
are from the Route A completed evaluation outputs. Route B's top-level tables
are saved evaluation outputs. Numeric CSV bytes are unchanged.

Path-normalized config copies and modified defaults are marked in the source
manifest. Historical source-hash checks remain in the entrypoints. Some
cross-project paths still require explicit resolution to the original frozen
bundle; the relocated pipeline has not been executed. Large canonical assets,
image/group manifests, annotations, models/preprocessors, prepared arrays,
prediction curves and selected-record parquet are deferred. There is no
established public download URL for them in this batch.

The recorded selection stack is Python 3.10.20, torch 2.11.0+cu128, CUDA 12.8,
numpy 2.2.6, pandas 2.3.3, pyarrow 24.0.0, scikit-learn 1.7.2, joblib 1.5.3,
RTX 5080 Laptop GPU. Some historical stages used a separate existing pycocotools
site; its exact version is unrecorded. This upload does not run any of the stages.
