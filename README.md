# M11-Output-Budget-Allocation
Code, configurations, and reproducibility materials for Learning Marginal Utility for Resource-Aware Output Budget Allocation in Object Detection.

M11 allocates a limited number of output records after a fixed object detector.
Each image keeps an original-score prefix; the allocator changes only its length.
The reference setting uses Top100 candidate records, per-image lengths 5–50,
40-image groups and exact mean budgets 10, 15, 20, 30 and 40. Candidate records
from the same query are retained separately. QUALITY is a class-weighted,
multi-IoU maximum-matching proxy; AP/AR use the original detector scores.

## Initial materials

This submission contains real historical source files, configurations and saved,
unrounded small result tables from the following branches:

| Directory | Contents | Paper connection |
|---|---|---|
| [RT-DETR/BDD](experiments/rtdetr_bdd/README.md) | P1 feature, label, model and exact-DP code; FIT settings; DEV results | Reference method and BDD development protocol |
| [COCO](experiments/coco/README.md) | Frozen LC transfer and COCO-fitted source/config/result sets | Table II and target-domain analysis |
| [Baselines and ablations](experiments/baselines_ablation/README.md) | Aggregation controls, weighted score, calibration adapters, retrained controls, group sensitivity and timing records | Tables I(b), V, VI |
| [Common contracts](common/README.md) | Data and evaluation conventions; dependencies on the separate historical implementations | Shared interpretation |
| [Manifests](manifests/README.md) | Source/release hashes and deferred large-asset identities | Publication provenance |

YOLOv8 and Faster R-CNN submissions have designated member directories described
in [CONTRIBUTING.md](CONTRIBUTING.md). Their materials are contributed through
separate pull requests. This first submission does not assess their completion.

## Scope and use

The CSV results are copied byte-for-byte; no result values were regenerated.
Machine-local paths were normalized only in selected publication copies. The
source manifest distinguishes the historical source hashes from release-copy
hashes. Configurations containing `external_assets/` are archival templates:
resolve their paths explicitly before any future execution.

This is an initial source/result archive, not a self-contained reproduction
bundle. Models, PCA/scaler binaries, candidate/native caches, annotations,
private evaluation payloads, split identities and selected-record arrays are
not included. No public download location for those assets is established by
this submission. See [release scope](common/RELEASE_SCOPE.md) and the manifests.

The upload was checked for file scope, machine paths, secret markers, syntax and
unchanged result bytes. Training, detector/allocator inference, allocation,
GT evaluation, bootstrap and timing were not executed for this submission.

## Interpretation limits

Three seeded models are allocated and evaluated independently before averaging
metrics. Core results average budgets 10/15/20 equally. DEV has been observed;
post-hoc controls are development evidence. Image groups are offline allocation
groups, not validated video or communication windows. Allocator timing excludes
the detector, and per-image figures are amortized group costs.

COCO frozen transfer and COCO-fitted adaptation are separate assets. Adaptation
followed observation of the frozen-transfer results; val2017 was also used for
detector development/selection. The COCO-fitted core-budget results (10/15/20)
include a QUALITY versus coverage/AP/AR trade-off and should not be described
as uniform improvement.

The BDD confirmation has a disclosed early truncated-GT-retrieval deviation;
the historical report states it was not used to change the frozen method. The
confirmation should retain that qualification rather than claim a completely
untouched end-to-end procedure. Confirmation-specific tables and private
payloads are not part of this initial submission.

This repository does not claim detector acceleration, real-time operation,
communication savings, or a COCO80 leaderboard result.
