# Team contributions

Submit changes through a normal pull request; do not merge another member's
branch or replace their experiment directory.

| Owner | Branch convention | Directory |
|---|---|---|
| Lejian Min | `upload/lejian-initial` | `common/`, `experiments/rtdetr_bdd/`, `experiments/coco/`, `experiments/baselines_ablation/`, `manifests/` |
| Ruiqi Guo | `upload/yolo-initial` | `experiments/yolov8/` |
| Zhaolin Dong | `upload/faster-rcnn-initial` | `experiments/faster_rcnn/` |

For each experiment, preserve its actual source version, configuration, seed
identities, preprocessing, candidate stream, grouping and original result units.
Include small raw result tables and a README connecting them to the relevant
paper items. Describe dependencies that have not yet been released. Do not
silently replace a historical helper with a different shared-library version.

Keep three-model metric averaging distinct from a prediction ensemble. Record
development-set exposure, target-domain adaptation, evaluation qualifications,
unequal-budget references and class trade-offs where relevant.

Candidate/native caches, annotations, checkpoints, private split identities,
credentials, environments and internal audit logs are outside this initial
source upload. Register large assets with their existing identities and an
accurate download status; do not invent a public URL. Respect original component
licenses and data permissions. Shared-file changes belong in the contributor's
PR description and are reviewed separately.

When publishing path-edited copies, retain both historical and release hashes.
Do not describe a static upload check as completed scientific reproduction.
