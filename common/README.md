# Shared scientific contracts

The reference implementation is the historical P1 code under
`../experiments/rtdetr_bdd/research_LC_ALLOC_P1/scripts/`:

- `p1_core.py`: ordered 90D features, same-class maximum matching, prefix labels,
  temperature handling and utility helpers.
- `dp_solver.py`: float64 exact multiple-choice allocation and declared tie rule.
- `train_models.py`: 90→128→64→10 MLP, LayerNorm/GELU, natural unweighted BCE,
  early stopping and per-model temperature fitting.
- `release_io.py`: record/query joins for the frozen RT-DETR release.

These are referenced in their original experiment location rather than copied
into a new unified library. P1A, BENCH and ABLATION retain their own historical
helpers. The internal handoff library is not included in this public submission.

The original ranks 1–5 participate in maximum matching and are mandatory outputs.
Labels for ranks 6–50 are the change in prefix maximum cardinality, not a single
candidate's matchability. Ten thresholds span IoU .50:.05:.95. Class weights affect
inference utility and its QUALITY evaluation, not BCE. Native features join by
image and query identity, never rank; multiple class records can share a query.

DP optimizes predicted utility within the declared discrete feasible set.
`Uhat(5)=0` omits a common optimization constant, not the first five outputs.
Maximum matching and legacy greedy TP are separate measurements. AP/AR retain
the original detector score, class and box.
