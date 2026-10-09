# Baselines, ablations and development analyses

Each historical directory retains its own source and result relationships.
The similarly named `common.py` files are separate implementations; they were
not merged into a new shared library.

| Directory | Actual comparison | Paper relation |
|---|---|---|
| `research_LC_ALLOC_P1A` | M00/M10/M01/M11 value aggregation from the same ten-output models | Table V mechanism controls |
| `research_LC_ALLOC_P1B` | frozen FIT class weight × raw score, exact prefix DP | Class-score in Table I(b) |
| `research_LC_ALLOC_P2` | global score temperature and isotonic allocation sanity checks | Calibration development context |
| `research_LC_ALLOC_BENCH_P0` | class-specific Platt fixed-prefix adapters, group sensitivity, primary timing | Tables I(b), VI and grouping discussion |
| `research_LC_ALLOC_ABLATION_P0` | retrained NO_EMBED/NO_PREFIX/MATCHABILITY_TARGET and FULL next-slot greedy | Table V and separate matched profile |

M00 uses p(.50), M10 uses class-weighted p(.50), M01 uses the ten-probability
mean, and M11 class-weights that mean. This tests inference aggregation, not
the necessity of multi-threshold training. The three models are never averaged
at the prediction stage.

NO_EMBED masks 33 standardized columns (PCA32 and native-vector norm), retaining
class signals. NO_PREFIX masks 12 standardized prefix columns, retaining rank
and image context. Each is trained from the beginning with the same 90D network.
MATCHABILITY_TARGET uses candidate-level matchability instead of prefix gains.
Next-slot greedy reuses FULL predictions and changes only allocation. These
post-confirmation DEV controls are exploratory; none inherits the original
confirmation identity or changes the frozen main method.

Global temperature is a score-only sanity check where its allocation equals
Score under the frozen exact-budget/tie protocol. It is not the class-specific
Platt baseline. BENCH supplies two adapters from one class-specific calibrator:
`PS_PREFIX` and `PS_CLASS_PREFIX`. Their saved aggregate results remain together.
The historical M11-minus-weighted-Platt comparison had negative coverage-recall
differences for bicycle and motorcycle; the detailed Platt class-result asset
is not included in this batch. The fitting source/third-party port is
deferred; see [calibration provenance](CALIBRATION_PROVENANCE.md).

BENCH grouping changes use the same DEV2000 under three deterministic
permutations and sizes 10/20/40/80; these are not 6,000 independent images.
Each configuration bootstraps its own complete groups. Saved positive-resample
fractions are not conventional p-values and multiple exploratory intervals do
not claim global error-rate control.

## Historical timing sets

Table VI uses BENCH FIT400, K15, seed530101, 3 warm-ups and 10 measurements per
actual group. At n40 this is 10 groups and 100 recorded calls per policy.
The separate matched ablation profile uses FIT200, five groups, 5 warm-ups and
20 measurements per group. Keep their `runtime_*` files separate.
Both start with required candidate/native state already in host memory and end
at returned K and record IDs. Detector execution, disk loading, initialization,
GT evaluation and output saving are excluded. A 40-image cost divided by 40 is
amortized cost, not single-frame response latency. Cached component medians
cannot be summed to create a full-path median.

## Dependencies and entrypoints

The uploaded code includes actual allocation/evaluation/summary/profile stages,
and the three ablation preparation/training stages. P1A/P1B/P2/BENCH/ABLATION
depend on their specifically referenced P1 sources and historical bundles,
R0 groups/score implementation/evaluator, prepared data and saved indices.
Those external assets and some stage dependencies are not included. The
follow-up also supplies P1B's saved evaluation, analysis, profiling and summary
stages, implementation parity table, runtime samples and runtime reconciliation
note. P1A's single-model profile and P2's calibration analysis/evaluation/profile
stages and saved timing tables are included without running them again.

Source/default paths changed only in new publication copies. Config placeholders
and retained historical hash checks must be resolved before execution; no
full command or relocated-run success is claimed here. The source manifest
records exact original/release hashes. The CSV files remain byte-identical.
