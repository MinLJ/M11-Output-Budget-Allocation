# External calibration source and adapter boundary

Source: *On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines*,
ECCV 2024.

- Paper: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf>
- Official implementation: <https://github.com/fiveai/detection_calibration>
- Reviewed source snapshot: `bf59c92f5416ac4b66a7c2c67b44487777ff2ea4`.

The original calibration script has CC BY-NC-SA 4.0 terms; other components in
the original repository have their own terms. No blanket license is assigned
to these components by this upload. The original fitting source and the
Windows matching/oLRP port are not included in this submission.

The saved calibrator uses per-class nonnegative Platt scale and bias, fitted
on FIT8000 records with the original calibration target and filtering/threshold
protocol. CALIBRATION1000 is diagnostic-only, with no refit or parameter
selection. The included author-maintained adapter applies the frozen mapping
to all Top100 records, including records below the FIT filtering threshold;
that application is explicit extrapolation beyond the fitting selection.
The adapter optionally multiplies frozen LC class weights and uses
the existing fixed-prefix exact DP. It does not delete candidates using the
original method's operating threshold or change COCO evaluation scores.

The Windows port was source-inspected; saved dynamic exact-parity evidence
against the original native extension is unavailable. The results are
class-specific Platt calibration with a fixed-prefix budget adapter, not a
claim to reproduce the paper's entire benchmark or dominate all calibration.
