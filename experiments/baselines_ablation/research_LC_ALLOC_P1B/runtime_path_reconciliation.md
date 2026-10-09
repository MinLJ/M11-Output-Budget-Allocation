# Runtime path reconciliation

`historical_timing_status = PARTIALLY_EXPLAINED`

## What the historical numbers measured

- R0 `PILOT_FIXED_CACHED_40` reported **6.598 ms median** for one cached 40-image list-of-dicts input, one ranking pass, all five budgets, CPU/BLAS one thread, and K allocations only; it did not materialize selected record IDs.
- P1A reported roughly **64–65 ms median** per single budget for `S_ADAPT`. Its timed path rebuilt 4,000 full Python dictionaries (about 56,000 fields) from 40 DataFrames, ranked them, solved one budget, materialized record IDs, and was interleaved with GPU model paths under 12 torch threads.
- R0's full-DEV score allocation was 0.3032 s for 50 groups and five budgets, or about 6.063 ms/group, consistent with its cached PILOT boundary.

These are different boundaries; `64/6.598` is not a valid speed-regression ratio. The near-constant P1A latency over K10/K15/K20 also points to adapter/object creation rather than capacity-dependent allocation.

## Current same-machine, same-input, same-boundary measurements

Every row below starts from its policy's required host-resident raw state and ends with K, ordered candidate record IDs, and objective. Each median/p95 uses 100 samples over the same five frozen FIT groups; budgets are solved separately.

| Path | K10 median / p95 ms | K15 median / p95 ms | K20 median / p95 ms |
|---|---:|---:|---:|
| S ref | 24.591 / 26.224 | 24.633 / 26.297 | 24.720 / 25.772 |
| S opt | 1.239 / 1.519 | 1.292 / 1.443 | 1.293 / 1.496 |
| S_CLASS | 21.153 / 21.674 | 24.266 / 25.052 | 25.690 / 26.310 |
| M11 ref | 165.255 / 171.652 | 168.838 / 180.216 | 170.710 / 180.262 |
| M11 opt | 122.163 / 127.795 | 125.836 / 130.423 | 127.205 / 132.331 |

The accepted score fast path first proves that detector score order equals frozen `road8_rank`, then calls the frozen exact score allocator and materializes the same prefix IDs. Across all profile groups/budgets, K, ordered IDs, objective float64 bytes and capacity were identical.

The accepted M11 implementation batches the unchanged PCA/scaler work. The 30,000 DEV K cells and corresponding candidate-ID prefixes had zero mismatch. Saved-probability differences were at most 2.452e-7, within atol=1e-6/rtol=1e-5; this is tolerance equivalence plus exact selection parity, not a claim of universal bitwise identity.

P1B does not fully recreate the historical scheduling epoch or R0's one-thread/no-ID boundary. The historical gap is therefore structurally explained but not converted into a same-boundary numerical comparison; the honest status is `PARTIALLY_EXPLAINED`.
