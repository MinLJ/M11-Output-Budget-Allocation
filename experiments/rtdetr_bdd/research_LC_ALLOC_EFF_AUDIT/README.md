# Existing full-pipeline latency material

This directory publishes the original owned `scripts/profile_full_pipeline.py`
and `scripts/prepare_outputs.py`, saved `run_config.json`, `environment.json`
and `runtime_pipeline_results.csv`. Result bytes are unchanged. No timing was
run during publication.

The historical profile measures fixed RT-DETR plus M11 processing from
host-resident preprocessed tensors, with detector-only and full-pipeline
boundaries kept separate. It is not camera-to-output latency or evidence of
detector acceleration. Consult the actual source/configuration for exclusions
and environment; historical absolute defaults remain in these unmodified files.

The frozen detector implementation/checkpoint, image inputs, release assets,
P1/P1A/P1B model/preprocessor bundles and original profile inputs are external
dependencies, not included here. This archive does not claim a newly executed
or standalone reproduction.
