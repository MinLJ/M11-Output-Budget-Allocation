# Existing COCO canonical export implementation

`working/canonical_export_runner.py` and `canonical_export_validator.py` are the
actual owned export/readback stages. `working/run_config.json`,
`working/runtime_records.csv` and `runtime_summary.csv` are historical records,
not a new export. All five files are copied without altering their bytes.

The runner consumes image-only inputs and fixed detector/export identities;
it does not take annotation/GT, allocator or evaluator inputs. The canonical
protocol produces Top300 candidate records and 300 native rows per image; M11
consumes Top100. See the separate frozen-contract directory for the full schema.

The implementation depends on the external frozen DetectorSnapshot module,
RT-DETR repository/checkpoint, COCO images and image-only manifests. These are
not bundled. Original path defaults and source/hash checks remain: no current
portability or replay claim is made and neither entrypoint was executed here.
