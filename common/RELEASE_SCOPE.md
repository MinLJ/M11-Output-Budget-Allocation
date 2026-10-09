# Initial publication scope

This submission publishes author-maintained source, selected configurations and
saved small results. It does not assign a repository-wide MIT or other license,
and does not override component-specific rights. Third-party code, weights and
data remain subject to their own licenses and release permissions.

The internal M11 handoff package is withheld. The ECCV 2024 calibration fitting
implementation and the source-inspected Windows matching/oLRP port are withheld
pending separate component handling. The owner-maintained fixed-prefix budget
adapter, frozen calibration parameters and their saved results are included;
this is not a complete reproduction of that paper's experiments.

Source filenames and hashes in `manifests/` identify historical originals.
Some publication copies have path-only edits. Integrity-locked historical
entrypoints retain their historical hash checks; changing paths in a release
copy does not make it equivalent to the historical bound bytes. Those paths
and external bundles must be explicitly resolved and source bindings reviewed
before later execution. No scientific execution of relocated copies is claimed.

`M11_EXPERIMENT_ROOT` in path-edited source defaults denotes an external asset
workspace containing the historical project subdirectories. `M11_COCO_ROOT`
denotes a separately obtained COCO data root. `M11_ENVIRONMENT_ROOT` and
`M11_COCO_SITE_PACKAGES` only locate an existing dependency environment. These
variables do not download assets or alter the scientific recipe.

Do not use template configs or default historical entrypoints against published
result directories: some stages create outputs or refuse existing completed
runs. This first batch provides source and provenance, not turnkey commands
with all withheld inputs. Original CSVs are immutable archival results.
