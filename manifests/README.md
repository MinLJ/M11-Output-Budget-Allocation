# Publication manifests

`source_release_manifest.json` lists each copied source/config/result with its
historical project-relative locator, historical SHA, new copy SHA, size and
path-only changes. It contains no machine root or internal inventory IDs.
CSV results have `csv_bytes_unchanged: true`.

`deferred_assets.json` registers selected model/preprocessor/prepared/selection
assets using existing identities, sizes and checksums. No large payloads were
loaded or rehashed during upload. `public_download_url: null` means no public
download location has been established; it is not a fabricated URL or proof
that the historical asset never existed.

These manifests support source and release provenance. They are not evidence
that training, frozen inference or scientific evaluation was reproduced in
this upload. Current publication hashes do not replace historical bindings.
