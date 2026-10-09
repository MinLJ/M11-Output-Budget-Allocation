"""Freeze R0/R1/R2 nested DEV grouping manifests before allocations."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from common import ROOT, R0, append_log, load_dev_prediction_arrays, sha256_file, write_json, write_parquet


SIZES = (10, 20, 40, 80)


def key(prefix: str, image_id: str) -> str:
    return hashlib.sha256((prefix + image_id).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    start = time.perf_counter()
    append_log("STAGE prepare_groups START")

    ids, _, _, _, _ = load_dev_prediction_arrays()
    if len(ids) != 2000 or len(set(ids)) != 2000:
        raise RuntimeError("DEV identity count mismatch")
    frozen = pq.read_table(R0 / "group_manifest.parquet").to_pandas()
    frozen["image_id"] = frozen["image_id"].astype(str)
    frozen = frozen.sort_values(["group_id", "position_in_group"], kind="stable")
    r0_order = frozen["image_id"].tolist()
    if set(r0_order) != set(ids) or len(r0_order) != len(ids):
        raise RuntimeError("R0 and frozen prediction identity sets differ")

    orders = {
        "R0": [(str(frozen.iloc[i].group_key), r0_order[i]) for i in range(2000)],
        "R1": sorted([(key("LC_ALLOC_BENCH_P0_R1|", x), x) for x in ids], key=lambda z: (z[0], z[1])),
        "R2": sorted([(key("LC_ALLOC_BENCH_P0_R2|", x), x) for x in ids], key=lambda z: (z[0], z[1])),
    }
    rows = []
    for permutation, ordered in orders.items():
        for n in SIZES:
            for position, (sort_key, image_id) in enumerate(ordered):
                group_id = position // n
                members = sorted(x[1] for x in ordered[group_id * n:(group_id + 1) * n])
                rows.append({
                    "permutation": permutation,
                    "group_size": n,
                    "group_id": group_id,
                    "position_in_permutation": position,
                    "position_in_group": position % n,
                    "caller_position": members.index(image_id),
                    "image_id": image_id,
                    "sort_key": sort_key,
                })
    manifest = pd.DataFrame(rows)
    if len(manifest) != 24000:
        raise RuntimeError(f"group manifest rows={len(manifest)}")
    for (p, n), d in manifest.groupby(["permutation", "group_size"]):
        if len(d) != 2000 or d["image_id"].nunique() != 2000 or not (d.groupby("group_id").size() == n).all():
            raise RuntimeError(f"group coverage invariant failed {p} n={n}")
    r0n40 = manifest[(manifest.permutation == "R0") & (manifest.group_size == 40)]
    recovered = {x: set(d.image_id) for x, d in r0n40.groupby("group_id")}
    expected = {x: set(d.image_id) for x, d in frozen.groupby("group_id")}
    if recovered != expected:
        raise RuntimeError("R0 n=40 member parity failed")

    out = root / "outputs" / "group_manifest.parquet"
    write_parquet(manifest, out)
    write_json(root / "qa" / "group_manifest_validation.json", {
        "status": "PASS",
        "rows": len(manifest),
        "dev_unique_images_per_config": 2000,
        "configs": 12,
        "r0_n40_exact_membership_parity": True,
        "r0_source_sha256": sha256_file(R0 / "group_manifest.parquet"),
        "manifest_sha256": sha256_file(out),
        "elapsed_seconds": time.perf_counter() - start,
    })
    append_log(f"STAGE prepare_groups COMPLETE rows={len(manifest)} sha256={sha256_file(out)} elapsed_seconds={time.perf_counter()-start:.6f}")


if __name__ == "__main__":
    main()
