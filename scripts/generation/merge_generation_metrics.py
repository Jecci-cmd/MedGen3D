#!/usr/bin/env python3
"""Merge StyleGAN-V/TorchMetrics generation metric feature shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_generation_metrics import PROTOCOL, styleganv_frechet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    args = parser.parse_args()

    keys = ("real_fid", "fake_fid", "real_fvd", "fake_fvd", "clip_score")
    arrays = {key: [] for key in keys}
    rows = []
    for index in range(args.num_shards):
        path = args.features_root / f"features_{index}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        for key in keys:
            arrays[key].append(data[key])
        rows.extend(json.loads(str(data["rows_json"])))
    rows.sort(key=lambda row: row["index"])
    if len(rows) != args.expected_samples or len({row["case_id"] for row in rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation patients")
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    result = {"protocol": PROTOCOL, "samples": len(rows), "summary": {
        "fid": styleganv_frechet(merged["real_fid"], merged["fake_fid"]),
        "fvd_i3d_16f": styleganv_frechet(merged["real_fvd"], merged["fake_fvd"]),
        "clip_score": float(np.mean(merged["clip_score"])),
    }, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
