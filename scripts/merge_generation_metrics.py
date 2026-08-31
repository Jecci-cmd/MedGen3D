#!/usr/bin/env python3
"""Merge feature shards from evaluate_generation_metrics.py and score all four metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

PROTOCOL = "ctrate_v2_inceptionv3_fid_i3d_fvd_ctclip_t2i_i2i_v1"


def frechet(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Fréchet metrics require at least two volumes")
    mu1, mu2 = first.mean(0), second.mean(0)
    cov1, cov2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    eye = np.eye(cov1.shape[0]) * 1e-6
    covmean = sqrtm((cov1 + eye) @ (cov2 + eye))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - 2 * covmean))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    args = parser.parse_args()
    keys = ("real_fid", "fake_fid", "real_fvd", "fake_fvd", "t2i", "i2i")
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
        "fid": frechet(merged["real_fid"], merged["fake_fid"]),
        "fvd": frechet(merged["real_fvd"], merged["fake_fvd"]),
        "ct_clip_t2i": float(np.mean(merged["t2i"])),
        "ct_clip_i2i": float(np.mean(merged["i2i"])),
    }, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
