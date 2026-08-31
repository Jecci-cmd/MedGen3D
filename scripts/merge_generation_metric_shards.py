#!/usr/bin/env python3
"""Merge FID/FVD-CT feature shards written by evaluate_generation_metrics.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm


def frechet(first: np.ndarray, second: np.ndarray) -> float:
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
    arrays = {key: [] for key in ("real_2d", "fake_2d", "real_3d", "fake_3d")}
    rows = []
    for index in range(args.num_shards):
        path = args.features_root / f"features_{index}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        for key in arrays:
            arrays[key].append(data[key])
        rows.extend(json.loads(str(data["rows_json"])))
    rows.sort(key=lambda row: row["index"])
    if len(rows) != args.expected_samples or len({row["case_id"] for row in rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation patients")
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    result = {"protocol": "ctclip3d_fvd_inceptionv3_slice_fid_v1", "samples": len(rows),
              "summary": {"fid": frechet(merged["real_2d"], merged["fake_2d"]),
                          "fvd_ct": frechet(merged["real_3d"], merged["fake_3d"])}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
