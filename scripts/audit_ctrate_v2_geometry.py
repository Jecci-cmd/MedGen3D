#!/usr/bin/env python3
"""Fail closed unless every CT-RATE V2 volume has the canonical geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, default=1.5)
    parser.add_argument("--xy-shape", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    expected = {"train": 1000, "val": 100, "test": 200}
    patients: dict[str, set[str]] = {}
    depth_min, depth_max, checked = float("inf"), 0, 0
    for split, count in expected.items():
        manifest = root / "manifests" / f"{split}.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != count:
            raise RuntimeError(f"{split}: expected {count} rows, found {len(rows)}")
        ids = {str(row.get("patient_id")) for row in rows}
        if len(ids) != count:
            raise RuntimeError(f"{split}: patient IDs must be unique and non-empty")
        patients[split] = ids
        for row in rows:
            if row.get("task") != "generation" or row.get("split") != split:
                raise RuntimeError(f"Malformed {split} row: {row.get('case_id')}")
            meta, inverse = row.get("metadata", {}), row.get("inverse_transform", {})
            array = np.load(resolve(root, row["target"]), mmap_mode="r")
            image = nib.load(resolve(root, inverse["canonical_target_path"]))
            expected_dhw = tuple(reversed(tuple(int(v) for v in inverse["canonical_shape_xyz"])))
            if tuple(array.shape) != expected_dhw or tuple(array.shape) != (array.shape[0], *args.xy_shape):
                raise RuntimeError(f"Shape mismatch for {row['case_id']}: npy={array.shape}, expected={expected_dhw}")
            if tuple(image.shape) != tuple(reversed(array.shape)):
                raise RuntimeError(f"NIfTI/npy axis mismatch for {row['case_id']}")
            if "".join(nib.aff2axcodes(image.affine)) != "RAS":
                raise RuntimeError(f"Non-RAS volume: {row['case_id']}")
            spacing = image.header.get_zooms()[:3]
            if not np.allclose(spacing, (args.spacing_mm,) * 3, atol=1e-4):
                raise RuntimeError(f"Spacing mismatch for {row['case_id']}: {spacing}")
            if not np.allclose(image.affine, np.asarray(inverse["canonical_affine"]), atol=1e-4):
                raise RuntimeError(f"Affine mismatch for {row['case_id']}")
            if meta.get("orientation") != "RAS" or not np.allclose(meta.get("spacing_xyz_mm"), (args.spacing_mm,) * 3):
                raise RuntimeError(f"Manifest geometry mismatch for {row['case_id']}")
            depth_min, depth_max = min(depth_min, array.shape[0]), max(depth_max, array.shape[0])
            checked += 1
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Patient leakage across splits")
    result = {"status": "passed", "version": "ctrate-v2-geometry-audit-v1", "checked": checked,
              "splits": expected, "orientation": "RAS", "spacing_xyz_mm": [args.spacing_mm] * 3,
              "xy_shape": list(args.xy_shape), "z_depth_range": [depth_min, depth_max]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
