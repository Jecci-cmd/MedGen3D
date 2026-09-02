#!/usr/bin/env python3
"""Create the CT-CLIP geometry CSV for a frozen generation manifest.

Use this only when a dataset does not supply CT-RATE-style DICOM metadata.
The output records each NIfTI target's native voxel spacing and NIfTI scale
fields, so that CT-CLIP preprocessing is explicit and offline reproducible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != args.expected_samples:
        raise SystemExit(f"Expected {args.expected_samples} manifest rows, found {len(records)}")
    rows = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        inverse = record.get("inverse_transform")
        if not case_id or not isinstance(inverse, dict) or not isinstance(inverse.get("target_path"), str):
            raise SystemExit("Each manifest row requires case_id and inverse_transform.target_path")
        path = Path(inverse["target_path"])
        image = nib.load(str(path))
        zooms = image.header.get_zooms()
        slope, intercept = image.header.get_slope_inter()
        rows.append({"VolumeName": case_id, "ZSpacing": float(zooms[2]),
                     "XYSpacing": f"({float(zooms[0])}, {float(zooms[1])})",
                     "RescaleSlope": 1.0 if slope is None else float(slope),
                     "RescaleIntercept": 0.0 if intercept is None else float(intercept)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["VolumeName", "ZSpacing", "XYSpacing", "RescaleSlope", "RescaleIntercept"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} with {len(rows)} geometry rows")


if __name__ == "__main__":
    main()
