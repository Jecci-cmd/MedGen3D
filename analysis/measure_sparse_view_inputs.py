#!/usr/bin/env python3
"""Measure untrained sparse-view FBP input quality on the fixed test cohort."""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


VIEWS = (10, 15, 18, 20)
PATCH_SHAPE = (97, 96, 96)
HU_CLIP = (-1000.0, 1000.0)
DATA_RANGE_HU = 2000.0


def crop_bounds(start: tuple[int, int, int], shape: tuple[int, int, int], available: tuple[int, int, int]):
    source, destination = [], []
    for origin, length, limit in zip(start, shape, available):
        lo, hi = max(0, origin), min(limit, origin + length)
        source.append(slice(lo, hi))
        destination.append(slice(lo - origin, hi - origin))
    return tuple(source), tuple(destination)


def sparse_view_slice(hu: np.ndarray, views: int, pixel_size_mm: float) -> np.ndarray:
    from skimage.transform import iradon, radon

    mu = np.clip(0.02 * (hu.astype(np.float32) / 1000.0 + 1.0), 0.0, None)
    height, width = mu.shape
    side = max(height, width)
    before_y, before_x = (side - height) // 2, (side - width) // 2
    padded = np.pad(
        mu,
        ((before_y, side - height - before_y), (before_x, side - width - before_x)),
        mode="constant",
    )
    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32)
    sino = radon(padded * pixel_size_mm, theta=angles, circle=False, preserve_range=True)
    reconstruction = iradon(
        sino, theta=angles, filter_name="ramp", circle=False,
        output_size=side, preserve_range=True,
    )
    reconstruction = reconstruction[
        before_y:before_y + height, before_x:before_x + width
    ] / pixel_size_mm
    return (reconstruction.astype(np.float32) / 0.02 - 1.0) * 1000.0


def evaluate(payload: tuple[str, str, str, int]) -> dict[str, float | str | int]:
    case_id, ct_path, mask_path, views = payload
    ct_image = nib.load(ct_path)
    ct_xyz = np.asanyarray(ct_image.dataobj).astype(np.float32)
    label_zyx = np.transpose(np.asanyarray(nib.load(mask_path).dataobj), (2, 1, 0))
    foreground = np.argwhere(label_zyx > 0)
    center = (
        np.rint((foreground.min(0) + foreground.max(0)) / 2).astype(int)
        if len(foreground) else np.asarray(label_zyx.shape) // 2
    )
    start = tuple(int(c - size // 2) for c, size in zip(center, PATCH_SHAPE))
    source, _ = crop_bounds(start, PATCH_SHAPE, label_zyx.shape)
    z_slice, y_slice, x_slice = source
    target = np.transpose(ct_xyz[x_slice, y_slice, z_slice], (2, 1, 0))
    reconstructed = []
    pixel_size_mm = float(ct_image.header.get_zooms()[0])
    for z in range(z_slice.start, z_slice.stop):
        axial = ct_xyz[:, :, z].T
        reconstructed.append(sparse_view_slice(axial, views, pixel_size_mm)[y_slice, x_slice])
    prediction = np.stack(reconstructed)
    target = np.clip(target, *HU_CLIP).astype(np.float64)
    prediction = np.clip(prediction, *HU_CLIP).astype(np.float64)
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error * error))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(DATA_RANGE_HU) - 10.0 * math.log10(mse)
    return {
        "case_id": case_id, "views": views, "mae_hu": mae, "psnr_hu": psnr,
        "voxels": int(target.size), "squared_error_sum": float(np.sum(error * error)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in
        (args.dataset_root / "processed/manifests/test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[:args.limit]
    payloads = []
    for record in records:
        ct = Path(record["image"]); mask = Path(record["mask"])
        if not ct.is_absolute(): ct = args.dataset_root / ct
        if not mask.is_absolute(): mask = args.dataset_root / mask
        payloads.extend((str(record["case_id"]), str(ct), str(mask), views) for views in VIEWS)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate, payload): payload for payload in payloads}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result(); rows.append(row)
            print(json.dumps({"done": done, "total": len(payloads), **row}), flush=True)
    rows.sort(key=lambda row: (int(row["views"]), str(row["case_id"])))
    summary = {}
    for views in VIEWS:
        selected = [row for row in rows if int(row["views"]) == views]
        voxels = sum(int(row["voxels"]) for row in selected)
        pooled_mse = sum(float(row["squared_error_sum"]) for row in selected) / voxels
        summary[str(views)] = {
            "num_cases": len(selected),
            "mae_hu_mean_patient": float(np.mean([row["mae_hu"] for row in selected])),
            "mae_hu_median_patient": float(np.median([row["mae_hu"] for row in selected])),
            "psnr_hu_mean_patient": float(np.mean([row["psnr_hu"] for row in selected])),
            "psnr_hu_median_patient": float(np.median([row["psnr_hu"] for row in selected])),
            "mae_hu_pooled_voxel": sum(float(row["absolute_error_sum"]) for row in selected) / voxels,
            "psnr_hu_pooled_voxel": 20.0 * math.log10(DATA_RANGE_HU) - 10.0 * math.log10(pooled_mse),
        }
    output = {
        "protocol": {
            "views": list(VIEWS), "num_cases": len(records), "patch_shape_zyx": list(PATCH_SHAPE),
            "geometry": "parallel_beam", "angles_deg": "[0,180), uniform, endpoint=False",
            "filter": "ramp", "mu_water_per_mm": 0.02, "hu_clip": list(HU_CLIP),
            "metric_data_range_hu": DATA_RANGE_HU,
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
