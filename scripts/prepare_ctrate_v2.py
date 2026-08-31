#!/usr/bin/env python3
"""Create a physically canonical CT-RATE corpus for full-volume generation.

The old preparation path resized image indices only.  This script first
reorients every scan to RAS and resamples it to a fixed physical spacing, then
uses a fixed in-plane field of view while retaining the complete z extent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import pandas as pd


def patient_id(name: str) -> str:
    match = re.match(r"((?:train|valid)_\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"Cannot derive patient from {name}")
    return match.group(1)


def report_text(row: pd.Series) -> str:
    findings = str(row.get("Findings_EN", row.get("Findings", ""))).strip()
    impression = str(row.get("Impressions_EN", row.get("Impression", ""))).strip()
    return f"Findings: {findings}\nImpression: {impression}".strip()


def select_one(table: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    table = table.copy()
    table["patient_id"] = table["VolumeName"].map(patient_id)
    table = table.sort_values("VolumeName").drop_duplicates("patient_id", keep="first")
    return table.sample(n=count, random_state=seed, replace=False).sort_values("patient_id")


def crop_or_pad_xy(image: nib.Nifti1Image, xy_shape: tuple[int, int]) -> nib.Nifti1Image:
    """Keep a fixed physical XY field centered on the patient's body."""
    array = np.asarray(image.dataobj, dtype=np.float32)
    body = array > -800.0
    coordinates = np.where(body)
    if not coordinates[0].size:
        raise ValueError("No body voxels found after resampling")
    # Robust quantiles ignore thin table/edge artefacts without requiring a
    # task-specific lung segmentation model.
    center = [int(round((np.quantile(axis, .05) + np.quantile(axis, .95)) / 2))
              for axis in coordinates[:2]]
    starts = [point - size // 2 for point, size in zip(center, xy_shape)]
    output = np.full((xy_shape[0], xy_shape[1], array.shape[2]), -1000.0, dtype=np.float32)
    source_slices, target_slices = [], []
    for start, size, available in zip(starts, xy_shape, array.shape[:2]):
        source_low, source_high = max(start, 0), min(start + size, available)
        target_low = source_low - start
        source_slices.append(slice(source_low, source_high))
        target_slices.append(slice(target_low, target_low + max(source_high - source_low, 0)))
    output[target_slices[0], target_slices[1], :] = array[source_slices[0], source_slices[1], :]
    translation = np.eye(4, dtype=np.float64)
    translation[:2, 3] = starts
    return nib.Nifti1Image(output, image.affine @ translation)


def canonicalize(source: Path, spacing: float, xy_shape: tuple[int, int]) -> nib.Nifti1Image:
    image = nib.as_closest_canonical(nib.load(source))
    image = resample_to_output(image, voxel_sizes=(spacing, spacing, spacing), order=1)
    image = crop_or_pad_xy(image, xy_shape)
    if "".join(nib.aff2axcodes(image.affine)) != "RAS":
        raise AssertionError("Canonical image is not RAS")
    zooms = image.header.get_zooms()[:3]
    if not np.allclose(zooms, (spacing, spacing, spacing), atol=1e-4):
        raise AssertionError(f"Unexpected canonical spacing: {zooms}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-root", type=Path, required=True)
    parser.add_argument("--train-reports", type=Path, required=True)
    parser.add_argument("--valid-reports", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-spacing-mm", type=float, default=1.5)
    parser.add_argument("--xy-shape", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.target_spacing_mm <= 0 or args.workers < 1 or any(v <= 0 for v in args.xy_shape):
        parser.error("target spacing, workers, and xy shape must be positive")
    spacing = float(args.target_spacing_mm)
    xy_shape = tuple(int(v) for v in args.xy_shape)
    train = select_one(pd.read_csv(args.train_reports), 1000, args.seed)
    valid_pool = select_one(pd.read_csv(args.valid_reports), 300, args.seed)
    cohorts = {"train": train, "val": valid_pool.iloc[:100], "test": valid_pool.iloc[100:300]}
    volume_index: dict[str, Path] = {}
    for path in args.files_root.rglob("*.nii.gz"):
        if path.name in volume_index:
            raise RuntimeError(f"Duplicate CT-RATE volume basename: {path.name}")
        volume_index[path.name] = path
    (args.output_root / "manifests").mkdir(parents=True, exist_ok=True)
    patient_sets: dict[str, set[str]] = {}
    for split, table in cohorts.items():
        def prepare_row(item: tuple[int, pd.Series]) -> dict[str, object]:
            _, row = item
            source = volume_index.get(str(row.VolumeName))
            if source is None:
                raise FileNotFoundError(f"Missing CT-RATE volume {row.VolumeName}")
            original = nib.load(source)
            canonical = canonicalize(source, spacing, xy_shape)
            xyz = np.asarray(canonical.dataobj, dtype=np.float32)
            dhw = np.moveaxis(np.clip(xyz, -1000.0, 1000.0) / 1000.0, (0, 1, 2), (2, 1, 0))
            pid = str(row.patient_id)
            output = args.output_root / "volumes" / pid
            output.mkdir(parents=True, exist_ok=True)
            target, canonical_target = output / "ct.npy", output / "ct_ras_1p5mm.nii.gz"
            np.save(target, dhw.astype(np.float16))
            nib.save(canonical, canonical_target)
            noise_seed = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "big")
            condition = output / "fixed_noise.npy"
            np.save(condition, np.random.default_rng(noise_seed).uniform(-1, 1, dhw.shape).astype(np.float16))
            return {
                "case_id": pid, "patient_id": pid, "split": split, "task": "generation",
                "condition": str(condition.relative_to(args.output_root)),
                "target": str(target.relative_to(args.output_root)), "prompt": report_text(row),
                "source_modality": "fixed_noise", "target_modality": "chest_ct",
                "metadata": {
                    "prepared_shape_dhw": list(dhw.shape), "orientation": "RAS",
                    "spacing_xyz_mm": [spacing, spacing, spacing],
                    "physical_xy_fov_mm": [spacing * xy_shape[0], spacing * xy_shape[1]],
                    "z_resampling": "linear_to_fixed_spacing",
                },
                "inverse_transform": {
                    "original_shape_xyz": list(original.shape), "original_affine": original.affine.tolist(),
                    "target_path": str(source), "hu_clip": [-1000, 1000], "noise_seed": noise_seed,
                    "canonical_shape_xyz": list(canonical.shape), "canonical_affine": canonical.affine.tolist(),
                    "canonical_target_path": str(canonical_target),
                    "canonical_spacing_xyz_mm": [spacing, spacing, spacing],
                },
            }
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(prepare_row, table.iterrows()))
        patient_sets[split] = set(table.patient_id)
        (args.output_root / "manifests" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    if any(patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Patient leakage")
    audit = {
        "version": "ctrate-canonical-v2", "seed": args.seed,
        "counts": {split: len(values) for split, values in patient_sets.items()},
        "orientation": "RAS", "spacing_xyz_mm": [spacing, spacing, spacing],
        "xy_shape": list(xy_shape), "official_source": {"train": "train", "val": "valid", "test": "valid"},
    }
    (args.output_root / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
