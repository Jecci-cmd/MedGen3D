#!/usr/bin/env python3
"""Prepare one CT-RATE scan per patient with official train/valid separation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

XY_SHAPE = (256, 256)


def patient_id(name: str) -> str:
    match = re.match(r"((?:train|valid)_\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"Cannot derive patient from {name}")
    return match.group(1)


def report_text(row: pd.Series) -> str:
    findings = str(row.get("Findings_EN", row.get("Findings", ""))).strip()
    impression = str(row.get("Impressions_EN", row.get("Impression", ""))).strip()
    return f"Findings: {findings}\nImpression: {impression}".strip()


def resize_ct_xy(array_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to DHW and resize only XY, retaining the complete z axis."""
    dhw = np.moveaxis(np.asarray(array_xyz, dtype=np.float32), (0, 1, 2), (2, 1, 0))
    tensor = torch.from_numpy(dhw)[:, None]
    return F.interpolate(
        tensor, XY_SHAPE, mode="bilinear", align_corners=False
    )[:, 0].numpy()


def select_one(table: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    table = table.copy()
    table["patient_id"] = table["VolumeName"].map(patient_id)
    table = table.sort_values("VolumeName").drop_duplicates("patient_id", keep="first")
    return table.sample(n=count, random_state=seed, replace=False).sort_values("patient_id")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--files-root", type=Path, required=True)
    p.add_argument("--train-reports", type=Path, required=True)
    p.add_argument("--valid-reports", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    torch.set_num_threads(1)
    train = select_one(pd.read_csv(args.train_reports), 1000, args.seed)
    valid_pool = select_one(pd.read_csv(args.valid_reports), 300, args.seed)
    cohorts = {"train": train, "val": valid_pool.iloc[:100], "test": valid_pool.iloc[100:300]}
    volume_index: dict[str, Path] = {}
    for path in args.files_root.rglob("*.nii.gz"):
        if path.name in volume_index:
            raise RuntimeError(f"Duplicate CT-RATE volume basename: {path.name}")
        volume_index[path.name] = path
    (args.output_root / "manifests").mkdir(parents=True, exist_ok=True)
    all_patients: dict[str, set[str]] = {}
    for split, table in cohorts.items():
        def prepare_row(item: tuple[int, pd.Series]) -> dict[str, object]:
            _, row = item
            source = volume_index.get(str(row.VolumeName))
            if source is None:
                raise FileNotFoundError(f"Missing CT-RATE volume {row.VolumeName}")
            image = nib.load(source); ct = resize_ct_xy(np.asarray(image.dataobj))
            ct = np.clip(ct, -1000.0, 1000.0) / 1000.0
            pid = str(row.patient_id); out = args.output_root / "volumes" / pid
            out.mkdir(parents=True, exist_ok=True)
            target = out / "ct.npy"; condition = out / "fixed_noise.npy"
            np.save(target, ct.astype(np.float16))
            noise_seed = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "big")
            np.save(condition, np.random.default_rng(noise_seed).uniform(-1, 1, ct.shape).astype(np.float16))
            return {
                "case_id": pid, "patient_id": pid, "split": split, "task": "generation",
                "condition": str(condition.relative_to(args.output_root)),
                "target": str(target.relative_to(args.output_root)), "prompt": report_text(row),
                "source_modality": "fixed_noise", "target_modality": "chest_ct",
                "metadata": {"prepared_shape_dhw": list(ct.shape),
                    "xy_resize": list(XY_SHAPE), "z_resampling": "none"},
                "inverse_transform": {"original_shape_xyz": list(image.shape),
                    "original_affine": image.affine.tolist(), "hu_clip": [-1000, 1000],
                    "target_path": str(source), "noise_seed": noise_seed},
            }
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(prepare_row, table.iterrows()))
        all_patients[split] = set(table.patient_id)
        (args.output_root / "manifests" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if any(all_patients[a] & all_patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Patient leakage")
    (args.output_root / "split_audit.json").write_text(json.dumps({
        "seed": args.seed, "counts": {k: len(v) for k, v in all_patients.items()},
        "official_source": {"train": "train", "val": "valid", "test": "valid"},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
