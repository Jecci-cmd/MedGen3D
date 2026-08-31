#!/usr/bin/env python3
"""Freeze the BraTS 2021 T1->T2 cohort used by the five-task main run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


XY_SHAPE = (256, 256)


def resize_xy(array: np.ndarray) -> np.ndarray:
    """Convert XYZ to DHW and resize only XY, retaining the complete z axis."""
    dhw = np.moveaxis(np.asarray(array, dtype=np.float32), (0, 1, 2), (2, 1, 0))
    tensor = torch.from_numpy(dhw)[:, None]
    return F.interpolate(
        tensor, XY_SHAPE, mode="bilinear", align_corners=False
    )[:, 0].numpy()


def normalize(array: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    foreground = array[array > 0]
    lo, hi = (np.percentile(foreground, (0.5, 99.5)) if foreground.size
              else np.percentile(array, (0.5, 99.5)))
    if hi <= lo:
        raise ValueError("Degenerate MRI intensity range")
    value = np.clip(array, lo, hi)
    return (2.0 * (value - lo) / (hi - lo) - 1.0), {"lower": float(lo), "upper": float(hi)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    torch.set_num_threads(1)
    cases = sorted({path.parent for path in args.input_root.rglob("*_t1.nii.gz")})
    complete = [case for case in cases if next(case.glob("*_t2.nii.gz"), None)]
    if len(complete) < 1250:
        raise RuntimeError(f"Expected at least 1250 complete BraTS cases, found {len(complete)}")
    rng = np.random.default_rng(args.seed)
    ordered = [complete[i] for i in rng.permutation(len(complete))]
    split_cases = {"train": ordered[:1000], "val": ordered[1000:1050], "test": ordered[1050:1250]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifests").mkdir(exist_ok=True)
    for split, selected in split_cases.items():
        def prepare_case(case: Path) -> dict[str, object]:
            patient = case.name
            t1_path, t2_path = next(case.glob("*_t1.nii.gz")), next(case.glob("*_t2.nii.gz"))
            t1_img, t2_img = nib.load(t1_path), nib.load(t2_path)
            t1, t1_norm = normalize(np.asarray(t1_img.dataobj))
            t2, t2_norm = normalize(np.asarray(t2_img.dataobj))
            out = args.output_root / "volumes" / patient
            out.mkdir(parents=True, exist_ok=True)
            t1_resized, t2_resized = resize_xy(t1), resize_xy(t2)
            np.save(out / "t1.npy", t1_resized.astype(np.float16))
            np.save(out / "t2.npy", t2_resized.astype(np.float16))
            return {
                "case_id": patient, "patient_id": patient, "split": split, "task": "synthesis",
                "condition": str((out / "t1.npy").relative_to(args.output_root)),
                "target": str((out / "t2.npy").relative_to(args.output_root)),
                "prompt": "Synthesize the T2 MRI volume from this T1 MRI volume.",
                "source_modality": "T1", "target_modality": "T2",
                "metadata": {"prepared_shape_dhw": list(t2_resized.shape),
                    "xy_resize": list(XY_SHAPE), "z_resampling": "none"},
                "inverse_transform": {"original_shape_xyz": list(t2_img.shape),
                    "original_affine": t2_img.affine.tolist(), "normalization": t2_norm,
                    "source_normalization": t1_norm, "target_path": str(t2_path)},
            }
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(prepare_case, selected))
        manifest = args.output_root / "manifests" / f"{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256("\n".join(case.name for case in ordered).encode()).hexdigest()
    (args.output_root / "split_audit.json").write_text(json.dumps({
        "seed": args.seed, "counts": {k: len(v) for k, v in split_cases.items()},
        "unused": len(ordered) - 1250, "ordered_patient_sha256": digest,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
