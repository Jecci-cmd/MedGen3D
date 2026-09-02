#!/usr/bin/env python3
"""Render MVG ID/OOD result JSON into the three matching paper-table rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(payload: dict, label: str = "MVG") -> str:
    tasks = payload["tasks"]
    seg = tasks["segmentation"]["summary"]["overall"]
    sm = seg["metrics"]
    restore = tasks["restoration"]["summary"]["metrics"]
    synth = tasks["synthesis"]["summary"]
    return "\n".join([
        f"{label} & {sm['dice']['mean']:.3f} & {sm['nsd']['mean']:.3f} & {sm['hd95_mm']['finite_mean']:.2f} & {sm['assd_mm']['finite_mean']:.2f} & {100 * seg['empty_prediction_rate']:.1f} \\\\",
        f"{label} & {restore['mae_hu']['model_mean']:.2f} & {restore['rmse_hu']['model_mean']:.2f} & {restore['psnr_hu']['model_mean']:.2f} & {restore['ssim']['model_mean']:.3f} \\\\",
        f"{label} & {synth['mae']:.3f} & {synth['psnr']:.2f} & {synth['ssim']:.3f} \\\\",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=Path, required=True); ap.add_argument("--ood", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--label", default="MVG")
    args = ap.parse_args()
    text = "% ID\n" + row(json.loads(args.id.read_text()), args.label) + "\n\n% OOD\n" + row(json.loads(args.ood.read_text()), args.label) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
