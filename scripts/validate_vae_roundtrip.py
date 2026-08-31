#!/usr/bin/env python3
"""Measure the frozen Wan VAE representation ceiling on real medical patches."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config
from medgen3d.data import DynamicCaseDataset
from medgen3d.wan import FrozenWanVAE, load_official_vae


def denormalize(value: np.ndarray, hu_clip: list[float]) -> np.ndarray:
    low, high = hu_clip
    return (value + 1.0) * .5 * (high - low) + low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/vae_roundtrip.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.wan_repo.resolve()))
    config = load_experiment_config(args.config)
    data = config["data"]
    device = torch.device("cuda")
    official_vae = load_official_vae(args.checkpoint_dir, str(device), torch.bfloat16)
    vae = FrozenWanVAE(official_vae)
    rows: list[dict[str, float | str]] = []

    for task in config["experiment"]["tasks"]:
        dataset = DynamicCaseDataset(
            data["root"], data["manifests"]["val"], "val", data["patch_size_dhw"],
            evaluation_task=task, segmentation_target=data.get("segmentation_target", "sdf"),
            hu_clip=data["hu_clip"], output_range=data["ct_normalization"]["output_range"],
            num_samples=args.samples_per_task,
        )
        for index in range(args.samples_per_task):
            sample = dataset[index]
            for role in ("condition", "target"):
                original = sample[role].unsqueeze(0).to(device)
                latent, padding = vae.encode_volume(original, pad_value=1.0 if task == "segmentation" and role == "target" else -1.0)
                decoded = vae.decode_volume(latent, padding)
                source = original.float().cpu().numpy()[0, 0]
                prediction = decoded.float().cpu().numpy()[0, 0]
                record: dict[str, float | str] = {
                    "task": task, "role": role, "case_id": sample["case_id"],
                    "mae_normalized": float(np.mean(np.abs(prediction - source))),
                }
                if task == "segmentation" and role == "target":
                    truth, pred = source < 0, prediction < 0
                    denominator = truth.sum() + pred.sum()
                    record["dice_zero_level"] = float(1.0 if denominator == 0 else 2 * np.logical_and(truth, pred).sum() / denominator)
                else:
                    truth_hu = denormalize(source, data["hu_clip"])
                    pred_hu = denormalize(prediction, data["hu_clip"])
                    record["mae_hu"] = float(np.mean(np.abs(pred_hu - truth_hu)))
                    record["rmse_hu"] = float(np.sqrt(np.mean((pred_hu - truth_hu) ** 2)))
                rows.append(record)

    grouped: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        buckets[f"{row['task']}/{row['role']}"] .append(row)
    for name, records in buckets.items():
        numeric = sorted({key for record in records for key, value in record.items() if isinstance(value, float)})
        grouped[name] = {key: float(np.mean([record[key] for record in records if key in record])) for key in numeric}
    payload = {"samples_per_task": args.samples_per_task, "patch_size_dhw": data["patch_size_dhw"],
               "summary": grouped, "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(grouped, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
