#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config
from medgen3d.data import DynamicCaseDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml"),
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    data = config["data"]
    root = Path(data["root"])
    dataset = DynamicCaseDataset(
        root, data["manifests"]["train"], "train", data["patch_size_dhw"],
        task_weights={"segmentation": 1.0},
        segmentation_target=data.get("segmentation_target", "sdf"),
        foreground_probability=data.get("foreground_probability", .7),
        segmentation_foreground_probability=data.get("segmentation_foreground_probability", 1.0),
        segmentation_surface_probability=data.get("segmentation_surface_probability", 0.0),
        segmentation_foreground_warmup_probability=data.get(
            "segmentation_foreground_warmup_probability"
        ),
        segmentation_surface_warmup_probability=data.get(
            "segmentation_surface_warmup_probability"
        ),
        segmentation_foreground_warmup_fraction=data.get(
            "segmentation_foreground_warmup_fraction", 0.0
        ),
        segmentation_center_jitter_zyx=data.get("segmentation_center_jitter_zyx", (12, 12, 12)),
        segmentation_surface_center_jitter_zyx=data.get(
            "segmentation_surface_center_jitter_zyx", (2, 2, 2)
        ),
        segmentation_surface_band_mm=data.get("segmentation_surface_band_mm", 2.0),
        segmentation_sdf_clip_mm=data.get("sdf", {}).get("clip_distance_mm", 8.0),
        segmentation_min_foreground_voxels=data.get("segmentation_min_foreground_voxels", 100),
        segmentation_organ_sampling=data.get("segmentation_organ_sampling", "case_uniform"),
        segmentation_case_sampling=data.get("segmentation_case_sampling", "shuffled_cycle"),
        segmentation_zoom=data.get("segmentation_zoom"),
        spatial_flip_probability=0,
        seed=config["train"]["seed"], num_samples=args.samples,
    )
    modes: Counter[str] = Counter()
    organs: Counter[int] = Counter()
    cases_by_organ: dict[int, set[str]] = {
        class_id: set() for class_id in dataset.segmentation_classes
    }
    foreground_counts: list[int] = []
    zoomed_organs: Counter[int] = Counter()
    failures: list[dict[str, object]] = []
    for index in range(args.samples):
        try:
            sample = dataset[index]
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})
            continue
        metadata = sample["metadata"]
        class_id = int(metadata["organ_id"])
        count = int(metadata["foreground_voxels"])
        mode = str(metadata["sampling_mode"])
        modes[mode] += 1; organs[class_id] += 1
        if metadata.get("segmentation_zoom_applied"):
            zoomed_organs[class_id] += 1
        cases_by_organ.setdefault(class_id, set()).add(str(sample["case_id"]))
        if mode == "target_foreground_centered":
            foreground_counts.append(count)
    report = {
        "samples": args.samples,
        "mode_counts": dict(modes),
        "mode_fractions": {key: value / args.samples for key, value in modes.items()},
        "organ_counts": {str(key): value for key, value in sorted(organs.items())},
        "organ_count_range": {
            "min": min(organs.values()) if organs else None,
            "max": max(organs.values()) if organs else None,
        },
        "organ_pool_sizes": {
            str(key): len(value) for key, value in dataset.segmentation_case_pools.items()
        },
        "zoomed_organ_counts": {
            str(key): value for key, value in sorted(zoomed_organs.items())
        },
        "unique_cases_sampled_by_organ": {
            str(key): len(value) for key, value in sorted(cases_by_organ.items())
        },
        "foreground_voxels": {
            "min": min(foreground_counts) if foreground_counts else None,
            "median": float(np.median(foreground_counts)) if foreground_counts else None,
            "p05": float(np.percentile(foreground_counts, 5)) if foreground_counts else None,
        },
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
