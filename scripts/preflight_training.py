#!/usr/bin/env python3
"""Fail-fast, read-only checks before allocating a 4- or 8-GPU training job."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config
from medgen3d.data import DynamicCaseDataset, audit_disjoint_splits, build_task_dataset


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/main5task_feedforward_h200x8.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    data, model = config["data"], config["model"]
    root = Path(data["root"])

    patch = tuple(data["patch_size_dhw"])
    if (patch[0] - 1) % 4 or patch[1] % 32 or patch[2] % 32:
        raise SystemExit(f"Invalid Wan patch {patch}: require D=4n+1 and H/W multiples of 32")
    audit_disjoint_splits(root, data["manifests"], data["splits"])

    split_sets: dict[str, set[str]] = {}
    reconstruction_views: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        frozen = set(lines(root / data["splits"][split]))
        records = [json.loads(row) for row in lines(root / data["manifests"][split])]
        observed = {str(row["case_id"]) for row in records}
        if observed != frozen:
            raise SystemExit(f"{split}: manifest/split mismatch ({len(observed)} vs {len(frozen)})")
        required = {"image", "mask", "sdf", "ldct", "sparse_view_ct"}
        incomplete = [row["case_id"] for row in records if not required <= row.keys()]
        if incomplete:
            raise SystemExit(f"{split}: incomplete three-task cases: {incomplete[:5]}")
        split_sets[split] = observed
        reconstruction_views[split] = {
            DynamicCaseDataset._reconstruction_views(row, str(row["sparse_view_ct"][0]))
            for row in records
        }

    total = sum(map(len, split_sets.values()))
    counts = {key: len(value) for key, value in split_sets.items()}
    expected = {key: int(value) for key, value in data["expected_split_counts"].items()}
    if counts != expected:
        raise SystemExit(f"Expected frozen patient-level split {expected}, observed {counts}")

    reconstruction = data.get("reconstruction", {})
    expected_views = {
        "train": set(map(int, reconstruction.get("train_views", [reconstruction.get("views", 18)]))),
        "val": set(map(int, reconstruction.get("validation_views", [reconstruction.get("views", 18)]))),
        # The canonical test manifest is the in-distribution primary test.
        "test": {int(reconstruction.get("views", 18))},
    }
    if reconstruction_views != expected_views:
        raise SystemExit(
            f"Reconstruction view protocol mismatch: expected {expected_views}, "
            f"observed {reconstruction_views}"
        )

    for task in config["experiment"]["tasks"]:
        sample = build_task_dataset(
            data, task, "val", seed=int(config["train"]["seed"]),
            num_samples=1, training=False,
        )[0]
        if tuple(sample["condition"].shape[-3:]) != patch or tuple(sample["target"].shape[-3:]) != patch:
            raise SystemExit(f"{task}: bad sample geometry")

    required_weights = ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"]
    missing = [name for name in required_weights if not (args.checkpoint_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Checkpoint files missing: {missing}")
    if not (args.wan_repo / "wan").is_dir():
        raise SystemExit(f"Official Wan package missing under {args.wan_repo}")
    print(f"PASS: {total} disjoint patients {counts}; three tasks load at {patch}")
    print(f"PASS: reconstruction view protocol {reconstruction_views}")
    print(f"PASS: checkpoint and Wan repository paths exist; pinned commit={model['wan_repo_commit']}")


if __name__ == "__main__":
    main()
