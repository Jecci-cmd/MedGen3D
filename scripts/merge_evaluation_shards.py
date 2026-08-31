#!/usr/bin/env python3
"""Merge and audit parallel fixed-cohort MedGen3D evaluation shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medgen3d.evaluation import summarize_paired_ct, summarize_segmentation_by_class

ALL_TASKS = ("segmentation", "restoration", "reconstruction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))
    args = parser.parse_args()
    paths = [args.input_root / f"shard_{index}" / "results.json" for index in range(args.num_shards)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing shard result files: {missing}")
    payloads: list[dict[str, Any]] = [json.loads(path.read_text()) for path in paths]
    reference = payloads[0]
    invariants = ("checkpoint", "samples_per_task", "sampling_steps", "num_shards", "objective")
    for payload in payloads:
        for key in invariants:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"Shard mismatch for {key}")
    if {int(payload["shard_index"]) for payload in payloads} != set(range(args.num_shards)):
        raise ValueError("Shard indices are incomplete")
    rows = [row for payload in payloads for row in payload["rows"]]
    observed = {(str(row["task"]), int(row["index"])) for row in rows}
    tasks = tuple(args.tasks)
    expected = {(task, index) for task in tasks for index in range(args.samples_per_task)}
    if len(observed) != len(rows) or observed != expected:
        raise ValueError(f"Evaluation cohort mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")
    rows.sort(key=lambda row: (tasks.index(str(row["task"])), int(row["index"])))
    summary: dict[str, Any] = {}
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        if task == "segmentation":
            summary[task] = summarize_segmentation_by_class(selected)
        else:
            summary[task] = summarize_paired_ct([row["metrics"] for row in selected], seed=args.seed)
    if "reconstruction" in tasks:
        reconstruction = [row for row in rows if row["task"] == "reconstruction"]
        summary["reconstruction_by_views"] = {}
        observed_views = sorted({int(row["reconstruction_views"]) for row in reconstruction})
        for views in observed_views:
            selected = [row for row in reconstruction if int(row["reconstruction_views"]) == views]
            if not selected:
                raise ValueError(f"No reconstruction cases for {views} views")
            summary["reconstruction_by_views"][str(views)] = summarize_paired_ct(
                [row["metrics"] for row in selected], seed=args.seed + views)
    merged = {key: reference[key] for key in invariants}
    merged.update({"summary": summary, "rows": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
