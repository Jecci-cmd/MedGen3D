#!/usr/bin/env python3
"""Build per-organ MedGen3D vs TotalSegmentator paper-table statistics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.evaluation import (
    SEGMENTATION_METRIC_KEYS,
    canonical_segmentation_class,
    summarize_segmentation_by_class,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON row list in {path}")
    selected = [row for row in rows if row.get("task", "segmentation") == "segmentation"]
    if not selected:
        raise ValueError(f"No segmentation rows in {path}")
    return selected


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("case_id", "")), int(row.get("index", -1))


def harmonize_class_labels(*collections: list[dict[str, Any]]) -> None:
    """Fill missing legacy labels from the matching model's cohort row."""
    labels: dict[tuple[str, int], str] = {}
    for rows in collections:
        for row in rows:
            value = row.get("structure") or row.get("organ") or row.get("class_name")
            if value:
                labels[row_key(row)] = canonical_segmentation_class(value)
    for rows in collections:
        for row in rows:
            if not (row.get("structure") or row.get("organ") or row.get("class_name")):
                label = labels.get(row_key(row))
                if label is None:
                    raise ValueError(f"Cannot recover class label for row {row_key(row)}")
                row["structure"] = label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medgen-results", type=Path, required=True)
    parser.add_argument("--totalseg-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    medgen_rows = read_rows(args.medgen_results)
    totalseg_rows = read_rows(args.totalseg_results)
    harmonize_class_labels(medgen_rows, totalseg_rows)
    summaries = {
        "MedGen3D": summarize_segmentation_by_class(medgen_rows),
        "TotalSegmentator": summarize_segmentation_by_class(totalseg_rows),
    }
    classes = sorted(set().union(*(summary["by_class"] for summary in summaries.values())))
    rows: list[dict[str, Any]] = []
    for class_name in classes + ["Average"]:
        output: dict[str, Any] = {"class": class_name}
        for model, summary in summaries.items():
            stats = summary["overall"] if class_name == "Average" else summary["by_class"].get(class_name)
            output[f"{model}_num_cases"] = 0 if stats is None else stats["num_cases"]
            output[f"{model}_zero_dice_cases"] = 0 if stats is None else stats["zero_dice_cases"]
            for metric in SEGMENTATION_METRIC_KEYS:
                metric_stats = None if stats is None else stats["metrics"][metric]
                output[f"{model}_{metric}_mean"] = None if metric_stats is None else metric_stats["mean"]
                output[f"{model}_{metric}_std"] = None if metric_stats is None else metric_stats["std"]
                output[f"{model}_{metric}_finite_mean"] = None if metric_stats is None else metric_stats["finite_mean"]
                output[f"{model}_{metric}_nonfinite_count"] = 0 if metric_stats is None else metric_stats["nonfinite_count"]
        rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"models": summaries, "paper_table_rows": rows}
    (args.output_dir / "segmentation_by_class.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "segmentation_by_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
