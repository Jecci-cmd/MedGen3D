#!/usr/bin/env python3
"""Build a strict 200-case generation ``results.json`` from a frozen manifest.

This adapter is used only for external baselines whose prediction directories
do not already follow MedGen3D's evaluator output format.  It never guesses a
case order: every prediction must map uniquely to a manifest ``case_id``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def keys(path: Path) -> set[str]:
    values = {nifti_stem(path), path.parent.name}
    output: set[str] = set()
    for value in values:
        if value.startswith("samples."):
            value = value.removeprefix("samples.")
        output.add(value)
        output.add(re.sub(r"_\d+$", "", value))
    return output


def target_path(record: dict[str, object]) -> str:
    inverse = record.get("inverse_transform")
    if not isinstance(inverse, dict):
        raise KeyError(f"Missing inverse_transform for {record.get('case_id')}")
    # The original CT and its rescale metadata are the reference used by the
    # fixed CT-CLIP protocol.  FID itself resamples this path to 1 mm.
    path = inverse.get("target_path")
    if not isinstance(path, str):
        raise KeyError(f"Missing inverse_transform.target_path for {record.get('case_id')}")
    return path


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != args.expected_samples:
        raise SystemExit(f"Expected {args.expected_samples} manifest rows, found {len(records)}")
    case_ids = [str(record.get("case_id")) for record in records]
    if len(set(case_ids)) != len(case_ids) or any(case_id == "None" for case_id in case_ids):
        raise SystemExit("Manifest case_id values must be unique and non-empty")

    by_case: dict[str, list[Path]] = {case_id: [] for case_id in case_ids}
    for prediction in args.predictions.rglob("*.nii*"):
        for key in keys(prediction):
            if key in by_case:
                by_case[key].append(prediction)
    invalid = {case_id: paths for case_id, paths in by_case.items() if len(paths) != 1}
    if invalid:
        detail = ", ".join(f"{case_id}={len(paths)}" for case_id, paths in list(invalid.items())[:10])
        raise SystemExit(f"Every manifest case requires exactly one prediction; violations: {detail}")

    rows = []
    for index, record in enumerate(records):
        case_id = case_ids[index]
        target = Path(target_path(record))
        prediction = by_case[case_id][0]
        prompt = str(record.get("prompt", "")).strip()
        if not target.is_file() or not prediction.is_file() or not prompt:
            raise SystemExit(f"Missing target, prediction, or prompt for {case_id}")
        rows.append({"index": index, "case_id": case_id, "task": "generation",
                     "volume": str(prediction), "target_volume": str(target), "prompt": prompt})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol_input": "frozen_generation_manifest_v1", "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} exact case matches")


if __name__ == "__main__":
    main()
