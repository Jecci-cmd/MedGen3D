#!/usr/bin/env python3
"""Freeze the approved old-1000 train/new-300 validation-test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def case_id(number: int) -> str:
    return f"BDMAP_{number:08d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    splits = {
        "train": [case_id(i) for i in range(1, 1001)],
        "val": [case_id(i) for i in range(1001, 1101)],
        "test": [case_id(i) for i in range(1101, 1301)],
    }
    available: set[str] = set()
    for source in (args.root / "raw/extracted", args.root / "canonical"):
        if source.is_dir():
            available.update(path.name for path in source.iterdir() if path.is_dir())
    missing = sorted(set().union(*map(set, splits.values())) - available)
    if missing:
        raise RuntimeError(f"Cannot freeze split; {len(missing)} cases are missing, first={missing[:5]}")
    out = args.root / "splits"; out.mkdir(parents=True, exist_ok=True)
    for split, values in splits.items():
        (out / f"{split}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
    (out / "unused.txt").write_text("", encoding="utf-8")
    (out / "main1300.json").write_text(json.dumps({
        "name": "main1300", "policy": "cases_1_1000_train_1001_1100_val_1101_1300_test",
        "counts": {key: len(value) for key, value in splits.items()}, "splits": splits,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
