#!/usr/bin/env python3
"""Download only the 1300 patient-level CT-RATE volumes used by the main run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def patient_id(name: str) -> str:
    match = re.match(r"((?:train|valid)_\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"Cannot derive patient from {name}")
    return match.group(1)


def select_one(table: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    table = table.copy()
    table["patient_id"] = table["VolumeName"].map(patient_id)
    table = table.sort_values("VolumeName").drop_duplicates("patient_id", keep="first")
    return table.sample(n=count, random_state=seed, replace=False).sort_values("patient_id")


def repo_path(volume: str) -> str:
    stem = volume.removesuffix(".nii.gz")
    parts = stem.split("_")
    patient = "_".join(parts[:2])
    scan = "_".join(parts[:3])
    top = "train_fixed" if parts[0] == "train" else "valid"
    return f"dataset/{top}/{patient}/{scan}/{volume}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-reports", type=Path, required=True)
    p.add_argument("--valid-reports", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    train = select_one(pd.read_csv(args.train_reports), 1000, args.seed)
    valid = select_one(pd.read_csv(args.valid_reports), 300, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_root / "selected_train_reports.csv", index=False)
    valid.to_csv(args.output_root / "selected_valid_reports.csv", index=False)
    rows = [*train.to_dict("records"), *valid.to_dict("records")]

    # Submit only genuinely missing volumes.  Besides avoiding a full 1300-item
    # cache scan after every watchdog restart, randomising the remaining work
    # prevents the same small set of slow Hugging Face/Xet objects from
    # occupying every worker on each retry.
    files_root = args.output_root / "files"
    rows = [
        row for row in rows
        if not (files_root / repo_path(str(row["VolumeName"]))).is_file()
    ]
    random.SystemRandom().shuffle(rows)
    print(f"remaining {len(rows)}/1300", flush=True)

    def download(row: dict) -> str:
        path = repo_path(str(row["VolumeName"]))
        hf_hub_download("ibrahimhamamci/CT-RATE", path, repo_type="dataset",
                        local_dir=files_root)
        return str(row["VolumeName"])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(f"downloaded {index}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
