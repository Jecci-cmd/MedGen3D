#!/usr/bin/env python3
"""Download an explicit contiguous AbdomenAtlas Mini patient range."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "AbdomenAtlas/AbdomenAtlas1.0Mini"
REVISION = "4dff62f03f7e4f17cd8c62617bc75fde9893a1e9"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--first-case", type=int, default=1)
    parser.add_argument("--last-case", type=int, default=1000)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "data/AbdomenAtlas1.0Mini")
    args = parser.parse_args()
    if args.first_case < 1 or args.last_case < args.first_case:
        parser.error("Require 1 <= first-case <= last-case")
    root = args.output_root
    destination = root / "raw/extracted"
    metadata = root / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    patterns = [
        f"BDMAP_{case_number:08d}/{filename}"
        for case_number in range(args.first_case, args.last_case + 1)
        for filename in ("ct.nii.gz", "combined_labels.nii.gz")
    ]
    def download(path: str) -> str:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=path,
            repo_type="dataset",
            revision=REVISION,
            local_dir=destination,
        )
        return path

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, path): path for path in patterns}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == len(patterns):
                print(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"downloaded {completed}/{len(patterns)} files",
                    flush=True,
                )
    missing = [
        pattern
        for pattern in patterns
        if not (destination / pattern).is_file()
        or (destination / pattern).stat().st_size == 0
    ]
    report = {
        "repository": REPO_ID,
        "revision": REVISION,
        "first_case": args.first_case,
        "last_case": args.last_case,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "expected_files": len(patterns),
        "present_files": len(patterns) - len(missing),
        "missing": missing,
        "passed": not missing,
    }
    (metadata / "direct_download.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        raise RuntimeError(f"Direct download incomplete: {len(missing)} files missing")


if __name__ == "__main__":
    main()
