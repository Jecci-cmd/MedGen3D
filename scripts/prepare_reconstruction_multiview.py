#!/usr/bin/env python3
"""Build severe sparse-view inputs: train/val=18 views, test=10/18/20 views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from medicalmodel_data.degradation import sparse_view_slice


TRAIN_VIEWS = (18,)
TEST_VIEWS = (10, 18, 20)
SPLIT_SIZES = {"train": 1000, "val": 100, "test": 200}
SEED = 20260825


def assignments(dataset_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, expected_count in SPLIT_SIZES.items():
        ids = [x.strip() for x in (dataset_root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]
        if len(ids) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} cases, found {len(ids)}")
        # Preserve the original patient split and manifest order.  View count
        # is the only experimental variable in this protocol.
        views_for_split = TEST_VIEWS if split == "test" else TRAIN_VIEWS
        for views in views_for_split:
            for case_id in ids:
                rows.append({"case_id": case_id, "split": split, "views": views})
    return rows


def worker(payload: tuple[np.ndarray, int, float]) -> np.ndarray:
    axial, views, pixel_size_mm = payload
    return sparse_view_slice(axial, views, mu_water=0.02, pixel_size_mm=pixel_size_mm)


def save_atomic(image: nib.Nifti1Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial.nii.gz")
    nib.save(image, temporary)
    temporary.replace(output)


def publish_task_alias(dataset_root: Path, output: Path, row: dict[str, object]) -> None:
    """Expose the formal 18-view input through the canonical case manifest."""
    if int(row["views"]) != 18:
        return
    case_id = str(row["case_id"])
    alias = dataset_root / "tasks/reconstruction" / case_id / "source_hu.nii.gz"
    alias.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias.with_name(alias.name + ".main18.partial")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(output)
    temporary.replace(alias)
    (alias.parent / "meta.json").write_text(json.dumps({
        **row, "views": 18, "geometry": "parallel_beam", "mu_water_per_mm": 0.02,
        "protocol_sha256": protocol_hash(row), "source": str(output),
        "formal_protocol": "reconstruction_main18_v1",
    }, indent=2) + "\n")


def generate_case(dataset_root: Path, output_root: Path, row: dict[str, object], workers: int, force: bool) -> None:
    case_id, views = str(row["case_id"]), int(row["views"])
    output = output_root / str(row["split"]) / f"views_{views:03d}" / case_id / "source_hu.nii.gz"
    metadata = output.parent / "meta.json"
    if output.exists() and metadata.exists() and not force:
        current = json.loads(metadata.read_text())
        if current.get("protocol_sha256") == protocol_hash(row):
            publish_task_alias(dataset_root, output, row)
            return
        # Split membership does not alter an 18-view FBP volume. Reuse a
        # previously verified physical output when only the formal split and
        # protocol bookkeeping changed.
        if int(current.get("views", -1)) == views:
            source = dataset_root / "canonical" / case_id / "ct_hu.nii.gz"
            metadata.write_text(json.dumps({
                **row, "geometry": "parallel_beam", "mu_water_per_mm": 0.02,
                "pixel_size_mm": float(nib.load(source).header.get_zooms()[0]),
                "angles_deg": [0.0, 180.0], "protocol_sha256": protocol_hash(row),
                "source": str(source), "reused_existing_physical_output": True,
            }, indent=2) + "\n")
            publish_task_alias(dataset_root, output, row)
            return
    if not force and not output.exists():
        alternatives = list(output_root.glob(f"*/views_{views:03d}/{case_id}/source_hu.nii.gz"))
        if alternatives:
            output.parent.mkdir(parents=True, exist_ok=True)
            os.link(alternatives[0], output)
            source = dataset_root / "canonical" / case_id / "ct_hu.nii.gz"
            metadata.write_text(json.dumps({
                **row, "geometry": "parallel_beam", "mu_water_per_mm": 0.02,
                "pixel_size_mm": float(nib.load(source).header.get_zooms()[0]),
                "angles_deg": [0.0, 180.0], "protocol_sha256": protocol_hash(row),
                "source": str(source), "reused_existing_physical_output": True,
            }, indent=2) + "\n")
            publish_task_alias(dataset_root, output, row)
            return
    source = dataset_root / "canonical" / case_id / "ct_hu.nii.gz"
    image = nib.load(source)
    ct_zyx = np.transpose(np.asanyarray(image.dataobj).astype(np.float32), (2, 1, 0))
    pixel_size = float(image.header.get_zooms()[0])
    payloads = ((axial, views, pixel_size) for axial in ct_zyx)
    if workers == 1:
        slices = [worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            slices = list(executor.map(worker, payloads, chunksize=1))
    result_xyz = np.transpose(np.stack(slices).astype(np.float32), (2, 1, 0))
    header = image.header.copy(); header.set_data_dtype(np.float32)
    save_atomic(nib.Nifti1Image(result_xyz, image.affine, header), output)
    metadata.write_text(json.dumps({
        **row, "geometry": "parallel_beam", "mu_water_per_mm": 0.02,
        "pixel_size_mm": pixel_size, "angles_deg": [0.0, 180.0],
        "protocol_sha256": protocol_hash(row), "source": str(source),
    }, indent=2) + "\n")
    publish_task_alias(dataset_root, output, row)


def protocol_hash(row: dict[str, object]) -> str:
    payload = {"version": "severe-sparse-v1", "seed": SEED, "geometry": "parallel_beam",
               "mu_water_per_mm": 0.02, **row}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_protocol(dataset_root: Path, output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "assignments.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "views"])
        writer.writeheader(); writer.writerows(rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    restoration_entries = []
    restoration_missing = []
    for split in SPLIT_SIZES:
        for case_id in [x.strip() for x in (dataset_root / "splits" / f"{split}.txt").read_text().splitlines() if x.strip()]:
            path = dataset_root / "tasks/restoration" / case_id / "source_hu.nii.gz"
            if not path.exists():
                # Reconstruction depends only on the canonical clean CT.  The
                # restoration snapshot is an integrity guard, not an input
                # dependency, so allow both datasets to be generated in
                # parallel and record which restoration cases were pending.
                restoration_missing.append(case_id)
                continue
            stat = path.stat()
            restoration_entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    restoration_fingerprint = hashlib.sha256(
        json.dumps(sorted(restoration_entries)).encode()).hexdigest()
    (output_root / "protocol.json").write_text(json.dumps({
        "version": "severe-sparse-v1", "assignment_seed": SEED,
        "train_views": list(TRAIN_VIEWS), "test_views": list(TEST_VIEWS),
        "patient_counts": SPLIT_SIZES,
        "assignment_sha256": digest, "output_root": str(output_root),
        "restoration_read_only_fingerprint": restoration_fingerprint,
        "restoration_file_count": len(restoration_entries),
        "restoration_missing_count": len(restoration_missing),
        "restoration_snapshot_complete": not restoration_missing,
    }, indent=2) + "\n")


def write_manifests(dataset_root: Path, output_root: Path, rows: list[dict[str, object]]) -> None:
    available = {
        (str(row["split"]), str(row["case_id"]), int(row["views"]))
        for row in rows
    }
    target_dir = dataset_root / "processed" / "manifests_recon_severe_v1"
    target_dir.mkdir(parents=True, exist_ok=True)
    # The formal 1300-case split files are the sole source of patient
    # membership.  Older preprocessing manifests may contain the same cases
    # under an earlier train/val/test assignment, so first build a split-free
    # case index and then materialize each current split in its frozen order.
    record_by_case: dict[str, dict[str, object]] = {}
    for source in sorted((dataset_root / "processed" / "manifests").glob("*.jsonl")):
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record["case_id"])
            record_by_case.setdefault(case_id, record)
    if len(record_by_case) != sum(SPLIT_SIZES.values()):
        raise ValueError(
            f"Expected {sum(SPLIT_SIZES.values())} unique source records, "
            f"found {len(record_by_case)}"
        )
    for split in SPLIT_SIZES:
        split_ids = [
            value.strip()
            for value in (dataset_root / "splits" / f"{split}.txt").read_text().splitlines()
            if value.strip()
        ]
        records = []
        for case_id in split_ids:
            if case_id not in record_by_case:
                raise ValueError(f"Missing source manifest record for {case_id}")
            record = json.loads(json.dumps(record_by_case[case_id]))
            record["split"] = split
            records.append(record)
        views_for_split = TEST_VIEWS if split == "test" else TRAIN_VIEWS
        for views in views_for_split:
            updated = []
            for original in records:
                record = json.loads(json.dumps(original))
                case_id = str(record["case_id"])
                if (split, case_id, views) not in available:
                    raise ValueError(f"Missing assignment for {split}/{case_id}/{views}")
                path = output_root / split / f"views_{views:03d}" / case_id / "source_hu.nii.gz"
                record["sparse_view_ct"] = [str(path)]
                record.setdefault("metadata", {})["reconstruction_severe_v1"] = {
                    "views": views, "training_views": list(TRAIN_VIEWS),
                    "test_only": split == "test" and views not in TRAIN_VIEWS,
                    "assignment_seed": SEED, "protocol_sha256": protocol_hash(
                        {"case_id": case_id, "split": split, "views": views})
                }
                updated.append(record)
            name = f"test_views_{views:03d}.jsonl" if split == "test" else f"{split}.jsonl"
            (target_dir / name).write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in updated))
    # Canonical in-distribution test alias is an ordinary file for portability.
    canonical = target_dir / "test_views_018.jsonl"
    (target_dir / "test.jsonl").write_bytes(canonical.read_bytes())


def audit_outputs(dataset_root: Path, output_root: Path, rows: list[dict[str, object]]) -> None:
    counts: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        case_id, split, views = str(row["case_id"]), str(row["split"]), int(row["views"])
        key = f"{split}/views_{views:03d}"
        output = output_root / split / f"views_{views:03d}" / case_id / "source_hu.nii.gz"
        metadata = output.parent / "meta.json"
        if not output.exists() or not metadata.exists():
            raise FileNotFoundError(f"Incomplete reconstruction output: {output}")
        expected_meta = protocol_hash(row)
        if json.loads(metadata.read_text()).get("protocol_sha256") != expected_meta:
            raise ValueError(f"Protocol mismatch: {metadata}")
        source = nib.load(dataset_root / "canonical" / case_id / "ct_hu.nii.gz")
        generated = nib.load(output)
        if source.shape != generated.shape or not np.allclose(source.affine, generated.affine):
            raise ValueError(f"Geometry mismatch: {output}")
        if generated.get_data_dtype() != np.dtype(np.float32):
            raise ValueError(f"Non-float32 output: {output}")
        if not np.isfinite(np.asanyarray(generated.dataobj)).all():
            raise ValueError(f"Non-finite output: {output}")
        counts[key] = counts.get(key, 0) + 1
        if index % 50 == 0 or index == len(rows):
            print(json.dumps({"audit_done": index, "audit_total": len(rows)}), flush=True)
    (output_root / "audit.json").write_text(json.dumps({
        "passed": True, "num_outputs": len(rows), "counts": counts,
        "train_views": list(TRAIN_VIEWS), "test_views": list(TEST_VIEWS),
        "assignment_sha256": hashlib.sha256((output_root / "assignments.csv").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--write-manifests-only", action="store_true")
    parser.add_argument("--skip-write-manifests", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = assignments(args.dataset_root)
    write_protocol(args.dataset_root, args.output_root, rows)
    if args.write_manifests_only:
        write_manifests(args.dataset_root, args.output_root, rows)
        return
    if args.finalize:
        audit_outputs(args.dataset_root, args.output_root, rows)
        write_manifests(args.dataset_root, args.output_root, rows)
        return
    selected = [row for row in rows if args.split == "all" or row["split"] == args.split]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    selected = selected[args.shard_index::args.shard_count]
    if args.limit is not None:
        selected = selected[:args.limit]
    for index, row in enumerate(selected, 1):
        generate_case(args.dataset_root, args.output_root, row, args.workers, args.force)
        print(json.dumps({"done": index, "total": len(selected), **row}), flush=True)
    if args.limit is None and args.shard_count == 1 and not args.skip_write_manifests:
        write_manifests(args.dataset_root, args.output_root, rows)


if __name__ == "__main__":
    main()
