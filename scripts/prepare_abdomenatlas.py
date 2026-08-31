#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicalmodel_data.config import load_config
from medicalmodel_data.geometry import (
    crop_or_pad,
    label_to_sdf,
    normalize_hu,
    sdf_to_label,
)
from medicalmodel_data.io import (
    directory_sha256,
    save_nifti_atomic,
    save_npz_atomic,
    sha256sum,
    write_csv,
    write_json,
    write_jsonl,
)
from medicalmodel_data.case_pipeline import build_case_manifests, generate_full_volume_sdf
from medicalmodel_data.layout import ensure_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AbdomenAtlas for MedGen3D.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/data/abdomenatlas1300_prepare.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, action="append")
    subparsers.add_parser("verify-archives")
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--checksum", action="store_true")
    split = subparsers.add_parser("split")
    split.add_argument("--pilot", action="store_true")
    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    canonicalize.add_argument("--limit", type=int)
    canonicalize.add_argument("--force", action="store_true")
    derive = subparsers.add_parser("derive")
    derive.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    derive.add_argument("--limit", type=int)
    derive.add_argument(
        "--task", choices=["segmentation", "restoration", "reconstruction", "all"], default="all"
    )
    derive.add_argument("--force", action="store_true")
    derive.add_argument("--workers", type=int)
    patches = subparsers.add_parser("patches")
    patches.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    patches.add_argument("--limit", type=int)
    patches.add_argument("--force", action="store_true")
    sdf_volumes = subparsers.add_parser("sdf-volumes")
    sdf_volumes.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    sdf_volumes.add_argument("--limit", type=int)
    sdf_volumes.add_argument("--num-shards", type=int, default=1)
    sdf_volumes.add_argument("--shard-index", type=int, default=0)
    sdf_volumes.add_argument(
        "--refresh-stale", action="store_true",
        help="Regenerate only SDF files whose metadata differs from the active config.",
    )
    sdf_volumes.add_argument("--force", action="store_true")
    case_manifests = subparsers.add_parser("case-manifests")
    case_manifests.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    qc = subparsers.add_parser("qc")
    qc.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    qc.add_argument("--limit", type=int)
    subparsers.add_parser("calibrate")
    subparsers.add_parser("smoke-test")
    subparsers.add_parser("status")
    return parser.parse_args()


def get_paths(config: dict) -> dict[str, Path]:
    root = PROJECT_ROOT / config["dataset"]["root"]
    return ensure_layout(root)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"Archive contains unsupported special/link member: {member.name}"
                )
        handle.extractall(destination, filter="data")


def command_verify_archives(config: dict, paths: dict[str, Path]) -> None:
    expected = config["dataset"]["archives"]
    records = []
    failures = []
    for name, metadata in expected.items():
        path = paths["raw/archives"] / name
        try:
            if path.with_suffix(path.suffix + ".aria2").exists():
                raise RuntimeError("aria2 control file exists; download is incomplete")
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size != metadata["bytes"]:
                raise RuntimeError(
                    f"size {path.stat().st_size} != expected {metadata['bytes']}"
                )
            digest = sha256sum(path)
            if digest != metadata["etag_sha256"]:
                raise RuntimeError(
                    f"sha256 {digest} != expected etag {metadata['etag_sha256']}"
                )
            with tarfile.open(path, "r:gz") as handle:
                members = handle.getmembers()
                if not members:
                    raise RuntimeError("archive is empty")
                for member in members:
                    if not (member.isfile() or member.isdir()):
                        raise RuntimeError(f"unsupported member: {member.name}")
                case_pattern = re.compile(r"(?:^|/)(BDMAP_\d{8})(?:/|$)")
                case_ids = {
                    match.group(1)
                    for member in members
                    if (match := case_pattern.search(member.name))
                }
                expected_case_ids = {
                    f"BDMAP_{case_number:08d}"
                    for case_number in range(
                        metadata["case_start"], metadata["case_end"] + 1
                    )
                }
                if case_ids != expected_case_ids:
                    missing = sorted(expected_case_ids - case_ids)
                    extra = sorted(case_ids - expected_case_ids)
                    raise RuntimeError(
                        f"archive case range mismatch: missing={missing[:10]}, "
                        f"extra={extra[:10]}"
                    )
            records.append(
                {
                    "archive": name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "tar_members": len(members),
                    "case_count": len(case_ids),
                    "case_start": min(case_ids),
                    "case_end": max(case_ids),
                    "passed": True,
                }
            )
        except Exception as error:
            failures.append({"archive": name, "error": repr(error)})
    all_case_ids = [
        case_id
        for record in records
        for case_id in (
            f"BDMAP_{case_number:08d}"
            for case_number in range(
                expected[record["archive"]]["case_start"],
                expected[record["archive"]]["case_end"] + 1,
            )
        )
    ]
    if len(all_case_ids) != len(set(all_case_ids)):
        failures.append(
            {"archive": "<config>", "error": "Configured archive case ranges overlap"}
        )
    write_json(
        paths["metadata"] / "archive_verification.json",
        {"records": records, "failures": failures, "passed": not failures},
    )
    if failures:
        raise RuntimeError(
            "Archive verification failed; see metadata/archive_verification.json"
        )


def command_extract(config: dict, paths: dict[str, Path], archives: list[Path] | None) -> None:
    verification_path = paths["metadata"] / "archive_verification.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.exists()
        else {}
    )
    if not verification.get("passed"):
        raise RuntimeError("Archives must pass verify-archives before extraction")
    verified_records = {
        record["archive"]: record for record in verification["records"]
    }
    configured = [
        paths["raw/archives"] / name for name in config["dataset"]["archives"]
    ]
    selected = [path.resolve() for path in (archives or configured)]
    allowed = {path.resolve() for path in configured}
    if not set(selected).issubset(allowed):
        raise RuntimeError("Refusing to extract an archive not listed in config")
    if not selected:
        raise FileNotFoundError(f"No tar.gz archives found in {paths['raw/archives']}")
    records = []
    for archive in selected:
        archive = archive.resolve()
        configured_metadata = config["dataset"]["archives"][archive.name]
        current_digest = sha256sum(archive)
        if (
            archive.stat().st_size != configured_metadata["bytes"]
            or current_digest != configured_metadata["etag_sha256"]
            or verified_records.get(archive.name, {}).get("sha256")
            != current_digest
        ):
            raise RuntimeError(
                f"{archive.name}: archive changed or no longer matches verification"
            )
        staging_root = paths["root"] / "raw/extract_staging"
        staging = staging_root / archive.name.replace(".tar.gz", "")
        if staging.exists():
            if not staging.resolve().is_relative_to(staging_root.resolve()):
                raise RuntimeError("Unsafe staging path")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        safe_extract(archive, staging)
        staged_cases = discover_cases(staging)
        expected = config["dataset"]["archives"][archive.name]
        expected_count = expected["case_end"] - expected["case_start"] + 1
        if len(staged_cases) != expected_count:
            raise RuntimeError(
                f"{archive.name}: extracted {len(staged_cases)} cases, "
                f"expected {expected_count}"
            )
        moved = 0
        skipped_existing = 0
        for case_id, ct_path, _ in staged_cases:
            source_case = ct_path.parent
            destination_case = paths["raw/extracted"] / case_id
            if destination_case.exists():
                existing = discover_cases(destination_case)
                if len(existing) != 1 or existing[0][0] != case_id:
                    raise RuntimeError(f"Existing incomplete/colliding case: {case_id}")
                if directory_sha256(destination_case) != directory_sha256(source_case):
                    raise RuntimeError(
                        f"Existing case differs from verified archive: {case_id}"
                    )
                skipped_existing += 1
                continue
            shutil.move(str(source_case), str(destination_case))
            moved += 1
        shutil.rmtree(staging)
        records.append(
            {
                "archive": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": verified_records[archive.name]["sha256"],
                "cases": len(staged_cases),
                "moved": moved,
                "skipped_existing": skipped_existing,
            }
        )
    extracted_cases = discover_cases(paths["raw/extracted"])
    if len(extracted_cases) != config["dataset"]["expected_cases"]:
        raise RuntimeError(
            f"Extracted case count {len(extracted_cases)} != "
            f"expected {config['dataset']['expected_cases']}"
        )
    write_json(paths["metadata"] / "archives.json", records)


def discover_cases(extracted: Path) -> list[tuple[str, Path, Path]]:
    cases = []
    ct_names = ("ct.nii.gz", "CT.nii.gz", "image.nii.gz")
    label_names = ("combined_labels.nii.gz", "label.nii.gz", "combined_label.nii.gz")
    for directory, _, filenames in os.walk(extracted):
        names = set(filenames)
        ct_name = next((name for name in ct_names if name in names), None)
        label_name = next((name for name in label_names if name in names), None)
        if ct_name and label_name:
            case_dir = Path(directory)
            cases.append((case_dir.name, case_dir / ct_name, case_dir / label_name))
    cases = sorted(cases)
    ids = [case_id for case_id, _, _ in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise RuntimeError(
            "Duplicate leaf case IDs detected; splitting would be unsafe: "
            + ", ".join(duplicates[:20])
        )
    return cases


def command_inventory(config: dict, paths: dict[str, Path], checksum: bool) -> None:
    import nibabel as nib

    rows = []
    exclusions = []
    for case_id, ct_path, label_path in discover_cases(paths["raw/extracted"]):
        try:
            ct = nib.load(ct_path)
            label = nib.load(label_path)
            ct_data = np.asanyarray(ct.dataobj)
            label_data = np.asanyarray(label.dataobj)
            labels = sorted(int(value) for value in np.unique(label_data))
            spacing = np.asarray(ct.header.get_zooms()[:3], dtype=float)
            affine_det = float(np.linalg.det(ct.affine[:3, :3]))
            affine_aligned = np.allclose(ct.affine, label.affine, atol=1e-3)
            label_affine = np.asarray(label.affine)
            reset_policy = config["dataset"].get("label_reset_header_policy", {})
            expected_signs = np.asarray(
                reset_policy.get("expected_axis_signs_xyz", []), dtype=float
            )
            expected_reset_linear = (
                np.diag(expected_signs * label.header.get_zooms()[:3])
                if expected_signs.shape == (3,)
                else None
            )
            label_affine_reset = (
                reset_policy.get("enabled") is True
                and bool(config["dataset"].get("source_repository"))
                and bool(config["dataset"].get("source_revision"))
                and ct.shape == label.shape
                and np.allclose(
                    ct.header.get_zooms()[:3],
                    label.header.get_zooms()[:3],
                    rtol=1e-5,
                    atol=1e-5,
                )
                and np.allclose(label_affine[:3, 3], 0.0, atol=1e-5)
                and expected_reset_linear is not None
                and np.allclose(
                    label_affine[:3, :3],
                    expected_reset_linear,
                    rtol=1e-5,
                    atol=1e-5,
                )
            )
            label_geometry_mode = (
                "native_affine"
                if affine_aligned
                else "voxel_aligned_reset_header"
                if label_affine_reset
                else "invalid"
            )
            valid = (
                ct.shape == label.shape
                and label_geometry_mode != "invalid"
                and set(labels).issubset(range(10))
                and np.isfinite(ct_data).all()
                and np.isfinite(label_data).all()
                and np.isfinite(spacing).all()
                and np.all(spacing > 0)
                and np.isfinite(affine_det)
                and abs(affine_det) > 1e-8
            )
            row = {
                "case_id": case_id,
                "ct_path": str(ct_path),
                "label_path": str(label_path),
                "shape_xyz": "x".join(map(str, ct.shape)),
                "spacing_xyz_mm": ",".join(f"{x:.6g}" for x in ct.header.get_zooms()[:3]),
                "orientation": "".join(nib.aff2axcodes(ct.affine)),
                "labels": ",".join(map(str, labels)),
                "foreground_fraction": float(np.mean(label_data > 0)),
                "ct_min_hu": float(np.min(ct_data)),
                "ct_max_hu": float(np.max(ct_data)),
                "affine_determinant": affine_det,
                "label_geometry_mode": label_geometry_mode,
                "label_geometry_policy_version": (
                    reset_policy.get("version", "")
                    if label_geometry_mode == "voxel_aligned_reset_header"
                    else ""
                ),
                "valid": int(valid),
                "ct_sha256": sha256sum(ct_path) if checksum else "",
                "label_sha256": sha256sum(label_path) if checksum else "",
            }
            rows.append(row)
            if not valid:
                exclusions.append({"case_id": case_id, "reason": "geometry_or_label_validation"})
        except Exception as error:
            exclusions.append({"case_id": case_id, "reason": repr(error)})
    write_csv(paths["metadata"] / "inventory.csv", rows)
    if exclusions:
        write_csv(paths["metadata"] / "exclusions.csv", exclusions)
    write_json(
        paths["metadata"] / "inventory_summary.json",
        {"discovered": len(rows), "valid": sum(row["valid"] for row in rows), "excluded": len(exclusions)},
    )


def read_valid_inventory(inventory_path: Path) -> list[dict[str, str]]:
    import csv

    with inventory_path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["valid"] == "1"]


def stratified_case_order(rows: list[dict[str, str]], seed: int) -> list[str]:
    if not rows:
        return []
    z_sizes = np.asarray([int(row["shape_xyz"].split("x")[2]) for row in rows])
    z_spacing = np.asarray(
        [float(row["spacing_xyz_mm"].split(",")[2]) for row in rows]
    )
    foreground = np.asarray([float(row["foreground_fraction"]) for row in rows])

    def bins(values: np.ndarray) -> np.ndarray:
        edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        return np.digitize(values, edges, right=True)

    strata: dict[tuple[int, int, int], list[str]] = {}
    for row, key in zip(
        rows, zip(bins(z_sizes), bins(z_spacing), bins(foreground))
    ):
        strata.setdefault(tuple(int(x) for x in key), []).append(row["case_id"])
    rng = random.Random(seed)
    for ids in strata.values():
        rng.shuffle(ids)
    ordered = []
    keys = sorted(strata)
    while any(strata.values()):
        rng.shuffle(keys)
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop())
    return ordered


def command_split(config: dict, paths: dict[str, Path], pilot: bool) -> None:
    rows = read_valid_inventory(paths["metadata"] / "inventory.csv")
    case_ids = stratified_case_order(rows, config["split"]["seed"])
    if pilot:
        counts = config["split"]["pilot_counts"]
        total = sum(counts.values())
        if len(case_ids) < total:
            raise RuntimeError(f"Need {total} valid cases, found {len(case_ids)}")
        splits = {}
        offset = 0
        for name in ("train", "val", "test"):
            splits[name] = sorted(case_ids[offset : offset + counts[name]])
            offset += counts[name]
        split_name = "pilot"
    else:
        counts = config["split"].get("full_counts")
        if counts:
            requested = sum(counts.values())
            if len(case_ids) < requested:
                raise RuntimeError(
                    f"Configured full_counts require at least {requested} valid cases, "
                    f"but inventory has {len(case_ids)}"
                )
            train_end = counts["train"]
            val_end = train_end + counts["val"]
            test_end = val_end + counts["test"]
        else:
            ratios = config["split"]["full_ratios"]
            train_end = round(len(case_ids) * ratios["train"])
            val_end = train_end + round(len(case_ids) * ratios["val"])
        splits = {
            "train": sorted(case_ids[:train_end]),
            "val": sorted(case_ids[train_end:val_end]),
            "test": sorted(case_ids[val_end:test_end] if counts else case_ids[val_end:]),
        }
        unused = sorted(case_ids[test_end:]) if counts else []
        split_name = "full"
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    payload = {
        "name": split_name,
        "seed": config["split"]["seed"],
        "config_hash": config["_config_hash"],
        "grouping": "case_id",
        "stratification": ["z_size_quartile", "spacing_z_quartile", "foreground_fraction_quartile"],
        "limitation": (
            "AbdomenAtlas archive exposes volume IDs but no reliable patient identifier; "
            "the split is case-level and cannot prove absence of repeated-patient scans."
        ),
        "splits": splits,
        "unused": unused if not pilot else [],
    }
    write_json(paths["splits"] / f"{split_name}.json", payload)
    for name, ids in splits.items():
        (paths["splits"] / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    if not pilot:
        (paths["splits"] / "unused.txt").write_text("\n".join(unused) + ("\n" if unused else ""), encoding="utf-8")


def selected_ids(paths: dict[str, Path], split: str, limit: int | None) -> list[str]:
    names = ("train", "val", "test") if split == "all" else (split,)
    ids = []
    for name in names:
        file = paths["splits"] / f"{name}.txt"
        ids.extend(line.strip() for line in file.read_text().splitlines() if line.strip())
    return ids[:limit] if limit else ids


def inventory_lookup(path: Path) -> dict[str, dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def command_canonicalize(
    config: dict,
    paths: dict[str, Path],
    split: str,
    limit: int | None,
    force: bool,
) -> None:
    import nibabel as nib
    from nibabel.processing import resample_from_to, resample_to_output

    inventory = inventory_lookup(paths["metadata"] / "inventory.csv")
    target_spacing = tuple(config["canonical"]["target_spacing_xyz_mm"])
    for case_id in selected_ids(paths, split, limit):
        # The first 1000 pilot cases intentionally retain only their verified
        # canonical volumes; raw duplicates were removed to reclaim storage.
        # The added 300 cases still have raw inventory rows.  Reuse a valid
        # canonical asset when its raw row is absent, but fail loudly if
        # neither a valid canonical asset nor raw source exists.
        row = inventory.get(case_id)
        case_dir = paths["canonical"] / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        meta_path = case_dir / "meta.json"
        cached_meta: dict = {}
        if meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_meta = {}
        # Canonical CT/label geometry does not depend on downstream SDF,
        # restoration, reconstruction, or patch-sampling parameters.  Using
        # the hash of the entire preparation config here needlessly rebuilt
        # all volumes whenever one of those task settings changed.
        canonical_geometry_valid = (
            cached_meta.get("canonical_orientation") == config["canonical"]["orientation"]
            and len(cached_meta.get("canonical_spacing_xyz_mm", ())) == 3
            and np.allclose(cached_meta["canonical_spacing_xyz_mm"], target_spacing, atol=1e-6)
        )
        canonical_source_valid = (
            row is None
            or (
                cached_meta.get("source_ct") == row["ct_path"]
                and cached_meta.get("source_label") == row["label_path"]
                and cached_meta.get("label_geometry_mode", "native_affine")
                == row.get("label_geometry_mode", "native_affine")
                and cached_meta.get("label_geometry_policy_version", "")
                == row.get("label_geometry_policy_version", "")
            )
        )
        if (
            not force
            and (case_dir / "ct_hu.nii.gz").exists()
            and (case_dir / "label_id.nii.gz").exists()
            and canonical_geometry_valid
            and canonical_source_valid
        ):
            continue
        if row is None:
            raise FileNotFoundError(
                f"{case_id}: raw inventory row is absent and cached canonical "
                "CT/label failed geometry validation"
            )
        ct_original = nib.load(row["ct_path"])
        label_original = nib.load(row["label_path"])
        label_geometry_mode = row.get("label_geometry_mode", "native_affine")
        label_geometry_policy_version = row.get(
            "label_geometry_policy_version", ""
        )
        original_label_affine = np.asarray(label_original.affine).copy()
        if label_geometry_mode == "voxel_aligned_reset_header":
            label_original = nib.Nifti1Image(
                np.asanyarray(label_original.dataobj),
                ct_original.affine,
                header=label_original.header.copy(),
            )
        elif label_geometry_mode != "native_affine":
            raise ValueError(
                f"{case_id}: unsupported label geometry mode {label_geometry_mode!r}"
            )
        ct_ras = nib.as_closest_canonical(ct_original)
        label_ras = nib.as_closest_canonical(label_original)
        ct_resampled = resample_to_output(ct_ras, voxel_sizes=target_spacing, order=1)
        label_resampled = resample_from_to(label_ras, ct_resampled, order=0)
        ct_float = np.asanyarray(ct_resampled.dataobj)
        if not np.isfinite(ct_float).all():
            raise ValueError(f"{case_id}: non-finite HU after resampling")
        int16_limits = np.iinfo(np.int16)
        if ct_float.min() < int16_limits.min or ct_float.max() > int16_limits.max:
            raise ValueError(
                f"{case_id}: HU range [{ct_float.min()}, {ct_float.max()}] "
                "cannot be represented as int16"
            )
        ct_data = np.rint(ct_float).astype(np.int16)
        label_data = np.rint(np.asanyarray(label_resampled.dataobj)).astype(np.uint8)
        if not set(np.unique(label_data)).issubset(range(10)):
            raise ValueError(f"{case_id}: labels outside 0..9 after resampling")
        if "".join(nib.aff2axcodes(ct_resampled.affine)) != config["canonical"][
            "orientation"
        ]:
            raise ValueError(f"{case_id}: failed to produce configured orientation")
        original_label = np.asanyarray(label_original.dataobj).astype(np.uint8)
        original_voxel_mm3 = float(
            np.prod(ct_original.header.get_zooms()[:3], dtype=np.float64)
        )
        canonical_voxel_mm3 = float(np.prod(target_spacing, dtype=np.float64))
        original_volumes = {
            str(class_id): float(np.count_nonzero(original_label == class_id) * original_voxel_mm3)
            for class_id in range(1, 10)
        }
        canonical_volumes = {
            str(class_id): float(np.count_nonzero(label_data == class_id) * canonical_voxel_mm3)
            for class_id in range(1, 10)
        }
        volume_change = {
            str(class_id): (
                float(
                    (canonical_volumes[str(class_id)] - original_volumes[str(class_id)])
                    / original_volumes[str(class_id)]
                )
                if original_volumes[str(class_id)] > 0
                else None
            )
            for class_id in range(1, 10)
        }
        ct_out = nib.Nifti1Image(ct_data, ct_resampled.affine)
        label_out = nib.Nifti1Image(label_data, ct_resampled.affine)
        ct_out.header.set_zooms(target_spacing)
        label_out.header.set_zooms(target_spacing)
        ct_out.set_qform(ct_resampled.affine, code=1)
        ct_out.set_sform(ct_resampled.affine, code=1)
        label_out.set_qform(ct_resampled.affine, code=1)
        label_out.set_sform(ct_resampled.affine, code=1)
        save_nifti_atomic(ct_out, case_dir / "ct_hu.nii.gz")
        save_nifti_atomic(label_out, case_dir / "label_id.nii.gz")
        write_json(
            case_dir / "meta.json",
            {
                "case_id": case_id,
                "source_ct": row["ct_path"],
                "source_label": row["label_path"],
                "original_shape_xyz": [int(x) for x in ct_original.shape],
                "original_spacing_xyz_mm": [
                    float(x) for x in ct_original.header.get_zooms()[:3]
                ],
                "original_orientation": "".join(nib.aff2axcodes(ct_original.affine)),
                "original_affine": np.asarray(ct_original.affine).tolist(),
                "original_label_affine": original_label_affine.tolist(),
                "label_geometry_mode": label_geometry_mode,
                "label_geometry_policy_version": label_geometry_policy_version,
                "label_header_repaired": (
                    label_geometry_mode == "voxel_aligned_reset_header"
                ),
                "effective_label_affine": np.asarray(label_original.affine).tolist(),
                "original_extent_xyz_mm": [
                    float(size * spacing)
                    for size, spacing in zip(
                        ct_original.shape, ct_original.header.get_zooms()[:3]
                    )
                ],
                "canonical_shape_xyz": [int(x) for x in ct_data.shape],
                "canonical_spacing_xyz_mm": [float(x) for x in target_spacing],
                "canonical_orientation": "".join(nib.aff2axcodes(ct_resampled.affine)),
                "canonical_affine": np.asarray(ct_resampled.affine).tolist(),
                "canonical_extent_xyz_mm": [
                    float(size * spacing)
                    for size, spacing in zip(ct_data.shape, target_spacing)
                ],
                "present_labels": [int(x) for x in np.unique(label_data)],
                "organ_volume_mm3_original": original_volumes,
                "organ_volume_mm3_canonical": canonical_volumes,
                "organ_volume_relative_change": volume_change,
                "organ_volume_change_warning_classes": [
                    int(class_id)
                    for class_id, change in volume_change.items()
                    if change is not None and abs(change) > 0.15
                ],
                "config_hash": config["_config_hash"],
            },
        )


def choose_patch_start(
    label_zyx: np.ndarray,
    patch_shape: tuple[int, int, int],
    foreground_probability: float,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    if rng.random() < foreground_probability and np.any(label_zyx > 0):
        locations = np.argwhere(label_zyx > 0)
        center = locations[rng.integers(0, len(locations))]
    else:
        center = np.array([rng.integers(0, max(1, size)) for size in label_zyx.shape])
    return tuple(int(c - p // 2) for c, p in zip(center, patch_shape))


def center_patch_start(
    label_zyx: np.ndarray, patch_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    foreground = np.argwhere(label_zyx > 0)
    center = (
        np.rint((foreground.min(axis=0) + foreground.max(axis=0)) / 2).astype(int)
        if len(foreground)
        else np.asarray(label_zyx.shape) // 2
    )
    return tuple(int(c - p // 2) for c, p in zip(center, patch_shape))


def stable_seed(global_seed: int, case_id: str, task: str) -> int:
    payload = f"{global_seed}\0{case_id}\0{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def restoration_worker(payload: tuple[np.ndarray, dict, int, float]) -> np.ndarray:
    from medicalmodel_data.degradation import synthetic_low_dose_slice

    axial, params, seed, pixel_size_mm = payload
    return synthetic_low_dose_slice(
        axial,
        params["views"],
        params["incident_photons"],
        params["mu_water_per_mm"],
        np.random.default_rng(seed),
        params["electronic_noise_std"],
        pixel_size_mm,
    )


def reconstruction_worker(payload: tuple[np.ndarray, dict, float]) -> np.ndarray:
    from medicalmodel_data.degradation import sparse_view_slice

    axial, params, pixel_size_mm = payload
    return sparse_view_slice(
        axial,
        params["views"],
        params["mu_water_per_mm"],
        pixel_size_mm,
    )


def command_derive(
    config: dict,
    paths: dict[str, Path],
    split: str,
    limit: int | None,
    task: str,
    force: bool,
    workers: int,
) -> None:
    import nibabel as nib

    task_names = (
        ("segmentation", "restoration", "reconstruction") if task == "all" else (task,)
    )
    for case_id in selected_ids(paths, split, limit):
        pending = []
        for task_name in task_names:
            task_dir = paths[f"tasks/{task_name}"] / case_id
            expected_output = (
                task_dir / "meta.json"
                if task_name == "segmentation"
                else task_dir / "source_hu.nii.gz"
            )
            meta_path = task_dir / "meta.json"
            valid_cached = False
            if not force and expected_output.exists() and meta_path.exists():
                try:
                    cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    valid_cached = cached_meta.get("config_hash") == config["_config_hash"]
                    if task_name == "restoration" and not valid_cached:
                        # The main run intentionally preserves the prior
                        # restoration protocol while changing cohort/split and
                        # reconstruction settings. Reuse iff every physical
                        # low-dose parameter is identical.
                        params = config["tasks"]["restoration"]
                        physical = ("geometry", "views", "incident_photons",
                                    "mu_water_per_mm", "electronic_noise_std")
                        valid_cached = all(cached_meta.get(key) == params.get(key)
                                           for key in physical)
                except (OSError, json.JSONDecodeError):
                    valid_cached = False
            if not valid_cached:
                pending.append(task_name)
        if not pending:
            continue
        canonical = paths["canonical"] / case_id
        ct_img = nib.load(canonical / "ct_hu.nii.gz")
        label_img = nib.load(canonical / "label_id.nii.gz")
        ct_xyz = np.asanyarray(ct_img.dataobj).astype(np.float32)
        label_xyz = np.asanyarray(label_img.dataobj).astype(np.uint8)
        ct_zyx = np.transpose(ct_xyz, (2, 1, 0))
        label_zyx = np.transpose(label_xyz, (2, 1, 0))
        seed = stable_seed(config["processed"]["seed"], case_id, "shared")
        if "segmentation" in pending:
            params = config["tasks"]["segmentation"]
            out = paths["tasks/segmentation"] / case_id
            out.mkdir(parents=True, exist_ok=True)
            write_json(
                out / "meta.json",
                {
                    "seed": stable_seed(config["processed"]["seed"], case_id, "segmentation"),
                    "representation": "patch_local_sdf_with_clip_halo",
                    "authoritative_ground_truth": str(canonical / "label_id.nii.gz"),
                    "config_hash": config["_config_hash"],
                    **params,
                },
            )
        for task_name in ("restoration", "reconstruction"):
            if task_name not in pending:
                continue
            params = config["tasks"][task_name]
            task_seed = stable_seed(config["processed"]["seed"], case_id, task_name)
            pixel_size = config["canonical"]["target_spacing_xyz_mm"][0]
            if task_name == "restoration":
                payloads = (
                    (
                        axial,
                        params,
                        stable_seed(task_seed, str(z), "slice"),
                        pixel_size,
                    )
                    for z, axial in enumerate(ct_zyx)
                )
                worker = restoration_worker
            else:
                payloads = ((axial, params, pixel_size) for axial in ct_zyx)
                worker = reconstruction_worker
            if workers == 1:
                slices = [worker(payload) for payload in payloads]
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    slices = list(executor.map(worker, payloads, chunksize=1))
            result = np.stack(slices).astype(np.float32, copy=False)
            out = paths[f"tasks/{task_name}"] / case_id
            out.mkdir(parents=True, exist_ok=True)
            result_xyz = np.transpose(result, (2, 1, 0))
            save_nifti_atomic(
                nib.Nifti1Image(result_xyz, ct_img.affine),
                out / "source_hu.nii.gz",
            )
            write_json(
                out / "meta.json",
                {
                    "seed": task_seed,
                    "rng": "numpy.PCG64",
                    "projection_workers": workers,
                    "slice_seed_scheme": "sha256(global_seed,case_id,task,z)",
                    "angles_deg": {
                        "start": 0.0,
                        "stop_exclusive": 180.0,
                        "count": params["views"],
                    },
                    "config_hash": config["_config_hash"],
                    **params,
                },
            )


def command_patches(
    config: dict,
    paths: dict[str, Path],
    split: str,
    limit: int | None,
    force: bool,
) -> None:
    import nibabel as nib

    patch_shape = tuple(config["processed"]["patch_shape_zyx"])
    hu_clip = tuple(config["processed"]["hu_clip"])
    output_range = tuple(config["processed"]["output_range"])
    manifest = []
    split_names = ("train", "val", "test") if split == "all" else (split,)
    id_to_split = {}
    for split_name in split_names:
        for case_id in selected_ids(paths, split_name, None):
            id_to_split[case_id] = split_name
    for case_id in selected_ids(paths, split, limit):
        canonical = paths["canonical"] / case_id
        ct = np.transpose(np.asanyarray(nib.load(canonical / "ct_hu.nii.gz").dataobj), (2, 1, 0))
        label = np.transpose(
            np.asanyarray(nib.load(canonical / "label_id.nii.gz").dataobj), (2, 1, 0)
        )
        if id_to_split[case_id] == "train":
            rng = np.random.default_rng(
                stable_seed(config["processed"]["seed"], case_id, "patch")
            )
            start = choose_patch_start(
                label,
                patch_shape,
                config["processed"]["foreground_probability"],
                rng,
            )
        else:
            start = center_patch_start(label, patch_shape)
        target = normalize_hu(
            crop_or_pad(ct, start, patch_shape, fill_value=-1000),
            hu_clip,
            output_range,
        )
        case_output = paths["processed"] / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        for task_name in ("segmentation", "restoration", "reconstruction"):
            if task_name == "segmentation":
                source = target
                sdf_config = config["tasks"]["segmentation"]
                spacing_zyx = tuple(
                    reversed(config["canonical"]["target_spacing_xyz_mm"])
                )
                halo_zyx = tuple(
                    int(np.ceil(sdf_config["sdf_clip_mm"] / spacing))
                    for spacing in spacing_zyx
                )
                extended_start = tuple(s - h for s, h in zip(start, halo_zyx))
                extended_shape = tuple(
                    p + 2 * h for p, h in zip(patch_shape, halo_zyx)
                )
                extended_label = crop_or_pad(
                    label, extended_start, extended_shape, fill_value=0
                )
                extended_sdf = label_to_sdf(
                    extended_label,
                    spacing_zyx,
                    sdf_config["sdf_classes"],
                    sdf_config["sdf_clip_mm"],
                    sdf_config["sdf_positive_inside"],
                )
                task_target = crop_or_pad(
                    extended_sdf,
                    halo_zyx,
                    patch_shape,
                    fill_value=-1,
                )
            else:
                source_img = nib.load(paths[f"tasks/{task_name}"] / case_id / "source_hu.nii.gz")
                source_volume = np.transpose(np.asanyarray(source_img.dataobj), (2, 1, 0))
                source = normalize_hu(
                    crop_or_pad(
                        source_volume, start, patch_shape, fill_value=-1000
                    ),
                    hu_clip,
                    output_range,
                )
                task_target = target
            output = case_output / f"{task_name}.npz"
            valid_output = False
            if not force and output.exists():
                try:
                    cached = np.load(output)
                    valid_output = str(cached["config_hash"].item()) == config[
                        "_config_hash"
                    ]
                except (OSError, KeyError, ValueError):
                    valid_output = False
            if not valid_output:
                save_npz_atomic(
                    output,
                    source=source.astype(np.float16),
                    target=task_target.astype(np.float16),
                    crop_start_zyx=np.asarray(start, dtype=np.int32),
                    config_hash=np.asarray(config["_config_hash"]),
                )
            task_meta_path = paths[f"tasks/{task_name}"] / case_id / "meta.json"
            task_meta = (
                json.loads(task_meta_path.read_text(encoding="utf-8"))
                if task_meta_path.exists()
                else {}
            )
            manifest.append(
                {
                    "sample_id": f"{case_id}__{task_name}__crop00",
                    "case_id": case_id,
                    "split": id_to_split[case_id],
                    "task": task_name,
                    "instruction": {
                        "segmentation": "Segment the nine abdominal organs in this CT volume.",
                        "restoration": "Restore this low-dose CT volume to normal-dose image quality.",
                        "reconstruction": "Reconstruct a high-quality CT volume from this sparse-view FBP volume.",
                    }[task_name],
                    "source": {"path": str(output), "key": "source"},
                    "target": {"path": str(output), "key": "target"},
                    "label": (
                        {
                            "path": str(canonical / "label_id.nii.gz"),
                            "crop_start_zyx": list(start),
                            "patch_shape_zyx": list(patch_shape),
                        }
                        if task_name == "segmentation"
                        else None
                    ),
                    "source_target_npz": str(output),
                    "canonical_ct": str(canonical / "ct_hu.nii.gz"),
                    "clean_target_id": (
                        f"{canonical / 'ct_hu.nii.gz'}::"
                        f"{','.join(map(str, start))}::{','.join(map(str, patch_shape))}"
                    ),
                    "ground_truth_label": (
                        str(canonical / "label_id.nii.gz")
                        if task_name == "segmentation"
                        else None
                    ),
                    "crop_start_zyx": list(start),
                    "patch_shape_zyx": list(patch_shape),
                    "hu_clip": list(hu_clip),
                    "output_range": list(output_range),
                    "spacing_xyz_mm": config["canonical"][
                        "target_spacing_xyz_mm"
                    ],
                    "degradation": task_meta,
                    "config_hash": config["_config_hash"],
                }
            )
    for split_name in split_names:
        records = [row for row in manifest if row["split"] == split_name]
        write_jsonl(paths["manifests"] / f"{split_name}.jsonl", records)


def command_qc(
    config: dict, paths: dict[str, Path], split: str, limit: int | None
) -> None:
    import nibabel as nib
    import imageio.v3 as iio
    from PIL import Image
    from skimage.metrics import structural_similarity

    records = []
    failures = []
    selected_case_ids = selected_ids(paths, split, limit)
    overlay_cases: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_file = paths["splits"] / f"{split_name}.txt"
        if split_file.exists():
            ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
            if ids:
                overlay_cases[ids[0]] = split_name
    for case_id in selected_case_ids:
        try:
            canonical = paths["canonical"] / case_id
            ct_img = nib.load(canonical / "ct_hu.nii.gz")
            label_img = nib.load(canonical / "label_id.nii.gz")
            case_meta = json.loads(
                (canonical / "meta.json").read_text(encoding="utf-8")
            )
            ct = np.asanyarray(ct_img.dataobj)
            label = np.asanyarray(label_img.dataobj)
            if ct.shape != label.shape or not np.allclose(ct_img.affine, label_img.affine):
                raise ValueError("canonical CT/label geometry mismatch")
            if not np.isfinite(ct).all():
                raise ValueError("non-finite canonical CT")
            if not set(np.unique(label)).issubset(range(10)):
                raise ValueError("label outside 0..9")
            if "".join(nib.aff2axcodes(ct_img.affine)) != config["canonical"][
                "orientation"
            ]:
                raise ValueError("canonical orientation mismatch")
            if not np.allclose(
                ct_img.header.get_zooms()[:3],
                config["canonical"]["target_spacing_xyz_mm"],
                atol=1e-5,
            ):
                raise ValueError("canonical spacing mismatch")
            if ct_img.get_qform(coded=True)[1] == 0 or ct_img.get_sform(coded=True)[1] == 0:
                raise ValueError("qform/sform codes must be set")
            record = {
                "case_id": case_id,
                "shape_xyz": "x".join(map(str, ct.shape)),
                "ct_min_hu": float(ct.min()),
                "ct_max_hu": float(ct.max()),
                "present_labels": ",".join(map(str, sorted(int(x) for x in np.unique(label)))),
                "volume_change_warning_classes": ",".join(
                    map(str, case_meta["organ_volume_change_warning_classes"])
                ),
            }
            for class_id, volume_mm3 in case_meta[
                "organ_volume_mm3_canonical"
            ].items():
                record[f"label_{class_id}_volume_mm3"] = volume_mm3
            body = ct > -500
            for task_name in ("restoration", "reconstruction"):
                source_path = paths[f"tasks/{task_name}"] / case_id / "source_hu.nii.gz"
                source_img = nib.load(source_path)
                source = np.asanyarray(source_img.dataobj)
                if source.shape != ct.shape or not np.allclose(source_img.affine, ct_img.affine):
                    raise ValueError(f"{task_name} geometry mismatch")
                if not np.isfinite(source).all():
                    raise ValueError(f"{task_name} contains non-finite values")
                roi = body if np.any(body) else np.ones(ct.shape, dtype=bool)
                record[f"{task_name}_mae_hu"] = float(
                    np.mean(np.abs(source[roi].astype(np.float32) - ct[roi].astype(np.float32)))
                )
                clean_clipped = np.clip(ct.astype(np.float32), -1000.0, 1000.0)
                source_clipped = np.clip(
                    source.astype(np.float32), -1000.0, 1000.0
                )
                mse = float(
                    np.mean(
                        (
                            source_clipped[roi]
                            - clean_clipped[roi]
                        )
                        ** 2
                    )
                )
                record[f"{task_name}_psnr_body_db"] = (
                    float("inf")
                    if mse == 0
                    else float(20.0 * np.log10(2000.0) - 10.0 * np.log10(mse))
                )
                body_coordinates = np.argwhere(roi)
                lower = body_coordinates.min(axis=0)
                upper = body_coordinates.max(axis=0) + 1
                body_slices = tuple(
                    slice(int(start), int(stop))
                    for start, stop in zip(lower, upper)
                )
                clean_bbox = clean_clipped[body_slices]
                source_bbox = source_clipped[body_slices]
                if min(clean_bbox.shape) >= 7:
                    record[f"{task_name}_ssim_body_bbox"] = float(
                        structural_similarity(
                            clean_bbox,
                            source_bbox,
                            data_range=2000.0,
                        )
                    )
                else:
                    record[f"{task_name}_ssim_body_bbox"] = None
                if record[f"{task_name}_mae_hu"] <= 0:
                    raise ValueError(f"{task_name} unexpectedly identical to target")
            segmentation_patch = paths["processed"] / case_id / "segmentation.npz"
            if segmentation_patch.exists():
                segmentation_arrays = np.load(segmentation_patch)
                sdf = segmentation_arrays["target"]
                if sdf.shape[0] != 9:
                    raise ValueError("SDF channel mismatch")
                if not np.isfinite(sdf).all() or np.max(np.abs(sdf)) > 1:
                    raise ValueError("invalid normalized SDF")
                crop_start = tuple(
                    int(x) for x in segmentation_arrays["crop_start_zyx"]
                )
                patch_shape = tuple(config["processed"]["patch_shape_zyx"])
                label_zyx = np.transpose(label, (2, 1, 0))
                label_patch = crop_or_pad(
                    label_zyx, crop_start, patch_shape, fill_value=0
                )
                decoded = sdf_to_label(
                    sdf, config["tasks"]["segmentation"]["sdf_classes"]
                )
                sdf_accuracy = float(np.mean(decoded == label_patch))
                record["sdf_decode_accuracy"] = sdf_accuracy
                if sdf_accuracy != 1.0:
                    raise ValueError(
                        f"SDF decode differs from label crop: accuracy={sdf_accuracy}"
                    )
            records.append(record)
            if case_id in overlay_cases:
                foreground_xyz = np.argwhere(label > 0)
                center_xyz = (
                    np.rint(np.median(foreground_xyz, axis=0)).astype(int)
                    if len(foreground_xyz)
                    else np.asarray(ct.shape) // 2
                )
                x_index, y_index, z_index = (int(x) for x in center_xyz)

                def planes(volume: np.ndarray) -> list[np.ndarray]:
                    return [
                        volume[:, :, z_index].T,
                        volume[:, y_index, :].T,
                        volume[x_index, :, :].T,
                    ]

                def gray(image: np.ndarray) -> np.ndarray:
                    return (
                        np.clip((image + 1000.0) / 2000.0, 0.0, 1.0) * 255
                    ).astype(np.uint8)

                def resize_rgb(image: np.ndarray) -> np.ndarray:
                    return np.asarray(
                        Image.fromarray(image).resize(
                            (256, 256), resample=Image.Resampling.BILINEAR
                        )
                    )

                palette = np.asarray(
                    [
                        [0, 0, 0],
                        [230, 25, 75],
                        [60, 180, 75],
                        [255, 225, 25],
                        [0, 130, 200],
                        [245, 130, 48],
                        [145, 30, 180],
                        [70, 240, 240],
                        [240, 50, 230],
                        [210, 245, 60],
                    ],
                    dtype=np.uint8,
                )
                segmentation_planes = []
                for clean_plane, label_plane in zip(planes(ct), planes(label)):
                    base = np.repeat(
                        gray(clean_plane.astype(np.float32))[..., None], 3, axis=2
                    )
                    label_plane = label_plane.astype(np.uint8)
                    color = palette[label_plane]
                    foreground_mask = label_plane > 0
                    overlay = base.copy()
                    overlay[foreground_mask] = (
                        0.55 * base[foreground_mask]
                        + 0.45 * color[foreground_mask]
                    ).astype(np.uint8)
                    segmentation_planes.append(resize_rgb(overlay))
                segmentation = np.concatenate(segmentation_planes, axis=1)
                split_name = overlay_cases[case_id]
                iio.imwrite(
                    paths["qc/overlays"] / f"{split_name}_{case_id}_segmentation.png",
                    segmentation,
                )
                for task_name in ("restoration", "reconstruction"):
                    source = np.asanyarray(
                        nib.load(
                            paths[f"tasks/{task_name}"]
                            / case_id
                            / "source_hu.nii.gz"
                        ).dataobj
                    )
                    rows = []
                    for clean_plane, source_plane in zip(planes(ct), planes(source)):
                        clean_plane = clean_plane.astype(np.float32)
                        source_plane = source_plane.astype(np.float32)
                        clean_rgb = np.repeat(
                            gray(clean_plane)[..., None], 3, axis=2
                        )
                        source_rgb = np.repeat(
                            gray(source_plane)[..., None], 3, axis=2
                        )
                        difference = np.abs(source_plane - clean_plane)
                        difference_rgb = np.repeat(
                            (
                                np.clip(difference / 500.0, 0.0, 1.0) * 255
                            ).astype(np.uint8)[..., None],
                            3,
                            axis=2,
                        )
                        rows.append(
                            np.concatenate(
                                [
                                    resize_rgb(clean_rgb),
                                    resize_rgb(source_rgb),
                                    resize_rgb(difference_rgb),
                                ],
                                axis=1,
                            )
                        )
                    montage = np.concatenate(rows, axis=0)
                    iio.imwrite(
                        paths["qc/overlays"]
                        / f"{split_name}_{case_id}_{task_name}.png",
                        montage,
                    )
        except Exception as error:
            failures.append({"case_id": case_id, "error": repr(error)})
    write_csv(paths["qc"] / "stats.csv", records)
    write_json(
        paths["qc"] / "report.json",
        {
            "checked": len(records) + len(failures),
            "passed": len(records),
            "failed": len(failures),
            "failures": failures,
            "selected_case_ids": selected_case_ids,
            "split_counts": {
                split_name: len(
                    [
                        line
                        for line in (paths["splits"] / f"{split_name}.txt")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    ]
                )
                for split_name in ("train", "val", "test")
            },
            "split_file_sha256": {
                split_name: sha256sum(paths["splits"] / f"{split_name}.txt")
                for split_name in ("train", "val", "test")
            },
            "overlays": [
                {
                    "path": str(path),
                    "sha256": sha256sum(path),
                }
                for path in sorted(paths["qc/overlays"].glob("*.png"))
            ],
            "config_hash": config["_config_hash"],
        },
    )
    if failures:
        raise RuntimeError(f"QC failed for {len(failures)} case(s); see qc/report.json")


def command_status(config: dict, paths: dict[str, Path]) -> None:
    archives = []
    expected_archives = config["dataset"]["archives"]
    for name, expected in expected_archives.items():
        expected_bytes = expected["bytes"]
        path = paths["raw/archives"] / name
        control = path.with_suffix(path.suffix + ".aria2")
        logical_bytes = path.stat().st_size if path.exists() else 0
        allocated_bytes = (
            path.stat().st_blocks * 512 if path.exists() else 0
        )
        archives.append(
            {
                "name": name,
                "logical_bytes": logical_bytes,
                "allocated_bytes": allocated_bytes,
                "expected_bytes": expected_bytes,
                "expected_etag_sha256": expected["etag_sha256"],
                "allocated_percent": round(100.0 * allocated_bytes / expected_bytes, 3),
                "complete": (
                    path.exists()
                    and not control.exists()
                    and logical_bytes == expected_bytes
                ),
            }
        )
    incomplete_root = PROJECT_ROOT / config["dataset"]["root"] / "archives_http/.cache/huggingface/download"
    for path in sorted(incomplete_root.glob("*.incomplete")):
        archives.append({"name": path.name, "bytes": path.stat().st_size, "complete": False})
    status = {
        "archives": archives,
        "extracted_cases": len(discover_cases(paths["raw/extracted"])),
        "canonical_cases": len(list(paths["canonical"].glob("*/ct_hu.nii.gz"))),
        "segmentation_cases": len(list(paths["tasks/segmentation"].glob("*/meta.json"))),
        "restoration_cases": len(list(paths["tasks/restoration"].glob("*/source_hu.nii.gz"))),
        "reconstruction_cases": len(list(paths["tasks/reconstruction"].glob("*/source_hu.nii.gz"))),
        "processed_cases": len(list(paths["processed"].glob("*/*.npz"))),
    }
    print(json.dumps(status, indent=2))


def command_calibrate(config: dict, paths: dict[str, Path]) -> None:
    from medicalmodel_data.degradation import sparse_view_slice

    size = 128
    yy, xx = np.ogrid[:size, :size]
    phantom = np.full((size, size), -1000.0, dtype=np.float32)
    body = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size * 0.36) ** 2
    insert = (xx - size * 0.58) ** 2 + (yy - size * 0.48) ** 2 < (
        size * 0.08
    ) ** 2
    phantom[body] = 0.0
    phantom[insert] = 500.0
    params = config["tasks"]["restoration"]
    reconstruction = sparse_view_slice(
        phantom,
        views=params["views"],
        mu_water=params["mu_water_per_mm"],
        pixel_size_mm=config["canonical"]["target_spacing_xyz_mm"][0],
    )
    report = {
        "phantom_size": size,
        "views": params["views"],
        "body_mae_hu": float(np.mean(np.abs(reconstruction[body] - phantom[body]))),
        "body_bias_hu": float(np.mean(reconstruction[body] - phantom[body])),
        "insert_mae_hu": float(np.mean(np.abs(reconstruction[insert] - phantom[insert]))),
        "insert_bias_hu": float(np.mean(reconstruction[insert] - phantom[insert])),
        "finite": bool(np.isfinite(reconstruction).all()),
        "acceptance": {"body_mae_hu_lt": 50.0, "abs_body_bias_hu_lt": 30.0},
        "config_hash": config["_config_hash"],
    }
    report["passed"] = (
        report["finite"]
        and report["body_mae_hu"] < 50.0
        and abs(report["body_bias_hu"]) < 30.0
    )
    write_json(paths["qc"] / "projection_calibration.json", report)
    if not report["passed"]:
        raise RuntimeError("Projection calibration failed; see qc/projection_calibration.json")


def command_smoke_test(config: dict, paths: dict[str, Path]) -> None:
    from medicalmodel_data.loader import (
        TASK_IDS,
        collate_same_task,
        load_sample,
        make_task_balanced_batch,
    )

    report: dict[str, object] = {"splits": {}, "config_hash": config["_config_hash"]}
    expected_patch = tuple(config["processed"]["patch_shape_zyx"])
    failures = []
    split_case_ids: dict[str, set[str]] = {}
    all_sample_ids: set[str] = set()
    for split_name in ("train", "val", "test"):
        split_case_ids[split_name] = {
            line.strip()
            for line in (paths["splits"] / f"{split_name}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        manifest_path = paths["manifests"] / f"{split_name}.jsonl"
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tasks = {record["task"] for record in records}
        split_summary = {
            "cases": len(split_case_ids[split_name]),
            "records": len(records),
            "expected_records": len(split_case_ids[split_name]) * 3,
            "tasks": sorted(tasks),
        }
        report["splits"][split_name] = split_summary
        if len(records) != len(split_case_ids[split_name]) * 3:
            failures.append(f"{split_name}: unexpected manifest record count")
        if tasks != {"segmentation", "restoration", "reconstruction"}:
            failures.append(f"{split_name}: missing task records")
            continue
        case_tasks: dict[str, set[str]] = {}
        per_task_samples: dict[str, list[object]] = {
            task_name: [] for task_name in tasks
        }
        clean_targets: dict[str, dict[str, np.ndarray]] = {}
        for record in records:
            sample_id = record["sample_id"]
            if sample_id in all_sample_ids:
                failures.append(f"duplicate sample_id: {sample_id}")
            all_sample_ids.add(sample_id)
            case_id = record["case_id"]
            task_name = record["task"]
            if record["split"] != split_name or case_id not in split_case_ids[split_name]:
                failures.append(f"{sample_id}: split/case ownership mismatch")
            case_tasks.setdefault(case_id, set()).add(task_name)
            if record.get("config_hash") != config["_config_hash"]:
                failures.append(f"{sample_id}: stale config hash")
            if not isinstance(record.get("source"), dict) or not isinstance(
                record.get("target"), dict
            ):
                failures.append(f"{sample_id}: missing explicit source/target")
            arrays = np.load(record["source"]["path"])
            if record["source"]["key"] not in arrays or record["target"]["key"] not in arrays:
                failures.append(f"{sample_id}: source/target key missing")
                continue
            source = arrays[record["source"]["key"]]
            target = arrays[record["target"]["key"]]
            if len(per_task_samples[task_name]) < 2:
                per_task_samples[task_name].append(load_sample(record))
            if source.shape != expected_patch:
                failures.append(
                    f"{split_name}/{task_name}: source {source.shape} != {expected_patch}"
                )
            expected_target = (
                (9, *expected_patch) if task_name == "segmentation" else expected_patch
            )
            if target.shape != expected_target:
                failures.append(
                    f"{split_name}/{task_name}: target {target.shape} != {expected_target}"
                )
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                failures.append(f"{split_name}/{task_name}: non-finite patch")
            if source.min() < -1 or source.max() > 1:
                failures.append(f"{split_name}/{task_name}: source outside [-1,1]")
            if target.min() < -1 or target.max() > 1:
                failures.append(f"{split_name}/{task_name}: target outside [-1,1]")
            if task_name == "segmentation":
                if record.get("label") is None or not Path(record["label"]["path"]).exists():
                    failures.append(f"{sample_id}: missing authoritative label")
                clean_targets.setdefault(case_id, {})["segmentation"] = source
            else:
                clean_targets.setdefault(case_id, {})[task_name] = target
        for case_id in split_case_ids[split_name]:
            if case_tasks.get(case_id) != {
                "segmentation",
                "restoration",
                "reconstruction",
            }:
                failures.append(f"{split_name}/{case_id}: not exactly three tasks")
            targets = clean_targets.get(case_id, {})
            if set(targets) == {"segmentation", "restoration", "reconstruction"}:
                if not np.array_equal(
                    targets["segmentation"], targets["restoration"]
                ) or not np.array_equal(
                    targets["segmentation"], targets["reconstruction"]
                ):
                    failures.append(f"{split_name}/{case_id}: clean targets differ")
        batches = {}
        balanced_candidates = []
        for task_name, samples in per_task_samples.items():
            if samples:
                batch = collate_same_task(samples)
                batches[task_name] = {
                    "source_bcthw": list(batch["source"].shape),
                    "target_bcthw": list(batch["target"].shape),
                    "instructions": len(batch["instruction"]),
                    "task_ids": batch["task_id"].tolist(),
                    "case_ids": len(batch["case_id"]),
                    "sample_ids": len(batch["sample_id"]),
                }
                balanced_candidates.append(samples[0])
        split_summary["per_task_batch"] = batches
        try:
            balanced = make_task_balanced_batch(balanced_candidates)
            balanced_task_ids = balanced["task_id"].tolist()
            if set(balanced_task_ids) != set(TASK_IDS.values()):
                failures.append(f"{split_name}: task-balanced IDs are invalid")
            if any(not instruction for instruction in balanced["instruction"]):
                failures.append(f"{split_name}: empty instruction in balanced batch")
            if len(balanced["case_id"]) != 3 or len(balanced["sample_id"]) != 3:
                failures.append(f"{split_name}: invalid balanced batch metadata")
            split_summary["task_balanced_batch"] = {
                "task_ids": balanced_task_ids,
                "instructions": balanced["instruction"],
                "case_ids": balanced["case_id"],
                "sample_ids": balanced["sample_id"],
                "per_task_source_shapes": {
                    task: list(values["source"].shape)
                    for task, values in balanced["per_task"].items()
                },
                "per_task_target_shapes": {
                    task: list(values["target"].shape)
                    for task, values in balanced["per_task"].items()
                },
            }
        except ValueError as error:
            failures.append(f"{split_name}: {error}")
    names = ("train", "val", "test")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_case_ids[left] & split_case_ids[right]
            if overlap:
                failures.append(f"split leakage {left}/{right}: {sorted(overlap)[:10]}")
    report["failures"] = failures
    report["passed"] = not failures
    write_json(paths["qc"] / "dataloader_smoke_test.json", report)
    if failures:
        raise RuntimeError("Smoke test failed; see qc/dataloader_smoke_test.json")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = get_paths(config)
    if args.command == "setup":
        write_json(paths["metadata"] / "class_map.json", config["dataset"]["class_map"])
        write_json(
            paths["metadata"] / "palette.json",
            {
                "0": [0, 0, 0],
                "1": [230, 25, 75],
                "2": [60, 180, 75],
                "3": [255, 225, 25],
                "4": [0, 130, 200],
                "5": [245, 130, 48],
                "6": [145, 30, 180],
                "7": [70, 240, 240],
                "8": [240, 50, 230],
                "9": [210, 245, 60],
            },
        )
        write_json(paths["metadata"] / "pipeline_config.json", config)
    elif args.command == "extract":
        command_extract(config, paths, args.archive)
    elif args.command == "verify-archives":
        command_verify_archives(config, paths)
    elif args.command == "inventory":
        command_inventory(config, paths, args.checksum)
    elif args.command == "split":
        command_split(config, paths, args.pilot)
    elif args.command == "canonicalize":
        command_canonicalize(config, paths, args.split, args.limit, args.force)
    elif args.command == "derive":
        workers = args.workers or config["runtime"]["projection_workers"]
        if workers < 1:
            raise ValueError("--workers must be >= 1")
        command_derive(
            config,
            paths,
            args.split,
            args.limit,
            args.task,
            args.force,
            workers,
        )
    elif args.command == "patches":
        command_patches(config, paths, args.split, args.limit, args.force)
    elif args.command == "sdf-volumes":
        ids = selected_ids(paths, args.split, args.limit)
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("--shard-index must be in [0, --num-shards)")
        ids = ids[args.shard_index :: args.num_shards]
        params = config["tasks"]["segmentation"]
        count = generate_full_volume_sdf(
            paths["root"], ids, params["sdf_classes"], params["sdf_clip_mm"],
            params["sdf_positive_inside"], args.force, args.refresh_stale,
        )
        print(f"full-volume SDF complete: {count}")
    elif args.command == "case-manifests":
        splits = ("train", "val", "test") if args.split == "all" else (args.split,)
        positive_inside = config["tasks"]["segmentation"]["sdf_positive_inside"]
        print(json.dumps(build_case_manifests(paths["root"], splits, positive_inside), sort_keys=True))
    elif args.command == "qc":
        command_qc(config, paths, args.split, args.limit)
    elif args.command == "calibrate":
        command_calibrate(config, paths)
    elif args.command == "smoke-test":
        command_smoke_test(config, paths)
    elif args.command == "status":
        command_status(config, paths)


if __name__ == "__main__":
    main()
