from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import write_json, write_jsonl


PROMPTS = {
    "restoration": "Restore the low-dose CT into a normal-dose CT volume.",
    "reconstruction": "Reconstruct a full-view CT volume from the sparse-view CT.",
}


def _case_ids(root: Path, splits: Iterable[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for split in splits:
        split_path = root / "splits" / f"{split}.txt"
        result.extend((case_id, split) for case_id in split_path.read_text().splitlines() if case_id)
    return result


def generate_full_volume_sdf(
    root: Path,
    case_ids: Iterable[str],
    classes: list[int],
    clip_mm: float,
    positive_inside: bool,
    force: bool = False,
    refresh_stale: bool = False,
) -> int:
    """Generate one CxDxHxW float16 SDF from each complete canonical mask."""
    import nibabel as nib

    completed = 0
    for case_id in case_ids:
        mask_path = root / "canonical" / case_id / "label_id.nii.gz"
        output = root / "tasks" / "segmentation" / "sdf" / f"{case_id}.npy"
        meta_path = output.with_suffix(".json")
        if output.exists() and meta_path.exists() and not force:
            metadata = json.loads(meta_path.read_text())
            expected = {
                "classes": [int(value) for value in classes],
                "clip_mm": float(clip_mm),
                "positive_inside": bool(positive_inside),
            }
            actual = {key: metadata.get(key) for key in expected}
            if actual != expected and not refresh_stale:
                raise ValueError(
                    f"Stale SDF representation for {case_id}: expected {expected}, "
                    f"found {actual}. Re-run sdf-volumes with --force."
                )
            if actual == expected:
                completed += 1
                continue
        image = nib.load(mask_path)
        label = np.transpose(np.asanyarray(image.dataobj), (2, 1, 0)).astype(np.uint8, copy=False)
        spacing_zyx = tuple(float(x) for x in image.header.get_zooms()[:3][::-1])
        from scipy.ndimage import distance_transform_edt

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        sdf_shape = (len(classes), *label.shape)
        sdf = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16,
            shape=sdf_shape,
        )
        try:
            for channel, class_id in enumerate(classes):
                inside = label == class_id
                if not np.any(inside):
                    sdf[channel].fill(-1.0 if positive_inside else 1.0)
                    continue
                distance_inside = distance_transform_edt(inside, sampling=spacing_zyx)
                distance_outside = distance_transform_edt(~inside, sampling=spacing_zyx)
                distance_inside -= distance_outside
                if not positive_inside:
                    distance_inside *= -1
                np.clip(distance_inside, -clip_mm, clip_mm, out=distance_inside)
                distance_inside /= clip_mm
                sdf[channel] = distance_inside
            sdf.flush()
            del sdf
            temporary.replace(output)
        except BaseException:
            del sdf
            temporary.unlink(missing_ok=True)
            raise
        write_json(meta_path, {
            "case_id": case_id,
            "source_mask": str(mask_path.relative_to(root)),
            "shape_czyx": list(sdf_shape),
            "spacing_zyx_mm": list(spacing_zyx),
            "classes": classes,
            "clip_mm": clip_mm,
            "positive_inside": positive_inside,
            "dtype": "float16",
            "representation": "full_volume_sdf",
        })
        completed += 1
        print(f"[{completed}] {case_id}: {output}", flush=True)
    return completed


def _variants(directory: Path, case_id: str, legacy_name: str) -> list[str]:
    candidates = sorted(directory.glob(f"{case_id}*.npy"))
    legacy = directory / case_id / legacy_name
    if legacy.exists():
        candidates.append(legacy)
    return [str(path.relative_to(directory.parents[1])) for path in candidates]


def build_case_manifests(
    root: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    sdf_positive_inside: bool = True,
) -> dict[str, int]:
    """Write one manifest row per case, containing all available task variants."""
    counts: dict[str, int] = {}
    class_map_path = root / "metadata" / "class_map.json"
    class_map = json.loads(class_map_path.read_text()) if class_map_path.exists() else {}
    for split in splits:
        rows = []
        for case_id, _ in _case_ids(root, (split,)):
            canonical = root / "canonical" / case_id
            sdf = root / "tasks" / "segmentation" / "sdf" / f"{case_id}.npy"
            ldct = _variants(root / "tasks" / "restoration", case_id, "source_hu.nii.gz")
            sparse = _variants(root / "tasks" / "reconstruction", case_id, "source_hu.nii.gz")
            required = [canonical / "ct_hu.nii.gz", canonical / "label_id.nii.gz", sdf]
            if not all(path.exists() for path in required) or not ldct or not sparse:
                missing = [str(path) for path in required if not path.exists()]
                if not ldct: missing.append(f"restoration/{case_id}")
                if not sparse: missing.append(f"reconstruction/{case_id}")
                raise FileNotFoundError(f"Incomplete case {case_id}: {missing}")
            meta_path = canonical / "meta.json"
            metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            rows.append({
                "case_id": case_id,
                "split": split,
                "image": str((canonical / "ct_hu.nii.gz").relative_to(root)),
                "mask": str((canonical / "label_id.nii.gz").relative_to(root)),
                "sdf": str(sdf.relative_to(root)),
                "ldct": ldct,
                "sparse_view_ct": sparse,
                "spacing_xyz_mm": metadata.get("canonical_spacing_xyz_mm"),
                "sdf_classes": [int(x) for x in sorted(class_map, key=int) if int(x) != 0],
                "sdf_positive_inside": sdf_positive_inside,
                "label_map": class_map,
                "metadata": {"canonical": metadata},
            })
        output = root / "processed" / "manifests" / f"{split}.jsonl"
        write_jsonl(output, rows)
        counts[split] = len(rows)
    return counts
