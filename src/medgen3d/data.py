from __future__ import annotations

import json
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


CT_TASKS = ("segmentation", "restoration", "reconstruction")
VOLUME_TASKS = ("synthesis", "generation")
TASKS = (*CT_TASKS, *VOLUME_TASKS)

DEFAULT_PROMPTS = {
    "segmentation": "Segment the {structure} in this CT volume.",
    "restoration": "Restore the low-dose CT into a normal-dose CT volume.",
    "reconstruction": "Reconstruct a full-view CT volume from the sparse-view CT.",
    "synthesis": "Synthesize the T2 MRI volume from this T1 MRI volume.",
    "generation": "Generate a 3D chest CT volume matching this radiology report.",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sliding_window_starts(depth: int, window: int, stride: int) -> tuple[int, ...]:
    """Cover a volume from first to last slice, including a flush final window."""
    if depth <= window:
        return (0,)
    starts = list(range(0, depth - window + 1, stride))
    final = depth - window
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _crop_or_pad_depth(array: np.ndarray, start: int, depth: int, fill: float) -> np.ndarray:
    """Crop the D axis of [D,H,W] or [C,D,H,W], padding only short volumes."""
    depth_axis = array.ndim - 3
    available = array.shape[depth_axis]
    stop = min(available, start + depth)
    selector = [slice(None)] * array.ndim
    selector[depth_axis] = slice(start, stop)
    cropped = array[tuple(selector)]
    missing = depth - cropped.shape[depth_axis]
    if missing <= 0:
        return np.ascontiguousarray(cropped)
    pad_width = [(0, 0)] * array.ndim
    pad_width[depth_axis] = (0, missing)
    return np.pad(cropped, pad_width, mode="constant", constant_values=fill)


def _resize_xy(array: np.ndarray, shape_hw: Sequence[int], mode: str) -> np.ndarray:
    """Resize each axial slice while preserving depth and optional channels."""
    value = np.asarray(array, dtype=np.float32)
    had_channel = value.ndim == 4
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4:
        raise ValueError(f"Expected [D,H,W] or [C,D,H,W], got {value.shape}")
    channels, depth, height, width = value.shape
    tensor = torch.from_numpy(value).permute(1, 0, 2, 3)
    kwargs = {"size": tuple(int(x) for x in shape_hw), "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    resized = torch.nn.functional.interpolate(tensor, **kwargs)
    result = resized.permute(1, 0, 2, 3).numpy()
    return result if had_channel else result[0]


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_array(root: Path, spec: str | dict[str, Any]) -> np.ndarray:
    if isinstance(spec, str):
        path, key, channel = _resolve(root, spec), None, None
    else:
        path = _resolve(root, str(spec["path"]))
        key, channel = spec.get("key"), spec.get("channel")
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        archive = np.load(path, mmap_mode="r")
        key = key or next(iter(archive.files))
        array = archive[key]
    elif path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        array = np.asanyarray(nib.load(path).dataobj)
        array = np.moveaxis(array, (0, 1, 2), (2, 1, 0))  # XYZ -> DHW
    else:
        raise ValueError(f"Unsupported volume file: {path}")
    if channel is not None:
        array = array[int(channel)]
    return np.asarray(array)


class UnifiedCTDataset(Dataset[dict[str, Any]]):
    """One manifest-backed interface for every MedGen3D task.

    Offline manifests own crop coordinates and degradations. The loader never
    manufactures validation/test patches or projection-domain corruptions.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str,
        split_file: str | Path | None = None,
        training: bool | None = None,
        spatial_flip_probability: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.manifest = _resolve(self.root, str(manifest))
        self.split = split
        self.training = split == "train" if training is None else training
        if split != "train" and self.training:
            raise ValueError("Validation/test datasets cannot enable training transforms")
        self.flip_probability = spatial_flip_probability
        self.seed = seed
        self.epoch = 0
        self.records = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        allowed: set[str] | None = None
        if split_file:
            allowed = {x.strip() for x in _resolve(self.root, str(split_file)).read_text().splitlines() if x.strip()}
        for record in self.records:
            if record.get("split") != split:
                raise ValueError(f"Manifest contains {record.get('split')} record in {split} dataset")
            if allowed is not None and str(record["case_id"]) not in allowed:
                raise ValueError(f"Case {record['case_id']} is absent from frozen {split} split")
            if record["task"] not in TASKS:
                raise ValueError(f"Unknown task {record['task']}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        condition = _load_array(self.root, record.get("condition", record.get("source"))).astype(np.float32, copy=True)
        target = _load_array(self.root, record["target"]).astype(np.float32, copy=True)
        if condition.ndim == 3:
            condition = condition[None]
        if target.ndim == 3:
            target = target[None]
        # The preprocessing archive is positive-inside v1. The model contract
        # is negative-inside v1 so inference uses the explicit SDF < 0 rule.
        if record["task"] == "segmentation" and record.get("degradation", {}).get("sdf_positive_inside", False):
            target = -target
        if condition.shape != target.shape:
            raise ValueError(f"Unaligned sample {record.get('sample_id')}: {condition.shape} != {target.shape}")
        valid = np.ones((1, *condition.shape[-3:]), dtype=np.float32)
        if record.get("valid_mask"):
            valid = _load_array(self.root, record["valid_mask"]).astype(np.float32, copy=True)
            if valid.ndim == 3:
                valid = valid[None]
        elif record.get("crop_start_zyx") is not None and record.get("canonical_ct"):
            # Offline crops may extend beyond the canonical volume. Recover the
            # exact geometrically valid region instead of training on fill voxels.
            import nibabel as nib
            canonical_xyz = nib.load(record["canonical_ct"]).shape[:3]
            canonical_zyx = canonical_xyz[::-1]
            for axis, (start, size, available) in enumerate(zip(record["crop_start_zyx"], condition.shape[-3:], canonical_zyx)):
                lo, hi = max(0, -int(start)), min(size, int(available) - int(start))
                selector = [slice(None)] * 4
                selector[axis + 1] = slice(0, lo); valid[tuple(selector)] = 0
                selector[axis + 1] = slice(max(0, hi), size); valid[tuple(selector)] = 0
        if self.training and self.flip_probability:
            rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
            for axis in (-1, -2, -3):
                if rng.random() < self.flip_probability:
                    condition, target, valid = (np.flip(x, axis=axis).copy() for x in (condition, target, valid))
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("spacing", record.get("spacing", record.get("spacing_xyz_mm")))
        metadata.setdefault("structure", record.get("structure"))
        metadata.setdefault("degradation", record.get("degradation", {}))
        metadata["sample_id"] = record.get("sample_id")
        metadata["split"] = self.split
        metadata["prompt_template_version"] = record.get("prompt_template_version")
        metadata["prompt_embedding_cache"] = record.get("prompt_embedding_cache")
        return {
            "case_id": str(record["case_id"]),
            "task": str(record["task"]),
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target),
            "prompt": str(record.get("prompt", record.get("instruction", ""))),
            "valid_mask": torch.from_numpy(valid),
            "metadata": metadata,
        }


class ManifestVolumeDataset(Dataset[dict[str, Any]]):
    """Frozen whole-volume samples for MRI synthesis and text-to-CT generation.

    The preprocessing stage owns orientation, resampling and normalization.
    Runtime loading is deliberately simple so validation and test can never
    manufacture a different target grid.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str,
        task: str,
        num_samples: int | None = None,
        training: bool | None = None,
        spatial_flip_probability: float = 0.0,
        volume_shape_dhw: Sequence[int] | None = None,
        sliding_window_stride: int | None = None,
        whole_volume: bool = False,
        seed: int = 0,
    ) -> None:
        if task not in VOLUME_TASKS:
            raise ValueError(f"ManifestVolumeDataset only supports {VOLUME_TASKS}, got {task}")
        self.root = Path(root)
        self.manifest = _resolve(self.root, str(manifest))
        self.split = split
        self.task = task
        self.training = split == "train" if training is None else bool(training)
        if split != "train" and self.training:
            raise ValueError("Validation/test volume datasets cannot enable training transforms")
        self.flip_probability = float(spatial_flip_probability) if self.training else 0.0
        self.seed = int(seed)
        self.volume_shape = tuple(int(value) for value in (volume_shape_dhw or (97, 96, 96)))
        if len(self.volume_shape) != 3 or any(value <= 0 for value in self.volume_shape):
            raise ValueError("volume_shape_dhw must contain three positive integers")
        self.sliding_window_stride = int(sliding_window_stride or self.volume_shape[0])
        if self.sliding_window_stride <= 0:
            raise ValueError("sliding_window_stride must be positive")
        self.records = [
            json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"Empty {task} manifest: {self.manifest}")
        for record in self.records:
            if record.get("split") != split or record.get("task") != task:
                raise ValueError(f"Manifest record violates {split}/{task}: {record.get('case_id')}")
            if not record.get("patient_id"):
                raise ValueError(f"Manifest record has no patient_id: {record.get('case_id')}")
        self.num_samples = len(self.records) if num_samples is None else int(num_samples)
        self.whole_volume = bool(whole_volume)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index % len(self.records)]
        condition = _load_array(self.root, record["condition"]).astype(np.float32, copy=True)
        target = _load_array(self.root, record["target"]).astype(np.float32, copy=True)
        condition = condition[None] if condition.ndim == 3 else condition
        target = target[None] if target.ndim == 3 else target
        if condition.shape != target.shape:
            raise ValueError(
                f"Expected aligned condition/target for {record['case_id']}, "
                f"got {condition.shape} and {target.shape}"
            )
        if condition.shape[-2:] != self.volume_shape[-2:]:
            raise ValueError(
                f"Expected preprocessed XY shape {self.volume_shape[-2:]} for "
                f"{record['case_id']}, got {condition.shape[-2:]}"
            )
        valid = np.ones((1, *condition.shape[-3:]), dtype=np.float32)
        if record.get("valid_mask"):
            valid = _load_array(self.root, record["valid_mask"]).astype(np.float32, copy=True)
            valid = valid[None] if valid.ndim == 3 else valid
        full_depth = int(condition.shape[-3])
        starts = _sliding_window_starts(full_depth, self.volume_shape[0], self.sliding_window_stride)
        if self.whole_volume:
            z_start = 0
        else:
            if self.training:
                cycle = index // len(self.records)
                z_start = starts[cycle % len(starts)]
            else:
                z_start = starts[len(starts) // 2]
            condition = _crop_or_pad_depth(condition, z_start, self.volume_shape[0], -1.0)
            target = _crop_or_pad_depth(target, z_start, self.volume_shape[0], -1.0)
            valid = _crop_or_pad_depth(valid, z_start, self.volume_shape[0], 0.0)
        if self.training and self.flip_probability:
            rng = np.random.default_rng(stable_seed(self.seed, self.task, str(index)))
            for axis in (-1, -2, -3):
                if rng.random() < self.flip_probability:
                    condition, target, valid = (
                        np.flip(value, axis=axis).copy() for value in (condition, target, valid)
                    )
        metadata = dict(record.get("metadata", {}))
        metadata.update({
            "patient_id": str(record["patient_id"]),
            "split": self.split,
            "source_modality": record.get("source_modality"),
            "target_modality": record.get("target_modality"),
            "inverse_transform": record.get("inverse_transform"),
            "sliding_window_start_z": z_start,
            "sliding_window_depth": self.volume_shape[0],
            "sliding_window_stride": self.sliding_window_stride,
            "full_depth": full_depth,
            "z_start_fraction": z_start / max(full_depth - self.volume_shape[0], 1),
            "z_extent_fraction": min(self.volume_shape[0], full_depth) / max(full_depth, 1),
        })
        return {
            "case_id": str(record["case_id"]),
            "task": self.task,
            "condition": torch.from_numpy(np.asarray(condition, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "prompt": str(record["prompt"]),
            "valid_mask": torch.from_numpy(np.asarray(valid, dtype=np.float32)),
            "metadata": metadata,
        }


class BalancedMultiTaskDataset(Dataset[dict[str, Any]]):
    """Exact round-robin task balancing over independent task datasets."""

    def __init__(self, datasets: dict[str, Dataset], tasks: Sequence[str], num_samples: int) -> None:
        self.tasks = tuple(tasks)
        if not self.tasks or set(self.tasks) != set(datasets):
            raise ValueError("BalancedMultiTaskDataset requires exactly one dataset per requested task")
        self.datasets = dict(datasets)
        self.num_samples = int(num_samples)
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if any(len(dataset) <= 0 for dataset in self.datasets.values()):
            raise ValueError("Every task dataset must be non-empty")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = index % len(self.tasks)
        task = self.tasks[task_index]
        within_task_index = index // len(self.tasks)
        return self.datasets[task][within_task_index]


def build_task_dataset(
    data: dict[str, Any], task: str, split: str, *, seed: int,
    num_samples: int | None = None, training: bool | None = None,
    whole_volume: bool = False,
) -> Dataset:
    """Construct one task without leaking task-specific paths into trainers."""
    if task in CT_TASKS:
        if training is False and split == "train":
            raise ValueError("A train split cannot be forced into evaluation mode")
        return DynamicCaseDataset(
            data["root"], data["manifests"][split], split, data["patch_size_dhw"],
            task_weights={task: 1.0}, evaluation_task=None if split == "train" else task,
            segmentation_target=data.get("segmentation_target", "sdf"),
            foreground_probability=data.get("foreground_probability", .7),
            segmentation_foreground_probability=data.get("segmentation_foreground_probability", 1.0),
            segmentation_surface_probability=data.get("segmentation_surface_probability", 0.0),
            segmentation_foreground_warmup_probability=data.get(
                "segmentation_foreground_warmup_probability"
            ),
            segmentation_surface_warmup_probability=data.get(
                "segmentation_surface_warmup_probability"
            ),
            segmentation_foreground_warmup_fraction=(
                data.get("segmentation_foreground_warmup_fraction", 0.0)
                if split == "train" else 0.0
            ),
            segmentation_center_jitter_zyx=data.get("segmentation_center_jitter_zyx", (12, 12, 12)),
            segmentation_surface_center_jitter_zyx=data.get(
                "segmentation_surface_center_jitter_zyx", (2, 2, 2)
            ),
            segmentation_surface_band_mm=data.get("segmentation_surface_band_mm", 2.0),
            segmentation_sdf_clip_mm=data.get("sdf", {}).get("clip_distance_mm", 8.0),
            segmentation_min_foreground_voxels=data.get("segmentation_min_foreground_voxels", 100),
            segmentation_organ_sampling=(
                data.get("segmentation_organ_sampling", "case_uniform")
                if task == "segmentation" else "case_uniform"
            ),
            segmentation_case_sampling=data.get("segmentation_case_sampling", "shuffled_cycle"),
            segmentation_zoom=data.get("segmentation_zoom"),
            resize_xy=data.get("resize_xy"),
            sliding_window_depth=data.get("sliding_window", {}).get("depth"),
            sliding_window_stride=data.get("sliding_window", {}).get("stride"),
            hu_clip=data["hu_clip"], output_range=data["ct_normalization"]["output_range"],
            spatial_flip_probability=(data.get("spatial_flip_probability", .5) if split == "train" else 0.0),
            seed=seed, num_samples=num_samples,
        )
    if task in VOLUME_TASKS:
        spec = data["volume_tasks"][task]
        return ManifestVolumeDataset(
            spec["root"], spec["manifests"][split], split, task,
            num_samples=num_samples, training=training,
            spatial_flip_probability=(spec.get("spatial_flip_probability", 0.0) if split == "train" else 0.0),
            volume_shape_dhw=data.get("patch_size_dhw"),
            sliding_window_stride=data.get("sliding_window", {}).get("stride"),
            whole_volume=whole_volume,
            seed=seed,
        )
    raise ValueError(f"Unknown task {task}")


def audit_multitask_splits(data: dict[str, Any], tasks: Sequence[str]) -> None:
    """Reject patient leakage and malformed frozen manifests before GPU use."""
    if any(task in CT_TASKS for task in tasks):
        audit_disjoint_splits(data["root"], data["manifests"], data["splits"])
    for task in (name for name in tasks if name in VOLUME_TASKS):
        spec = data["volume_tasks"][task]
        root = Path(spec["root"])
        patient_sets: dict[str, set[str]] = {}
        expected = spec.get("expected_split_counts", {})
        for split in ("train", "val", "test"):
            path = _resolve(root, spec["manifests"][split])
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(row.get("task") != task or row.get("split") != split for row in records):
                raise ValueError(f"Malformed {task}/{split} manifest")
            patients = {str(row["patient_id"]) for row in records}
            if len(patients) != len(records):
                raise ValueError(f"{task}/{split} must contain exactly one volume per patient")
            if split in expected and len(patients) != int(expected[split]):
                raise ValueError(
                    f"{task}/{split} expected {expected[split]} patients, observed {len(patients)}"
                )
            patient_sets[split] = patients
        if any(patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Patient leakage detected in {task} manifests")


class DynamicCaseDataset(Dataset[dict[str, Any]]):
    """Case-indexed multi-task dataset with online paired patch processing.

    Training chooses task/configuration/patch deterministically from
    ``seed + epoch + index``. Validation and test require one explicit task and
    use a fixed foreground/volume-centred patch without augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str,
        patch_shape_zyx: Sequence[int],
        task_weights: dict[str, float] | None = None,
        evaluation_task: str | None = None,
        segmentation_target: str = "sdf",
        foreground_probability: float = 0.7,
        segmentation_foreground_probability: float = 1.0,
        segmentation_surface_probability: float = 0.0,
        segmentation_foreground_warmup_probability: float | None = None,
        segmentation_surface_warmup_probability: float | None = None,
        segmentation_foreground_warmup_fraction: float = 0.0,
        segmentation_center_jitter_zyx: Sequence[int] = (12, 12, 12),
        segmentation_surface_center_jitter_zyx: Sequence[int] = (2, 2, 2),
        segmentation_surface_band_mm: float = 2.0,
        segmentation_sdf_clip_mm: float = 8.0,
        segmentation_min_foreground_voxels: int = 100,
        segmentation_organ_sampling: str = "case_uniform",
        segmentation_case_sampling: str = "shuffled_cycle",
        segmentation_zoom: dict[str, Any] | None = None,
        resize_xy: Sequence[int] | None = None,
        sliding_window_depth: int | None = None,
        sliding_window_stride: int | None = None,
        hu_clip: Sequence[float] = (-1000.0, 1000.0),
        output_range: Sequence[float] = (-1.0, 1.0),
        spatial_flip_probability: float = 0.5,
        seed: int = 0,
        num_samples: int | None = None,
        case_ids: Sequence[str] | None = None,
    ) -> None:
        from medicalmodel_data.geometry import normalize_hu

        del normalize_hu  # import check gives a direct dependency error at construction
        self.root = Path(root)
        self.manifest = _resolve(self.root, str(manifest))
        self.split = split
        self.training = split == "train"
        self.patch_shape = tuple(int(x) for x in patch_shape_zyx)
        if len(self.patch_shape) != 3 or any(x <= 0 for x in self.patch_shape):
            raise ValueError("patch_shape_zyx must contain three positive integers")
        self.resize_xy = (tuple(int(value) for value in resize_xy)
                          if resize_xy is not None else None)
        self.sliding_window_depth = (int(sliding_window_depth)
                                     if sliding_window_depth is not None else None)
        self.sliding_window_stride = (int(sliding_window_stride)
                                      if sliding_window_stride is not None else None)
        if self.resize_xy is not None:
            if len(self.resize_xy) != 2 or any(value <= 0 for value in self.resize_xy):
                raise ValueError("resize_xy must contain two positive integers")
            if self.sliding_window_depth is None or self.sliding_window_stride is None:
                raise ValueError("resize_xy training requires sliding-window depth and stride")
            expected = (self.sliding_window_depth, *self.resize_xy)
            if self.patch_shape != expected:
                raise ValueError(
                    f"patch_size_dhw must equal sliding window plus resized XY: {expected}"
                )
            if self.sliding_window_depth <= 0 or self.sliding_window_stride <= 0:
                raise ValueError("sliding-window depth and stride must be positive")
        self.segmentation_target = segmentation_target
        if segmentation_target not in {"sdf", "mask"}:
            raise ValueError("segmentation_target must be 'sdf' or 'mask'")
        self.foreground_probability = float(foreground_probability)
        self.segmentation_foreground_probability = float(segmentation_foreground_probability)
        self.segmentation_surface_probability = float(segmentation_surface_probability)
        self.segmentation_foreground_warmup_probability = (
            self.segmentation_foreground_probability
            if segmentation_foreground_warmup_probability is None
            else float(segmentation_foreground_warmup_probability)
        )
        self.segmentation_surface_warmup_probability = (
            self.segmentation_surface_probability
            if segmentation_surface_warmup_probability is None
            else float(segmentation_surface_warmup_probability)
        )
        for name, foreground_value, surface_value in (
            ("mixed", self.segmentation_foreground_probability, self.segmentation_surface_probability),
            ("warmup", self.segmentation_foreground_warmup_probability,
             self.segmentation_surface_warmup_probability),
        ):
            if (foreground_value < 0.0 or surface_value < 0.0
                    or not np.isclose(foreground_value + surface_value, 1.0)):
                raise ValueError(
                    f"segmentation {name} foreground/surface probabilities must be non-negative "
                    "and sum to exactly 1; random segmentation crops are disabled"
                )
        self.segmentation_foreground_warmup_fraction = float(segmentation_foreground_warmup_fraction)
        if not 0.0 <= self.segmentation_foreground_warmup_fraction <= 1.0:
            raise ValueError("segmentation_foreground_warmup_fraction must be in [0, 1]")
        self.segmentation_center_jitter = tuple(int(x) for x in segmentation_center_jitter_zyx)
        if len(self.segmentation_center_jitter) != 3 or any(x < 0 for x in self.segmentation_center_jitter):
            raise ValueError("segmentation_center_jitter_zyx must contain three non-negative integers")
        self.segmentation_surface_center_jitter = tuple(
            int(x) for x in segmentation_surface_center_jitter_zyx
        )
        if (len(self.segmentation_surface_center_jitter) != 3
                or any(x < 0 for x in self.segmentation_surface_center_jitter)):
            raise ValueError(
                "segmentation_surface_center_jitter_zyx must contain three non-negative integers"
            )
        self.segmentation_surface_band_mm = float(segmentation_surface_band_mm)
        self.segmentation_sdf_clip_mm = float(segmentation_sdf_clip_mm)
        if self.segmentation_surface_band_mm <= 0 or self.segmentation_sdf_clip_mm <= 0:
            raise ValueError("segmentation SDF surface band and clip distance must be positive")
        if self.segmentation_surface_band_mm >= self.segmentation_sdf_clip_mm:
            raise ValueError("segmentation surface band must be smaller than the SDF clip distance")
        self.segmentation_min_foreground_voxels = int(segmentation_min_foreground_voxels)
        if self.segmentation_min_foreground_voxels < 1:
            raise ValueError("segmentation_min_foreground_voxels must be positive")
        self.segmentation_organ_sampling = str(segmentation_organ_sampling)
        if self.segmentation_organ_sampling not in {"case_uniform", "balanced_round_robin"}:
            raise ValueError(
                "segmentation_organ_sampling must be 'case_uniform' or 'balanced_round_robin'"
            )
        self.segmentation_case_sampling = str(segmentation_case_sampling)
        if self.segmentation_case_sampling != "shuffled_cycle":
            raise ValueError("segmentation_case_sampling must be 'shuffled_cycle'")
        zoom = dict(segmentation_zoom or {})
        self.segmentation_zoom_probability = float(zoom.get("probability", 0.0))
        if not 0.0 <= self.segmentation_zoom_probability <= 1.0:
            raise ValueError("segmentation zoom probability must be in [0, 1]")
        if self.resize_xy is not None and self.segmentation_zoom_probability:
            raise ValueError("segmentation zoom cannot be combined with full-XY sliding windows")
        self.segmentation_zoom_scales: dict[str, tuple[float, float, float]] = {}
        for structure, values in zoom.get("scale_zyx", {}).items():
            scales = tuple(float(value) for value in values)
            if len(scales) != 3 or any(value < 1.0 for value in scales):
                raise ValueError(
                    f"segmentation zoom scale for {structure} must contain three values >= 1"
                )
            self.segmentation_zoom_scales[str(structure)] = scales
        self.hu_clip = tuple(float(x) for x in hu_clip)
        self.output_range = tuple(float(x) for x in output_range)
        self.flip_probability = float(spatial_flip_probability) if self.training else 0.0
        self.seed, self.epoch = int(seed), 0
        self.records = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if case_ids is not None:
            allowed = {str(value) for value in case_ids}
            self.records = [record for record in self.records if str(record["case_id"]) in allowed]
        self.num_samples = int(num_samples) if num_samples is not None else len(self.records)
        if not self.records or self.num_samples <= 0:
            raise ValueError("DynamicCaseDataset requires cases and a positive sample count")
        if any(record.get("split") != split for record in self.records):
            raise ValueError(f"Manifest contains records outside {split}")
        weights = task_weights or {task: 1.0 for task in CT_TASKS}
        self.tasks = [task for task in CT_TASKS if float(weights.get(task, 0)) > 0]
        probabilities = np.asarray([weights[task] for task in self.tasks], dtype=np.float64)
        self.task_probabilities = probabilities / probabilities.sum()
        self.evaluation_task = evaluation_task
        if not self.training and evaluation_task not in CT_TASKS:
            raise ValueError("Validation/test DynamicCaseDataset requires evaluation_task")
        if (self.training and self.segmentation_organ_sampling == "balanced_round_robin"
                and self.tasks != ["segmentation"]):
            raise ValueError(
                "balanced_round_robin requires a segmentation-only DynamicCaseDataset"
            )
        self.segmentation_case_pools: dict[int, tuple[int, ...]] = {}
        self.segmentation_classes: tuple[int, ...] = ()
        self._organ_order_cache: dict[tuple[int, int], np.ndarray] = {}
        self._case_order_cache: dict[tuple[int, int, int], np.ndarray] = {}
        if self.training and self.segmentation_organ_sampling == "balanced_round_robin":
            self._build_segmentation_case_pools()

    def _record_segmentation_class_voxels(self, record: dict[str, Any]) -> dict[int, int]:
        canonical = record.get("metadata", {}).get("canonical", {})
        volumes = canonical.get("organ_volume_mm3_canonical")
        spacing = record.get("spacing_xyz_mm", canonical.get("canonical_spacing_xyz_mm"))
        if volumes and spacing and len(spacing) == 3:
            voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
            if voxel_volume > 0:
                return {
                    int(class_id): int(round(float(volume_mm3) / voxel_volume))
                    for class_id, volume_mm3 in volumes.items()
                }
        label = _load_array(self.root, record["mask"]).astype(np.uint8, copy=False)
        return {
            int(class_id): int(np.count_nonzero(label == int(class_id)))
            for class_id in record["sdf_classes"]
        }

    def _build_segmentation_case_pools(self) -> None:
        pools: dict[int, list[int]] = defaultdict(list)
        names: dict[int, set[str]] = defaultdict(set)
        for record_index, record in enumerate(self.records):
            class_voxels = self._record_segmentation_class_voxels(record)
            label_map = record.get("label_map", {})
            for value in (int(class_id) for class_id in record["sdf_classes"]):
                names[value].add(str(label_map.get(str(value), value)))
                if class_voxels.get(value, 0) >= self.segmentation_min_foreground_voxels:
                    pools[value].append(record_index)
        inconsistent = {value: sorted(values) for value, values in names.items() if len(values) != 1}
        if inconsistent:
            raise ValueError(f"Inconsistent segmentation label names across cases: {inconsistent}")
        empty = sorted(value for value in names if not pools[value])
        if empty:
            raise ValueError(
                "No eligible training cases for segmentation classes "
                f"{empty} at minimum {self.segmentation_min_foreground_voxels} voxels"
            )
        self.segmentation_case_pools = {
            value: tuple(indices) for value, indices in sorted(pools.items())
        }
        self.segmentation_classes = tuple(self.segmentation_case_pools)

    def _balanced_segmentation_pair(
        self, index: int,
    ) -> tuple[int, int, dict[str, int | str]]:
        if not self.segmentation_classes:
            raise RuntimeError("Balanced segmentation pools were not initialized")
        organ_cycle, organ_slot = divmod(int(index), len(self.segmentation_classes))
        organ_key = (self.epoch, organ_cycle)
        organ_order = self._organ_order_cache.get(organ_key)
        if organ_order is None:
            organ_rng = np.random.default_rng(stable_seed(
                self.seed, self.epoch, "segmentation-organ-order", organ_cycle
            ))
            organ_order = organ_rng.permutation(self.segmentation_classes)
            self._organ_order_cache[organ_key] = organ_order
        class_id = int(organ_order[organ_slot])
        pool = self.segmentation_case_pools[class_id]
        pool_cycle, pool_position = divmod(organ_cycle, len(pool))
        case_key = (self.epoch, class_id, pool_cycle)
        case_order = self._case_order_cache.get(case_key)
        if case_order is None:
            case_rng = np.random.default_rng(stable_seed(
                self.seed, self.epoch, "segmentation-case-order", class_id, pool_cycle
            ))
            case_order = case_rng.permutation(len(pool))
            self._case_order_cache[case_key] = case_order
        record_index = pool[int(case_order[pool_position])]
        return record_index, class_id, {
            "organ_sampling_strategy": self.segmentation_organ_sampling,
            "case_sampling_strategy": self.segmentation_case_sampling,
            "organ_cycle": organ_cycle,
            "organ_slot": organ_slot,
            "organ_pool_size": len(pool),
            "organ_pool_cycle": pool_cycle,
            "organ_pool_position": pool_position,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._organ_order_cache.clear()
        self._case_order_cache.clear()

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, index: int) -> np.random.Generator:
        payload = f"{self.seed}\0{self.epoch}\0{index}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return np.random.default_rng(value)

    @staticmethod
    def _with_channel(array: np.ndarray) -> np.ndarray:
        return array[None] if array.ndim == 3 else array

    def _patch_start(self, label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
        if self.training:
            foreground = np.argwhere(label > 0)
            if len(foreground) and rng.random() < self.foreground_probability:
                center = foreground[int(rng.integers(len(foreground)))]
            else:
                center = np.asarray([rng.integers(max(1, size)) for size in label.shape])
        else:
            foreground = np.argwhere(label > 0)
            center = (np.rint((foreground.min(0) + foreground.max(0)) / 2).astype(int)
                      if len(foreground) else np.asarray(label.shape) // 2)
        return tuple(int(c - size // 2) for c, size in zip(center, self.patch_shape))

    def _segmentation_patch_start(
        self, target_mask: np.ndarray, target_sdf: np.ndarray, rng: np.random.Generator,
        foreground_probability: float | None = None,
        surface_probability: float | None = None,
        patch_shape_zyx: Sequence[int] | None = None,
    ) -> tuple[tuple[int, int, int], int, str]:
        """Sample only the foreground or surface of the organ named by the prompt."""
        from medicalmodel_data.geometry import crop_or_pad

        patch_shape = tuple(int(value) for value in (patch_shape_zyx or self.patch_shape))
        if len(patch_shape) != 3 or any(value <= 0 for value in patch_shape):
            raise ValueError("segmentation patch shape must contain three positive integers")
        foreground = np.argwhere(target_mask)
        if not len(foreground):
            raise ValueError("Requested segmentation class is absent from the case")
        if not self.training:
            center = np.rint((foreground.min(0) + foreground.max(0)) / 2).astype(int)
            sampling_mode = "evaluation_foreground_center"
        else:
            foreground_probability = (
                self.segmentation_foreground_probability
                if foreground_probability is None else foreground_probability
            )
            surface_probability = (
                self.segmentation_surface_probability
                if surface_probability is None else surface_probability
            )
            draw = float(rng.random())
            if draw < foreground_probability:
                centers = foreground
                jitter_radius = self.segmentation_center_jitter
                sampling_mode = "target_foreground_centered"
            else:
                # On-disk SDF is normalized by its clip distance. Absolute
                # value makes this independent of the positive/negative-inside
                # convention while still using only the prompted class channel.
                surface_threshold = self.segmentation_surface_band_mm / self.segmentation_sdf_clip_mm
                centers = np.argwhere(np.abs(target_sdf) < surface_threshold)
                if not len(centers):
                    raise RuntimeError(
                        "Prompt-target SDF contains no surface voxels within "
                        f"{self.segmentation_surface_band_mm:g} mm"
                    )
                jitter_radius = self.segmentation_surface_center_jitter
                sampling_mode = "target_surface_centered"
        if sampling_mode in {"target_foreground_centered", "target_surface_centered"}:
            required = self.segmentation_min_foreground_voxels
            best: tuple[int, tuple[int, int, int]] | None = None
            # Retrying only enforces the explicit target-organ minimum after
            # mode-specific jitter; it never falls back to another class.
            for _ in range(16):
                center = centers[int(rng.integers(len(centers)))].copy()
                jitter = np.asarray([
                    rng.integers(-radius, radius + 1) if radius else 0
                    for radius in jitter_radius
                ])
                center += jitter
                candidate = tuple(int(c - size // 2) for c, size in zip(center, patch_shape))
                count = int(crop_or_pad(target_mask, candidate, patch_shape, fill_value=0).sum())
                if best is None or count > best[0]:
                    best = (count, candidate)
                if count >= required:
                    return candidate, count, sampling_mode
            assert best is not None
            start = best[1]
            foreground_voxels = best[0]
            if foreground_voxels < required:
                raise RuntimeError(
                    f"Could not construct a foreground-centred crop with at least {required} "
                    f"target voxels (best={foreground_voxels})"
                )
            return start, foreground_voxels, sampling_mode
        start = tuple(int(c - size // 2) for c, size in zip(center, patch_shape))
        foreground_voxels = int(crop_or_pad(target_mask, start, patch_shape, fill_value=0).sum())
        required = min(self.segmentation_min_foreground_voxels, int(target_mask.sum()))
        if foreground_voxels < required:
            raise RuntimeError(
                f"Foreground-centred segmentation crop contains {foreground_voxels} target voxels; "
                f"expected at least {required}"
            )
        return start, foreground_voxels, sampling_mode

    def _segmentation_zoom_parameters(
        self, structure: str, index: int,
    ) -> tuple[bool, tuple[float, float, float], tuple[int, int, int]]:
        scales = self.segmentation_zoom_scales.get(structure, (1.0, 1.0, 1.0))
        if (not self.training or scales == (1.0, 1.0, 1.0)
                or self.segmentation_zoom_probability <= 0.0):
            return False, scales, self.patch_shape
        zoom_rng = np.random.default_rng(stable_seed(
            self.seed, self.epoch, index, "segmentation-zoom"
        ))
        applied = bool(zoom_rng.random() < self.segmentation_zoom_probability)
        source_shape = (
            tuple(max(1, int(round(size / scale))) for size, scale in zip(self.patch_shape, scales))
            if applied else self.patch_shape
        )
        return applied, scales, source_shape

    @staticmethod
    def _resize_patch(
        array: np.ndarray, shape_zyx: tuple[int, int, int], mode: str,
    ) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))[None, None]
        kwargs = {"align_corners": False} if mode == "trilinear" else {}
        resized = torch.nn.functional.interpolate(
            tensor, size=shape_zyx, mode=mode, **kwargs
        )
        return resized[0, 0].numpy()

    @staticmethod
    def _reconstruction_views(record: dict[str, Any], configuration: str) -> int:
        candidates = [
            record.get("reconstruction_views"),
            record.get("num_views"),
            record.get("views"),
            record.get("metadata", {}).get("reconstruction_views"),
            record.get("metadata", {}).get("num_views"),
            record.get("metadata", {}).get("reconstruction_multiview_v2", {}).get("views"),
            record.get("degradation", {}).get("views"),
            record.get("degradation", {}).get("num_views"),
        ]
        for value in candidates:
            if value is not None:
                return int(value)
        for pattern in (r"(?:views?|view)[_-]?(\d+)", r"(\d+)[_-]?(?:views?|view)"):
            match = re.search(pattern, configuration, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        raise ValueError(f"Cannot infer reconstruction view count from {configuration}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        from medicalmodel_data.geometry import crop_or_pad, normalize_hu

        rng = self._rng(index)
        task = (str(rng.choice(self.tasks, p=self.task_probabilities))
                if self.training else str(self.evaluation_task))
        balanced_class_id = None
        balance_metadata: dict[str, int | str] = {}
        if (task == "segmentation" and self.training
                and self.segmentation_organ_sampling == "balanced_round_robin"):
            record_index, balanced_class_id, balance_metadata = self._balanced_segmentation_pair(index)
        else:
            record_index = index % len(self.records)
        record = self.records[record_index]
        clean = _load_array(self.root, record["image"])
        label = _load_array(self.root, record["mask"]).astype(np.uint8, copy=False)
        structure = None
        configuration = None
        class_id = None
        target_foreground_voxels = None
        sampling_mode = None
        reconstruction_views = None
        in_warmup = False
        active_segmentation_foreground_probability = None
        active_segmentation_surface_probability = None
        zoom_applied = False
        zoom_scale_zyx = (1.0, 1.0, 1.0)
        source_patch_shape = self.patch_shape
        recomputed_zoom_sdf = False
        if task == "segmentation":
            condition_volume = clean
            classes = [int(x) for x in record["sdf_classes"]]
            class_voxels = {value: int(np.count_nonzero(label == value)) for value in classes}
            present = [value for value in classes if class_voxels[value] > 0]
            if self.training:
                if balanced_class_id is not None:
                    class_id = int(balanced_class_id)
                    if class_voxels.get(class_id, 0) < self.segmentation_min_foreground_voxels:
                        raise RuntimeError(
                            f"Balanced pair {record['case_id']}/class-{class_id} no longer satisfies "
                            f"the {self.segmentation_min_foreground_voxels}-voxel eligibility threshold"
                        )
                else:
                    eligible = [
                        value for value in present
                        if class_voxels[value] >= self.segmentation_min_foreground_voxels
                    ]
                    if not eligible:
                        # Keep this patient available to restoration/reconstruction,
                        # but never turn an invalid tiny-label case into a broken
                        # segmentation sample. Deterministically advance to the
                        # next patient that has a usable prompted organ.
                        for offset in range(1, len(self.records)):
                            candidate = self.records[(index + offset) % len(self.records)]
                            candidate_label = _load_array(self.root, candidate["mask"]).astype(np.uint8, copy=False)
                            candidate_classes = [int(x) for x in candidate["sdf_classes"]]
                            candidate_voxels = {
                                value: int(np.count_nonzero(candidate_label == value))
                                for value in candidate_classes
                            }
                            candidate_eligible = [
                                value for value in candidate_classes
                                if candidate_voxels[value] >= self.segmentation_min_foreground_voxels
                            ]
                            if candidate_eligible:
                                record = candidate
                                label = candidate_label
                                classes = candidate_classes
                                class_voxels = candidate_voxels
                                present = [value for value in classes if class_voxels[value] > 0]
                                eligible = candidate_eligible
                                clean = _load_array(self.root, record["image"])
                                condition_volume = clean
                                break
                        if not eligible:
                            raise ValueError(
                                "No training case contains a segmentation class with at least "
                                f"{self.segmentation_min_foreground_voxels} voxels"
                            )
                    class_id = int(eligible[int(rng.integers(len(eligible)))])
            else:
                # Cover prompted organs deterministically across a held-out
                # evaluation set.  The previous `classes[0]` rule silently
                # evaluated only the aorta for every patient.
                if not present:
                    raise ValueError(f"No evaluation class is present in {record['case_id']}")
                class_id = present[index % len(present)]
            target_mask = label == class_id
            sdf = _load_array(self.root, record["sdf"])
            channel = classes.index(class_id)
            target_sdf = sdf[channel]
            structure = str(record.get("label_map", {}).get(str(class_id), class_id))
            zoom_applied, zoom_scale_zyx, source_patch_shape = (
                self._segmentation_zoom_parameters(structure, index)
            )
            stream_fraction = min(1.0, max(0.0, index / max(1, self.num_samples)))
            in_warmup = (
                self.training
                and stream_fraction < self.segmentation_foreground_warmup_fraction
            )
            active_segmentation_foreground_probability = (
                self.segmentation_foreground_warmup_probability
                if in_warmup else self.segmentation_foreground_probability
            )
            active_segmentation_surface_probability = (
                self.segmentation_surface_warmup_probability
                if in_warmup else self.segmentation_surface_probability
            )
            if self.resize_xy is not None:
                all_starts = _sliding_window_starts(
                    target_mask.shape[0], self.sliding_window_depth,
                    self.sliding_window_stride,
                )
                eligible_starts = tuple(
                    value for value in all_starts
                    if int(target_mask[value:value + self.sliding_window_depth].sum())
                    >= self.segmentation_min_foreground_voxels
                )
                if not eligible_starts:
                    raise RuntimeError(
                        f"No z-window contains enough foreground for {record['case_id']} "
                        f"class {class_id}"
                    )
                window_cycle = index // max(1, len(self.records))
                start = (eligible_starts[window_cycle % len(eligible_starts)], 0, 0)
                target_foreground_voxels = int(
                    target_mask[start[0]:start[0] + self.sliding_window_depth].sum()
                )
                sampling_mode = "z_sliding_window_foreground"
            else:
                start, target_foreground_voxels, sampling_mode = self._segmentation_patch_start(
                    target_mask, target_sdf, rng,
                    active_segmentation_foreground_probability,
                    active_segmentation_surface_probability,
                    source_patch_shape,
                )
            if self.segmentation_target == "mask":
                target_volume = target_mask.astype(np.uint8)
            else:
                target_volume = target_sdf
        elif task == "restoration":
            if self.resize_xy is not None:
                starts = _sliding_window_starts(
                    clean.shape[0], self.sliding_window_depth, self.sliding_window_stride
                )
                start = (starts[(index // max(1, len(self.records))) % len(starts)]
                         if self.training else starts[len(starts) // 2], 0, 0)
            else:
                start = self._patch_start(label, rng)
            variants = record["ldct"]
            configuration = str(variants[int(rng.integers(len(variants)))]) if self.training else str(variants[0])
            condition_volume, target_volume = _load_array(self.root, configuration), clean
        else:
            if self.resize_xy is not None:
                starts = _sliding_window_starts(
                    clean.shape[0], self.sliding_window_depth, self.sliding_window_stride
                )
                start = (starts[(index // max(1, len(self.records))) % len(starts)]
                         if self.training else starts[len(starts) // 2], 0, 0)
            else:
                start = self._patch_start(label, rng)
            variants = record["sparse_view_ct"]
            configuration = str(variants[int(rng.integers(len(variants)))]) if self.training else str(variants[0])
            reconstruction_views = self._reconstruction_views(record, configuration)
            condition_volume, target_volume = _load_array(self.root, configuration), clean

        if self.resize_xy is not None:
            z_start = start[0]
            condition = _crop_or_pad_depth(
                condition_volume, z_start, self.sliding_window_depth, -1000
            )
            condition = _resize_xy(condition, self.resize_xy, "bilinear")
            target_fill = (0 if self.segmentation_target == "mask" else
                           (-1 if task == "segmentation" else -1000))
            target = _crop_or_pad_depth(
                target_volume, z_start, self.sliding_window_depth, target_fill
            )
            target = _resize_xy(
                target, self.resize_xy,
                "nearest" if task == "segmentation" and self.segmentation_target == "mask"
                else "bilinear",
            )
            valid = _crop_or_pad_depth(
                np.ones(label.shape, np.float32), z_start, self.sliding_window_depth, 0
            )
            valid = _resize_xy(valid, self.resize_xy, "nearest")
            actual_foreground = (int((_resize_xy(
                _crop_or_pad_depth(
                    (label == class_id).astype(np.float32), z_start,
                    self.sliding_window_depth, 0,
                ), self.resize_xy, "nearest"
            ) >= 0.5).sum()) if task == "segmentation" else 0)
        elif task == "segmentation" and zoom_applied:
            from medicalmodel_data.geometry import label_to_sdf

            condition = crop_or_pad(
                condition_volume, start, source_patch_shape, fill_value=-1000
            )
            zoom_mask = crop_or_pad(
                target_mask, start, source_patch_shape, fill_value=0
            ).astype(np.float32)
            valid = crop_or_pad(
                np.ones(label.shape, np.float32), start, source_patch_shape, fill_value=0
            )
            condition = self._resize_patch(condition, self.patch_shape, "trilinear")
            zoom_mask = self._resize_patch(zoom_mask, self.patch_shape, "nearest") >= 0.5
            valid = self._resize_patch(valid, self.patch_shape, "nearest")
            if self.segmentation_target == "mask":
                target = zoom_mask.astype(np.float32)
            else:
                canonical = record.get("metadata", {}).get("canonical", {})
                spacing_xyz = record.get(
                    "spacing_xyz_mm", canonical.get("canonical_spacing_xyz_mm")
                )
                if not spacing_xyz or len(spacing_xyz) != 3:
                    raise ValueError(
                        f"Missing spacing_xyz_mm for zoomed segmentation case {record['case_id']}"
                    )
                spacing_zyx = tuple(float(value) for value in reversed(spacing_xyz))
                target = label_to_sdf(
                    zoom_mask.astype(np.uint8), spacing_zyx, [1],
                    self.segmentation_sdf_clip_mm, positive_inside=False,
                )[0]
                recomputed_zoom_sdf = True
            actual_foreground = int(zoom_mask.sum())
        else:
            condition = crop_or_pad(condition_volume, start, self.patch_shape, fill_value=-1000)
            target_fill = (0 if self.segmentation_target == "mask" else
                           (-1 if task == "segmentation" else -1000))
            target = crop_or_pad(target_volume, start, self.patch_shape, fill_value=target_fill)
            valid = crop_or_pad(
                np.ones(label.shape, np.float32), start, self.patch_shape, fill_value=0
            )
            actual_foreground = (int(crop_or_pad(
                label == class_id, start, self.patch_shape, fill_value=0
            ).sum()) if task == "segmentation" else 0)
        if task == "segmentation":
            if actual_foreground <= 0:
                raise RuntimeError(
                    f"Segmentation patch invariant failed for {record['case_id']} class {class_id}"
                )
            target_foreground_voxels = actual_foreground
        if (task == "segmentation" and self.segmentation_target == "sdf"
                and record.get("sdf_positive_inside", True) and not recomputed_zoom_sdf):
            # On-disk preprocessing is currently positive-inside; the model
            # contract is negative-inside with +1 exterior.
            target = -target
        condition = normalize_hu(condition, self.hu_clip, self.output_range)
        if task != "segmentation":
            target = normalize_hu(target, self.hu_clip, self.output_range)
        elif self.segmentation_target == "mask":
            target = target.astype(np.float32)
        else:
            target = target.astype(np.float32)
        condition, target, valid = map(self._with_channel, (condition, target, valid))
        if self.training and self.flip_probability:
            for axis in (-1, -2, -3):
                if rng.random() < self.flip_probability:
                    condition, target, valid = (np.flip(x, axis=axis).copy() for x in (condition, target, valid))
        prompt = (DEFAULT_PROMPTS[task].format(structure=structure)
                  if task == "segmentation" and structure else DEFAULT_PROMPTS[task])
        return {
            "case_id": str(record["case_id"]), "task": task,
            "condition": torch.from_numpy(np.asarray(condition, np.float32)),
            "target": torch.from_numpy(np.asarray(target, np.float32)),
            "prompt": prompt, "valid_mask": torch.from_numpy(np.asarray(valid, np.float32)),
            "metadata": {**record.get("metadata", {}), "split": self.split,
                         **balance_metadata,
                         "crop_start_zyx": start, "configuration": configuration,
                         "resize_xy": self.resize_xy,
                         "sliding_window_depth": self.sliding_window_depth,
                         "sliding_window_stride": self.sliding_window_stride,
                         "structure": structure, "class_id": class_id,
                         "organ_id": class_id, "sampling_mode": sampling_mode,
                         "segmentation_sampling_phase": (
                             "foreground_warmup" if task == "segmentation" and in_warmup
                             else "mixed" if task == "segmentation" and self.training
                             else "evaluation"
                         ),
                         "segmentation_foreground_probability": (
                             active_segmentation_foreground_probability
                         ),
                         "segmentation_surface_probability": (
                             active_segmentation_surface_probability
                         ),
                         "segmentation_random_probability": (
                             None if active_segmentation_foreground_probability is None else 0.0
                         ),
                         "segmentation_surface_band_mm": self.segmentation_surface_band_mm,
                         "segmentation_zoom_applied": zoom_applied,
                         "segmentation_zoom_probability": self.segmentation_zoom_probability,
                         "segmentation_zoom_scale_zyx": zoom_scale_zyx,
                         "segmentation_source_patch_shape_zyx": source_patch_shape,
                         "segmentation_zoom_sdf_recomputed": recomputed_zoom_sdf,
                         "target_present": (target_foreground_voxels or 0) > 0,
                         "foreground_voxels": target_foreground_voxels,
                         "target_foreground_voxels": target_foreground_voxels,
                         "reconstruction_views": reconstruction_views},
        }


def unified_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    shapes = {tuple(sample["condition"].shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"Batch contains incompatible shapes: {shapes}")
    return {
        "case_id": [x["case_id"] for x in samples],
        "task": [x["task"] for x in samples],
        "condition": torch.stack([x["condition"] for x in samples]),
        "target": torch.stack([x["target"] for x in samples]),
        "prompt": [x["prompt"] for x in samples],
        "valid_mask": torch.stack([x["valid_mask"] for x in samples]),
        "metadata": [x["metadata"] for x in samples],
    }


class TaskRatioSampler(Sampler[int]):
    """Replacement sampler with an exact categorical task distribution."""

    def __init__(self, records: Sequence[dict[str, Any]], ratios: dict[str, float], num_samples: int, seed: int = 0) -> None:
        self.by_task: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.by_task[str(record["task"])].append(index)
        self.tasks = [task for task, ratio in ratios.items() if ratio > 0]
        missing = [task for task in self.tasks if not self.by_task[task]]
        if missing:
            raise ValueError(f"No samples for requested tasks: {missing}")
        weights = torch.tensor([ratios[t] for t in self.tasks], dtype=torch.double)
        self.probabilities = weights / weights.sum()
        self.num_samples, self.seed, self.epoch = num_samples, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        task_ids = torch.multinomial(self.probabilities, self.num_samples, replacement=True, generator=generator)
        for task_id in task_ids.tolist():
            pool = self.by_task[self.tasks[task_id]]
            yield pool[torch.randint(len(pool), (), generator=generator).item()]

    def __len__(self) -> int:
        return self.num_samples


def audit_disjoint_splits(root: str | Path, manifests: dict[str, str], splits: dict[str, str]) -> None:
    root=Path(root); case_sets={}
    for name in ("train","val","test"):
        frozen={x.strip() for x in _resolve(root,splits[name]).read_text().splitlines() if x.strip()}
        records=[json.loads(x) for x in _resolve(root,manifests[name]).read_text().splitlines() if x.strip()]
        observed={str(x["case_id"]) for x in records}
        if not observed <= frozen: raise ValueError(f"{name} manifest contains cases outside frozen split")
        case_sets[name]=observed
    for left,right in (("train","val"),("train","test"),("val","test")):
        overlap=case_sets[left]&case_sets[right]
        if overlap: raise ValueError(f"Case leakage between {left}/{right}: {sorted(overlap)[:5]}")
