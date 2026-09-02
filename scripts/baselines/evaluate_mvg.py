#!/usr/bin/env python3
"""Evaluate a trained 2-D MVG checkpoint on frozen MedGen3D ID/OOD cohorts.

The query target is completely masked in MVG's label stream; only one fixed
labelled support slice from the ID training set remains visible.  Predictions
are reconstructed slice by slice into the original 3-D grid and then scored by
the same metric functions used by ``evaluate_medgen3d.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from medgen3d.evaluation import (  # noqa: E402  # pragma: no cover
    paired_ct_metrics, segmentation_metrics, summarize_paired_ct,
    summarize_segmentation_by_class, synthesis_metrics,
)

IMAGE_SIZE = 448
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
LABELS = {1: "aorta", 2: "gall_bladder", 3: "kidney_left", 4: "kidney_right", 5: "liver", 6: "pancreas", 7: "postcava", 8: "spleen", 9: "stomach"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)).transpose(2, 1, 0)


def encode(array: np.ndarray, mode: str) -> torch.Tensor:
    if mode == "ct":
        array = np.clip((array.astype(np.float32) + 1000.0) / 2000.0, 0, 1)
    elif mode == "mri":
        array = np.clip((array.astype(np.float32) + 1.0) / 2.0, 0, 1)
    elif mode == "seg":
        array = (array.astype(bool).astype(np.float32) * 10.0)
        x = torch.from_numpy(array)[None].repeat(3, 1, 1)
        x = F.interpolate(x[None], (IMAGE_SIZE, IMAGE_SIZE), mode="nearest")[0]
        return (x - MEAN) / STD
    else:
        raise ValueError(mode)
    x = torch.from_numpy(array)[None].repeat(3, 1, 1)
    x = F.interpolate(x[None], (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0]
    return (x - MEAN) / STD


def decode(tensor: torch.Tensor, mode: str, shape: tuple[int, int]) -> np.ndarray:
    x = (tensor.detach().float().cpu() * STD + MEAN)[0:1]
    x = F.interpolate(x[None], shape, mode="nearest" if mode == "seg" else "bilinear", align_corners=False if mode != "seg" else None)[0, 0].numpy()
    if mode == "ct":
        return np.clip(x, 0, 1) * 2000.0 - 1000.0
    if mode == "mri":
        # ``encode(..., "mri")`` maps the manifest's [-1, 1] MRI range to
        # [0, 1] before ImageNet normalization.  Return to that original
        # manifest range here; the synthesis metric below then converts both
        # prediction and target to [0, 1] exactly once.
        return np.clip(x, 0, 1) * 2.0 - 1.0
    return x


class Predictor:
    def __init__(self, args: argparse.Namespace) -> None:
        sys.path.insert(0, str(args.mvg_root))
        import models_mvg  # type: ignore
        self.device = torch.device(args.device)
        self.model = models_mvg.mvg_vit_large_patch16().to(self.device)
        state = torch.load(args.checkpoint, map_location="cpu")["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval(); self.batch_size = args.batch_size
        self.mask = torch.zeros((IMAGE_SIZE // 16 * 2, IMAGE_SIZE // 16), dtype=torch.bool, device=self.device)
        self.mask[IMAGE_SIZE // 16:] = True

    @torch.inference_mode()
    def predict(self, support_x: torch.Tensor, support_y: torch.Tensor, query: np.ndarray, source_mode: str, target_mode: str) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(query), self.batch_size):
            slices = query[start:start + self.batch_size]
            qs = torch.stack([encode(item, source_mode) for item in slices])
            # Ground truth only fills the lower target tensor.  Every lower-half
            # token is replaced by MVG's mask token in forward_encoder.
            qy = torch.stack([encode(np.zeros_like(item), target_mode) for item in slices])
            x = torch.cat((support_x[None].expand(len(qs), -1, -1, -1), qs), dim=2).to(self.device)
            y = torch.cat((support_y[None].expand(len(qs), -1, -1, -1), qy), dim=2).to(self.device)
            valid = torch.ones_like(y)
            _, patches, _ = self.model(x, y, bool_masked_pos=self.mask[None].expand(len(qs), -1, -1), valid=valid)
            pred = self.model.unpatchify(patches)[:, :, IMAGE_SIZE:, :]
            outputs.extend([decode(row, target_mode, tuple(item.shape)) for row, item in zip(pred, slices)])
        return np.stack(outputs)


def first_seg_support(rows: list[dict[str, Any]], root: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    for row in rows:
        source, mask = load(resolve(root, row["image"])), load(resolve(root, row["mask"]))
        z = np.flatnonzero(np.any(mask == label, axis=(1, 2)))
        if len(z): return source[int(z[len(z) // 2])], mask[int(z[len(z) // 2])] == label
    raise RuntimeError(f"No training support for label {label}")


def fixed_pair(row: dict[str, Any], root: Path, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "restoration":
        source, target = load(resolve(root, row["ldct"][0])), load(resolve(root, row["image"]))
    else:
        source, target = load(resolve(root, row["condition"])), load(resolve(root, row["target"]))
    z = min(len(source), len(target)) // 2
    return source[z], target[z]


def evaluate_segmentation(pred: Predictor, test: list[dict[str, Any]], train: list[dict[str, Any]], test_root: Path, train_root: Path,
                          tolerance: float) -> dict[str, Any]:
    supports = {label: first_seg_support(train, train_root, label) for label in LABELS}
    rows = []
    for index, row in enumerate(test):
        label = int(row.get("target_label_id", (index % len(LABELS)) + 1))
        source, target = load(resolve(test_root, row["image"])), load(resolve(test_root, row["mask"]))
        sx, sy = supports[label]
        raw = pred.predict(encode(sx, "ct"), encode(sy, "seg"), source, "ct", "seg")
        output = raw > 5.0
        spacing_xyz = row.get("spacing_xyz_mm", [1.5, 1.5, 1.5])
        metric = segmentation_metrics(output, target == label, tuple(reversed(tuple(map(float, spacing_xyz)))), tolerance)
        rows.append({"index": index, "case_id": row["case_id"], "structure": LABELS[label], "metrics": metric})
        print(json.dumps({"task": "segmentation", "index": index, "case_id": row["case_id"], "dice": metric["dice"]}), flush=True)
    return {"rows": rows, "summary": summarize_segmentation_by_class(rows)}


def evaluate_paired(pred: Predictor, test: list[dict[str, Any]], train: list[dict[str, Any]], test_root: Path, train_root: Path, task: str) -> dict[str, Any]:
    support_x, support_y = fixed_pair(train[0], train_root, task)
    mode = "ct" if task == "restoration" else "mri"
    rows = []
    for index, row in enumerate(test):
        if task == "restoration":
            source, target = load(resolve(test_root, row["ldct"][0])), load(resolve(test_root, row["image"]))
            output = pred.predict(encode(support_x, mode), encode(support_y, mode), source, mode, mode)
            metric = paired_ct_metrics(source, output, target)
        else:
            source, target = load(resolve(test_root, row["condition"])), load(resolve(test_root, row["target"]))
            output = pred.predict(encode(support_x, mode), encode(support_y, mode), source, mode, mode)
            metric = synthesis_metrics((output + 1.0) / 2.0, (target + 1.0) / 2.0)
        rows.append({"index": index, "case_id": row["case_id"], "metrics": metric})
        report = metric["model"]["mae_hu"] if task == "restoration" else metric["mae"]
        print(json.dumps({"task": task, "index": index, "case_id": row["case_id"], "mae": report}), flush=True)
    if task == "restoration":
        return {"rows": rows, "summary": summarize_paired_ct([row["metrics"] for row in rows])}
    return {"rows": rows, "summary": {key: float(np.mean([r["metrics"][key] for r in rows])) for key in ("mae", "psnr", "ssim")}}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mvg-root", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ct-root", type=Path, required=True, help="Root for the evaluated CT cohort."); p.add_argument("--ct-train-root", type=Path, help="Root for fixed ID support slices (default: --ct-root)")
    p.add_argument("--ct-test-manifest", type=Path, required=True, help="Fallback CT test manifest for both CT tasks.")
    p.add_argument("--seg-test-manifest", type=Path, help="Optional segmentation-specific CT test manifest.")
    p.add_argument("--restoration-test-manifest", type=Path, help="Optional restoration-specific CT test manifest.")
    p.add_argument("--ct-train-manifest", type=Path, required=True)
    p.add_argument("--synth-root", type=Path, required=True, help="Root for the evaluated synthesis cohort."); p.add_argument("--synth-train-root", type=Path, help="Root for fixed ID support slices (default: --synth-root)")
    p.add_argument("--synth-test-manifest", type=Path, required=True); p.add_argument("--synth-train-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--tasks", nargs="+", default=["segmentation", "restoration", "synthesis"])
    p.add_argument("--max-cases", type=int, default=0, help="Optional smoke-test cap; zero evaluates every frozen case.")
    p.add_argument("--seg-nsd-tolerance-mm", type=float, default=3.0); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    a = args(); model = Predictor(a)
    ct_test, ct_train = read_jsonl(a.ct_test_manifest), read_jsonl(a.ct_train_manifest)
    seg_test = read_jsonl(a.seg_test_manifest) if a.seg_test_manifest else ct_test
    restoration_test = read_jsonl(a.restoration_test_manifest) if a.restoration_test_manifest else ct_test
    synth_test, synth_train = read_jsonl(a.synth_test_manifest), read_jsonl(a.synth_train_manifest)
    ct_train_root, synth_train_root = a.ct_train_root or a.ct_root, a.synth_train_root or a.synth_root
    if a.max_cases:
        seg_test, restoration_test, synth_test = seg_test[:a.max_cases], restoration_test[:a.max_cases], synth_test[:a.max_cases]
    result: dict[str, Any] = {"checkpoint": str(a.checkpoint), "tasks": {}}
    if "segmentation" in a.tasks: result["tasks"]["segmentation"] = evaluate_segmentation(model, seg_test, ct_train, a.ct_root, ct_train_root, a.seg_nsd_tolerance_mm)
    if "restoration" in a.tasks: result["tasks"]["restoration"] = evaluate_paired(model, restoration_test, ct_train, a.ct_root, ct_train_root, "restoration")
    if "synthesis" in a.tasks: result["tasks"]["synthesis"] = evaluate_paired(model, synth_test, synth_train, a.synth_root, synth_train_root, "synthesis")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
