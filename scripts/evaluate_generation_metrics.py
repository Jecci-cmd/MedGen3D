#!/usr/bin/env python3
"""Evaluate report-to-CT volumes with FID, FVD, CT-CLIP T2I and CT-CLIP I2I.

The script operates only on the canonical full-volume NIfTI files emitted by
``evaluate_medgen3d.py`` under the CT-RATE V2 protocol.  FID uses frozen
ImageNet Inception-V3 features from five fixed axial slices per volume.  FVD
uses a user-supplied *frozen, TorchScript I3D* feature extractor.  CT-CLIP
scores are paired cosine similarities: generated-volume/report (T2I) and
generated-volume/reference-volume (I2I).  The I3D and CT-CLIP checkpoints are
explicit arguments so an experiment record states exactly which frozen models
were used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights, inception_v3


PROTOCOL = "ctrate_v2_inceptionv3_fid_i3d_fvd_ctclip_t2i_i2i_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="results.json produced by evaluate_medgen3d.py")
    parser.add_argument("--i3d-model", type=Path, required=True,
                        help="Frozen TorchScript I3D feature extractor; input [B,3,T,H,W]")
    parser.add_argument("--ctclip-repo", type=Path, required=True)
    parser.add_argument("--ctclip-backbone", type=Path, required=True)
    parser.add_argument("--ctclip-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--features-output", type=Path,
                        help="Write one feature shard; merge with merge_generation_metrics.py")
    return parser.parse_args()


def load_zyx(path: Path) -> np.ndarray:
    xyz = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {xyz.shape}: {path}")
    return np.transpose(np.nan_to_num(xyz, nan=-1000.0), (2, 1, 0))


def resize_volume(array: np.ndarray, shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    return F.interpolate(torch.from_numpy(array)[None, None].to(device), size=shape,
                         mode="trilinear", align_corners=False)


def ct_window(array: np.ndarray, device: torch.device, shape: tuple[int, int, int]) -> torch.Tensor:
    """Map Z/Y/X HU data to three channels in [-1, 1] on a fixed grid."""
    volume = resize_volume(array, shape, device).clamp(-1000.0, 1000.0).div(1000.0)
    return volume.repeat(1, 3, 1, 1, 1)


def ctclip_volume(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """CT-CLIP's established X/Y/Z input convention, normalized to [-1, 1]."""
    return resize_volume(array, (192, 512, 512), device).clamp(-1000.0, 1000.0).div(1000.0).permute(
        0, 1, 4, 3, 2
    ).contiguous()


def inception_features(model: torch.nn.Module, array: np.ndarray, device: torch.device) -> np.ndarray:
    indices = np.rint(np.linspace(0.1, 0.9, 5) * (array.shape[0] - 1)).astype(int)
    slices = torch.from_numpy(array[indices]).to(device)[:, None]
    slices = slices.clamp(-1000, 1000).add(1000).div(2000).repeat(1, 3, 1, 1)
    slices = F.interpolate(slices, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    return model((slices - mean) / std).float().cpu().numpy()


def normalize(feature: torch.Tensor) -> torch.Tensor:
    return F.normalize(feature.float().flatten(1), dim=1, eps=1e-8)


def frechet(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Fréchet metrics require at least two volumes")
    mu1, mu2 = first.mean(0), second.mean(0)
    cov1, cov2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    eye = np.eye(cov1.shape[0]) * 1e-6
    covmean = sqrtm((cov1 + eye) @ (cov2 + eye))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - 2 * covmean))


def load_ctclip(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(args.ctclip_repo))
    previous = Path.cwd()
    try:
        os.chdir(args.ctclip_repo)
        from core.cfg_helper import model_cfg_bank
        from core.models.common.get_model import get_model
        config = model_cfg_bank()("clip_3D")
        config.args.version = str(args.ctclip_backbone)
        model = get_model()(config)
        state = torch.load(args.ctclip_weights, map_location=device)
        model.load_state_dict(state.get("state_dict", state), strict=True)
        return model.to(device).eval()
    finally:
        os.chdir(previous)


def ctclip_feature(model: torch.nn.Module, value: Any, mode: str) -> torch.Tensor:
    """Keep the official CT-CLIP call surface explicit and fail loudly on API mismatch."""
    feature = model(value, mode)
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"CT-CLIP {mode} returned {type(feature)!r}, expected Tensor")
    return normalize(feature)


def i3d_feature(model: torch.jit.ScriptModule, array: np.ndarray, device: torch.device) -> torch.Tensor:
    feature = model(ct_window(array, device, (64, 224, 224)))
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"I3D model returned {type(feature)!r}, expected Tensor")
    return feature.float().flatten(1)


def collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise SystemExit("Generation metric evaluation requires CUDA")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num-shards > 0 and 0 <= shard-index < num-shards")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    all_rows = [row for row in payload["rows"] if row["task"] == "generation"]
    if len(all_rows) != args.expected_samples or len({row["case_id"] for row in all_rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation patients, found {len(all_rows)}")
    if any(not str(row.get("prompt", "")).strip() for row in all_rows):
        raise RuntimeError("Generation rows must include the report prompt; rerun evaluate_medgen3d.py with this release")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
    inception.fc = torch.nn.Identity()
    i3d = torch.jit.load(str(args.i3d_model), map_location=device).to(device).eval()
    ctclip = load_ctclip(args, device)
    features = {key: [] for key in ("real_fid", "fake_fid", "real_fvd", "fake_fvd", "t2i", "i2i")}
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(all_rows):
        if index % args.num_shards != args.shard_index:
            continue
        prediction_file, target_file = Path(row["volume"]), Path(row["target_volume"])
        if not prediction_file.exists() or not target_file.exists():
            raise FileNotFoundError((prediction_file, target_file))
        prediction, target = load_zyx(prediction_file), load_zyx(target_file)
        with torch.inference_mode():
            fake_clip = ctclip_feature(ctclip, ctclip_volume(prediction, device), "encode_vision")
            real_clip = ctclip_feature(ctclip, ctclip_volume(target, device), "encode_vision")
            text_clip = ctclip_feature(ctclip, [str(row["prompt"])], "encode_text")
            features["fake_fid"].append(inception_features(inception, prediction, device))
            features["real_fid"].append(inception_features(inception, target, device))
            features["fake_fvd"].append(i3d_feature(i3d, prediction, device).cpu().numpy())
            features["real_fvd"].append(i3d_feature(i3d, target, device).cpu().numpy())
            features["t2i"].append((fake_clip * text_clip).sum(dim=1).cpu().numpy())
            features["i2i"].append((fake_clip * real_clip).sum(dim=1).cpu().numpy())
        item = {"index": index, "case_id": str(row["case_id"]), "prediction": str(prediction_file),
                "target": str(target_file), "prompt": str(row["prompt"])}
        output_rows.append(item)
        print(json.dumps({"done": len(output_rows), "shard_total": (len(all_rows) + args.num_shards - 1) // args.num_shards,
                          **item}), flush=True)
    return {key: np.concatenate(value, axis=0) for key, value in features.items()}, output_rows


def summarize(features: dict[str, np.ndarray]) -> dict[str, float]:
    return {"fid": frechet(features["real_fid"], features["fake_fid"]),
            "fvd": frechet(features["real_fvd"], features["fake_fvd"]),
            "ct_clip_t2i": float(np.mean(features["t2i"])),
            "ct_clip_i2i": float(np.mean(features["i2i"]))}


def main() -> None:
    args = parse_args()
    features, rows = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_output, **features, rows_json=np.asarray(json.dumps(rows)))
        args.output.write_text(json.dumps({"protocol": PROTOCOL, "shard_index": args.shard_index,
                                           "num_shards": args.num_shards, "samples": len(rows)}) + "\n",
                               encoding="utf-8")
        return
    if args.num_shards != 1:
        raise SystemExit("Use --features-output and scripts/merge_generation_metrics.py for multiple shards")
    result = {"protocol": PROTOCOL, "samples": len(rows), "summary": summarize(features), "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
