#!/usr/bin/env python3
"""Compute the paper's FID and FVD-CT metrics from generated CT volumes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torchvision.models import Inception_V3_Weights, inception_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="results.json produced by evaluate_medgen3d.py")
    parser.add_argument("--ctclip-repo", type=Path, required=True)
    parser.add_argument("--ctclip-backbone", type=Path, required=True)
    parser.add_argument("--ctclip-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--features-output", type=Path,
                        help="Write feature shard instead of a final metric summary")
    parser.add_argument("--expected-samples", type=int, default=200)
    return parser.parse_args()


def load_zyx(path: Path) -> np.ndarray:
    xyz = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return np.transpose(np.nan_to_num(xyz, nan=-1000.0), (2, 1, 0))


def resize_volume(array: np.ndarray, shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    return F.interpolate(torch.from_numpy(array).to(device)[None, None], size=shape,
                         mode="trilinear", align_corners=False)


def ctclip_volume(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert Z/Y/X HU data to the frozen CT-CLIP X/Y/Z input convention."""
    volume = resize_volume(array, (192, 512, 512), device)
    return volume.clamp(-1000.0, 1000.0).div(1000.0).permute(0, 1, 4, 3, 2).contiguous()


def inception_features(model: torch.nn.Module, array: np.ndarray, device: torch.device) -> np.ndarray:
    indices = np.linspace(0.1, 0.9, 5) * (array.shape[0] - 1)
    slices = torch.from_numpy(array[np.rint(indices).astype(int)]).to(device)[:, None]
    slices = slices.clamp(-1000, 1000).add(1000).div(2000).repeat(1, 3, 1, 1)
    slices = F.interpolate(slices, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    return model((slices - mean) / std).float().cpu().numpy()


def frechet(first: np.ndarray, second: np.ndarray) -> float:
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
        model.load_state_dict(torch.load(args.ctclip_weights, map_location=device), strict=True)
        return model.to(device).eval()
    finally:
        os.chdir(previous)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if not torch.cuda.is_available():
        raise SystemExit("FID/FVD-CT evaluation requires CUDA")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    all_rows = [row for row in payload["rows"] if row["task"] == "generation"]
    if len(all_rows) != args.expected_samples or len({row["case_id"] for row in all_rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation patients")
    rows = [(index, row) for index, row in enumerate(all_rows)
            if index % args.num_shards == args.shard_index]
    device = torch.device("cuda")
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
    inception.fc = torch.nn.Identity()
    ctclip = load_ctclip(args, device)
    real_2d, fake_2d, real_3d, fake_3d, output_rows = [], [], [], [], []
    for done, (index, row) in enumerate(rows, start=1):
        prediction_file, target_file = Path(row["volume"]), Path(row["target_volume"])
        if not prediction_file.exists() or not target_file.exists():
            raise FileNotFoundError((prediction_file, target_file))
        prediction, target = load_zyx(prediction_file), load_zyx(target_file)
        with torch.inference_mode():
            fake_2d.append(inception_features(inception, prediction, device))
            real_2d.append(inception_features(inception, target, device))
            fake_3d.append(ctclip(ctclip_volume(prediction, device), "encode_vision").flatten(1).float().cpu().numpy())
            real_3d.append(ctclip(ctclip_volume(target, device), "encode_vision").flatten(1).float().cpu().numpy())
        output_rows.append({"index": index, "case_id": str(row["case_id"]),
                            "prediction": str(prediction_file), "target": str(target_file)})
        print(json.dumps({"done": done, "shard_total": len(rows), **output_rows[-1]}), flush=True)
    features = {"real_2d": np.concatenate(real_2d), "fake_2d": np.concatenate(fake_2d),
                "real_3d": np.concatenate(real_3d), "fake_3d": np.concatenate(fake_3d)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_output, **features, rows_json=np.asarray(json.dumps(output_rows)))
        args.output.write_text(json.dumps({"shard_index": args.shard_index, "num_shards": args.num_shards,
                                           "samples": len(output_rows)}) + "\n", encoding="utf-8")
        return
    if args.num_shards != 1:
        raise SystemExit("Use --features-output and merge_generation_metric_shards.py for multiple shards")
    result = {"protocol": "ctclip3d_fvd_inceptionv3_slice_fid_v1", "samples": len(output_rows),
              "summary": {"fid": frechet(features["real_2d"], features["fake_2d"]),
                          "fvd_ct": frechet(features["real_3d"], features["fake_3d"])}, "rows": output_rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
