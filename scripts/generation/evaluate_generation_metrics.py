#!/usr/bin/env python3
"""Retired StyleGAN-V evaluator retained only for historical reproducibility.

This evaluator deliberately follows the public protocol referenced by
GenerateCT: StyleGAN-V's TensorFlow-compatible Inception detector for FID,
StyleGAN-V's official FVD I3D detector for FVD-16, and TorchMetrics'
``CLIPScore`` for report-to-image alignment.  CT volumes are treated as axial
video sequences.  They are clipped to [-1000, 1000] HU and encoded as uint8
grayscale RGB frames, exactly the input domain expected by the StyleGAN-V
detectors.

Unlike the former CT-CLIP protocol, this produces one text--image score:
``clip_score``.  It does not define an image--image CLIP metric.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from scipy.linalg import sqrtm


PROTOCOL = "ctrate_v2_styleganv_fid_fvd16_torchmetrics_clipscore_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="results.json produced by evaluate_medgen3d.py")
    parser.add_argument("--fid-detector", type=Path, required=True,
                        help="StyleGAN-V inception-2015-12-05.pkl detector")
    parser.add_argument("--i3d-model", type=Path, required=True,
                        help="StyleGAN-V i3d_torchscript.pt FVD detector")
    parser.add_argument("--styleganv-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / ".cache" / "stylegan-v",
                        help="StyleGAN-V checkout; needed to unpickle its FID detector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument("--fvd-frames", type=int, default=16,
                        help="Number of axial frames for FVD (StyleGAN-V FVD-16 default)")
    parser.add_argument("--clip-model-name-or-path", default="openai/clip-vit-large-patch14",
                        help="TorchMetrics CLIPScore model_name_or_path")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--features-output", type=Path,
                        help="Write a feature shard; merge with merge_generation_metrics.py")
    return parser.parse_args()


def load_zyx(path: Path) -> np.ndarray:
    """Load a NIfTI volume as axial Z/Y/X data."""
    xyz = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {xyz.shape}: {path}")
    return np.transpose(np.nan_to_num(xyz, nan=-1000.0), (2, 1, 0))


def hu_to_uint8(frames_zyx: np.ndarray) -> torch.Tensor:
    """Convert axial HU frames to StyleGAN-V's uint8 RGB image convention."""
    values = np.clip(frames_zyx, -1000.0, 1000.0)
    values = np.rint((values + 1000.0) * (255.0 / 2000.0)).astype(np.uint8)
    return torch.from_numpy(values)[:, None].repeat(1, 3, 1, 1)


def uniformly_sample_frames(volume: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("frame count must be positive")
    indices = np.rint(np.linspace(0, volume.shape[0] - 1, count)).astype(np.int64)
    return volume[indices]


def load_pickle_detector(path: Path, styleganv_root: Path, device: torch.device) -> torch.nn.Module:
    """Load StyleGAN-V's pickled TensorFlow-compatible Inception detector."""
    if not path.is_file():
        raise FileNotFoundError(path)
    source_root = styleganv_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"StyleGAN-V source is required to load {path.name}; expected {source_root}. "
            "Clone https://github.com/universome/stylegan-v or pass --styleganv-root."
        )
    # The pickled detector imports both ``torch_utils`` (from src/) and
    # ``src.dnnlib`` (from the repository root).
    sys.path.insert(0, str(styleganv_root))
    sys.path.insert(0, str(source_root))
    with path.open("rb") as handle:
        detector = pickle.load(handle)
    if not isinstance(detector, torch.nn.Module):
        raise TypeError(f"FID detector must be a torch.nn.Module, got {type(detector)!r}")
    return detector.to(device).eval()


def detector_features(detector: torch.nn.Module, images: torch.Tensor, **kwargs: Any) -> np.ndarray:
    output = detector(images, **kwargs)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Detector returned {type(output)!r}, expected Tensor")
    return output.detach().float().flatten(1).cpu().numpy()


def fid_features(detector: torch.nn.Module, volume: np.ndarray, device: torch.device) -> np.ndarray:
    # StyleGAN-V's image-dataset FID flattens the dataset into individual frames.
    images = hu_to_uint8(volume).to(device)
    return detector_features(detector, images, return_features=True)


def fvd_feature(detector: torch.jit.ScriptModule, volume: np.ndarray, device: torch.device,
                num_frames: int) -> np.ndarray:
    # Match fvd2048_16f's detector invocation: uint8 [B, C, T, H, W], with
    # detector-side rescale and resize enabled.
    frames = hu_to_uint8(uniformly_sample_frames(volume, num_frames))
    video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(device)
    return detector_features(detector, video, rescale=True, resize=True, return_features=True)


def styleganv_frechet(first: np.ndarray, second: np.ndarray) -> float:
    """The StyleGAN-V population-covariance Fréchet computation."""
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Fréchet metrics require at least two feature vectors")
    mu_first, mu_second = first.mean(axis=0), second.mean(axis=0)
    # StyleGAN-V FeatureStats divides by N, not N - 1.
    cov_first = np.cov(first, rowvar=False, bias=True)
    cov_second = np.cov(second, rowvar=False, bias=True)
    covmean, _ = sqrtm(cov_second @ cov_first, disp=False)
    value = np.square(mu_second - mu_first).sum() + np.trace(cov_second + cov_first - 2 * covmean)
    return float(np.real(value))


def make_clip_metric(model_name_or_path: str, device: torch.device) -> Any:
    try:
        from torchmetrics.multimodal import CLIPScore
    except ImportError as exc:
        raise SystemExit("TorchMetrics is required: pip install 'torchmetrics[image]' transformers") from exc
    return CLIPScore(model_name_or_path=model_name_or_path).to(device).eval()


def clip_score(metric: Any, volume: np.ndarray, prompt: str, device: torch.device,
               batch_size: int) -> float:
    """Mean TorchMetrics CLIPScore over all axial frames of one CT volume."""
    images = hu_to_uint8(volume)
    metric.reset()
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        captions = [prompt] * len(batch)
        metric.update(batch, captions)
    return float(metric.compute().detach().cpu())


def collect(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise SystemExit("Generation metric evaluation requires CUDA")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if args.fvd_frames != 16:
        raise SystemExit("This protocol is StyleGAN-V FVD-16; require --fvd-frames 16")

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    all_rows = [row for row in payload["rows"] if row["task"] == "generation"]
    if len(all_rows) != args.expected_samples or len({row["case_id"] for row in all_rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation patients, found {len(all_rows)}")
    if any(not str(row.get("prompt", "")).strip() for row in all_rows):
        raise RuntimeError("Generation rows must include a non-empty report prompt")

    device = torch.device("cuda")
    fid_detector = load_pickle_detector(args.fid_detector, args.styleganv_root, device)
    i3d_detector = torch.jit.load(str(args.i3d_model), map_location=device).to(device).eval()
    clip_metric = make_clip_metric(args.clip_model_name_or_path, device)
    features = {key: [] for key in ("real_fid", "fake_fid", "real_fvd", "fake_fvd", "clip_score")}
    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(all_rows):
        if index % args.num_shards != args.shard_index:
            continue
        prediction_file, target_file = Path(row["volume"]), Path(row["target_volume"])
        if not prediction_file.exists() or not target_file.exists():
            raise FileNotFoundError((prediction_file, target_file))
        prediction, target = load_zyx(prediction_file), load_zyx(target_file)
        prompt = str(row["prompt"])
        with torch.inference_mode():
            features["fake_fid"].append(fid_features(fid_detector, prediction, device))
            features["real_fid"].append(fid_features(fid_detector, target, device))
            features["fake_fvd"].append(fvd_feature(i3d_detector, prediction, device, args.fvd_frames))
            features["real_fvd"].append(fvd_feature(i3d_detector, target, device, args.fvd_frames))
            features["clip_score"].append(np.asarray([clip_score(
                clip_metric, prediction, prompt, device, args.clip_batch_size)], dtype=np.float32))
        item = {"index": index, "case_id": str(row["case_id"]), "prediction": str(prediction_file),
                "target": str(target_file), "prompt": prompt}
        output_rows.append(item)
        print(json.dumps({"done": len(output_rows), "shard_total": (len(all_rows) + args.num_shards - 1) // args.num_shards,
                          **item}), flush=True)
    return {key: np.concatenate(value, axis=0) for key, value in features.items()}, output_rows


def summarize(features: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "fid": styleganv_frechet(features["real_fid"], features["fake_fid"]),
        "fvd_i3d_16f": styleganv_frechet(features["real_fvd"], features["fake_fvd"]),
        "clip_score": float(np.mean(features["clip_score"])),
    }


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
        raise SystemExit("Use --features-output and scripts/generation/merge_generation_metrics.py for multiple shards")
    result = {"protocol": PROTOCOL, "samples": len(rows), "summary": summarize(features), "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    # New evaluations must use the medical-image protocol.  Keep the legacy
    # functions above readable so prior JSON artifacts remain auditable.
    from evaluate_maisi_fid_2p5d import main as maisi_main
    maisi_main()
