#!/usr/bin/env python3
"""Evaluate generated CT volumes with the official MONAI MAISI 2.5D FID.

This is a results-manifest adapter for MONAI's
``generation/maisi/scripts/compute_fid_2-5d_ct.py``.  It deliberately keeps
the MAISI feature network and protocol: HU clipping, RAS orientation,
optional 1-mm resampling, pad/crop to a common 3-D field of view, global
per-volume min--max normalization followed by the RadImageNet channel mean,
and FID on all axial/coronal/sagittal slices.  The only difference is that
our evaluator reads the frozen ``evaluate_medgen3d.py`` JSON manifest rather
than two directory/file-list pairs.

Run under torchrun.  Rank zero writes the result JSON.  Feature extraction is
partitioned by volume, whereas FID is computed over all slices pooled across
the frozen cohort, exactly as in the MAISI reference implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import monai
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.metrics.fid import FIDMetric
from monai.transforms import Compose


PROTOCOL = "monai_maisi_radimagenet_resnet50_2p5d_fid_v1"


def parse_shape(value: str) -> tuple[int, int, int]:
    out = tuple(int(x) for x in value.lower().split("x"))
    if len(out) != 3 or min(out) <= 0:
        raise argparse.ArgumentTypeError("target shape must be XxYxZ")
    return out


def parse_spacing(value: str) -> tuple[float, float, float]:
    out = tuple(float(x) for x in value.lower().split("x"))
    if len(out) != 3 or min(out) <= 0:
        raise argparse.ArgumentTypeError("spacing must be XxYxZ")
    return out


def args_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True,
                   help="Frozen generation results.json from evaluate_medgen3d.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-samples", type=int, default=200)
    p.add_argument("--num-images", type=int, default=200,
                   help="Maximum sorted cohort members; use 200 for the frozen protocol")
    p.add_argument("--target-shape", type=parse_shape, default=parse_shape("512x512x512"))
    p.add_argument("--resampling-spacing", type=parse_spacing, default=parse_spacing("1x1x1"))
    p.add_argument("--center-slices-ratio", type=float, default=0.4,
                   help="MAISI README reference setting; 1.0 uses every slice")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def normalise_radimagenet(images: torch.Tensor) -> torch.Tensor:
    """The exact 4-D branch of MAISI's radimagenet_intensity_normalisation."""
    maxval, minval = images.max(), images.min()
    images = (images - minval) / (maxval - minval + 1e-10)
    means = images.new_tensor([0.406, 0.456, 0.485])[None, :, None, None]
    return images - means


@torch.inference_mode()
def plane_features(image: torch.Tensor, network: torch.nn.Module,
                   center_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract MAISI features for XY, YZ and ZX planes from [1,1,H,W,D]."""
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1, 1)
    image = image[:, [2, 1, 0], ...]  # MAISI RGB-to-BGR convention.
    _, _, h, w, d = image.shape

    def centered(dim: int, length: int) -> torch.Tensor:
        start = int((1.0 - center_ratio) * length / 2.0)
        end = int((1.0 + center_ratio) * length / 2.0)
        return torch.unbind(image.narrow(dim, start, max(1, end - start)), dim=dim)

    def encode(slices: Iterable[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(tuple(slices), dim=0)
        x = normalise_radimagenet(x)
        feat = network.forward(x)
        return feat.mean((2, 3)) if feat.ndim == 4 else feat.flatten(1)

    return encode(centered(4, d)), encode(centered(2, h)), encode(centered(3, w))


def gather_uneven(local: torch.Tensor, world_size: int) -> torch.Tensor:
    size = torch.tensor([len(local)], dtype=torch.long, device=local.device)
    sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size)
    largest = max(int(x.item()) for x in sizes)
    padded = F.pad(local, (0, 0, 0, largest - len(local)))
    pieces = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(pieces, padded)
    return torch.vstack([x[: int(n.item())] for x, n in zip(pieces, sizes)])


def build_transform(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> Compose:
    return Compose([
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes="RAS"),
        monai.transforms.Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
        monai.transforms.SpatialPadd(keys="image", spatial_size=shape, mode="constant", value=-1000),
        monai.transforms.CenterSpatialCropd(keys="image", roi_size=shape),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1000,
                                             b_min=-1000, b_max=1000, clip=True),
    ])


def main() -> None:
    a = args_parse()
    if not 0 < a.center_slices_ratio <= 1:
        raise SystemExit("--center-slices-ratio must lie in (0, 1]")
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(device)
    payload = json.loads(a.results.read_text(encoding="utf-8"))
    rows = sorted((x for x in payload["rows"] if x.get("task") == "generation"), key=lambda x: str(x["case_id"]))
    if len(rows) != a.expected_samples or len({str(x["case_id"]) for x in rows}) != len(rows):
        raise SystemExit(f"Expected {a.expected_samples} unique generation rows, got {len(rows)}")
    rows = rows[:a.num_images]
    if len(rows) < 2:
        raise SystemExit("FID requires at least two paired volumes")
    if any(not Path(x["volume"]).is_file() or not Path(x["target_volume"]).is_file() for x in rows):
        raise SystemExit("A prediction or target referenced by results.json is missing")
    network = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50",
                             verbose=(rank == 0), trust_repo=True).to(device).eval()
    transform = build_transform(a.target_shape, a.resampling_spacing)
    local = rows[rank::world]
    real, synth = [[], [], []], [[], [], []]
    for rec in local:
        target = transform({"image": str(rec["target_volume"])})["image"].as_tensor()[None].to(device)
        pred = transform({"image": str(rec["volume"])})["image"].as_tensor()[None].to(device)
        for i, feat in enumerate(plane_features(target, network, a.center_slices_ratio)): real[i].append(feat)
        for i, feat in enumerate(plane_features(pred, network, a.center_slices_ratio)): synth[i].append(feat)
        print(json.dumps({"rank": rank, "case_id": rec["case_id"]}), flush=True)
    all_real = [gather_uneven(torch.vstack(x), world) for x in real]
    all_synth = [gather_uneven(torch.vstack(x), world) for x in synth]
    if rank == 0:
        fid = FIDMetric()
        values = [float(fid(fake, truth).cpu()) for fake, truth in zip(all_synth, all_real)]
        result = {"protocol": PROTOCOL, "samples": len(rows), "summary": {
            "fid_xy": values[0], "fid_yz": values[1], "fid_zx": values[2], "fid_average": float(np.mean(values))},
            "settings": {"feature_network": "Warvito/radimagenet-models:radimagenet_resnet50",
                         "target_shape_xyz": a.target_shape, "resampling_spacing_xyz": a.resampling_spacing,
                         "center_slices_ratio": a.center_slices_ratio, "planes": ["XY", "YZ", "ZX"]},
            "case_ids": [str(x["case_id"]) for x in rows]}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["summary"], indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
