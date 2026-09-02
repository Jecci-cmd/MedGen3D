#!/usr/bin/env python3
"""Compute the official MAISI 2.5D RadImageNet FID for generated CT volumes.

This is an adapter around MONAI MAISI's ``compute_fid_2-5d_ct.py`` for the
``results.json`` emitted by MedGen3D.  It uses the MAISI reference protocol:
RAS orientation, 1-mm resampling, 512^3 center crop/pad, the central 40% of
all XY/YZ/ZX slices, RadImageNet ResNet-50 features, and one FID per plane.

All model assets must already exist locally.  The evaluator never downloads
weights; set ``MEDGEN3D_EVAL_ASSETS`` or use ``--assets-root``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import monai
import numpy as np
import torch
import torch.distributed as dist
from monai.metrics.fid import FIDMetric
from monai.transforms import Compose


PROTOCOL = "monai_maisi_radimagenet_resnet50_2p5d_fid_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = Path(os.environ.get("MEDGEN3D_EVAL_ASSETS", REPOSITORY_ROOT / "evaluation_assets"))


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(part) for part in value.lower().split("x"))
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must be XxYxZ with positive integers")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--target-shape", type=parse_shape, default=parse_shape("512x512x512"))
    parser.add_argument("--center-slices-ratio", type=float, default=0.4)
    parser.add_argument("--slice-batch-size", type=int, default=32)
    parser.add_argument("--keep-feature-shards", action="store_true")
    return parser.parse_args()


def distributed_device() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        if not torch.cuda.is_available():
            raise SystemExit("MAISI FID requires CUDA")
        return 0, 1, torch.device("cuda:0")
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    return rank, world, device


def finish_distributed(world: int) -> None:
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def build_transform(shape: tuple[int, int, int]) -> Compose:
    class CorrectCorruptedBIMCVZSpacingd(monai.transforms.MapTransform):
        """Repair the documented BIMCV-R header error before physical resampling.

        A small subset of BIMCV-R NIfTIs stores a 1.86--3.72 mm slice
        spacing as 186--372 mm.  Their in-plane spacing remains normal, so
        this is unambiguously a factor-100 header error rather than anatomy.
        Correcting it here keeps the MAISI 1 mm preprocessing physical while
        leaving all normally encoded volumes untouched.
        """

        def __call__(self, data: object) -> object:
            result = dict(data)  # type: ignore[arg-type]
            for key in self.keys:
                image = result[key]
                affine = image.affine.clone()
                spacing_z = float(torch.linalg.vector_norm(affine[:3, 2]))
                spacing_xy = float(torch.linalg.vector_norm(affine[:3, 0]))
                if spacing_z > 20.0 and 0.2 <= spacing_xy <= 2.0:
                    affine[:3, 2] /= 100.0
                    image.affine = affine
                    image.meta["affine"] = affine
                    image.meta["pixdim"][3] = spacing_z / 100.0
                result[key] = image
            return result

    return Compose([
        monai.transforms.LoadImaged(keys="image"),
        monai.transforms.EnsureChannelFirstd(keys="image"),
        CorrectCorruptedBIMCVZSpacingd(keys="image"),
        monai.transforms.Orientationd(keys="image", axcodes="RAS"),
        monai.transforms.Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        monai.transforms.SpatialPadd(keys="image", spatial_size=shape, mode="constant", value=-1000),
        monai.transforms.CenterSpatialCropd(keys="image", roi_size=shape),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1000,
                                             b_min=-1000, b_max=1000, clip=True),
    ])


def load_radimagenet(assets_root: Path, device: torch.device) -> torch.nn.Module:
    source = assets_root / "maisi" / "radimagenet-models"
    weights = source / "weights" / "RadImageNet-ResNet50_notop.pth"
    if not source.is_dir() or not weights.is_file():
        raise FileNotFoundError("Missing local MAISI RadImageNet assets: " + str(weights))
    sys.path.insert(0, str(source))
    from radimagenet_models.models.resnet import radimagenet_resnet50  # type: ignore[import-not-found]
    return radimagenet_resnet50(model_dir=str(weights.parent), file_name=weights.name).to(device).eval()


def normalize_radimagenet(images: torch.Tensor) -> torch.Tensor:
    # MAISI's 4-D branch: volume-wise min/max for each extracted slice set.
    images = (images - images.min()) / (images.max() - images.min() + 1e-10)
    means = images.new_tensor([0.406, 0.456, 0.485])[None, :, None, None]
    return images - means


@torch.inference_mode()
def encode_slices(slices: Iterable[torch.Tensor], network: torch.nn.Module, batch_size: int) -> torch.Tensor:
    image = torch.cat(tuple(slices), dim=0)
    pieces: list[torch.Tensor] = []
    for start in range(0, len(image), batch_size):
        batch = normalize_radimagenet(image[start:start + batch_size])
        feature = network(batch)
        pieces.append(feature.mean((2, 3)) if feature.ndim == 4 else feature.flatten(1))
    return torch.cat(pieces).float().cpu()


@torch.inference_mode()
def plane_features(image: torch.Tensor, network: torch.nn.Module, center_ratio: float,
                   batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1, 1)
    image = image[:, [2, 1, 0], ...]  # MAISI RGB-to-BGR convention.

    def central(dim: int) -> tuple[torch.Tensor, ...]:
        length = image.shape[dim]
        start = int((1.0 - center_ratio) * length / 2.0)
        end = int((1.0 + center_ratio) * length / 2.0)
        return torch.unbind(image.narrow(dim, start, max(1, end - start)), dim=dim)

    return (
        encode_slices(central(4), network, batch_size),
        encode_slices(central(2), network, batch_size),
        encode_slices(central(3), network, batch_size),
    )


def load_generation_rows(results: Path, expected_samples: int, num_images: int) -> list[dict[str, object]]:
    payload = json.loads(results.read_text(encoding="utf-8"))
    rows = sorted((row for row in payload["rows"] if row.get("task") == "generation"),
                  key=lambda row: str(row["case_id"]))
    if len(rows) != expected_samples or len({str(row["case_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"Expected {expected_samples} unique generation cases, found {len(rows)}")
    rows = rows[:num_images]
    if len(rows) < 2:
        raise RuntimeError("FID requires at least two paired volumes")
    for row in rows:
        if not Path(str(row["volume"])).is_file() or not Path(str(row["target_volume"])).is_file():
            raise FileNotFoundError((row.get("volume"), row.get("target_volume")))
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.center_slices_ratio <= 1.0:
        raise SystemExit("--center-slices-ratio must be in (0, 1]")
    rank, world, device = distributed_device()
    rows = load_generation_rows(args.results, args.expected_samples, args.num_images)
    transform, network = build_transform(args.target_shape), load_radimagenet(args.assets_root, device)
    local_real = [[], [], []]
    local_fake = [[], [], []]
    for row in rows[rank::world]:
        real = transform({"image": str(row["target_volume"])})["image"].as_tensor()[None].to(device)
        fake = transform({"image": str(row["volume"])})["image"].as_tensor()[None].to(device)
        for plane, feature in enumerate(plane_features(real, network, args.center_slices_ratio, args.slice_batch_size)):
            local_real[plane].append(feature.numpy())
        for plane, feature in enumerate(plane_features(fake, network, args.center_slices_ratio, args.slice_batch_size)):
            local_fake[plane].append(feature.numpy())
        print(json.dumps({"rank": rank, "case_id": row["case_id"]}), flush=True)
    shard = args.output.parent / f".{args.output.stem}.maisi_rank{rank}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, **{f"real_{i}": np.concatenate(local_real[i]) for i in range(3)},
                        **{f"fake_{i}": np.concatenate(local_fake[i]) for i in range(3)})
    if world > 1:
        dist.barrier()
    if rank == 0:
        loaded = [np.load(args.output.parent / f".{args.output.stem}.maisi_rank{i}.npz") for i in range(world)]
        fid = FIDMetric()
        values = []
        for plane in range(3):
            real = torch.from_numpy(np.concatenate([item[f"real_{plane}"] for item in loaded]))
            fake = torch.from_numpy(np.concatenate([item[f"fake_{plane}"] for item in loaded]))
            values.append(float(fid(fake, real).cpu()))
        summary = {"fid_xy": values[0], "fid_yz": values[1], "fid_zx": values[2],
                   "fid_average": float(np.mean(values))}
        result = {"protocol": PROTOCOL, "samples": len(rows), "summary": summary,
                  "settings": {"feature_network": "RadImageNet-ResNet50", "target_shape_xyz": args.target_shape,
                               "resampling_spacing_xyz": [1.0, 1.0, 1.0],
                               "center_slices_ratio": args.center_slices_ratio, "planes": ["XY", "YZ", "ZX"]},
                  "case_ids": [str(row["case_id"]) for row in rows]}
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if not args.keep_feature_shards:
            for item in loaded:
                item.close()
            for index in range(world):
                (args.output.parent / f".{args.output.stem}.maisi_rank{index}.npz").unlink(missing_ok=True)
    finish_distributed(world)


if __name__ == "__main__":
    main()
