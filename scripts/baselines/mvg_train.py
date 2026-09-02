#!/usr/bin/env python3
"""Train the released 2-D MVG architecture on MedGen3D-compatible tasks.

MVG is deliberately kept a two-dimensional, paired in-context model.  This
adapter only changes data I/O: every sample contains a labelled support slice
in the upper half and a query slice in the lower half, exactly as in MVG's
``med_dataset.py``.  The three table-overlapping tasks are segmentation,
low-dose CT restoration, and T1-to-T2 synthesis.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler


IMAGE_SIZE = 448
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
SEGMENTATION_LABELS = {
    1: "aorta", 2: "gall_bladder", 3: "kidney_left", 4: "kidney_right",
    5: "liver", 6: "pancreas", 7: "postcava", 8: "spleen", 9: "stomach",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


class VolumeCache:
    """Small per-worker LRU cache; avoids repeated NIfTI decompression."""

    def __init__(self, max_items: int = 4) -> None:
        self.max_items = max_items
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        if path.suffix == ".npy":
            array = np.load(path).astype(np.float32)
        else:
            array = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)).transpose(2, 1, 0)
        self.items[key] = array
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return array


def tensor_from_slice(array: np.ndarray, *, mode: str, nearest: bool = False) -> torch.Tensor:
    """Convert one source/target slice to MVG's ImageNet-normalized RGB input."""
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode == "ct":
        array = np.clip((array + 1000.0) / 2000.0, 0.0, 1.0)
    elif mode == "mri":
        array = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    elif mode == "seg":
        # This intentionally follows the released MVG label encoding:
        # binary PNG -> ToTensor() -> multiply by 10 -> ImageNet normalization.
        array = (array > 0).astype(np.float32) * 10.0
        tensor = torch.from_numpy(array)[None].repeat(3, 1, 1)
        tensor = F.interpolate(tensor[None], (IMAGE_SIZE, IMAGE_SIZE), mode="nearest")[0]
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(f"Unknown slice mode: {mode}")
    tensor = torch.from_numpy(array)[None].repeat(3, 1, 1)
    tensor = F.interpolate(tensor[None], (IMAGE_SIZE, IMAGE_SIZE), mode="nearest" if nearest else "bilinear", align_corners=False if not nearest else None)[0]
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@dataclass(frozen=True)
class TaskRecord:
    task: str
    root: Path
    record: dict[str, Any]
    label_id: int | None = None


class MVGMedGenDataset(Dataset[dict[str, torch.Tensor]]):
    """Balanced random slice-pair stream for MVG's original paired layout."""

    def __init__(self, *, ct_root: Path, ct_manifest: Path, synth_root: Path,
                 synth_manifest: Path, samples_per_epoch: int, seed: int) -> None:
        self.cache = VolumeCache()
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch
        self.epoch = 0
        ct_rows = read_jsonl(ct_manifest)
        synth_rows = read_jsonl(synth_manifest)
        self.seg = [TaskRecord("segmentation", ct_root, row, label_id)
                    for row in ct_rows for label_id in SEGMENTATION_LABELS]
        self.restore = [TaskRecord("restoration", ct_root, row) for row in ct_rows]
        self.synth = [TaskRecord("synthesis", synth_root, row) for row in synth_rows]
        if not self.seg or not self.restore or not self.synth:
            raise ValueError("All three task manifests must contain records")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, index: int) -> random.Random:
        worker = torch.utils.data.get_worker_info()
        return random.Random(self.seed + self.epoch * 1_000_000_007 + index * 1009 + (worker.id if worker else 0) * 10_000_019)

    def _paths(self, item: TaskRecord) -> tuple[Path, Path, str, int | None]:
        row = item.record
        if item.task == "segmentation":
            return resolve(item.root, row["image"]), resolve(item.root, row["mask"]), "ct", item.label_id
        if item.task == "restoration":
            return resolve(item.root, row["ldct"][0]), resolve(item.root, row["image"]), "ct", None
        return resolve(item.root, row["condition"]), resolve(item.root, row["target"]), "mri", None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = self._rng(index)
        choices = (self.seg, self.restore, self.synth)
        item = choices[index % 3][rng.randrange(len(choices[index % 3]))]
        source_path, target_path, mode, label_id = self._paths(item)
        source, target = self.cache.load(source_path), self.cache.load(target_path)
        depth = min(source.shape[0], target.shape[0])
        if depth < 2:
            raise RuntimeError(f"Volume has fewer than two slices: {source_path}")
        candidates = np.arange(depth)
        if label_id is not None:
            foreground = np.flatnonzero(np.any(target[:depth] == label_id, axis=(1, 2)))
            if len(foreground):
                candidates = foreground
        qz = int(candidates[rng.randrange(len(candidates))])
        sz = int(candidates[rng.randrange(len(candidates))])
        if sz == qz and len(candidates) > 1:
            sz = int(candidates[(np.where(candidates == qz)[0][0] + 1) % len(candidates)])
        source_mode = mode
        target_mode = "seg" if label_id is not None else mode
        support_x = tensor_from_slice(source[sz], mode=source_mode)
        query_x = tensor_from_slice(source[qz], mode=source_mode)
        support_y = tensor_from_slice(target[sz] == label_id if label_id is not None else target[sz], mode=target_mode)
        query_y = tensor_from_slice(target[qz] == label_id if label_id is not None else target[qz], mode=target_mode)
        image, label = torch.cat((support_x, query_x), dim=1), torch.cat((support_y, query_y), dim=1)
        mask = torch.zeros((IMAGE_SIZE // 16 * 2, IMAGE_SIZE // 16), dtype=torch.bool)
        mask[IMAGE_SIZE // 16:, :] = True
        valid = torch.ones_like(label)
        if label_id is not None:
            valid[label < ((1e-4 - IMAGENET_MEAN) / IMAGENET_STD)] = 0.1
        return {"image": image, "label": label, "mask": mask, "valid": valid}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mvg-root", type=Path, required=True)
    p.add_argument("--ct-root", type=Path, required=True)
    p.add_argument("--ct-train-manifest", type=Path, required=True)
    p.add_argument("--synth-root", type=Path, required=True)
    p.add_argument("--synth-train-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--finetune", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--samples-per-epoch", type=int, default=24_576)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-iter", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-every", type=int, default=5)
    return p.parse_args()


def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    return rank, world, device


def main() -> None:
    args = parse_args()
    rank, world, device = init_distributed()
    sys.path.insert(0, str(args.mvg_root))
    import models_mvg  # type: ignore
    import utils.lr_decay as lr_decay  # type: ignore

    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    dataset = MVGMedGenDataset(ct_root=args.ct_root, ct_manifest=args.ct_train_manifest,
                               synth_root=args.synth_root, synth_manifest=args.synth_train_manifest,
                               samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    # Workers are recreated each epoch so the epoch-dependent random slice
    # stream is visible to every worker without sharing mutable state.
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=False, drop_last=True)
    model = models_mvg.mvg_vit_large_patch16().to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state = checkpoint["model"]
        incompatible = model.load_state_dict(state, strict=False)
        if rank == 0:
            print("Loaded MAE initialization", incompatible, flush=True)
        start_epoch = 0
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    raw_model = model.module if world > 1 else model
    groups = lr_decay.param_groups_lrd(raw_model, args.weight_decay, raw_model.no_weight_decay(), 0.8)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch); dataset.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []; start = time.time()
        for step, batch in enumerate(loader):
            image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            mask, valid = batch["mask"].to(device, non_blocking=True), batch["valid"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(image, label, bool_masked_pos=mask, valid=valid)
                loss = loss / args.accum_iter
            scaler.scale(loss).backward()
            if (step + 1) % args.accum_iter == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().item() * args.accum_iter))
        average = float(np.mean(losses))
        if world > 1:
            tensor = torch.tensor([average], device=device); dist.all_reduce(tensor); average = float(tensor.item() / world)
        if rank == 0:
            row = {"epoch": epoch, "loss": average, "seconds": time.time() - start, "world_size": world}
            with log_path.open("a") as handle: handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if epoch % args.save_every == 0 or epoch + 1 == args.epochs:
                # Snapshots used for evaluation do not need Adam moments. Keep
                # exactly one resumable optimizer checkpoint plus compact
                # model-only milestones to avoid multiplying 4-GB optimizer
                # states over a long baseline run.
                torch.save({"model": raw_model.state_dict(), "epoch": epoch, "args": vars(args)},
                           args.output_dir / f"model-{epoch:03d}.pt")
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args)}, args.output_dir / "checkpoint-latest.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
