#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config, save_resolved_config
from medgen3d.data import (
    BalancedMultiTaskDataset,
    audit_multitask_splits,
    build_task_dataset,
    unified_collate,
)
from medgen3d.trainer import FeedForwardTrainer, FlowTrainer, append_jsonl_log, build_optimizer, build_scheduler, seed_everything
from medgen3d.wan import (FrozenTextEncoder, FrozenWanVAE, MedicalVolumeCodec,
                          MedicalWanDiT, configure_dit_finetuning,
                          load_official_components)
from medgen3d.numerics import pad_volume


class MockDiT(nn.Module):
    def __init__(self, channels: int = 4) -> None:
        super().__init__(); self.net=nn.Conv3d(channels*2,channels,1)
    def forward(self,z,condition,timestep,text,view_ratio=None,volume_position=None): return self.net(torch.cat([z,condition],1))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml"),
    )
    parser.add_argument("--mock",action="store_true",help="Run one CPU latent sanity step without Wan weights")
    parser.add_argument("--checkpoint-dir",type=Path); parser.add_argument("--max-steps",type=int)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--export-portable",type=Path,
                        help="After --resume, consolidate ZeRO optimizer state into a world-size-independent checkpoint and exit")
    parser.add_argument("--limit-train-cases",type=int,help="Deterministic tiny-set overfit check")
    parser.add_argument("--run-name",help="Override experiment name to isolate probes from formal outputs")
    parser.add_argument(
        "--task-sampling-ratio", nargs="+", metavar="TASK=WEIGHT",
        help="Exact positive integer task ratio, e.g. segmentation=3 restoration=1 "
             "reconstruction=1 synthesis=1 generation=1.",
    )
    args=parser.parse_args(); config=load_experiment_config(args.config); train=config["train"]
    if args.run_name: config["experiment"]["name"] = args.run_name
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        # Large ZeRO checkpoints can take more than the PyTorch default
        # collective timeout to deserialize uneven per-rank optimizer shards.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    seed_everything(int(train["seed"]) + rank); run_dir=Path(train["output_dir"])/config["experiment"]["name"]
    if rank == 0: save_resolved_config(config,run_dir/"resolved_config.yaml")
    if args.mock:
        model=MockDiT(); optimizer=build_optimizer(model,train)
        trainer_class = (FeedForwardTrainer if train.get("objective") == "feed_forward_latent_regression"
                         else FlowTrainer)
        trainer=trainer_class(model,optimizer,None,train,torch.device("cpu"),scheduler=build_scheduler(optimizer,train))
        batch={"condition":torch.randn(1,4,2,4,4),"target":torch.randn(1,4,2,4,4),
               "valid_mask":torch.ones(1,1,2,4,4),"prompt":["Restore this low-dose CT volume."]}
        loss=trainer.train_microbatch(batch,text_context=[torch.zeros(1,1)])
        append_jsonl_log(run_dir/"metrics.jsonl",{"step":trainer.step,"loss":loss,"mode":"mock"})
        trainer.save_checkpoint(run_dir/"mock_checkpoint.pt",config); print(f"mock step passed: loss={loss:.6f}"); return
    if args.checkpoint_dir is None: parser.error("--checkpoint-dir is required unless --mock is used")
    if train.get("require_clean_git",True) and config["_provenance"]["git_dirty"]:
        parser.error("Formal runs require a clean committed project worktree")
    if not config["model"].get("model_revision") or not config["model"].get("wan_repo_commit"):
        parser.error("Formal runs require pinned model_revision and wan_repo_commit")
    if not torch.cuda.is_available(): parser.error("Formal training requires CUDA")
    torch.cuda.set_device(local_rank)
    device=torch.device("cuda", local_rank); dtype=torch.bfloat16 if train["precision"]=="bf16" else torch.float16
    base,official_vae,official_text=load_official_components(args.checkpoint_dir,device=str(device),dtype=dtype)
    vae=FrozenWanVAE(official_vae); text=FrozenTextEncoder(official_text); model=MedicalWanDiT(
        base, timestep_scale=config["model"]["pretrained_timestep_scale"],
        view_fourier_bands=int(config["model"].get("view_fourier_bands", 8)),
    )
    lora_modules = configure_dit_finetuning(model, train.get("finetuning", {"mode": "full"}))
    if rank == 0:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(json.dumps({"finetuning_mode": train.get("finetuning", {}).get("mode", "full"),
                          "lora_linear_modules": len(lora_modules), "trainable_parameters": trainable,
                          "total_parameters": total, "trainable_fraction": trainable / total}))
    model = model.to(device)
    model.enable_gradient_checkpointing(config["model"].get("gradient_checkpointing",True))
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                        broadcast_buffers=False, find_unused_parameters=False,
                                        gradient_as_bucket_view=True)
    tasks=tuple(config["experiment"]["tasks"])
    ratios=dict(config["experiment"].get("task_sampling_ratio",train["task_sampling_ratio"]))
    if args.task_sampling_ratio:
        parsed: dict[str, float] = {}
        for item in args.task_sampling_ratio:
            if "=" not in item:
                parser.error(f"Invalid --task-sampling-ratio item: {item!r}")
            task, value = item.split("=", 1)
            try:
                parsed[task] = float(value)
            except ValueError:
                parser.error(f"Invalid task sampling weight: {item!r}")
        if set(parsed) != set(tasks):
            parser.error(f"Task ratio must specify exactly {list(tasks)}")
        ratios = parsed
        config["experiment"]["task_sampling_ratio"] = ratios
    ratio_counts: dict[str, int] = {}
    for task in tasks:
        weight=float(ratios.get(task, 0.0))
        count=int(weight)
        if weight <= 0 or count != weight:
            parser.error("Task sampling weights must be positive integers for an exact schedule")
        ratio_counts[task] = count
    task_schedule=tuple(task for task in tasks for _ in range(ratio_counts[task]))
    audit_multitask_splits(config["data"], tasks)
    num_samples=int(train["max_steps"])*int(train["batch_size"])*int(train["gradient_accumulation_steps"])*world_size
    per_task_samples={
        task: (num_samples * ratio_counts[task] + len(task_schedule) - 1) // len(task_schedule)
        for task in tasks
    }
    task_datasets = {
        task: build_task_dataset(
            config["data"], task, "train", seed=int(train["seed"]),
            num_samples=(min(per_task_samples[task], int(args.limit_train_cases))
                         if args.limit_train_cases else per_task_samples[task]),
        )
        for task in tasks
    }
    dataset=BalancedMultiTaskDataset(task_datasets, task_schedule, num_samples)
    optimizer=build_optimizer(model,train)
    codec=MedicalVolumeCodec(vae)
    objective = str(train.get("objective", "rectified_flow_matching"))
    def encode_pair(condition,target,task,valid_mask):
        zc,z0,info=codec.encode_pair(condition,target,task)
        if (objective == "rectified_flow_matching" and
                config["data"].get("reconstruction", {}).get("residual_prediction") == "latent"):
            reconstruction = torch.tensor([name == "reconstruction" for name in task], device=z0.device)
            z0 = torch.where(reconstruction[:, None, None, None, None], z0 - zc, z0)
        mask=pad_volume(valid_mask.to(device),info,0.0)
        mask=torch.nn.functional.interpolate(mask,size=zc.shape[-3:],mode="nearest")
        return zc,z0,mask
    trainer_config = {**train, "reconstruction_full_views":
                      config["data"].get("reconstruction", {}).get("full_views", 720)}
    trainer_class = FeedForwardTrainer if objective == "feed_forward_latent_regression" else FlowTrainer
    trainer=trainer_class(model,optimizer,text,trainer_config,device,
                          scheduler=build_scheduler(optimizer,train),pair_encoder=encode_pair)
    if args.resume: trainer.load_checkpoint(args.resume)
    if args.export_portable:
        if not args.resume:
            parser.error("--export-portable requires --resume")
        trainer.save_portable_checkpoint(args.export_portable, config, rank)
        if distributed:
            dist.destroy_process_group()
        return
    consumed_global_samples = (
        trainer.step * trainer.accum * int(train["batch_size"]) * world_size
    )
    if consumed_global_samples >= len(dataset):
        raise RuntimeError(
            f"Resume step {trainer.step} has already consumed the configured sample stream"
        )
    training_dataset = (Subset(dataset, range(consumed_global_samples, len(dataset)))
                        if consumed_global_samples else dataset)
    sampler = (DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, shuffle=False)
               if distributed else None)
    loader=DataLoader(training_dataset,batch_size=train["batch_size"],shuffle=False,sampler=sampler,
                      collate_fn=unified_collate,num_workers=int(train.get("num_workers",4)),
                      pin_memory=True,persistent_workers=int(train.get("num_workers",4)) > 0)
    limit=args.max_steps or int(train["max_steps"])
    def reduce_mean(value: float) -> float:
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        if distributed: dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor / world_size)

    def reduce_optional(metrics: dict[str, float], key: str) -> float | None:
        packed = torch.tensor([metrics.get(key, 0.0), float(key in metrics)], device=device, dtype=torch.float64)
        if distributed: dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return float(packed[0] / packed[1]) if packed[1] else None

    def validate() -> dict[str, float]:
        results: dict[str, float] = {}
        validation_counts = train.get("validation_samples", {})
        for task in tasks:
            val_dataset = build_task_dataset(
                config["data"], task, "val", seed=int(train["seed"]),
                num_samples=int(validation_counts.get(task, train.get("validation_samples_per_task", 8))),
                training=False,
            )
            # DistributedSampler pads when the sample count is not divisible by
            # world_size, which would silently duplicate validation cases.  A
            # strided subset evaluates each of the requested cases exactly once.
            val_rank_dataset = (Subset(val_dataset, range(rank, len(val_dataset), world_size))
                                if distributed else val_dataset)
            val_loader = DataLoader(val_rank_dataset, batch_size=1, shuffle=False,
                                    collate_fn=unified_collate, num_workers=0)
            total = count = 0.0
            for val_batch in val_loader:
                total += trainer.validation_microbatch(val_batch)["loss"]
                count += 1
            packed = torch.tensor([total, count], device=device, dtype=torch.float64)
            if distributed: dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            results[f"val_loss/{task}"] = float(packed[0] / packed[1].clamp_min(1))
        results["val_loss/mean"] = sum(results.values()) / len(results)
        return results

    best_val_loss = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    if rank == 0 and metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            try:
                previous = json.loads(line)
                value = previous.get("val_loss/mean")
                if value is not None:
                    best_val_loss = min(best_val_loss, float(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if distributed:
        packed_best = torch.tensor(best_val_loss, device=device, dtype=torch.float64)
        dist.broadcast(packed_best, src=0)
        best_val_loss = float(packed_best)
    if args.resume:
        if distributed: dist.barrier()
        resume_validation = {"step": trainer.step, "phase": "resume_validation", **validate()}
        if rank == 0:
            append_jsonl_log(run_dir/"metrics.jsonl", resume_validation)
            if float(resume_validation["val_loss/mean"]) < best_val_loss:
                best_val_loss = float(resume_validation["val_loss/mean"])
                trainer.save_model_checkpoint(run_dir/"best_model.pt", config, resume_validation)
        if distributed: dist.barrier()
    started = time.perf_counter()
    for batch in loader:
        loss=trainer.train_microbatch(batch)
        optimizer_step = trainer.step and trainer.micro_step % trainer.accum == 0
        if optimizer_step and trainer.step % train["log_every_steps"]==0:
            record={"step":trainer.step,"loss":reduce_mean(loss),"seconds_per_step":(time.perf_counter()-started)/train["log_every_steps"]}
            metric_keys = [*(f"loss/{task}" for task in sorted(tasks))]
            if objective == "rectified_flow_matching":
                metric_keys += [f"loss/t_{lower:.2f}_{upper:.2f}" for lower,upper in ((0.,.25),(.25,.5),(.5,.75),(.75,1.))]
            for key in metric_keys:
                value = reduce_optional(trainer.last_metrics, key)
                if value is not None: record[key] = value
            record["grad_norm"] = reduce_mean(trainer.last_metrics["grad_norm"])
            for key in sorted(value for value in trainer.last_metrics if value.startswith("lr/")):
                record[key] = reduce_mean(trainer.last_metrics[key])
            if rank == 0: append_jsonl_log(run_dir/"metrics.jsonl",record)
            started=time.perf_counter()
        if optimizer_step and trainer.step % train["validation_every_steps"]==0:
            record={"step":trainer.step,"phase":"validation",**validate()}
            if rank == 0:
                append_jsonl_log(run_dir/"metrics.jsonl",record)
                if float(record["val_loss/mean"]) < best_val_loss:
                    best_val_loss = float(record["val_loss/mean"])
                    trainer.save_model_checkpoint(run_dir/"best_model.pt", config, record)
            if distributed: dist.barrier()
        model_interval = int(train.get("model_checkpoint_every_steps", 0))
        if optimizer_step and model_interval > 0 and trainer.step % model_interval == 0:
            if rank == 0:
                trainer.save_model_checkpoint(
                    run_dir / f"model_step_{trainer.step:08d}.pt", config,
                    {"step": trainer.step, "phase": "periodic_model_checkpoint"},
                )
            if distributed: dist.barrier()
        full_interval = int(train.get(
            "full_checkpoint_every_steps", train.get("checkpoint_every_steps", 0)
        ))
        if optimizer_step and full_interval > 0 and trainer.step % full_interval == 0:
            checkpoint = run_dir / f"step_{trainer.step:08d}.sharded"
            trainer.save_sharded_checkpoint(checkpoint, config, rank, world_size)
            if rank == 0:
                checkpoints = sorted(run_dir.glob("step_*.sharded"))
                keep = int(train.get("keep_last_checkpoints", 0))
                if keep > 0:
                    for stale in checkpoints[:-keep]:
                        import shutil
                        shutil.rmtree(stale)
        if trainer.step>=limit: break
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__=="__main__": main()
