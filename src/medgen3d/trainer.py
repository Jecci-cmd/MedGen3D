from __future__ import annotations

import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .flow import construct_flow_batch, masked_flow_mse


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("No trainable parameters")
    opt = config["optimizer"]
    if opt["name"].lower() != "adamw":
        raise ValueError(f"Unsupported optimizer {opt['name']}")
    base_lr = float(opt["learning_rate"])
    new_lr = float(opt.get("new_module_learning_rate", base_lr))
    patterns = tuple(opt.get("new_module_patterns", ()))
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in named_params:
        is_new = any(pattern in name for pattern in patterns)
        no_decay = parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower()
        groups.setdefault((is_new, no_decay), []).append(parameter)
    parameter_groups = [
        {"params": params, "lr": new_lr if is_new else base_lr,
         "weight_decay": 0.0 if no_decay else float(opt["weight_decay"]),
         "group_name": ("new" if is_new else "pretrained") + ("_no_decay" if no_decay else "_decay")}
        for (is_new, no_decay), params in groups.items()
    ]
    optimizer_kwargs = {
        "lr": base_lr,
        "betas": tuple(opt["betas"]),
        "eps": opt["eps"],
    }
    if bool(opt.get("zero_redundancy", False)):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("optimizer.zero_redundancy requires an initialized distributed process group")
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            parameter_groups,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
    return torch.optim.AdamW(parameter_groups, **optimizer_kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler = config["scheduler"]; warmup, total = int(scheduler["warmup_steps"]), int(config["max_steps"])
    if scheduler["name"] != "cosine": raise ValueError(f"Unsupported scheduler {scheduler['name']}")
    def factor(step: int) -> float:
        if step < warmup: return max(step, 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class FlowTrainer:
    """Minimal accelerator-agnostic trainer; real 5B distributed launch is external."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 text_encoder: Callable[[list[str], torch.device], Any] | None,
                 config: dict[str, Any], device: torch.device,
                 latent_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 pair_encoder: Callable[..., tuple[torch.Tensor, ...]] | None = None) -> None:
        self.model, self.optimizer, self.text_encoder = model, optimizer, text_encoder
        self.config, self.device, self.latent_encoder = config, device, latent_encoder
        self.scheduler = scheduler
        self.pair_encoder = pair_encoder
        self.step = 0; self.micro_step = 0
        self.accum = int(config.get("gradient_accumulation_steps", 1))
        precision = config.get("precision", "fp32")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
        self.last_metrics: dict[str, float] = {}

    def _encode(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(self.device)
        return self.latent_encoder(value) if self.latent_encoder else value

    def _view_ratio(self, batch: dict[str, Any]) -> torch.Tensor:
        full_views = float(self.config.get("reconstruction_full_views", 720))
        if full_views <= 0:
            raise ValueError("reconstruction_full_views must be positive")
        values = []
        for task, metadata in zip(batch.get("task", []), batch.get("metadata", [])):
            views = metadata.get("reconstruction_views") if task == "reconstruction" else None
            values.append(float(views) / full_views if views is not None else 0.0)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _volume_position(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return normalized [z-start, z-extent] for generation windows.

        Non-generation tasks deliberately receive zeros to preserve their prior
        conditioning behaviour.
        """
        values = []
        for task, metadata in zip(batch.get("task", []), batch.get("metadata", [])):
            if task == "generation":
                values.append((float(metadata.get("z_start_fraction", 0.0)),
                               float(metadata.get("z_extent_fraction", 0.0))))
            else:
                values.append((0.0, 0.0))
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _model_checkpoint_payload(self) -> tuple[dict[str, torch.Tensor], str]:
        """Avoid duplicating frozen 5B weights in parameter-efficient checkpoints."""
        named_parameters = dict(self.model.named_parameters())
        trainable_names = {name for name, parameter in named_parameters.items()
                           if parameter.requires_grad}
        state = self.model.state_dict()
        if len(trainable_names) == len(named_parameters):
            return state, "full"
        trainable_state = {name: value for name, value in state.items() if name in trainable_names}
        if not trainable_state:
            raise RuntimeError("Parameter-efficient model has no trainable checkpoint tensors")
        return trainable_state, "trainable_only"

    def _load_model_checkpoint_payload(self, state: dict[str, Any]) -> None:
        checkpoint_format = state.get("model_format", "full")
        incompatible = self.model.load_state_dict(
            state["model"], strict=checkpoint_format == "full"
        )
        if checkpoint_format == "trainable_only":
            trainable_names = {name for name, parameter in self.model.named_parameters()
                               if parameter.requires_grad}
            # Pre-position-conditioning adapter checkpoints are valid initial
            # states: the new branch is zero-initialized and starts learning on
            # the V2 corpus.
            optional = {name for name in trainable_names if name.startswith("position_embedding.")}
            missing_trainable = (trainable_names - optional).intersection(incompatible.missing_keys)
            if incompatible.unexpected_keys or missing_trainable:
                raise RuntimeError(
                    f"Invalid adapter checkpoint: unexpected={incompatible.unexpected_keys}, "
                    f"missing_trainable={sorted(missing_trainable)}"
                )

    def train_microbatch(self, batch: dict[str, Any], text_context: Any | None = None) -> float:
        self.model.train()
        if self.pair_encoder:
            encoded=self.pair_encoder(batch["condition"].to(self.device),batch["target"].to(self.device),list(batch["task"]),batch.get("valid_mask"))
            zc,z0=encoded[:2]
            if len(encoded)==3: batch["latent_valid_mask"]=encoded[2]
        else:
            zc, z0 = self._encode(batch["condition"]), self._encode(batch["target"])
        flow = construct_flow_batch(z0, zc)
        if text_context is None:
            if self.text_encoder is None:
                raise ValueError("Provide text_context or a frozen text encoder")
            prompts = list(batch["prompt"])
            dropout = float(self.config.get("text_cfg_dropout", 0.0))
            prompts = ["" if torch.rand(()).item() < dropout else p for p in prompts]
            text_context = self.text_encoder(prompts, self.device)
        context = (torch.autocast(self.device.type, dtype=self.autocast_dtype)
                   if self.autocast_dtype is not None else nullcontext())
        with context:
            prediction = self.model(flow.noisy_target, flow.clean_condition, flow.timestep,
                                    text_context, self._view_ratio(batch), self._volume_position(batch))
            mask = batch.get("latent_valid_mask", batch.get("valid_mask")) if self.config.get("use_padding_loss_mask", True) else None
            if mask is not None: mask=mask.to(self.device)
            if mask is not None and mask.shape[-3:] != prediction.shape[-3:]:
                mask = torch.nn.functional.interpolate(mask.float().to(self.device), size=prediction.shape[-3:], mode="nearest")
            loss = masked_flow_mse(prediction, flow.velocity_target, mask) / self.accum
        self.scaler.scale(loss).backward()
        self.micro_step += 1
        raw_loss = float(loss.detach()) * self.accum
        metrics = {"loss": raw_loss}
        for task in sorted(set(batch.get("task", []))):
            indices = [index for index, value in enumerate(batch["task"]) if value == task]
            metrics[f"loss/{task}"] = float(masked_flow_mse(
                prediction[indices], flow.velocity_target[indices],
                mask[indices] if mask is not None else None,
            ).detach())
        for lower, upper in ((0.0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
            selected = (flow.timestep >= lower) & (flow.timestep <= upper if upper == 1.0 else flow.timestep < upper)
            if selected.any():
                metrics[f"loss/t_{lower:.2f}_{upper:.2f}"] = float(masked_flow_mse(
                    prediction[selected], flow.velocity_target[selected],
                    mask[selected] if mask is not None else None,
                ).detach())
        if self.micro_step % self.accum == 0:
            self.scaler.unscale_(self.optimizer)
            max_norm = float(self.config.get("max_grad_norm", 0.0))
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm or float("inf"))
            metrics["grad_norm"] = float(grad_norm.detach())
            self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None: self.scheduler.step()
            self.step += 1
            for index, group in enumerate(self.optimizer.param_groups):
                metrics[f"lr/{group.get('group_name', index)}"] = float(group["lr"])
        self.last_metrics = metrics
        return raw_loss

    @torch.no_grad()
    def validation_microbatch(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.eval()
        if self.pair_encoder:
            encoded = self.pair_encoder(batch["condition"].to(self.device), batch["target"].to(self.device),
                                        list(batch["task"]), batch.get("valid_mask"))
            zc, z0 = encoded[:2]
            mask = encoded[2] if len(encoded) == 3 else batch.get("valid_mask")
        else:
            zc, z0 = self._encode(batch["condition"]), self._encode(batch["target"])
            mask = batch.get("valid_mask")
        flow = construct_flow_batch(z0, zc)
        if self.text_encoder is None:
            raise ValueError("Validation requires a text encoder")
        text_context = self.text_encoder(list(batch["prompt"]), self.device)
        context = (torch.autocast(self.device.type, dtype=self.autocast_dtype)
                   if self.autocast_dtype is not None else nullcontext())
        with context:
            prediction = self.model(flow.noisy_target, flow.clean_condition, flow.timestep,
                                    text_context, self._view_ratio(batch), self._volume_position(batch))
        if mask is not None:
            mask = mask.to(self.device)
            if mask.shape[-3:] != prediction.shape[-3:]:
                mask = torch.nn.functional.interpolate(mask.float(), size=prediction.shape[-3:], mode="nearest")
        return {"loss": float(masked_flow_mse(prediction, flow.velocity_target, mask))}

    def save_checkpoint(self, path: str | Path, resolved_config: dict[str, Any]) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        try:
            model_state, model_format = self._model_checkpoint_payload()
            torch.save({"model": model_state, "model_format": model_format,
                        "optimizer": self.optimizer.state_dict(),
                        "scaler": self.scaler.state_dict(), "scheduler": self.scheduler.state_dict() if self.scheduler else None, "step": self.step,
                        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
                        "micro_step": self.micro_step, "config": resolved_config}, temporary)
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def save_model_checkpoint(
        self, path: str | Path, resolved_config: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        """Atomically save inference weights for the best validation step."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        try:
            model_state, model_format = self._model_checkpoint_payload()
            torch.save({"model": model_state, "model_format": model_format, "step": self.step,
                        "config": resolved_config, "validation": metrics}, temporary)
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def save_portable_checkpoint(
        self, path: str | Path, resolved_config: dict[str, Any], rank: int = 0
    ) -> None:
        """Consolidate ZeRO state so a different DDP world size can resume."""
        consolidate = getattr(self.optimizer, "consolidate_state_dict", None)
        if consolidate is not None:
            consolidate(to=0)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            try:
                model_state, model_format = self._model_checkpoint_payload()
                torch.save({
                    "format": "medgen3d-portable-zero-v1",
                    "model": model_state,
                    "model_format": model_format,
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                    "step": self.step,
                    "micro_step": self.micro_step,
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "python": random.getstate()},
                    "config": resolved_config,
                }, temporary)
                temporary.replace(path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            self.load_sharded_checkpoint(path)
            return
        # Distributed optimizer checkpoints contain the consolidated AdamW
        # state. Load on CPU first so every rank does not transiently place
        # the full tens-of-GiB state on its GPU before ZeRO takes its shard.
        # Portable checkpoints are ~60 GiB. mmap keeps the read-only storage
        # file-backed and shared across local ranks while ZeRO copies only the
        # optimizer states owned by each rank to its GPU.
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        self._load_model_checkpoint_payload(state); self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"]); self.step = int(state["step"])
        self.micro_step = int(state.get("micro_step", self.step * self.accum))
        if self.scheduler is not None and state.get("scheduler") is not None: self.scheduler.load_state_dict(state["scheduler"])
        torch_rng_state = state["rng"]["torch"]
        if not isinstance(torch_rng_state, torch.Tensor):
            torch_rng_state = torch.as_tensor(torch_rng_state, dtype=torch.uint8)
        torch.set_rng_state(torch_rng_state.detach().to(device="cpu", dtype=torch.uint8))
        np.random.set_state(state["rng"]["numpy"]); random.setstate(state["rng"]["python"])

    @staticmethod
    def _optimizer_local_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        local_optimizer = getattr(optimizer, "optim", None)
        return local_optimizer.state_dict() if local_optimizer is not None else optimizer.state_dict()

    @staticmethod
    def _load_optimizer_local_state(optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> None:
        local_optimizer = getattr(optimizer, "optim", None)
        if local_optimizer is not None:
            local_optimizer.load_state_dict(state)
        else:
            optimizer.load_state_dict(state)

    def save_sharded_checkpoint(
        self, path: str | Path, resolved_config: dict[str, Any], rank: int, world_size: int
    ) -> None:
        """Atomically save one optimizer shard per rank and one replicated model copy."""
        path = Path(path)
        temporary = path.with_name(path.name + ".partial")
        if rank == 0:
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        rng = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        rank_payload: dict[str, Any] = {
            "optimizer_local": self._optimizer_local_state(self.optimizer),
            "rng": rng,
            "rank": rank,
            "world_size": world_size,
            "step": self.step,
            "micro_step": self.micro_step,
        }
        if rank == 0:
            model_state, model_format = self._model_checkpoint_payload()
            rank_payload.update({
                "model": model_state,
                "model_format": model_format,
                "scaler": self.scaler.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "config": resolved_config,
            })
        shard = temporary / f"rank_{rank:05d}.pt"
        shard_tmp = shard.with_suffix(".pt.tmp")
        torch.save(rank_payload, shard_tmp)
        shard_tmp.replace(shard)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            manifest = {
                "format": "medgen3d-ddp-zero-local-v1",
                "step": self.step,
                "world_size": world_size,
                "rank_files": [f"rank_{value:05d}.pt" for value in range(world_size)],
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            (temporary / "_SUCCESS").write_text("complete\n")
            if path.exists():
                shutil.rmtree(path)
            temporary.replace(path)
        if dist.is_initialized():
            dist.barrier()

    def load_sharded_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not (path / "_SUCCESS").is_file():
            raise ValueError(f"Incomplete sharded checkpoint: {path}")
        manifest = json.loads((path / "manifest.json").read_text())
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if int(manifest["world_size"]) != world_size:
            raise ValueError(
                f"Checkpoint world size {manifest['world_size']} does not match current {world_size}"
            )
        shared = torch.load(path / "rank_00000.pt", map_location="cpu", weights_only=False)
        local = shared if rank == 0 else torch.load(
            path / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False
        )
        self._load_model_checkpoint_payload(shared)
        self._load_optimizer_local_state(self.optimizer, local["optimizer_local"])
        self.scaler.load_state_dict(shared["scaler"])
        if self.scheduler is not None and shared.get("scheduler") is not None:
            self.scheduler.load_state_dict(shared["scheduler"])
        self.step = int(shared["step"])
        self.micro_step = int(shared.get("micro_step", self.step * self.accum))
        rng = local["rng"]
        torch.set_rng_state(torch.as_tensor(rng["torch"], dtype=torch.uint8, device="cpu"))
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state(torch.as_tensor(rng["cuda"], dtype=torch.uint8, device="cpu"))
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])


class FeedForwardTrainer(FlowTrainer):
    """One-pass latent regression while retaining shared task/view conditioning."""

    def _prepare_pair(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.pair_encoder:
            encoded = self.pair_encoder(
                batch["condition"].to(self.device), batch["target"].to(self.device),
                list(batch["task"]), batch.get("valid_mask"),
            )
            zc, z0 = encoded[:2]
            mask = encoded[2] if len(encoded) == 3 else batch.get("valid_mask")
        else:
            zc, z0 = self._encode(batch["condition"]), self._encode(batch["target"])
            mask = batch.get("valid_mask")
        return zc, z0, mask

    def _predict(self, zc: torch.Tensor, text_context: Any, batch: dict[str, Any]) -> torch.Tensor:
        timestep = torch.full(
            (zc.shape[0],), float(self.config.get("fixed_timestep", 0.0)),
            device=zc.device, dtype=torch.float32,
        )
        return self.model(zc, torch.zeros_like(zc), timestep, text_context,
                          self._view_ratio(batch), self._volume_position(batch))

    def train_microbatch(self, batch: dict[str, Any], text_context: Any | None = None) -> float:
        self.model.train()
        zc, z0, mask = self._prepare_pair(batch)
        if text_context is None:
            if self.text_encoder is None:
                raise ValueError("Provide text_context or a frozen text encoder")
            prompts = list(batch["prompt"])
            dropout = float(self.config.get("text_cfg_dropout", 0.0))
            prompts = ["" if torch.rand(()).item() < dropout else value for value in prompts]
            text_context = self.text_encoder(prompts, self.device)
        context = (torch.autocast(self.device.type, dtype=self.autocast_dtype)
                   if self.autocast_dtype is not None else nullcontext())
        with context:
            prediction = self._predict(zc, text_context, batch)
            if mask is not None:
                mask = mask.to(self.device)
                if mask.shape[-3:] != prediction.shape[-3:]:
                    mask = torch.nn.functional.interpolate(
                        mask.float(), size=prediction.shape[-3:], mode="nearest"
                    )
            loss = masked_flow_mse(prediction, z0, mask) / self.accum
        self.scaler.scale(loss).backward()
        self.micro_step += 1
        raw_loss = float(loss.detach()) * self.accum
        metrics = {"loss": raw_loss}
        for task in sorted(set(batch.get("task", []))):
            indices = [index for index, value in enumerate(batch["task"]) if value == task]
            metrics[f"loss/{task}"] = float(masked_flow_mse(
                prediction[indices], z0[indices], mask[indices] if mask is not None else None
            ).detach())
        if self.micro_step % self.accum == 0:
            self.scaler.unscale_(self.optimizer)
            max_norm = float(self.config.get("max_grad_norm", 0.0))
            metrics["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm or float("inf")
            ).detach())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.step += 1
            for index, group in enumerate(self.optimizer.param_groups):
                metrics[f"lr/{group.get('group_name', index)}"] = float(group["lr"])
        self.last_metrics = metrics
        return raw_loss

    @torch.no_grad()
    def validation_microbatch(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.eval()
        zc, z0, mask = self._prepare_pair(batch)
        if self.text_encoder is None:
            raise ValueError("Validation requires a text encoder")
        text_context = self.text_encoder(list(batch["prompt"]), self.device)
        context = (torch.autocast(self.device.type, dtype=self.autocast_dtype)
                   if self.autocast_dtype is not None else nullcontext())
        with context:
            prediction = self._predict(zc, text_context, batch)
        if mask is not None:
            mask = mask.to(self.device)
            if mask.shape[-3:] != prediction.shape[-3:]:
                mask = torch.nn.functional.interpolate(mask.float(), size=prediction.shape[-3:], mode="nearest")
        return {"loss": float(masked_flow_mse(prediction, z0, mask))}


def append_jsonl_log(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")
