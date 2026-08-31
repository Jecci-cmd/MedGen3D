from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlowBatch:
    noisy_target: torch.Tensor
    clean_condition: torch.Tensor
    velocity_target: torch.Tensor
    timestep: torch.Tensor
    noise: torch.Tensor


def construct_flow_batch(z0: torch.Tensor, zc: torch.Tensor, generator: torch.Generator | None = None,
                         timestep: torch.Tensor | None = None) -> FlowBatch:
    if z0.shape != zc.shape:
        raise ValueError(f"z0 and zc differ: {z0.shape} vs {zc.shape}")
    if not z0.is_floating_point() or not zc.is_floating_point():
        raise TypeError("Flow latents must be floating point")
    if not torch.isfinite(z0).all() or not torch.isfinite(zc).all():
        raise ValueError("Flow latents contain NaN/Inf")
    b = z0.shape[0]
    t = timestep if timestep is not None else torch.rand(b, device=z0.device, dtype=torch.float32, generator=generator)
    if t.shape != (b,) or torch.any((t < 0) | (t > 1)):
        raise ValueError("timestep must have shape [B] and lie in [0,1]")
    noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    broadcast_t = t.to(z0.dtype).view(b, *([1] * (z0.ndim - 1)))
    zt = (1 - broadcast_t) * z0 + broadcast_t * noise
    return FlowBatch(zt, zc, noise - z0, t, noise)


def masked_flow_mse(prediction: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    error = (prediction.float() - target.float()).square()
    if mask is None:
        loss = error.mean()
    else:
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(1)
        mask = torch.broadcast_to(mask.to(error.dtype), error.shape)
        denominator = mask.sum()
        if denominator <= 0:
            raise ValueError("Loss mask has no valid elements")
        loss = (error * mask).sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("Flow matching loss is NaN/Inf")
    return loss

