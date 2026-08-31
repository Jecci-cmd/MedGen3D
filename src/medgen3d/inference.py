from __future__ import annotations

from typing import Any, Callable

import torch


@torch.no_grad()
def euler_flow_sample(model: Callable[..., torch.Tensor], condition_latent: torch.Tensor,
                      text_context: Any, steps: int = 30, seed: int = 0,
                      cfg_scale: float = 1.0, empty_text_context: Any | None = None,
                      view_ratio: torch.Tensor | None = None,
                      volume_position: torch.Tensor | None = None,
                      callback: Callable[[int, torch.Tensor], None] | None = None) -> torch.Tensor:
    """Integrate from noise at t=1 to data at t=0 using explicit Euler."""
    if steps <= 0: raise ValueError("steps must be positive")
    if cfg_scale != 1.0 and empty_text_context is None:
        raise ValueError("Text CFG requires empty_text_context")
    generator = torch.Generator(device=condition_latent.device).manual_seed(seed)
    z = torch.randn(condition_latent.shape, device=condition_latent.device,
                    dtype=condition_latent.dtype, generator=generator)
    dt = -1.0 / steps
    for index in range(steps):
        t = torch.full((z.shape[0],), 1.0 - index / steps, device=z.device)
        velocity = model(z, condition_latent, t, text_context, view_ratio, volume_position)
        if cfg_scale != 1.0:
            unconditional = model(z, condition_latent, t, empty_text_context, view_ratio, volume_position)
            velocity = unconditional + cfg_scale * (velocity - unconditional)
        z = z + dt * velocity
        if callback: callback(index, z)
    return z


@torch.no_grad()
def predict_volume(condition: torch.Tensor, prompt: str, vae: Any, text_encoder: Any,
                   model: Any, steps: int = 30, seed: int = 0, cfg_scale: float = 1.0,
                   reconstruction_views: int | None = None, full_views: int = 720,
                   residual_reconstruction: bool = False,
                   volume_position: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    zc, pad_info = vae.encode_volume(condition)
    context = text_encoder([prompt], zc.device)
    empty = text_encoder([""], zc.device) if cfg_scale != 1.0 else None
    view_ratio = (torch.tensor([reconstruction_views / full_views], device=zc.device)
                  if reconstruction_views is not None else torch.zeros(1, device=zc.device))
    prediction_latent = euler_flow_sample(
        model, zc, context, steps, seed, cfg_scale, empty, view_ratio=view_ratio,
        volume_position=volume_position,
    )
    if residual_reconstruction:
        prediction_latent = zc + prediction_latent
    return vae.decode_volume(prediction_latent, pad_info), prediction_latent


@torch.no_grad()
def predict_feed_forward_volume(
    condition: torch.Tensor, prompt: str, vae: Any, text_encoder: Any, model: Any,
    reconstruction_views: int | None = None, full_views: int = 720,
    fixed_timestep: float = 0.0,
    volume_position: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the GenCeption-style one-pass latent-regression formulation."""
    zc, pad_info = vae.encode_volume(condition)
    context = text_encoder([prompt], zc.device)
    view_ratio = (torch.tensor([reconstruction_views / full_views], device=zc.device)
                  if reconstruction_views is not None else torch.zeros(1, device=zc.device))
    timestep = torch.full((zc.shape[0],), fixed_timestep, device=zc.device, dtype=torch.float32)
    prediction_latent = model(zc, torch.zeros_like(zc), timestep, context, view_ratio, volume_position)
    return vae.decode_volume(prediction_latent, pad_info), prediction_latent


@torch.no_grad()
def predict_feed_forward_sliding_volume(
    condition: torch.Tensor, prompt: str, vae: Any, text_encoder: Any, model: Any,
    window_depth: int, stride: int, fixed_timestep: float = 0.0,
) -> torch.Tensor:
    """Predict a complete volume through overlapping, position-conditioned z windows."""
    if condition.ndim != 5 or condition.shape[0] != 1:
        raise ValueError("Expected one full volume shaped [1,C,D,H,W]")
    if window_depth <= 0 or stride <= 0:
        raise ValueError("window_depth and stride must be positive")
    depth = int(condition.shape[-3])
    starts = list(range(0, max(depth - window_depth, 0) + 1, stride))
    final_start = max(depth - window_depth, 0)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    output = torch.zeros_like(condition)
    weights = torch.zeros_like(condition)
    for start in starts:
        patch = condition[..., start:start + window_depth, :, :]
        available = patch.shape[-3]
        if available < window_depth:
            patch = torch.nn.functional.pad(patch, (0, 0, 0, 0, 0, window_depth - available), value=-1.0)
        position = torch.tensor(
            [[start / max(depth - window_depth, 1), min(window_depth, depth) / max(depth, 1)]],
            device=condition.device, dtype=torch.float32,
        )
        prediction, _ = predict_feed_forward_volume(
            patch, prompt, vae, text_encoder, model, fixed_timestep=fixed_timestep,
            volume_position=position,
        )
        output[..., start:start + available, :, :] += prediction[..., :available, :, :]
        weights[..., start:start + available, :, :] += 1
    return output / weights.clamp_min(1)


@torch.no_grad()
def sliding_window_predict(volume: torch.Tensor, predictor: Callable[[torch.Tensor, tuple[int, int, int]], torch.Tensor],
                           window_dhw: tuple[int, int, int], overlap: float = 0.25) -> torch.Tensor:
    """Blend aligned volume predictions; predictor receives patch and start DHW."""
    if volume.ndim != 5 or not 0 <= overlap < 1: raise ValueError("Invalid volume or overlap")
    spatial=volume.shape[-3:]; strides=[max(1,int(size*(1-overlap))) for size in window_dhw]
    starts=[]
    for available,size,stride in zip(spatial,window_dhw,strides):
        axis=list(range(0,max(available-size,0)+1,stride)); last=max(available-size,0)
        if not axis or axis[-1]!=last: axis.append(last)
        starts.append(axis)
    output=torch.zeros_like(volume); weight=torch.zeros_like(volume)
    for d in starts[0]:
        for h in starts[1]:
            for w in starts[2]:
                slices=(slice(d,d+window_dhw[0]),slice(h,h+window_dhw[1]),slice(w,w+window_dhw[2]))
                patch=volume[(...,*slices)]; prediction=predictor(patch,(d,h,w))
                if prediction.shape!=patch.shape: raise ValueError("Sliding-window predictor changed patch shape")
                output[(...,*slices)]+=prediction; weight[(...,*slices)]+=1
    return output/weight.clamp_min(1)
