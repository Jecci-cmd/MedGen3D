from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .numerics import VolumePadInfo, compute_wan_padding, crop_padding, pad_volume


class LoRALinear(nn.Module):
    """Parameter-efficient residual update for an existing linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.dropout(value))) * self.scaling
        return self.base(value) + residual.to(self.base.weight.dtype)


def inject_all_linear_lora(module: nn.Module, rank: int, alpha: float,
                           dropout: float = 0.0) -> list[str]:
    """Replace every existing linear descendant with LoRA, exactly once."""

    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(module)
    if not replaced:
        raise ValueError("LoRA all-linear mode found no nn.Linear modules in the DiT")
    return replaced


def download_official_checkpoint(model_id: str, revision: str, local_dir: str | Path, token: str | None = None) -> Path:
    """Download an immutable official snapshot; credentials are never persisted."""
    if not revision:
        raise ValueError("A pinned Hugging Face revision is required")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=model_id, revision=revision, local_dir=local_dir, token=token))


def load_official_components(checkpoint_dir: str | Path, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> tuple[Any, Any, Any]:
    """Load the official DiT, VAE and UMT5 classes from a checked-out Wan repo."""
    force_fa2 = os.environ.get("MEDGEN3D_FORCE_FLASH_ATTN2", "1") != "0"
    if force_fa2:
        # The source image exposes an FA3 egg even though its runtime is not
        # compatible with this workload.  Wan auto-prefers FA3 whenever the
        # module imports, so explicitly make that optional import unavailable
        # before Wan's attention module is loaded.  It then takes its official
        # FA2 branch without patching third-party source files.
        sys.modules["flash_attn_interface"] = None
    try:
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE
        if force_fa2:
            from wan.modules import attention as wan_attention
            if wan_attention.FLASH_ATTN_3_AVAILABLE or not wan_attention.FLASH_ATTN_2_AVAILABLE:
                raise RuntimeError("MEDGEN3D_FORCE_FLASH_ATTN2 failed to select FlashAttention 2")
    except ImportError as exc:
        raise RuntimeError("Official Wan2.2 repository must be importable as `wan`") from exc
    root = Path(checkpoint_dir)
    dit = WanModel.from_pretrained(root)
    vae = Wan2_2_VAE(vae_pth=root / "Wan2.2_VAE.pth", device=device, dtype=dtype)
    text = T5EncoderModel(text_len=512, dtype=dtype, device=torch.device(device),
                          checkpoint_path=root / "models_t5_umt5-xxl-enc-bf16.pth",
                          tokenizer_path=root / "google/umt5-xxl")
    text.model.eval().requires_grad_(False)
    return dit, vae, text


def load_official_vae(checkpoint_dir: str | Path, device: str = "cpu",
                      dtype: torch.dtype = torch.bfloat16) -> Any:
    """Load only the VAE for representation checks without allocating the 5B DiT/T5."""
    try:
        from wan.modules.vae2_2 import Wan2_2_VAE
    except ImportError as exc:
        raise RuntimeError("Official Wan2.2 repository must be importable as `wan`") from exc
    return Wan2_2_VAE(vae_pth=Path(checkpoint_dir) / "Wan2.2_VAE.pth", device=device, dtype=dtype)


class FrozenWanVAE(nn.Module):
    def __init__(self, official_vae: Any) -> None:
        super().__init__()
        self.official_vae = official_vae
        self.model = official_vae.model
        self.model.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenWanVAE":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_volume(self, volume: torch.Tensor, pad_value: float = -1.0) -> tuple[torch.Tensor, VolumePadInfo]:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(f"Expected [B,1,D,H,W], got {tuple(volume.shape)}")
        if not torch.isfinite(volume).all() or volume.min() < -1.01 or volume.max() > 1.01:
            raise ValueError("VAE input must be finite and normalized to approximately [-1,1]")
        info = compute_wan_padding(tuple(volume.shape[-3:]))
        rgb = pad_volume(volume, info, pad_value).repeat(1, 3, 1, 1, 1)
        encoded = self.official_vae.encode([item for item in rgb])
        if encoded is None:
            raise RuntimeError("Official Wan VAE encode failed")
        return torch.stack(encoded), info

    @torch.no_grad()
    def decode_volume(self, latent: torch.Tensor, pad_info: VolumePadInfo) -> torch.Tensor:
        decoded = self.official_vae.decode([item for item in latent])
        if decoded is None:
            raise RuntimeError("Official Wan VAE decode failed")
        rgb = torch.stack(decoded)
        return crop_padding(rgb.mean(dim=1, keepdim=True), pad_info)


class MedicalVolumeCodec:
    """Task-aware numerical boundary around the shared frozen Wan VAE."""

    def __init__(self, vae: FrozenWanVAE) -> None:
        self.vae = vae

    @torch.no_grad()
    def encode_pair(self, condition: torch.Tensor, target: torch.Tensor,
                    task: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, VolumePadInfo]:
        if condition.shape != target.shape or len(task) != condition.shape[0]:
            raise ValueError("Pair geometry and task batch must agree")
        zc, info = self.vae.encode_volume(condition, pad_value=-1.0)
        target_latents=[]
        for index, name in enumerate(task):
            # CT air is -1; negative-inside SDF exterior is +1.
            target_pad = 1.0 if name == "segmentation" else -1.0
            z0, target_info = self.vae.encode_volume(target[index:index + 1], pad_value=target_pad)
            if target_info != info: raise AssertionError("Condition/target padding differs")
            target_latents.append(z0[0])
        return zc, torch.stack(target_latents), info


class FrozenTextEncoder(nn.Module):
    def __init__(self, official_encoder: Any) -> None:
        super().__init__()
        self.encoder = official_encoder
        self.encoder.model.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenTextEncoder":
        super().train(False)
        self.encoder.model.eval()
        return self

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], device: torch.device) -> list[torch.Tensor]:
        return self.encoder(list(prompts), device)


def _sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.to(torch.float64)
    omega = torch.pow(10000, -torch.arange(half, device=position.device).div(half))
    return torch.cat([torch.cos(torch.outer(position, omega)), torch.sin(torch.outer(position, omega))], dim=1)


class MedicalWanDiT(nn.Module):
    """Wan DiT with a distinct, zero-initialized full-volume condition path.

    This class intentionally has no reference_image/first-frame argument. Native
    Wan I2V remains in official inference code and is never reused here.
    """

    conditioning_mode = "full_volume"

    def __init__(self, base: nn.Module, latent_channels: int | None = None,
                 timestep_scale: float = 1000.0, view_fourier_bands: int = 8) -> None:
        super().__init__()
        self.base = base
        channels = latent_channels or int(base.in_dim)
        if channels != int(base.in_dim):
            raise ValueError(f"Observed latent channels {channels} != Wan in_dim {base.in_dim}")
        self.condition_patch_embedding = nn.Conv3d(channels, base.dim,
                                                    kernel_size=base.patch_size,
                                                    stride=base.patch_size, bias=False)
        nn.init.zeros_(self.condition_patch_embedding.weight)
        if view_fourier_bands < 1:
            raise ValueError("view_fourier_bands must be positive")
        self.register_buffer("view_frequencies", 2.0 ** torch.arange(view_fourier_bands), persistent=False)
        self.view_embedding = nn.Sequential(
            nn.Linear(2 * view_fourier_bands + 2, base.dim),
            nn.SiLU(),
            nn.Linear(base.dim, base.dim),
        )
        nn.init.zeros_(self.view_embedding[-1].weight)
        nn.init.zeros_(self.view_embedding[-1].bias)
        self.timestep_scale = float(timestep_scale)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, noisy_target_latent: torch.Tensor, condition_latent: torch.Tensor,
                timestep: torch.Tensor, text_context: Sequence[torch.Tensor],
                view_ratio: torch.Tensor | None = None) -> torch.Tensor:
        if noisy_target_latent.shape != condition_latent.shape:
            raise ValueError("Target and condition latent shapes must match exactly")
        if noisy_target_latent.ndim != 5:
            raise ValueError("Latents must have shape [B,C,D,H,W]")
        b = noisy_target_latent.shape[0]
        target = self.base.patch_embedding(noisy_target_latent)
        condition = self.condition_patch_embedding(condition_latent.to(target.dtype))
        if target.shape != condition.shape:
            raise AssertionError("Condition and target token grids differ")
        grid = torch.tensor(target.shape[2:], device=target.device, dtype=torch.long).repeat(b, 1)
        x = (target + condition).flatten(2).transpose(1, 2)
        seq_len = x.shape[1]
        seq_lens = torch.full((b,), seq_len, device=x.device, dtype=torch.long)
        timestep = timestep * self.timestep_scale
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(b, seq_len)
        with torch.amp.autocast("cuda", enabled=False):
            flat_t = timestep.flatten()
            e = self.base.time_embedding(_sinusoidal_embedding(self.base.freq_dim, flat_t).float()).unflatten(0, (b, seq_len))
            if view_ratio is None:
                view_ratio = torch.zeros(b, device=timestep.device)
            view_ratio = view_ratio.to(device=timestep.device, dtype=torch.float32).flatten()
            if view_ratio.numel() != b:
                raise ValueError(f"Expected {b} view ratios, got {view_ratio.numel()}")
            active = (view_ratio > 0).float()
            angles = 2 * math.pi * view_ratio[:, None] * self.view_frequencies[None].float()
            view_features = torch.cat([
                torch.sin(angles), torch.cos(angles), view_ratio[:, None], active[:, None]
            ], dim=1)
            view_condition = self.view_embedding(view_features).to(e.dtype)
            e = e + view_condition[:, None, :]
            e0 = self.base.time_projection(e).unflatten(2, (6, self.base.dim))
        context = self.base.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.base.text_len - u.shape[0], u.shape[1])]) for u in text_context
        ]))
        freqs = self.base.freqs.to(x.device)
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid, freqs=freqs,
                      context=context, context_lens=None)
        for block in self.base.blocks:
            if getattr(self, "gradient_checkpointing", False) and self.training:
                x = checkpoint(block, x, use_reentrant=False, **kwargs)
            else:
                x = block(x, **kwargs)
        x = self.base.head(x, e)
        return torch.stack(self.base.unpatchify(x, grid))

    def medical_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "condition_patch_embedding.weight": self.condition_patch_embedding.weight.detach().cpu(),
            **{f"view_embedding.{key}": value.detach().cpu()
               for key, value in self.view_embedding.state_dict().items()},
        }


def configure_dit_finetuning(model: MedicalWanDiT, config: dict[str, Any]) -> list[str]:
    """Select full fine-tuning or all-linear LoRA without changing model semantics."""

    mode = str(config.get("mode", "full")).lower()
    if mode == "full":
        model.requires_grad_(True)
        return []
    if mode != "lora_all_linear":
        raise ValueError(f"Unsupported fine-tuning mode: {mode}")
    model.requires_grad_(False)
    replaced = inject_all_linear_lora(
        model.base,
        rank=int(config.get("rank", 16)),
        alpha=float(config.get("alpha", 32.0)),
        dropout=float(config.get("dropout", 0.0)),
    )
    # These branches are specific to MedGen3D and have no useful frozen
    # pretrained counterpart, so train them together with the adapters.
    model.condition_patch_embedding.requires_grad_(True)
    model.view_embedding.requires_grad_(True)
    return [f"base.{name}" for name in replaced]


def assert_zero_condition_equivalence(model: MedicalWanDiT) -> None:
    if torch.count_nonzero(model.condition_patch_embedding.weight).item() != 0:
        raise AssertionError("Condition branch is not zero-initialized")
