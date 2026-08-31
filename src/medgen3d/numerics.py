from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VolumePadInfo:
    original_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    depth_pad: tuple[int, int]
    height_pad: tuple[int, int]
    width_pad: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_wan_padding(shape_dhw: tuple[int, int, int]) -> VolumePadInfo:
    d, h, w = shape_dhw
    pd, ph, pw = (1 - d) % 4, (-h) % 32, (-w) % 32
    ht, hb = ph // 2, ph - ph // 2
    wl, wr = pw // 2, pw - pw // 2
    return VolumePadInfo((d, h, w), (d + pd, h + ph, w + pw), (0, pd), (ht, hb), (wl, wr))


def pad_volume(x: torch.Tensor, info: VolumePadInfo, value: float) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
    return F.pad(x, (info.width_pad[0], info.width_pad[1], info.height_pad[0],
                     info.height_pad[1], info.depth_pad[0], info.depth_pad[1]), value=value)


def crop_padding(x: torch.Tensor, info: VolumePadInfo) -> torch.Tensor:
    d, h, w = info.original_shape
    return x[..., :d, info.height_pad[0]:info.height_pad[0] + h,
             info.width_pad[0]:info.width_pad[0] + w]


def normalize_hu(x: torch.Tensor, clip: tuple[float, float]) -> torch.Tensor:
    lo, hi = clip
    return 2.0 * (x.float().clamp(lo, hi) - lo) / (hi - lo) - 1.0


def denormalize_hu(x: torch.Tensor, clip: tuple[float, float]) -> torch.Tensor:
    lo, hi = clip
    return ((x.float() + 1.0) * 0.5 * (hi - lo) + lo).clamp(lo, hi)


def sdf_to_mask(sdf: torch.Tensor, positive_inside: bool = False) -> torch.Tensor:
    return sdf > 0 if positive_inside else sdf < 0
