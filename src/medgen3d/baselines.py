from __future__ import annotations

import torch
from torch import nn


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.LeakyReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.LeakyReLU(inplace=True))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256), residual_output: bool = True) -> None:
        super().__init__(); self.residual_output = residual_output
        self.down = nn.ModuleList(); previous = in_channels
        for c in channels: self.down.append(_Block(previous, c)); previous = c
        self.pool = nn.MaxPool3d(2); self.up = nn.ModuleList()
        for c, skip in zip(reversed(channels[:-1]), reversed(channels[:-1])):
            self.up.append(nn.ModuleList([nn.ConvTranspose3d(previous, c, 2, 2), _Block(c + skip, c)])); previous = c
        self.head = nn.Conv3d(previous, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips=[]; h=x
        for i, block in enumerate(self.down):
            h=block(h)
            if i < len(self.down)-1: skips.append(h); h=self.pool(h)
        for (upsample, block), skip in zip(self.up, reversed(skips)):
            h=block(torch.cat([upsample(h), skip], dim=1))
        out=self.head(h)
        return x + out if self.residual_output and x.shape[1] == out.shape[1] else out

