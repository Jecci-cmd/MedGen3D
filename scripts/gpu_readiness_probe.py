#!/usr/bin/env python3
"""Finite 8-GPU NCCL/bf16 readiness probe while formal data is finalized."""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--matrix-size", type=int, default=8192)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import flash_attn

    if not flash_attn.__version__.startswith("2."):
        raise RuntimeError(f"Expected Flash-Attn 2, found {flash_attn.__version__}")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name:
        raise RuntimeError(f"Expected H200, found {name}")
    generator = torch.Generator(device=device).manual_seed(20260823 + rank)
    a = torch.randn(args.matrix_size, args.matrix_size, device=device,
                    dtype=torch.bfloat16, generator=generator)
    b = torch.randn_like(a)
    torch.cuda.synchronize()
    started = time.monotonic(); iterations = 0
    while time.monotonic() - started < args.seconds:
        c = a @ b
        a, b = b, c.mul_(1.0 / args.matrix_size)
        iterations += 1
        if iterations % 20 == 0:
            check = torch.tensor([float(iterations)], device=device)
            dist.all_reduce(check)
    torch.cuda.synchronize()
    if rank == 0:
        print({"status": "PASS", "world_size": dist.get_world_size(),
               "gpu": name, "flash_attn": flash_attn.__version__,
               "seconds": time.monotonic() - started, "iterations_per_rank": iterations},
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
