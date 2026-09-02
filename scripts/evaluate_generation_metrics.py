#!/usr/bin/env python3
"""Entry point for the MONAI MAISI-compatible CT generation evaluator.

The previous StyleGAN-V FID/FVD/CLIPScore protocol was retired because its
natural-video features are not comparable to 3-D CT-generation literature.
``generation.evaluate_maisi_fid_2p5d`` implements the official MAISI 2.5-D
RadImageNet FID protocol and is the only supported generation metric here.
"""
from generation.evaluate_maisi_fid_2p5d import main


if __name__ == "__main__":
    main()
