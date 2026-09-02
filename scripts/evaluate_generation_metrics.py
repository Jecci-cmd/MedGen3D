#!/usr/bin/env python3
"""Backward-compatible entry point for generation metric evaluation.

The implementation lives in ``scripts/generation`` with the other
generation-specific utilities.
"""
from generation.evaluate_generation_metrics import main


if __name__ == "__main__":
    main()
