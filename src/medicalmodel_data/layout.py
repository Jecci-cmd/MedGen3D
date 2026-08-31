from __future__ import annotations

from pathlib import Path


DIRECTORIES = (
    "raw/archives",
    "raw/extracted",
    "metadata",
    "splits",
    "canonical",
    "tasks/segmentation",
    "tasks/segmentation/sdf",
    "tasks/restoration",
    "tasks/reconstruction",
    "processed",
    "processed/cache",
    "processed/manifests",
    "processed/validation_inputs",
    "manifests",
    "qc",
    "qc/overlays",
    "logs",
)


def ensure_layout(root: Path) -> dict[str, Path]:
    root = root.resolve()
    paths = {"root": root}
    for relative in DIRECTORIES:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        paths[relative] = path
    return paths
