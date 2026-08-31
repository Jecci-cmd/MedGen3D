from __future__ import annotations

import hashlib
from pathlib import Path


PROMPT_VERSION = "medgen3d-prompts-v1"
TEMPLATES = {
    "segmentation": "Segment the {structure} in this CT volume.",
    "restoration": "Restore this low-dose CT volume.",
    "reconstruction": "Reconstruct this sparse-view CT volume.",
}


def render_prompt(task: str, structure: str | None = None) -> str:
    if task not in TEMPLATES:
        raise KeyError(f"Unknown task: {task}")
    return TEMPLATES[task].format(structure=structure or "specified organ")


def embedding_cache_path(root: str | Path, prompt: str) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
    return Path(root) / PROMPT_VERSION / f"{digest}.pt"

