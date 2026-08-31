from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Compose a small Hydra-like defaults list without adding a dependency."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config: dict[str, Any] = {}
    for default in raw.pop("defaults", []):
        config = _merge(config, load_experiment_config(path.parent / default))
    config = _merge(config, raw)
    config["_provenance"] = {
        "config_path": str(path),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git_commit": git_commit(path.parent),
        "git_dirty": git_is_dirty(path.parent),
    }
    return config


def git_commit(start: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=start, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-or-no-git-repository"


def git_is_dirty(start: Path) -> bool:
    try:
        return bool(subprocess.run(["git", "status", "--porcelain", "--", str(start)], cwd=start,
                                   check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def save_resolved_config(config: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
