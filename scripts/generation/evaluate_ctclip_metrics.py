#!/usr/bin/env python3
"""Evaluate report-conditioned CT generation with CT-CLIP features.

The script consumes the ``results.json`` produced by
``scripts/evaluate_medgen3d.py``.  It computes three whole-volume metrics on
the fixed CT-CLIP preprocessing grid:

* ``fvd_ctclip``: Fréchet distance between real and generated CT-CLIP image
  embeddings (lower is better);
* ``ctclip_t2i``: report-to-generated-volume cosine similarity x100 (higher
  is better);
* ``ctclip_i2i``: real-to-generated-volume cosine similarity x100 (higher
  is better).

CT-CLIP is an external dependency, so its checkout and checkpoint are explicit
arguments instead of machine-specific hard-coded paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from transformers import BertModel


PROTOCOL = "ctrate_v2_ctclip_fvd_t2i_i2i_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CTCLIP_ASSETS = REPOSITORY_ROOT / "evaluation_assets" / "ctclip"
DEFAULT_CTCLIP_ROOT = DEFAULT_CTCLIP_ASSETS / "CT-CLIP"
# Shared evaluation asset; do not fall back to an arbitrary local CLIP weight.
DEFAULT_CTCLIP_CHECKPOINT = Path(
    "/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main/"
    "evaluation_assets/ctclip/CT-CLIP_v2.pt"
)
DEFAULT_CXR_BERT = Path(
    "/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main/"
    "evaluation_assets/ctclip/BiomedVLP-CXR-BERT-specialized"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Generation results.json from evaluate_medgen3d.py")
    parser.add_argument("--metadata-csv", type=Path, required=True,
                        help="CT-RATE metadata CSV containing spacing and rescale fields")
    parser.add_argument("--ctclip-root", type=Path, default=DEFAULT_CTCLIP_ROOT,
                        help="CT-CLIP checkout (default: repository evaluation_assets/ctclip/CT-CLIP)")
    parser.add_argument("--ctclip-checkpoint", type=Path, default=DEFAULT_CTCLIP_CHECKPOINT,
                        help="CT-CLIP model checkpoint (default: fixed shared evaluation asset)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return parser.parse_args()


def volume_keys(value: str | Path) -> set[str]:
    """Return robust matching keys for CSV volume names and result case IDs."""
    name = Path(str(value)).name
    stem = name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return {name, stem, f"{stem}.nii", f"{stem}.nii.gz"}


def metadata_by_volume(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Metadata CSV is empty: {path}")
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("VolumeName")
        if not name:
            raise KeyError("Metadata CSV requires a VolumeName column")
        for key in volume_keys(name):
            metadata[key] = row
    return metadata


def parse_xy_spacing(value: str) -> float:
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", str(value))
    if not numbers:
        raise ValueError(f"Could not parse XYSpacing={value!r}")
    return float(numbers[0])


def lookup_metadata(metadata: dict[str, dict[str, str]], row: dict[str, Any]) -> dict[str, str]:
    for candidate in (row.get("case_id"), row.get("target_volume"), row.get("target")):
        if candidate is None:
            continue
        for key in volume_keys(str(candidate)):
            if key in metadata:
                return metadata[key]
    raise KeyError(f"No metadata entry for generation case {row.get('case_id')!r}")


def import_ctclip(root: Path) -> tuple[type[Any], Any, Any]:
    paths = (root / "CT_CLIP", root / "transformer_maskgit", root / "scripts")
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("CT-CLIP checkout is incomplete; missing " + ", ".join(missing))
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    from ct_clip import CTCLIP  # type: ignore[import-not-found]
    from data import resize_array  # type: ignore[import-not-found]
    from transformer_maskgit import CTViT  # type: ignore[import-not-found]
    return CTCLIP, CTViT, resize_array


def ctclip_tensor(path: Path, *, slope: float, intercept: float, z_spacing: float,
                  xy_spacing: float, resize_array: Any) -> torch.Tensor:
    """Apply the CT-CLIP evaluation preprocessing exactly once per volume."""
    array = nib.load(str(path)).get_fdata().astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D NIfTI, got {array.shape}: {path}")
    array = array * slope + intercept
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0).float()
    array = resize_array(tensor, (z_spacing, xy_spacing, xy_spacing), (1.5, 0.75, 0.75))[0][0]
    array = np.clip(np.transpose(array, (1, 2, 0)), -1000.0, 1000.0) / 1000.0
    tensor = torch.from_numpy(array.astype(np.float32))

    target_hwd = (480, 480, 240)
    slices = []
    pads: list[int] = []
    for size, target in zip(tensor.shape, target_hwd):
        start = max((size - target) // 2, 0)
        end = min(start + target, size)
        slices.append(slice(start, end))
        before = (target - (end - start)) // 2
        pads.extend((before, target - (end - start) - before))
    tensor = tensor[tuple(slices)]
    # F.pad consumes dimensions in reverse order: D, W, H.
    h_before, h_after, w_before, w_after, d_before, d_after = pads
    padding = (d_before, d_after, w_before, w_after, h_before, h_after)
    return F.pad(tensor, padding, value=-1.0).permute(2, 0, 1).unsqueeze(0)


def build_model(device: torch.device, checkpoint: Path, CTCLIP: type[Any], CTViT: Any) -> Any:
    if not (DEFAULT_CXR_BERT / "config.json").is_file():
        raise FileNotFoundError(f"Missing fixed BiomedVLP CXR-BERT asset: {DEFAULT_CXR_BERT}")
    text_encoder = BertModel.from_pretrained(str(DEFAULT_CXR_BERT), local_files_only=True).to(device)
    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
                          temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
                          dim_head=32, heads=8).to(device)
    model = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder,
                   dim_image=294_912, dim_text=768, dim_latent=512).to(device)
    model.load(str(checkpoint))
    return model.eval()


def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("FVD-CT requires at least two volumes")
    mu_real, mu_generated = real.mean(axis=0), generated.mean(axis=0)
    cov_real, cov_generated = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    covmean = sqrtm(cov_real @ cov_generated)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    difference = mu_real - mu_generated
    return float(difference @ difference + np.trace(cov_real + cov_generated - 2.0 * covmean))


def clean_prompt(text: str) -> str:
    return str(text).replace('"', "").replace("'", "").replace("(", "").replace(")", "")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CT-CLIP evaluation")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("task") == "generation"]
    if len(rows) != args.expected_samples or len({str(row.get("case_id")) for row in rows}) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique generation cases, found {len(rows)}")
    if any(not str(row.get("prompt", "")).strip() for row in rows):
        raise RuntimeError("Every generation row must contain a non-empty report prompt")

    device = torch.device(args.device)
    CTCLIP, CTViT, resize_array = import_ctclip(args.ctclip_root)
    model = build_model(device, args.ctclip_checkpoint, CTCLIP, CTViT)
    metadata = metadata_by_volume(args.metadata_csv)
    real_features, generated_features, details = [], [], []

    for index, row in enumerate(rows):
        prediction = Path(row["volume"])
        target = Path(row["target_volume"])
        if not prediction.is_file() or not target.is_file():
            raise FileNotFoundError((prediction, target))
        record = lookup_metadata(metadata, row)
        z_spacing = float(record["ZSpacing"])
        xy_spacing = parse_xy_spacing(record["XYSpacing"])
        prompt = clean_prompt(str(row["prompt"]))
        tokens = model.tokenizer(prompt, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=512).to(device)
        real = ctclip_tensor(target, slope=float(record["RescaleSlope"]),
                             intercept=float(record["RescaleIntercept"]), z_spacing=z_spacing,
                             xy_spacing=xy_spacing, resize_array=resize_array).unsqueeze(0).to(device)
        generated = ctclip_tensor(prediction, slope=1.0, intercept=0.0, z_spacing=z_spacing,
                                  xy_spacing=xy_spacing, resize_array=resize_array).unsqueeze(0).to(device)
        with torch.inference_mode():
            text_latent, real_latent, _ = model(tokens, real, return_latents=True, device=device)
            _, generated_latent, _ = model(tokens, generated, return_latents=True, device=device)
        t2i = float(100.0 * torch.sum(generated_latent * text_latent, dim=-1).item())
        i2i = float(100.0 * torch.sum(generated_latent * real_latent, dim=-1).item())
        real_features.append(real_latent.detach().float().cpu().numpy()[0])
        generated_features.append(generated_latent.detach().float().cpu().numpy()[0])
        details.append({"index": index, "case_id": str(row["case_id"]), "prediction": str(prediction),
                        "target": str(target), "ctclip_t2i": t2i,
                        "ctclip_t2i_clamped": max(t2i, 0.0), "ctclip_i2i": i2i,
                        "ctclip_mean": (max(t2i, 0.0) + i2i) / 2.0})
        print(json.dumps({"done": index + 1, "case_id": row["case_id"], "ctclip_t2i": t2i,
                          "ctclip_i2i": i2i}), flush=True)
        del real, generated, real_latent, generated_latent, text_latent
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"fvd_ctclip": frechet_distance(np.stack(real_features).astype(np.float64),
                                                np.stack(generated_features).astype(np.float64)),
               "ctclip_t2i": float(np.mean([detail["ctclip_t2i"] for detail in details])),
               "ctclip_t2i_clamped": float(np.mean([detail["ctclip_t2i_clamped"] for detail in details])),
               "ctclip_i2i": float(np.mean([detail["ctclip_i2i"] for detail in details])),
               "ctclip_mean": float(np.mean([detail["ctclip_mean"] for detail in details]))}
    result = {"protocol": PROTOCOL, "samples": len(details), "summary": summary, "rows": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
