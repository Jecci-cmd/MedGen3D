from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity


SEGMENTATION_METRIC_KEYS = ("dice", "nsd", "hd95_mm", "assd_mm")


def canonical_segmentation_class(value: Any) -> str:
    """Normalize dataset/baseline organ names for per-class comparisons."""
    label = str(value or "unknown").strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "kidney right": "Right kidney",
        "right kidney": "Right kidney",
        "kidney left": "Left kidney",
        "left kidney": "Left kidney",
        "gall bladder": "Gallbladder",
        "gallbladder": "Gallbladder",
        "postcava": "Inferior vena cava",
        "inferior vena cava": "Inferior vena cava",
        "ivc": "Inferior vena cava",
    }
    if label in aliases:
        return aliases[label]
    return label.capitalize()


def summarize_segmentation_by_class(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate case-level segmentation metrics overall and by target organ."""
    if not rows:
        raise ValueError("At least one segmentation row is required")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        class_name = canonical_segmentation_class(
            row.get("structure") or row.get("organ") or row.get("class_name") or row.get("class_id")
        )
        grouped.setdefault(class_name, []).append(row)

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key in SEGMENTATION_METRIC_KEYS:
            values = np.asarray([float(row["metrics"][key]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            metrics[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "finite_mean": float(finite.mean()) if len(finite) else None,
                "nonfinite_count": int(len(values) - len(finite)),
            }
        return {
            "num_cases": len(selected),
            "empty_prediction_count": int(sum(
                bool(row["metrics"].get("empty_prediction", False)) for row in selected
            )),
            "empty_prediction_rate": float(np.mean([
                bool(row["metrics"].get("empty_prediction", False)) for row in selected
            ])),
            "metrics": metrics,
        }

    return {
        "overall": aggregate(rows),
        "by_class": {name: aggregate(grouped[name]) for name in sorted(grouped)},
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) & ~binary_erosion(mask.astype(bool))


def segmentation_metrics(pred: np.ndarray, target: np.ndarray, spacing_zyx: tuple[float, float, float], nsd_tolerance_mm: float = 1.0) -> dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    denom = pred.sum() + target.sum()
    dice = 1.0 if denom == 0 else 2.0 * np.logical_and(pred, target).sum() / denom
    if not pred.any() or not target.any():
        distance = 0.0 if not pred.any() and not target.any() else float("inf")
        return {
            "dice": float(dice), "nsd": float(dice), "hd95_mm": distance,
            "assd_mm": distance, "empty_prediction": not pred.any(),
        }
    ps, ts = _surface(pred), _surface(target)
    d_to_t = distance_transform_edt(~ts, sampling=spacing_zyx)[ps]
    d_to_p = distance_transform_edt(~ps, sampling=spacing_zyx)[ts]
    distances = np.concatenate([d_to_t, d_to_p])
    nsd = ((d_to_t <= nsd_tolerance_mm).sum() + (d_to_p <= nsd_tolerance_mm).sum()) / (len(d_to_t) + len(d_to_p))
    return {
        "dice": float(dice), "nsd": float(nsd), "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float(distances.mean()), "empty_prediction": False,
    }


def ct_metrics(pred_hu: np.ndarray, target_hu: np.ndarray, data_range_hu: float = 2000.0) -> dict[str, float]:
    pred, target = pred_hu.astype(np.float64), target_hu.astype(np.float64)
    error = pred - target
    mse = float(np.mean(error ** 2)); rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 20 * math.log10(data_range_hu) - 10 * math.log10(mse)
    values = [structural_similarity(target[z], pred[z], data_range=data_range_hu) for z in range(target.shape[0])]
    return {"mae_hu": float(np.mean(np.abs(error))), "rmse_hu": rmse, "psnr_hu": psnr, "ssim": float(np.mean(values))}


def synthesis_metrics(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    """Paired MRI synthesis metrics in the restored original-volume grid."""
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    error = pred - target
    mse = float(np.mean(error ** 2))
    slice_ssim = [structural_similarity(target[z], pred[z], data_range=data_range)
                  for z in range(target.shape[0])]
    return {
        "mae": float(np.mean(np.abs(error))),
        "psnr": float("inf") if mse == 0 else 20 * math.log10(data_range) - 10 * math.log10(mse),
        "ssim": float(np.mean(slice_ssim)),
    }


def paired_ct_metrics(
    condition_hu: np.ndarray,
    prediction_hu: np.ndarray,
    target_hu: np.ndarray,
    data_range_hu: float = 2000.0,
) -> dict[str, dict[str, float]]:
    """Compare degraded input and model output against one paired target."""
    input_metrics = ct_metrics(condition_hu, target_hu, data_range_hu)
    model_metrics = ct_metrics(prediction_hu, target_hu, data_range_hu)
    lower_is_better = {"mae_hu", "rmse_hu"}
    improvement = {
        key: float(input_metrics[key] - model_metrics[key]
                   if key in lower_is_better else model_metrics[key] - input_metrics[key])
        for key in input_metrics
    }
    return {"input": input_metrics, "model": model_metrics, "improvement": improvement}


def summarize_paired_ct(
    records: list[dict[str, dict[str, float]]],
    seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Summarize paired CT metrics with patient-level bootstrap intervals."""
    if not records:
        raise ValueError("At least one paired CT record is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {"num_cases": len(records), "metrics": {}}
    for key in records[0]["input"]:
        input_values = np.asarray([row["input"][key] for row in records], dtype=np.float64)
        model_values = np.asarray([row["model"][key] for row in records], dtype=np.float64)
        improvements = np.asarray([row["improvement"][key] for row in records], dtype=np.float64)
        indices = rng.integers(0, len(records), size=(bootstrap_samples, len(records)))
        bootstrap_means = improvements[indices].mean(axis=1)
        summary["metrics"][key] = {
            "input_mean": float(input_values.mean()),
            "input_median": float(np.median(input_values)),
            "model_mean": float(model_values.mean()),
            "model_median": float(np.median(model_values)),
            "improvement_mean": float(improvements.mean()),
            "improvement_median": float(np.median(improvements)),
            "improvement_std": float(improvements.std(ddof=1)) if len(records) > 1 else 0.0,
            "improvement_fraction": float(np.mean(improvements > 0)),
            "improvement_ci95": [float(x) for x in np.percentile(bootstrap_means, [2.5, 97.5])],
        }
    return summary


def aggregate_by_patient(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows: grouped.setdefault(str(row["case_id"]), []).append(row["metrics"])
    return {case: {key: float(np.mean([r[key] for r in records])) for key in records[0]} for case, records in grouped.items()}


def save_triplanar_ct(condition: np.ndarray, prediction: np.ndarray, target: np.ndarray, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    d, h, w = target.shape
    views = [(d // 2, 0), (h // 2, 1), (w // 2, 2)]
    arrays = [condition, prediction, target, np.abs(prediction - target)]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row, (index, axis) in enumerate(views):
        for col, array in enumerate(arrays):
            image = np.take(array, index, axis=axis)
            axes[row, col].imshow(image, cmap="magma" if col == 3 else "gray")
            axes[row, col].axis("off")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_segmentation_overlay(ct: np.ndarray, prediction: np.ndarray, target: np.ndarray, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    d, h, w = ct.shape; slices = [(d // 2, 0), (h // 2, 1), (w // 2, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (index, axis) in zip(axes, slices):
        image = np.take(ct, index, axis=axis); gt = np.take(target, index, axis=axis); pr = np.take(prediction, index, axis=axis)
        ax.imshow(image, cmap="gray"); ax.contour(gt, levels=[0.5], colors="lime"); ax.contour(pr, levels=[0.5], colors="red"); ax.axis("off")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
