#!/usr/bin/env python3
"""Generate held-out test patches from a trained MedGen3D checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config
from medgen3d.data import TASKS, build_task_dataset
from medgen3d.evaluation import (
    delta_z_consistency_error,
    paired_ct_metrics,
    save_segmentation_overlay,
    save_triplanar_ct,
    segmentation_metrics,
    summarize_segmentation_by_class,
    summarize_paired_ct,
    synthesis_metrics,
)
from medgen3d.inference import predict_feed_forward_volume, predict_volume
from medgen3d.wan import (FrozenTextEncoder, FrozenWanVAE, MedicalWanDiT,
                          configure_dit_finetuning, load_official_components)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/main5task_feedforward_h200x8.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_trained_model(checkpoint_dir: Path, checkpoint: Path, device: torch.device,
                       finetuning: dict[str, Any] | None = None) -> tuple[Any, Any, Any]:
    base, official_vae, official_text = load_official_components(
        checkpoint_dir, device=str(device), dtype=torch.bfloat16
    )
    model = MedicalWanDiT(base)
    configure_dit_finetuning(model, finetuning or {"mode": "full"})
    model = model.to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_state = state["model"]
    if model_state and all(key.startswith("module.") for key in model_state):
        model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
    checkpoint_format = state.get("model_format", "full")
    incompatible = model.load_state_dict(model_state, strict=checkpoint_format == "full")
    if checkpoint_format == "trainable_only":
        trainable_names = {name for name, parameter in model.named_parameters()
                           if parameter.requires_grad}
        missing_trainable = trainable_names.intersection(incompatible.missing_keys)
        if incompatible.unexpected_keys or missing_trainable:
            raise RuntimeError(
                f"Invalid adapter checkpoint: unexpected={incompatible.unexpected_keys}, "
                f"missing_trainable={sorted(missing_trainable)}"
            )
    del state, model_state
    model.eval().requires_grad_(False)
    return model, FrozenWanVAE(official_vae), FrozenTextEncoder(official_text)


def denormalize(value: np.ndarray, hu_clip: list[float]) -> np.ndarray:
    low, high = map(float, hu_clip)
    return (np.clip(value, -1.0, 1.0) + 1.0) * 0.5 * (high - low) + low


def valid_crop(valid: np.ndarray) -> tuple[slice, slice, slice]:
    coordinates = np.where(valid)
    if not coordinates[0].size:
        raise ValueError("Evaluation patch has no valid voxels")
    return tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)  # type: ignore[return-value]


def resize_dhw(value: np.ndarray, shape_dhw: tuple[int, int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        tensor, shape_dhw, mode="trilinear", align_corners=False
    )[0, 0].numpy()


def save_original_grid_volume(
    prediction: np.ndarray, metadata: dict[str, Any], task: str, output: Path
) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    inverse = metadata["inverse_transform"]
    shape_xyz = tuple(int(x) for x in inverse["original_shape_xyz"])
    pred_norm = np.clip(resize_dhw(prediction, tuple(reversed(shape_xyz))), -1.0, 1.0)
    target_image = nib.load(inverse["target_path"])
    target_xyz = np.asarray(target_image.dataobj, dtype=np.float32)
    target_dhw = np.moveaxis(target_xyz, (0, 1, 2), (2, 1, 0))
    if task == "synthesis":
        norm = inverse["normalization"]
        low, high = float(norm["lower"]), float(norm["upper"])
        pred_native = (pred_norm + 1.0) * 0.5 * (high - low) + low
        target_metric = np.clip((target_dhw - low) / (high - low), 0.0, 1.0)
        pred_metric = np.clip((pred_norm + 1.0) * 0.5, 0.0, 1.0)
    else:
        pred_native = pred_norm * 1000.0
        target_metric = np.clip(target_dhw, -1000.0, 1000.0) / 1000.0
        pred_metric = pred_norm
    output.parent.mkdir(parents=True, exist_ok=True)
    native_xyz = np.moveaxis(pred_native, (0, 1, 2), (2, 1, 0))
    nib.save(nib.Nifti1Image(native_xyz.astype(np.float32), np.asarray(inverse["original_affine"])), output)
    return pred_metric, target_metric


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.samples_per_task <= 0 or args.sampling_steps <= 0:
        raise SystemExit("samples-per-task and sampling-steps must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires CUDA")

    device = torch.device("cuda", 0)
    config = load_experiment_config(args.config)
    data = config["data"]
    train = config["train"]
    objective = str(train.get("objective", "rectified_flow_matching"))
    model, vae, text_encoder = load_trained_model(
        args.checkpoint_dir, args.checkpoint, device, train.get("finetuning")
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "results.partial.json"
    rows: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if (partial.get("checkpoint") != str(args.checkpoint)
                or partial.get("sampling_steps") != args.sampling_steps
                or partial.get("seed") != args.seed
                or partial.get("num_shards", 1) != args.num_shards
                or partial.get("shard_index", 0) != args.shard_index):
            raise RuntimeError(f"Refusing to resume incompatible partial result: {partial_path}")
        rows = list(partial.get("rows", []))
    completed = {(row["task"], int(row["index"])) for row in rows}
    spacing_zyx = tuple(float(x) for x in reversed(data["target_spacing_xyz_mm"]))

    for task in args.tasks:
        dataset = build_task_dataset(
            data, task, "test", seed=args.seed,
            num_samples=args.samples_per_task, training=False,
        )
        for index in range(args.shard_index, args.samples_per_task, args.num_shards):
            if (task, index) in completed:
                print(json.dumps({"skipped_completed": True, "task": task, "index": index}), flush=True)
                continue
            sample = dataset[index]
            condition = sample["condition"].unsqueeze(0).to(device)
            reconstruction_views = sample.get("metadata", {}).get("reconstruction_views")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if objective == "feed_forward_latent_regression":
                    prediction, _ = predict_feed_forward_volume(
                        condition, sample["prompt"], vae, text_encoder, model,
                        reconstruction_views=reconstruction_views,
                        full_views=int(data.get("reconstruction", {}).get("full_views", 720)),
                        fixed_timestep=float(train.get("fixed_timestep", 0.0)),
                    )
                else:
                    prediction, _ = predict_volume(
                        condition, sample["prompt"], vae, text_encoder, model,
                        steps=args.sampling_steps, seed=args.seed + index, cfg_scale=1.0,
                        reconstruction_views=reconstruction_views,
                        full_views=int(data.get("reconstruction", {}).get("full_views", 720)),
                        residual_reconstruction=(
                            task == "reconstruction" and
                            data.get("reconstruction", {}).get("residual_prediction") == "latent"
                        ),
                    )
            valid = sample["valid_mask"].numpy()[0].astype(bool)
            crop = valid_crop(valid)
            pred = prediction.float().cpu().numpy()[0, 0]
            target = sample["target"].numpy()[0]
            condition_np = sample["condition"].numpy()[0]
            case_id = str(sample["case_id"])
            artifact = output_dir / "artifacts" / task / f"{case_id}_{index:03d}.png"

            if task == "segmentation":
                metrics = segmentation_metrics(pred[crop] < 0, target[crop] < 0, spacing_zyx)
                save_segmentation_overlay(condition_np, pred < 0, target < 0, artifact)
            elif task in {"restoration", "reconstruction"}:
                pred_hu = denormalize(pred, data["hu_clip"])
                target_hu = denormalize(target, data["hu_clip"])
                condition_hu = denormalize(condition_np, data["hu_clip"])
                metrics = paired_ct_metrics(
                    condition_hu[crop], pred_hu[crop], target_hu[crop]
                )
                save_triplanar_ct(condition_hu, pred_hu, target_hu, artifact)
            else:
                volume_path = output_dir / "volumes" / task / f"{case_id}.nii.gz"
                pred_metric, target_metric = save_original_grid_volume(
                    pred, sample["metadata"], task, volume_path
                )
                if task == "synthesis":
                    metrics = synthesis_metrics(pred_metric, target_metric, data_range=1.0)
                else:
                    metrics = {
                        "delta_z_consistency_error": delta_z_consistency_error(
                            pred_metric, target_metric
                        )
                    }
                save_triplanar_ct(condition_np, pred, target, artifact)

            row = {"case_id": case_id, "index": index, "task": task, "metrics": metrics,
                   "artifact": str(artifact), "sampling_steps": args.sampling_steps,
                   "seed": args.seed + index, "objective": objective,
                   "reconstruction_views": reconstruction_views,
                   "structure": sample.get("metadata", {}).get("structure"),
                   "class_id": sample.get("metadata", {}).get("class_id")}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            write_json_atomic(partial_path, {
                "checkpoint": str(args.checkpoint), "samples_per_task": args.samples_per_task,
                "sampling_steps": args.sampling_steps, "seed": args.seed,
                "num_shards": args.num_shards, "shard_index": args.shard_index, "rows": rows,
            })

    summary: dict[str, Any] = {}
    for task in args.tasks:
        task_rows = [row for row in rows if row["task"] == task]
        if task == "segmentation":
            summary[task] = summarize_segmentation_by_class(task_rows)
        elif task in {"restoration", "reconstruction"}:
            summary[task] = summarize_paired_ct(
                [row["metrics"] for row in task_rows], seed=args.seed
            )
        else:
            keys = sorted(task_rows[0]["metrics"])
            summary[task] = {
                key: float(np.mean([row["metrics"][key] for row in task_rows]))
                for key in keys
            }
    payload = {"checkpoint": str(args.checkpoint), "samples_per_task": args.samples_per_task,
               "sampling_steps": args.sampling_steps, "objective": objective,
               "shard_index": args.shard_index, "num_shards": args.num_shards,
               "summary": summary, "rows": rows}
    write_json_atomic(output_dir / "results.json", payload)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
