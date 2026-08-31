from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medgen3d.config import load_experiment_config
from medgen3d.data import DynamicCaseDataset, TaskRatioSampler, UnifiedCTDataset, unified_collate
from medgen3d.evaluation import (
    ct_metrics,
    paired_ct_metrics,
    segmentation_metrics,
    summarize_paired_ct,
    summarize_segmentation_by_class,
)
from medgen3d.flow import construct_flow_batch, masked_flow_mse
from medgen3d.inference import euler_flow_sample
from medgen3d.numerics import compute_wan_padding, crop_padding, pad_volume
from medgen3d.trainer import FlowTrainer, build_optimizer
from medgen3d.wan import (LoRALinear, MedicalWanDiT, assert_zero_condition_equivalence,
                          configure_dit_finetuning)


def test_configs_compose_and_record_provenance() -> None:
    config = load_experiment_config(
        Path(__file__).parents[1]
        / "configs/experiments/main5task_feedforward_h200x8.yaml"
    )
    assert config["data"]["patch_size_dhw"] == [97, 96, 96]
    assert config["data"]["sdf"] == {
        "clip_distance_mm": 8.0,
        "positive_inside": False,
        "foreground_rule": "sdf < 0",
        "normalized_range": [-1.0, 1.0],
        "version": "sdf-negative-inside-clip8-v3",
    }
    assert config["data"]["segmentation_foreground_warmup_probability"] == .65
    assert config["data"]["segmentation_surface_warmup_probability"] == .35
    assert config["data"]["segmentation_foreground_probability"] == .50
    assert config["data"]["segmentation_surface_probability"] == .50
    assert config["data"]["segmentation_organ_sampling"] == "balanced_round_robin"
    assert config["data"]["segmentation_case_sampling"] == "shuffled_cycle"
    assert config["data"]["segmentation_zoom"] == {
        "probability": .30,
        "scale_zyx": {
            "gall_bladder": [1.2, 1.2, 1.2],
            "pancreas": [1.2, 1.2, 1.2],
            "aorta": [1.0, 1.2, 1.2],
            "postcava": [1.0, 1.2, 1.2],
        },
    }
    assert config["data"]["segmentation_surface_band_mm"] == 2.0
    assert config["model"]["conditioning_mode"] == "full_volume"
    assert config["train"]["task_sampling_ratio"] == {
        "segmentation": 1.0,
        "restoration": 1.0,
        "reconstruction": 1.0,
        "synthesis": 1.0,
        "generation": 1.0,
    }
    assert len(config["_provenance"]["config_sha256"]) == 64


def test_segmentation_summary_is_grouped_by_canonical_class() -> None:
    rows = [
        {"structure": "kidney_right", "metrics": {"dice": .8, "nsd": .4, "hd95_mm": 5., "assd_mm": 2.}},
        {"organ": "right kidney", "metrics": {"dice": 0., "nsd": 0., "hd95_mm": 15., "assd_mm": 6.}},
        {"structure": "aorta", "metrics": {"dice": .9, "nsd": .5, "hd95_mm": 3., "assd_mm": 1.}},
    ]
    summary = summarize_segmentation_by_class(rows)
    assert set(summary["by_class"]) == {"Aorta", "Right kidney"}
    kidney = summary["by_class"]["Right kidney"]
    assert kidney["num_cases"] == 2
    assert kidney["zero_dice_cases"] == 1
    assert kidney["metrics"]["dice"]["mean"] == pytest.approx(.4)
    assert summary["overall"]["metrics"]["dice"]["mean"] == pytest.approx(1.7 / 3)


def _make_manifest(root: Path) -> Path:
    records=[]
    for index, task in enumerate(("segmentation", "restoration", "reconstruction")):
        condition=np.full((1, 4, 8, 8), index / 10, np.float32); target=condition + .1
        np.savez(root / f"{task}.npz", condition=condition, target=target)
        records.append({"sample_id": task, "case_id": f"case{index}", "split": "train", "task": task,
                        "condition": {"path": f"{task}.npz", "key": "condition"},
                        "target": {"path": f"{task}.npz", "key": "target"}, "prompt": task,
                        "spacing_xyz_mm": [1.5]*3, "degradation": {}})
    manifest=root / "train.jsonl"; manifest.write_text("\n".join(json.dumps(x) for x in records)+"\n")
    (root / "train.txt").write_text("case0\ncase1\ncase2\n")
    return manifest


def test_unified_dataset_contract_alignment_and_split(tmp_path: Path) -> None:
    manifest=_make_manifest(tmp_path)
    dataset=UnifiedCTDataset(tmp_path, manifest, "train", "train.txt", spatial_flip_probability=1, seed=7)
    batch=unified_collate([dataset[0], dataset[1], dataset[2]])
    assert batch["condition"].shape == batch["target"].shape == (3, 1, 4, 8, 8)
    assert batch["condition"].dtype == torch.float32
    torch.testing.assert_close(batch["target"] - batch["condition"], torch.full_like(batch["target"], .1))
    assert set(batch["case_id"]) == {"case0", "case1", "case2"}


def test_task_balanced_sampler_is_not_dataset_size_weighted(tmp_path: Path) -> None:
    dataset=UnifiedCTDataset(tmp_path, _make_manifest(tmp_path), "train")
    sampler=TaskRatioSampler(dataset.records, {task: 1 for task in ("segmentation", "restoration", "reconstruction")}, 6000, seed=2)
    counts={task:0 for task in sampler.tasks}
    for index in sampler: counts[dataset.records[index]["task"]] += 1
    assert all(1800 < value < 2200 for value in counts.values())


def test_balanced_segmentation_cycles_organs_and_cases_without_replacement(
    tmp_path: Path,
) -> None:
    rows = []
    for case_index in range(3):
        rows.append({
            "case_id": f"case{case_index}",
            "split": "train",
            "image": f"canonical/case{case_index}/ct_hu.npy",
            "mask": f"canonical/case{case_index}/label_id.npy",
            "sdf": f"tasks/segmentation/sdf/case{case_index}.npy",
            "ldct": [],
            "sparse_view_ct": [],
            "sdf_classes": [1, 2],
            "label_map": {"1": "aorta", "2": "kidney"},
            "spacing_xyz_mm": [1.0, 1.0, 1.0],
            "metadata": {"canonical": {"organ_volume_mm3_canonical": {
                "1": 1000.0,
                "2": 1000.0 if case_index < 2 else 0.0,
            }}},
        })
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    kwargs = dict(
        task_weights={"segmentation": 1.0},
        segmentation_organ_sampling="balanced_round_robin",
        segmentation_case_sampling="shuffled_cycle",
        segmentation_min_foreground_voxels=100,
        spatial_flip_probability=0,
        seed=17,
        num_samples=12,
    )
    dataset = DynamicCaseDataset(tmp_path, manifest, "train", (6, 6, 6), **kwargs)
    pairs = [dataset._balanced_segmentation_pair(index)[:2] for index in range(12)]
    classes = [class_id for _, class_id in pairs]
    assert all(set(classes[start:start + 2]) == {1, 2} for start in range(0, 12, 2))
    assert classes.count(1) == classes.count(2) == 6
    for class_id, expected_pool_size in ((1, 3), (2, 2)):
        case_indices = [record_index for record_index, value in pairs if value == class_id]
        assert len(set(case_indices[:expected_pool_size])) == expected_pool_size
        assert len(set(case_indices[expected_pool_size:2 * expected_pool_size])) == expected_pool_size
    duplicate = DynamicCaseDataset(tmp_path, manifest, "train", (6, 6, 6), **kwargs)
    assert pairs == [duplicate._balanced_segmentation_pair(index)[:2] for index in range(12)]


def test_balanced_segmentation_pair_drives_prompt_and_target(tmp_path: Path) -> None:
    rows = []
    for case_index in range(2):
        case_dir = tmp_path / "canonical" / f"case{case_index}"
        sdf_dir = tmp_path / "tasks" / "segmentation" / "sdf"
        case_dir.mkdir(parents=True)
        sdf_dir.mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 8), dtype=np.int16)
        label = np.zeros((8, 8, 8), dtype=np.uint8)
        label[1:4, 1:4, 1:4] = 1
        label[4:7, 4:7, 4:7] = 2
        np.save(case_dir / "ct_hu.npy", image)
        np.save(case_dir / "label_id.npy", label)
        np.save(sdf_dir / f"case{case_index}.npy", np.stack([
            np.where(label == class_id, .5, -.5).astype(np.float16)
            for class_id in (1, 2)
        ]))
        rows.append({
            "case_id": f"case{case_index}", "split": "train",
            "image": f"canonical/case{case_index}/ct_hu.npy",
            "mask": f"canonical/case{case_index}/label_id.npy",
            "sdf": f"tasks/segmentation/sdf/case{case_index}.npy",
            "ldct": [], "sparse_view_ct": [], "sdf_classes": [1, 2],
            "sdf_positive_inside": True,
            "label_map": {"1": "aorta", "2": "kidney"},
        })
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = DynamicCaseDataset(
        tmp_path, manifest, "train", (6, 6, 6),
        task_weights={"segmentation": 1.0}, num_samples=4, seed=23,
        segmentation_organ_sampling="balanced_round_robin",
        segmentation_foreground_probability=1.0,
        segmentation_center_jitter_zyx=(0, 0, 0),
        segmentation_min_foreground_voxels=1,
        spatial_flip_probability=0,
    )
    samples = [dataset[index] for index in range(4)]
    assert {sample["metadata"]["class_id"] for sample in samples[:2]} == {1, 2}
    assert all(sample["metadata"]["organ_sampling_strategy"] == "balanced_round_robin"
               for sample in samples)
    assert all(sample["metadata"]["structure"] in sample["prompt"] for sample in samples)
    assert all(sample["metadata"]["target_foreground_voxels"] >= 1 for sample in samples)


def test_aorta_zoom_preserves_depth_and_recomputes_sdf(tmp_path: Path) -> None:
    case_dir = tmp_path / "canonical" / "case0"
    sdf_dir = tmp_path / "tasks" / "segmentation" / "sdf"
    case_dir.mkdir(parents=True)
    sdf_dir.mkdir(parents=True)
    image = np.zeros((12, 12, 12), dtype=np.int16)
    label = np.zeros((12, 12, 12), dtype=np.uint8)
    label[2:10, 4:8, 4:8] = 1
    np.save(case_dir / "ct_hu.npy", image)
    np.save(case_dir / "label_id.npy", label)
    np.save(sdf_dir / "case0.npy", np.where(
        label[None] == 1, .5, -.5
    ).astype(np.float16))
    row = {
        "case_id": "case0", "split": "train",
        "image": "canonical/case0/ct_hu.npy",
        "mask": "canonical/case0/label_id.npy",
        "sdf": "tasks/segmentation/sdf/case0.npy",
        "ldct": [], "sparse_view_ct": [], "sdf_classes": [1],
        "sdf_positive_inside": True, "label_map": {"1": "aorta"},
        "spacing_xyz_mm": [1.5, 1.5, 1.5],
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    dataset = DynamicCaseDataset(
        tmp_path, manifest, "train", (9, 6, 6),
        task_weights={"segmentation": 1.0}, num_samples=1, seed=29,
        segmentation_foreground_probability=1.0,
        segmentation_center_jitter_zyx=(0, 0, 0),
        segmentation_min_foreground_voxels=1,
        segmentation_zoom={
            "probability": 1.0,
            "scale_zyx": {"aorta": [1.0, 1.2, 1.2]},
        },
        spatial_flip_probability=0,
    )
    sample = dataset[0]
    metadata = sample["metadata"]
    assert sample["condition"].shape == sample["target"].shape == (1, 9, 6, 6)
    assert metadata["segmentation_zoom_applied"]
    assert metadata["segmentation_zoom_scale_zyx"] == (1.0, 1.2, 1.2)
    assert metadata["segmentation_source_patch_shape_zyx"] == (9, 5, 5)
    assert metadata["segmentation_zoom_sdf_recomputed"]
    assert sample["target"].min() < 0 < sample["target"].max()


def test_dynamic_case_dataset_online_pair_processing(tmp_path: Path) -> None:
    import nibabel as nib

    case = tmp_path / "canonical" / "case0"; case.mkdir(parents=True)
    shape = (8, 10, 12)  # XYZ
    ct = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) - 1000
    label = np.zeros(shape, np.uint8); label[2:6, 3:8, 4:10] = 1
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    nib.save(nib.Nifti1Image(ct, affine), case / "ct_hu.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), case / "label_id.nii.gz")
    task = tmp_path / "tasks"; (task / "segmentation" / "sdf").mkdir(parents=True)
    (task / "restoration" / "case0").mkdir(parents=True)
    (task / "reconstruction" / "case0").mkdir(parents=True)
    sdf = np.where(np.transpose(label, (2, 1, 0))[None] == 1, 0.5, -0.5).astype(np.float16)
    np.save(task / "segmentation" / "sdf" / "case0.npy", sdf)
    nib.save(nib.Nifti1Image(ct + 10, affine), task / "restoration" / "case0" / "source_hu.nii.gz")
    nib.save(nib.Nifti1Image(ct + 20, affine), task / "reconstruction" / "case0" / "source_hu.nii.gz")
    row = {"case_id":"case0", "split":"train", "image":"canonical/case0/ct_hu.nii.gz",
           "mask":"canonical/case0/label_id.nii.gz", "sdf":"tasks/segmentation/sdf/case0.npy",
           "ldct":["tasks/restoration/case0/source_hu.nii.gz"],
           "sparse_view_ct":["tasks/reconstruction/case0/source_hu.nii.gz"],
           "sdf_classes":[1], "sdf_positive_inside":True, "label_map":{"1":"organ"}}
    manifest = tmp_path / "train.jsonl"; manifest.write_text(json.dumps(row) + "\n")
    dataset = DynamicCaseDataset(tmp_path, manifest, "train", (6, 8, 8),
                                 task_weights={"segmentation":1}, spatial_flip_probability=0, seed=3,
                                 segmentation_center_jitter_zyx=(0, 0, 0),
                                 segmentation_min_foreground_voxels=1)
    sample = dataset[0]
    assert sample["condition"].shape == sample["target"].shape == (1, 6, 8, 8)
    assert sample["valid_mask"].shape == (1, 6, 8, 8)
    assert sample["task"] == "segmentation" and "organ" in sample["prompt"]
    assert sample["target"].min() < 0 < sample["target"].max()


def test_segmentation_prompt_class_is_present_in_every_training_patch(tmp_path: Path) -> None:
    import nibabel as nib
    from medicalmodel_data.geometry import label_to_sdf

    case = tmp_path / "canonical" / "case0"; case.mkdir(parents=True)
    shape = (20, 20, 20)
    ct = np.zeros(shape, np.int16)
    label = np.zeros(shape, np.uint8)
    label[1:4, 1:4, 1:4] = 1
    label[15:19, 15:19, 15:19] = 2
    nib.save(nib.Nifti1Image(ct, np.eye(4)), case / "ct_hu.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), case / "label_id.nii.gz")
    sdf_dir = tmp_path / "tasks" / "segmentation" / "sdf"; sdf_dir.mkdir(parents=True)
    label_zyx = np.transpose(label, (2, 1, 0))
    np.save(sdf_dir / "case0.npy", label_to_sdf(
        label_zyx, (1.0, 1.0, 1.0), [1, 2], 16.0, positive_inside=True
    ))
    row = {"case_id":"case0", "split":"train", "image":"canonical/case0/ct_hu.nii.gz",
           "mask":"canonical/case0/label_id.nii.gz", "sdf":"tasks/segmentation/sdf/case0.npy",
           "ldct":[], "sparse_view_ct":[], "sdf_classes":[1, 2],
           "sdf_positive_inside":True, "label_map":{"1":"left", "2":"right"}}
    manifest = tmp_path / "train.jsonl"; manifest.write_text(json.dumps(row) + "\n")
    dataset = DynamicCaseDataset(
        tmp_path, manifest, "train", (6, 6, 6), task_weights={"segmentation":1},
        spatial_flip_probability=0, seed=11, num_samples=200,
        segmentation_foreground_probability=.50,
        segmentation_surface_probability=.50,
        segmentation_foreground_warmup_probability=.65,
        segmentation_surface_warmup_probability=.35,
        segmentation_foreground_warmup_fraction=.7,
        segmentation_center_jitter_zyx=(0, 0, 0),
        segmentation_surface_center_jitter_zyx=(0, 0, 0),
        segmentation_surface_band_mm=2.0,
        segmentation_sdf_clip_mm=8.0,
        segmentation_min_foreground_voxels=1,
    )
    seen = set(); modes = {
        "target_foreground_centered": 0,
        "target_surface_centered": 0,
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        if index < 140:
            assert sample["metadata"]["segmentation_sampling_phase"] == "foreground_warmup"
            assert sample["metadata"]["segmentation_foreground_probability"] == .65
            assert sample["metadata"]["segmentation_surface_probability"] == .35
            assert sample["metadata"]["segmentation_random_probability"] == 0.0
        else:
            assert sample["metadata"]["segmentation_sampling_phase"] == "mixed"
            assert sample["metadata"]["segmentation_foreground_probability"] == .50
            assert sample["metadata"]["segmentation_surface_probability"] == .50
            assert sample["metadata"]["segmentation_random_probability"] == 0.0
        seen.add(sample["metadata"]["class_id"])
        modes[sample["metadata"]["sampling_mode"]] += 1
        if sample["metadata"]["sampling_mode"] == "target_surface_centered":
            center = tuple(
                int(start) + 3 for start in sample["metadata"]["crop_start_zyx"]
            )
            channel = [1, 2].index(sample["metadata"]["class_id"])
            assert abs(float(np.load(dataset.root / row["sdf"])[channel][center])) < 2.0 / 16.0
        assert sample["metadata"]["foreground_voxels"] >= 1
        assert sample["metadata"]["target_present"]
        assert sample["metadata"]["structure"] in sample["prompt"]
    assert seen == {1, 2}
    assert 105 <= modes["target_foreground_centered"] <= 135
    assert 65 <= modes["target_surface_centered"] <= 95
    assert sum(modes.values()) == 200


def test_absent_segmentation_class_is_never_sampled(tmp_path: Path) -> None:
    import nibabel as nib

    case = tmp_path / "canonical" / "case0"; case.mkdir(parents=True)
    label = np.zeros((8, 8, 8), np.uint8); label[2:6, 2:6, 2:6] = 2
    nib.save(nib.Nifti1Image(np.zeros_like(label, dtype=np.int16), np.eye(4)), case / "ct_hu.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), case / "label_id.nii.gz")
    sdf_dir = tmp_path / "tasks" / "segmentation" / "sdf"; sdf_dir.mkdir(parents=True)
    label_zyx = np.transpose(label, (2, 1, 0))
    np.save(sdf_dir / "case0.npy", np.stack([
        np.full_like(label_zyx, -.5, dtype=np.float16),
        np.where(label_zyx == 2, .5, -.5).astype(np.float16),
    ]))
    row = {"case_id":"case0", "split":"train", "image":"canonical/case0/ct_hu.nii.gz",
           "mask":"canonical/case0/label_id.nii.gz", "sdf":"tasks/segmentation/sdf/case0.npy",
           "ldct":[], "sparse_view_ct":[], "sdf_classes":[1, 2],
           "sdf_positive_inside":True, "label_map":{"1":"absent", "2":"present"}}
    manifest = tmp_path / "train.jsonl"; manifest.write_text(json.dumps(row) + "\n")
    dataset = DynamicCaseDataset(tmp_path, manifest, "train", (6, 6, 6),
                                 task_weights={"segmentation":1}, spatial_flip_probability=0,
                                 segmentation_foreground_probability=1.0,
                                 segmentation_center_jitter_zyx=(0, 0, 0),
                                 segmentation_min_foreground_voxels=1, num_samples=20)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["metadata"]["class_id"] == 2
        assert sample["metadata"]["target_foreground_voxels"] > 0


def test_wan_padding_contract_roundtrip() -> None:
    x=torch.arange(2*1*66*35*34).reshape(2,1,66,35,34).float(); info=compute_wan_padding((66,35,34))
    assert info.padded_shape == (69,64,64)
    torch.testing.assert_close(crop_padding(pad_volume(x, info, -1), info), x)


def test_96_cube_only_needs_one_depth_padding_slice() -> None:
    info = compute_wan_padding((96, 96, 96))
    assert info.padded_shape == (97, 96, 96)
    assert info.depth_pad == (0, 1)
    assert info.height_pad == info.width_pad == (0, 0)


def test_xy256_z65_sliding_window_config() -> None:
    config = load_experiment_config(
        Path("configs/experiments/main5task_feedforward_lora_all_xy256_z65_h200x8.yaml")
    )
    assert config["data"]["patch_size_dhw"] == [65, 256, 256]
    assert config["data"]["resize_xy"] == [256, 256]
    assert config["data"]["sliding_window"] == {
        "axis": "z", "depth": 65, "stride": 32,
        "include_flush_final_window": True,
    }
    assert config["train"]["full_checkpoint_every_steps"] == 10_000
    assert config["train"]["model_checkpoint_every_steps"] == 0
    assert config["train"]["keep_last_checkpoints"] == 0


def test_manifest_volume_dataset_cycles_flush_aligned_z_windows(tmp_path: Path) -> None:
    from medgen3d.data import ManifestVolumeDataset

    volume_dir = tmp_path / "volumes" / "case0"
    volume_dir.mkdir(parents=True)
    volume = np.zeros((155, 256, 256), dtype=np.float16)
    np.save(volume_dir / "condition.npy", volume)
    np.save(volume_dir / "target.npy", volume)
    record = {
        "case_id": "case0", "patient_id": "case0", "split": "train",
        "task": "synthesis", "condition": "volumes/case0/condition.npy",
        "target": "volumes/case0/target.npy", "prompt": "Synthesize T2.",
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = ManifestVolumeDataset(
        tmp_path, manifest, "train", "synthesis", num_samples=4,
        volume_shape_dhw=(65, 256, 256), sliding_window_stride=32,
    )
    samples = [dataset[index] for index in range(4)]
    assert [sample["metadata"]["sliding_window_start_z"] for sample in samples] == [
        0, 32, 64, 90,
    ]
    assert all(tuple(sample["condition"].shape) == (1, 65, 256, 256)
               for sample in samples)


def test_flow_equations_are_per_sample_and_condition_is_clean() -> None:
    z0=torch.tensor([0., 2.]).view(2,1,1,1,1); zc=torch.tensor([5.,6.]).view_as(z0)
    flow=construct_flow_batch(z0, zc, timestep=torch.tensor([0.,1.]))
    torch.testing.assert_close(flow.noisy_target[0], z0[0]); torch.testing.assert_close(flow.noisy_target[1], flow.noise[1])
    torch.testing.assert_close(flow.clean_condition, zc)
    assert masked_flow_mse(flow.velocity_target, flow.velocity_target).item() == 0


class TinyFlow(nn.Module):
    def __init__(self) -> None: super().__init__(); self.scale=nn.Parameter(torch.tensor(0.))
    def forward(self, z, condition, timestep, context, view_ratio=None): return self.scale*z + 0*condition


def test_mock_trainer_backward_checkpoint_and_euler(tmp_path: Path) -> None:
    model=TinyFlow(); cfg={"optimizer":{"name":"adamw","learning_rate":1e-3,"weight_decay":0.,"betas":[.9,.99],"eps":1e-8},
                           "gradient_accumulation_steps":1,"precision":"fp32","use_padding_loss_mask":True}
    trainer=FlowTrainer(model, build_optimizer(model,cfg), None, cfg, torch.device("cpu"))
    batch={"condition":torch.randn(2,2,2,2,2),"target":torch.randn(2,2,2,2,2),
           "valid_mask":torch.ones(2,1,2,2,2),"prompt":["a","b"],
           "task":["restoration","reconstruction"],
           "metadata":[{}, {"reconstruction_views":36}]}
    loss=trainer.train_microbatch(batch, text_context=[torch.zeros(1,1)]*2); assert np.isfinite(loss); assert model.scale.item()!=0
    path=tmp_path/"state.pt"; trainer.save_checkpoint(path,{"test":True}); trainer.load_checkpoint(path)
    assert not (tmp_path/"state.pt.partial").exists()
    best=tmp_path/"best.pt"; trainer.save_model_checkpoint(best,{"test":True},{"val_loss/mean":.5})
    best_state=torch.load(best,weights_only=False)
    assert best_state["step"] == trainer.step and best_state["validation"]["val_loss/mean"] == .5
    result=euler_flow_sample(lambda z,c,t,e,v: torch.ones_like(z), torch.zeros(1,1,2,2,2), None, steps=4,seed=1)
    generator=torch.Generator().manual_seed(1); expected=torch.randn((1,1,2,2,2),generator=generator)-1
    torch.testing.assert_close(result,expected)


def test_optimizer_uses_higher_lr_for_new_condition_module() -> None:
    model = MedicalWanDiT(MockWan())
    cfg = {"optimizer": {"name": "adamw", "learning_rate": 1e-5,
                         "new_module_learning_rate": 1e-4,
                         "new_module_patterns": ["condition_patch_embedding", "view_embedding"],
                         "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8}}
    optimizer = build_optimizer(model, cfg)
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["new_decay"]["lr"] == 1e-4
    assert by_name["pretrained_decay"]["lr"] == 1e-5
    assert all(group["weight_decay"] == 0 for name, group in by_name.items() if name.endswith("no_decay"))


def test_lora_all_linear_freezes_base_and_trains_all_adapters() -> None:
    model = MedicalWanDiT(MockWan())
    replaced = configure_dit_finetuning(
        model, {"mode": "lora_all_linear", "rank": 2, "alpha": 4, "dropout": 0.0}
    )
    assert replaced
    assert all(isinstance(dict(model.base.named_modules())[name.removeprefix("base.")], LoRALinear)
               for name in replaced)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("lora_A" in name for name in trainable)
    assert any("lora_B" in name for name in trainable)
    assert "condition_patch_embedding.weight" in trainable
    assert any(name.startswith("view_embedding.") for name in trainable)
    assert not any(name.startswith("base.") and ".base." in name for name in trainable)


def test_full_finetuning_mode_remains_available() -> None:
    model = MedicalWanDiT(MockWan())
    assert configure_dit_finetuning(model, {"mode": "full"}) == []
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_lora_checkpoint_contains_only_trainable_parameters(tmp_path: Path) -> None:
    model = MedicalWanDiT(MockWan())
    configure_dit_finetuning(
        model, {"mode": "lora_all_linear", "rank": 2, "alpha": 4, "dropout": 0.0}
    )
    cfg = {"optimizer": {"name": "adamw", "learning_rate": 1e-3,
                          "weight_decay": 0.0, "betas": [.9, .99], "eps": 1e-8},
           "gradient_accumulation_steps": 1, "precision": "fp32"}
    trainer = FlowTrainer(model, build_optimizer(model, cfg), None, cfg, torch.device("cpu"))
    checkpoint = tmp_path / "adapter.pt"
    trainer.save_model_checkpoint(checkpoint, {"test": True}, {"val_loss/mean": 1.0})
    state = torch.load(checkpoint, weights_only=False)
    assert state["model_format"] == "trainable_only"
    assert set(state["model"]) == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    restored = MedicalWanDiT(MockWan())
    configure_dit_finetuning(
        restored, {"mode": "lora_all_linear", "rank": 2, "alpha": 4, "dropout": 0.0}
    )
    restored_trainer = FlowTrainer(
        restored, build_optimizer(restored, cfg), None, cfg, torch.device("cpu")
    )
    restored_trainer._load_model_checkpoint_payload(state)


def test_zero_redundancy_optimizer_requires_distributed_initialization() -> None:
    assert not dist.is_initialized()
    model = TinyFlow()
    cfg = {"optimizer": {"name": "adamw", "learning_rate": 1e-5,
                         "new_module_learning_rate": 1e-4, "new_module_patterns": [],
                         "weight_decay": 0.01, "betas": [.9, .999], "eps": 1e-8,
                         "zero_redundancy": True}}
    with pytest.raises(RuntimeError, match="initialized distributed process group"):
        build_optimizer(model, cfg)


class MockHead(nn.Module):
    def forward(self,x,e): return x
class MockBlock(nn.Module):
    def forward(self,x,**kwargs): return x + kwargs["e"][:, :, 0, :]
class MockWan(nn.Module):
    def __init__(self):
        super().__init__(); self.in_dim=self.out_dim=self.dim=2; self.patch_size=(1,1,1); self.freq_dim=2; self.text_len=2
        self.patch_embedding=nn.Conv3d(2,2,1,bias=False); self.time_embedding=nn.Linear(2,2); self.time_projection=nn.Linear(2,12)
        self.text_embedding=nn.Identity(); self.blocks=nn.ModuleList([MockBlock()]); self.head=MockHead(); self.freqs=torch.zeros(4,1,dtype=torch.complex128)
    def unpatchify(self,x,grid): return [u[:int(torch.prod(g))].T.reshape(2,*g.tolist()) for u,g in zip(x,grid)]


def test_medical_dit_condition_path_is_distinct_and_gets_gradient() -> None:
    model=MedicalWanDiT(MockWan()); assert_zero_condition_equivalence(model)
    z=torch.randn(1,2,2,2,2); context=[torch.zeros(2,2)]
    first=model(z,torch.randn_like(z),torch.tensor([.5]),context)
    first.sum().backward(); assert model.condition_patch_embedding.weight.grad is not None
    with torch.no_grad(): model.condition_patch_embedding.weight.fill_(.1)
    second=model(z,torch.ones_like(z),torch.tensor([.5]),context); assert not torch.allclose(first,second)


def test_view_condition_is_zero_initialized_and_trainable() -> None:
    model = MedicalWanDiT(MockWan(), view_fourier_bands=2)
    z = torch.randn(1, 2, 2, 2, 2); context = [torch.zeros(2, 2)]
    low = model(z, torch.zeros_like(z), torch.tensor([.5]), context, torch.tensor([18 / 720]))
    high = model(z, torch.zeros_like(z), torch.tensor([.5]), context, torch.tensor([72 / 720]))
    torch.testing.assert_close(low, high)
    low.sum().backward()
    assert model.view_embedding[-1].weight.grad is not None


def test_task_specific_metrics() -> None:
    mask=np.zeros((8,8,8),bool); mask[2:6,2:6,2:6]=1
    seg=segmentation_metrics(mask,mask,(1.,1.,1.)); assert seg == {"dice":1.,"nsd":1.,"hd95_mm":0.,"assd_mm":0.}
    ct=ct_metrics(np.zeros((3,16,16)),np.zeros((3,16,16))); assert ct["mae_hu"]==0 and ct["ssim"]==1


def test_paired_ct_metrics_and_summary() -> None:
    target = np.zeros((3, 16, 16), dtype=np.float32)
    condition = np.full_like(target, 2.0)
    prediction = np.full_like(target, 1.0)
    paired = paired_ct_metrics(condition, prediction, target)
    assert paired["input"]["mae_hu"] == 2.0
    assert paired["model"]["mae_hu"] == 1.0
    assert paired["improvement"]["mae_hu"] == 1.0
    assert paired["improvement"]["psnr_hu"] > 0
    summary = summarize_paired_ct([paired, paired], seed=7, bootstrap_samples=100)
    assert summary["num_cases"] == 2
    assert summary["metrics"]["mae_hu"]["improvement_mean"] == 1.0
    assert summary["metrics"]["mae_hu"]["improvement_fraction"] == 1.0
