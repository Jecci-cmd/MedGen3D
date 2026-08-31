from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medicalmodel_data.config import load_config
from medicalmodel_data.layout import ensure_layout
from scripts.prepare_abdomenatlas import (
    command_canonicalize,
    command_derive,
    command_inventory,
    command_patches,
    command_qc,
    command_smoke_test,
    command_split,
)


def test_three_case_end_to_end_pipeline(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic",
            "root": str(tmp_path / "dataset"),
            "expected_cases": 3,
            "archives": {},
            "class_map": {0: "background", 1: "aorta"},
        },
        "split": {
            "seed": 7,
            "pilot_counts": {"train": 1, "val": 1, "test": 1},
            "full_ratios": {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        },
        "canonical": {
            "orientation": "RAS",
            "target_spacing_xyz_mm": [1.5, 1.5, 1.5],
            "ct_dtype": "int16",
            "label_dtype": "uint8",
        },
        "tasks": {
            "segmentation": {
                "sdf_classes": list(range(1, 10)),
                "sdf_clip_mm": 4.5,
                "sdf_positive_inside": True,
            },
            "restoration": {
                "geometry": "parallel",
                "views": 30,
                "incident_photons": 10000,
                "mu_water_per_mm": 0.02,
                "electronic_noise_std": 0.0,
            },
            "reconstruction": {
                "geometry": "parallel",
                "views": 12,
                "mu_water_per_mm": 0.02,
            },
        },
        "processed": {
            "hu_clip": [-1000.0, 1000.0],
            "output_range": [-1.0, 1.0],
            "patch_shape_zyx": [4, 16, 16],
            "train_patches_per_case": 1,
            "foreground_probability": 0.7,
            "seed": 7,
        },
        "runtime": {"projection_workers": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config = load_config(config_path)
    paths = ensure_layout(Path(config["dataset"]["root"]))

    affine = np.asarray(
        [[-2.0, 0.0, 0.0, 30.0], [0.0, -2.0, 0.0, 30.0], [0.0, 0.0, 3.0, 0.0], [0, 0, 0, 1]]
    )
    for index in range(3):
        case_dir = paths["raw/extracted"] / f"case_{index:02d}"
        case_dir.mkdir(parents=True)
        ct = np.full((16, 16, 6), -1000, dtype=np.int16)
        ct[3:13, 3:13, 1:5] = 20 + index * 10
        label = np.zeros(ct.shape, dtype=np.uint8)
        label[6:10, 6:10, 2:4] = 1
        nib.save(nib.Nifti1Image(ct, affine), case_dir / "ct.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), case_dir / "combined_labels.nii.gz")

    command_inventory(config, paths, checksum=False)
    command_split(config, paths, pilot=True)
    command_canonicalize(config, paths, "all", None, force=False)
    command_derive(config, paths, "all", None, "all", force=False, workers=1)
    serial_restoration = {
        case_id: np.asanyarray(
            nib.load(paths["tasks/restoration"] / case_id / "source_hu.nii.gz").dataobj
        ).copy()
        for case_id in ("case_00", "case_01", "case_02")
    }
    command_derive(
        config,
        paths,
        "all",
        None,
        "restoration",
        force=True,
        workers=2,
    )
    for case_id, expected in serial_restoration.items():
        parallel = np.asanyarray(
            nib.load(paths["tasks/restoration"] / case_id / "source_hu.nii.gz").dataobj
        )
        np.testing.assert_array_equal(parallel, expected)
    command_patches(config, paths, "all", None, force=False)
    command_qc(config, paths, "all", None)
    command_smoke_test(config, paths)

    inventory = json.loads(
        (paths["metadata"] / "inventory_summary.json").read_text(encoding="utf-8")
    )
    assert inventory == {"discovered": 3, "excluded": 0, "valid": 3}
    assert json.loads((paths["qc"] / "report.json").read_text())["failed"] == 0
    assert json.loads(
        (paths["qc"] / "dataloader_smoke_test.json").read_text()
    )["passed"]
    for case_dir in paths["canonical"].iterdir():
        image = nib.load(case_dir / "ct_hu.nii.gz")
        assert nib.aff2axcodes(image.affine) == ("R", "A", "S")
        np.testing.assert_allclose(image.header.get_zooms()[:3], (1.5, 1.5, 1.5))


def test_reset_label_header_is_repaired_from_ct_grid(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic",
            "root": str(tmp_path / "dataset"),
            "expected_cases": 1,
            "archives": {},
            "class_map": {0: "background", 1: "aorta"},
            "source_repository": "synthetic/reset-header-fixture",
            "source_revision": "fixture-v1",
            "label_reset_header_policy": {
                "enabled": True,
                "version": "fixture-reset-lps-v1",
                "expected_axis_signs_xyz": [-1, -1, 1],
            },
        },
        "canonical": {
            "orientation": "RAS",
            "target_spacing_xyz_mm": [1.5, 1.5, 1.5],
            "ct_dtype": "int16",
            "label_dtype": "uint8",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config = load_config(config_path)
    paths = ensure_layout(Path(config["dataset"]["root"]))
    case_dir = paths["raw/extracted"] / "case_00"
    case_dir.mkdir(parents=True)
    shape = (12, 10, 8)
    ct_affine = np.asarray(
        [
            [-2.0, 0.0, 0.0, 24.0],
            [0.0, -2.0, 0.0, 20.0],
            [0.0, 0.0, 3.0, -12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reset_label_affine = np.diag([-2.0, -2.0, 3.0, 1.0])
    ct = np.full(shape, -1000, dtype=np.int16)
    ct[3:9, 2:8, 2:6] = 50
    label = np.zeros(shape, dtype=np.uint8)
    label[4:8, 3:7, 3:5] = 1
    nib.save(nib.Nifti1Image(ct, ct_affine), case_dir / "ct.nii.gz")
    nib.save(
        nib.Nifti1Image(label, reset_label_affine),
        case_dir / "combined_labels.nii.gz",
    )

    command_inventory(config, paths, checksum=False)
    inventory = json.loads(
        (paths["metadata"] / "inventory_summary.json").read_text(encoding="utf-8")
    )
    assert inventory == {"discovered": 1, "excluded": 0, "valid": 1}
    (paths["splits"] / "train.txt").write_text("case_00\n", encoding="utf-8")
    (paths["splits"] / "val.txt").write_text("", encoding="utf-8")
    (paths["splits"] / "test.txt").write_text("", encoding="utf-8")
    command_canonicalize(config, paths, "train", None, force=False)

    canonical = paths["canonical"] / "case_00"
    ct_out = nib.load(canonical / "ct_hu.nii.gz")
    label_out = nib.load(canonical / "label_id.nii.gz")
    np.testing.assert_allclose(ct_out.affine, label_out.affine)
    expected_label = nib.processing.resample_from_to(
        nib.as_closest_canonical(nib.Nifti1Image(label, ct_affine)),
        ct_out,
        order=0,
    )
    np.testing.assert_array_equal(
        np.asanyarray(label_out.dataobj),
        np.asanyarray(expected_label.dataobj).astype(np.uint8),
    )
    meta = json.loads((canonical / "meta.json").read_text(encoding="utf-8"))
    assert meta["label_header_repaired"] is True
    assert meta["label_geometry_mode"] == "voxel_aligned_reset_header"
    assert meta["label_geometry_policy_version"] == "fixture-reset-lps-v1"


def test_reset_label_header_requires_explicit_policy(tmp_path: Path) -> None:
    config = {
        "dataset": {
            "name": "synthetic",
            "root": str(tmp_path / "dataset"),
            "expected_cases": 1,
            "archives": {},
            "class_map": {0: "background", 1: "aorta"},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config = load_config(config_path)
    paths = ensure_layout(Path(config["dataset"]["root"]))
    case_dir = paths["raw/extracted"] / "case_00"
    case_dir.mkdir(parents=True)
    shape = (6, 6, 4)
    ct_affine = np.asarray(
        [
            [-2.0, 0.0, 0.0, 12.0],
            [0.0, -2.0, 0.0, 12.0],
            [0.0, 0.0, 3.0, -6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reset_label_affine = np.diag([-2.0, -2.0, 3.0, 1.0])
    nib.save(
        nib.Nifti1Image(np.zeros(shape, dtype=np.int16), ct_affine),
        case_dir / "ct.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), reset_label_affine),
        case_dir / "combined_labels.nii.gz",
    )
    command_inventory(config, paths, checksum=False)
    summary = json.loads(
        (paths["metadata"] / "inventory_summary.json").read_text(encoding="utf-8")
    )
    assert summary == {"discovered": 1, "excluded": 1, "valid": 0}
