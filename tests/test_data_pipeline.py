from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicalmodel_data.degradation import sparse_view_slice, synthetic_low_dose_slice
from medicalmodel_data.geometry import (
    crop_or_pad,
    label_to_sdf,
    normalize_hu,
    sdf_to_label,
)
from scripts.prepare_abdomenatlas import (
    center_patch_start,
    command_split,
    stable_seed,
    stratified_case_order,
)
from medicalmodel_data.io import write_csv
from medicalmodel_data.layout import ensure_layout


def test_normalize_hu_endpoints() -> None:
    values = np.array([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
    result = normalize_hu(values)
    np.testing.assert_allclose(result, [-1.0, -1.0, 0.0, 1.0, 1.0])


def test_crop_or_pad_preserves_requested_shape() -> None:
    array = np.ones((4, 5, 6), dtype=np.uint8)
    result = crop_or_pad(array, (-2, 1, 3), (8, 4, 5))
    assert result.shape == (8, 4, 5)
    assert result.sum() > 0


def test_crop_or_pad_uses_medically_correct_fill() -> None:
    ct = np.zeros((2, 2, 2), dtype=np.int16)
    padded = crop_or_pad(ct, (-1, -1, -1), (4, 4, 4), fill_value=-1000)
    assert padded[0, 0, 0] == -1000
    assert padded[1, 1, 1] == 0


def test_sdf_sign_and_absent_class() -> None:
    label = np.zeros((9, 9, 9), dtype=np.uint8)
    label[3:6, 3:6, 3:6] = 1
    sdf = label_to_sdf(label, (1.5, 1.5, 1.5), [1, 2], 32.0)
    assert sdf.shape == (2, 9, 9, 9)
    assert sdf[0, 4, 4, 4] > 0
    assert sdf[0, 0, 0, 0] < 0
    assert np.all(sdf[1] == -1)
    decoded = sdf_to_label(sdf, [1, 2])
    np.testing.assert_array_equal(decoded, label)


def test_projection_degradations_handle_rectangular_slice() -> None:
    hu = np.full((24, 32), -1000.0, dtype=np.float32)
    hu[7:17, 10:22] = 50.0
    sparse = sparse_view_slice(hu, views=30, mu_water=0.02)
    low_dose = synthetic_low_dose_slice(
        hu,
        views=60,
        incident_photons=1e4,
        mu_water=0.02,
        rng=np.random.default_rng(7),
    )
    assert sparse.shape == hu.shape
    assert low_dose.shape == hu.shape
    assert np.isfinite(sparse).all()
    assert np.isfinite(low_dose).all()
    assert not np.allclose(sparse, hu)
    assert not np.allclose(low_dose, hu)


def test_full_view_fbp_has_bounded_phantom_bias() -> None:
    size = 64
    yy, xx = np.ogrid[:size, :size]
    hu = np.full((size, size), -1000.0, dtype=np.float32)
    body = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size * 0.35) ** 2
    hu[body] = 0.0
    reconstruction = sparse_view_slice(hu, views=360, mu_water=0.02)
    assert np.mean(np.abs(reconstruction[body] - hu[body])) < 50.0
    assert abs(np.mean(reconstruction[body] - hu[body])) < 30.0


def test_validation_crop_is_deterministic_foreground_center() -> None:
    label = np.zeros((100, 300, 300), dtype=np.uint8)
    label[40:60, 100:200, 120:220] = 5
    assert center_patch_start(label, (64, 256, 256)) == (18, 22, 42)


def test_case_task_seed_is_stable_and_task_specific() -> None:
    first = stable_seed(20260727, "BDMAP_00000001", "restoration")
    assert first == stable_seed(20260727, "BDMAP_00000001", "restoration")
    assert first != stable_seed(20260727, "BDMAP_00000001", "reconstruction")


def test_stratified_order_is_stable_and_complete() -> None:
    rows = [
        {
            "case_id": f"case_{index:03d}",
            "shape_xyz": f"256x256x{80 + index}",
            "spacing_xyz_mm": f"1,1,{0.8 + index / 100:.3f}",
            "foreground_fraction": f"{0.01 + index / 1000:.4f}",
        }
        for index in range(32)
    ]
    first = stratified_case_order(rows, 7)
    second = stratified_case_order(rows, 7)
    assert first == second
    assert set(first) == {row["case_id"] for row in rows}


def test_configured_full_split_counts_are_exact(tmp_path: Path) -> None:
    paths = ensure_layout(tmp_path / "dataset")
    rows = [
        {
            "case_id": f"case_{index:03d}",
            "valid": 1,
            "shape_xyz": f"32x32x{20 + index}",
            "spacing_xyz_mm": f"1,1,{1 + index / 100}",
            "foreground_fraction": f"{0.01 + index / 1000}",
        }
        for index in range(10)
    ]
    write_csv(paths["metadata"] / "inventory.csv", rows)
    config = {
        "_config_hash": "test",
        "split": {
            "seed": 7,
            "pilot_counts": {"train": 2, "val": 1, "test": 1},
            "full_counts": {"train": 6, "val": 2, "test": 2},
            "full_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
    }
    command_split(config, paths, pilot=False)
    assert len((paths["splits"] / "train.txt").read_text().splitlines()) == 6
    assert len((paths["splits"] / "val.txt").read_text().splitlines()) == 2
    assert len((paths["splits"] / "test.txt").read_text().splitlines()) == 2


def test_configured_full_split_can_leave_cases_unused(tmp_path: Path) -> None:
    paths = ensure_layout(tmp_path / "dataset")
    rows = [{"case_id": f"case_{i:03d}", "valid": 1, "shape_xyz": f"32x32x{20+i}",
             "spacing_xyz_mm": f"1,1,{1+i/100}", "foreground_fraction": f"{.01+i/1000}"} for i in range(10)]
    write_csv(paths["metadata"] / "inventory.csv", rows)
    config = {"_config_hash": "test", "split": {"seed": 7,
              "pilot_counts": {"train": 2, "val": 1, "test": 1},
              "full_counts": {"train": 5, "val": 1, "test": 1}}}
    command_split(config, paths, pilot=False)
    assert [len((paths["splits"] / f"{name}.txt").read_text().splitlines())
            for name in ("train", "val", "test", "unused")] == [5, 1, 1, 3]
