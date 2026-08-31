#!/usr/bin/env python3
"""Interactive viewer for the five fixed-cohort segmentation failure cases."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st


DATA_ROOT = Path(
    "/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/data/AbdomenAtlas1.0Mini"
)
MEDGEN_RESULTS = Path(
    "/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-evals/"
    "feedforward_v3_best_seg_balanced100/results_merged.json"
)
TOTALSEG_ROOT = Path(
    "/inspire/qb-ilm/project/video-generation/public/lijiaxi/"
    "MedGen3D-baselines/TotalSegmentator/eval_balanced100"
)
TOTALSEG_RESULTS = TOTALSEG_ROOT / "results_merged.json"
PATCH_SHAPE = (97, 96, 96)

ORGAN_LABELS = {
    "gall_bladder": (2, "胆囊 / gall bladder"),
    "postcava": (7, "下腔静脉 / inferior vena cava"),
}


def import_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        base = Path(
            "/inspire/qb-ilm/project/video-generation/public/lijiaxi/"
            "MedGen3D-baselines/TotalSegmentator/venv_gpu_image/lib"
        )
        for site_packages in sorted(base.glob("python*/site-packages"), reverse=True):
            sys.path.insert(0, str(site_packages))
        import nibabel as nib
        return nib


nib = import_nibabel()


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner="读取病例体数据……")
def load_nifti_zyx(path: str) -> np.ndarray:
    array = np.asanyarray(nib.load(path).dataobj)
    return np.moveaxis(array, (0, 1, 2), (2, 1, 0))


def crop_or_pad(array: np.ndarray, start: tuple[int, int, int], shape: tuple[int, int, int], fill=0):
    output = np.full(shape, fill, dtype=array.dtype)
    src, dst = [], []
    for origin, length, limit in zip(start, shape, array.shape):
        lo, hi = max(0, origin), min(limit, origin + length)
        src.append(slice(lo, hi))
        dst.append(slice(lo - origin, hi - origin))
    output[tuple(dst)] = array[tuple(src)]
    return output


def gt_centered_patch(ct: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    foreground = np.argwhere(gt)
    center = np.rint((foreground.min(0) + foreground.max(0)) / 2).astype(int)
    start = tuple(int(value - size // 2) for value, size in zip(center, PATCH_SHAPE))
    return (
        crop_or_pad(ct, start, PATCH_SHAPE, fill=-1000),
        crop_or_pad(gt, start, PATCH_SHAPE, fill=False),
        crop_or_pad(pred, start, PATCH_SHAPE, fill=False),
        start,
    )


def take_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis)


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def base_rgb(ct_slice: np.ndarray, level: float, width: float) -> np.ndarray:
    low, high = level - width / 2, level + width / 2
    gray = np.clip((ct_slice.astype(np.float32) - low) / max(width, 1.0), 0, 1)
    return np.repeat(gray[..., None], 3, axis=-1)


def overlay(ct_slice: np.ndarray, gt=None, pred=None, level=50.0, width=400.0) -> np.ndarray:
    image = base_rgb(ct_slice, level, width)
    gt_edge = boundary(gt) if gt is not None else np.zeros(ct_slice.shape, bool)
    pred_edge = boundary(pred) if pred is not None else np.zeros(ct_slice.shape, bool)
    image[gt_edge] = (0.0, 1.0, 0.0)
    image[pred_edge] = (1.0, 0.1, 0.1)
    image[gt_edge & pred_edge] = (1.0, 1.0, 0.0)
    return image


def fmt(value: float) -> str:
    return "∞" if not math.isfinite(float(value)) else f"{float(value):.4f}"


def default_slice(mask: np.ndarray, axis: int) -> int:
    other_axes = tuple(value for value in range(3) if value != axis)
    areas = mask.sum(axis=other_axes)
    return int(np.argmax(areas))


st.set_page_config(page_title="Segmentation failure cases", layout="wide")
st.title("MedGen3D × TotalSegmentator：异常病例查看器")
st.caption(
    "固定 balanced-100 测试集。绿色=GT，红色=预测，黄色=边界重合。"
    "MedGen3D原始体素预测未落盘，因此下方展示评测时保存的原始三平面叠加图；"
    "TotalSegmentator可逐层查看其完整NIfTI预测。"
)

required = [DATA_ROOT, MEDGEN_RESULTS, TOTALSEG_RESULTS]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.error("缺少评测文件：\n" + "\n".join(missing))
    st.stop()

medgen_payload = load_json(str(MEDGEN_RESULTS))
totalseg_payload = load_json(str(TOTALSEG_RESULTS))
medgen_by_index = {int(row["index"]): row for row in medgen_payload["rows"]}
totalseg_by_index = {int(row["index"]): row for row in totalseg_payload["rows"]}
failure_indices = sorted(
    {index for index, row in medgen_by_index.items() if float(row["metrics"]["dice"]) == 0.0}
    | {index for index, row in totalseg_by_index.items() if float(row["metrics"]["dice"]) == 0.0}
)
manifest = {
    row["case_id"]: row
    for row in map(json.loads, (DATA_ROOT / "processed/manifests/test.jsonl").read_text().splitlines())
}

labels = {
    index: f"{totalseg_by_index[index]['case_id']} · {ORGAN_LABELS[totalseg_by_index[index]['organ']][1]}"
    for index in failure_indices
}
selected_index = st.sidebar.selectbox(
    "异常病例", failure_indices, format_func=lambda value: labels[value]
)
axis_name = st.sidebar.radio("观察方向", ["轴位 Z", "冠状位 Y", "矢状位 X"], horizontal=True)
axis = {"轴位 Z": 0, "冠状位 Y": 1, "矢状位 X": 2}[axis_name]
window_level = st.sidebar.slider("窗位 HU", -200, 200, 50, 10)
window_width = st.sidebar.slider("窗宽 HU", 100, 1000, 400, 25)

total_row = totalseg_by_index[selected_index]
medgen_row = medgen_by_index[selected_index]
case_id, organ = total_row["case_id"], total_row["organ"]
class_id, organ_display = ORGAN_LABELS[organ]
record = manifest[case_id]

ct = load_nifti_zyx(str(DATA_ROOT / record["image"]))
labels_volume = load_nifti_zyx(str(DATA_ROOT / record["mask"]))
gt_full = labels_volume == class_id
total_full = load_nifti_zyx(total_row["prediction"]) > 0
ct_patch, gt_patch, total_patch, crop_start = gt_centered_patch(ct, gt_full, total_full)

initial_slice = default_slice(gt_patch, axis)
slice_index = st.sidebar.slider(
    "Patch内切片", 0, ct_patch.shape[axis] - 1, initial_slice,
    key=f"slice-{selected_index}-{axis}",
)
ct_slice = take_slice(ct_patch, axis, slice_index)
gt_slice = take_slice(gt_patch, axis, slice_index)
total_slice = take_slice(total_patch, axis, slice_index)

med_metrics, total_metrics = medgen_row["metrics"], total_row["metrics"]
med_empty = float(med_metrics["dice"]) == 0.0 and not math.isfinite(float(med_metrics["hd95_mm"]))
total_pred_voxels = int(total_full.sum())
total_overlap = int(np.logical_and(total_patch, gt_patch).sum())

st.subheader(f"{case_id} · {organ_display}")
cols = st.columns(4)
for col, name, value in zip(
    cols,
    ["GT体素", "TotalSeg预测体素", "Patch重叠体素", "评测裁剪起点 Z/Y/X"],
    [int(gt_full.sum()), total_pred_voxels, total_overlap, str(crop_start)],
):
    col.metric(name, value)

left, right = st.columns(2)
with left:
    st.markdown("**MedGen3D（上一版 feed-forward best）**")
    st.dataframe(
        {"Dice": [fmt(med_metrics["dice"])], "NSD": [fmt(med_metrics["nsd"])],
         "HD95 mm": [fmt(med_metrics["hd95_mm"])], "ASSD mm": [fmt(med_metrics["assd_mm"])]},
        hide_index=True, use_container_width=True,
    )
    if med_empty:
        st.error("该病例的MedGen3D阈值化预测为空。")
with right:
    st.markdown("**TotalSegmentator 2.18.0（zero-shot）**")
    st.dataframe(
        {"Dice": [fmt(total_metrics["dice"])], "NSD": [fmt(total_metrics["nsd"])],
         "HD95 mm": [fmt(total_metrics["hd95_mm"])], "ASSD mm": [fmt(total_metrics["assd_mm"])]},
        hide_index=True, use_container_width=True,
    )
    if total_pred_voxels == 0:
        st.error("TotalSegmentator完整体积预测为空。")
    elif total_overlap == 0:
        st.warning(f"TotalSegmentator有{total_pred_voxels}个预测体素，但与GT没有重叠。")

view_cols = st.columns(4)
view_cols[0].image(base_rgb(ct_slice, window_level, window_width), caption="原始CT", use_container_width=True)
view_cols[1].image(
    overlay(ct_slice, gt=gt_slice, level=window_level, width=window_width),
    caption="GT（绿色）", use_container_width=True,
)
view_cols[2].image(
    overlay(ct_slice, pred=total_slice, level=window_level, width=window_width),
    caption="TotalSegmentator（红色）", use_container_width=True,
)
view_cols[3].image(
    overlay(ct_slice, gt=gt_slice, pred=total_slice, level=window_level, width=window_width),
    caption="GT × TotalSeg", use_container_width=True,
)

st.markdown("**MedGen3D评测时保存的三平面叠加图**")
artifact = Path(medgen_row["artifact"])
if artifact.exists():
    st.image(str(artifact), caption="绿色=GT；红色=MedGen3D预测", use_container_width=True)
else:
    st.warning(f"未找到MedGen3D叠加图：{artifact}")

with st.expander("这5例的整体对照", expanded=False):
    table = []
    for index in failure_indices:
        med, total = medgen_by_index[index], totalseg_by_index[index]
        table.append({
            "case": total["case_id"],
            "organ": ORGAN_LABELS[total["organ"]][1],
            "MedGen3D Dice": float(med["metrics"]["dice"]),
            "TotalSeg Dice": float(total["metrics"]["dice"]),
            "MedGen3D异常": float(med["metrics"]["dice"]) == 0.0,
            "TotalSeg异常": float(total["metrics"]["dice"]) == 0.0,
        })
    st.dataframe(table, hide_index=True, use_container_width=True)

st.info(
    "注意：这些异常来自上一版MedGen3D feed-forward最佳checkpoint与TotalSegmentator的固定100例评测；"
    "正在等待H200训练的新surface-aware版本不在本页面内。"
)
