from __future__ import annotations

import numpy as np


def normalize_hu(
    volume: np.ndarray,
    clip: tuple[float, float] = (-1000.0, 1000.0),
    output_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    low, high = clip
    out_low, out_high = output_range
    clipped = np.clip(volume.astype(np.float32), low, high)
    unit = (clipped - low) / (high - low)
    return (unit * (out_high - out_low) + out_low).astype(np.float32)


def label_to_sdf(
    label: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    classes: list[int],
    clip_mm: float,
    positive_inside: bool = True,
) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    channels = []
    for class_id in classes:
        inside = label == class_id
        if not np.any(inside):
            fill = -1.0 if positive_inside else 1.0
            channels.append(np.full(label.shape, fill, dtype=np.float16))
            continue
        distance_inside = distance_transform_edt(inside, sampling=spacing_zyx)
        distance_outside = distance_transform_edt(~inside, sampling=spacing_zyx)
        sdf = distance_inside - distance_outside
        if not positive_inside:
            sdf = -sdf
        sdf = np.clip(sdf, -clip_mm, clip_mm) / clip_mm
        channels.append(sdf.astype(np.float16))
    return np.stack(channels, axis=0)


def sdf_to_label(sdf: np.ndarray, classes: list[int]) -> np.ndarray:
    if sdf.ndim < 2 or sdf.shape[0] != len(classes):
        raise ValueError("SDF must have one leading channel per class")
    best_channel = np.argmax(sdf, axis=0)
    best_value = np.max(sdf, axis=0)
    class_values = np.asarray(classes, dtype=np.uint8)
    label = class_values[best_channel]
    return np.where(best_value > 0, label, 0).astype(np.uint8)


def crop_or_pad(
    array: np.ndarray,
    start_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
    fill_value: float | int = 0,
) -> np.ndarray:
    spatial_shape = array.shape[-3:]
    slices = []
    pads = [(0, 0)] * (array.ndim - 3)
    for start, size, available in zip(start_zyx, shape_zyx, spatial_shape):
        source_start = max(0, start)
        source_end = min(available, start + size)
        slices.append(slice(source_start, source_end))
        pads.append((max(0, -start), max(0, start + size - available)))
    cropped = array[(..., *slices)]
    if any(before or after for before, after in pads):
        cropped = np.pad(cropped, pads, mode="constant", constant_values=fill_value)
    return cropped
