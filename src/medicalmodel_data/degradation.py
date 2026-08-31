from __future__ import annotations

import numpy as np


def hu_to_mu(hu: np.ndarray, mu_water: float) -> np.ndarray:
    return np.clip(mu_water * (hu.astype(np.float32) / 1000.0 + 1.0), 0.0, None)


def mu_to_hu(mu: np.ndarray, mu_water: float) -> np.ndarray:
    return (mu.astype(np.float32) / mu_water - 1.0) * 1000.0


def _pad_square(image: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice]]:
    height, width = image.shape
    side = max(height, width)
    before_y = (side - height) // 2
    before_x = (side - width) // 2
    padded = np.pad(
        image,
        ((before_y, side - height - before_y), (before_x, side - width - before_x)),
        mode="constant",
    )
    return padded, (slice(before_y, before_y + height), slice(before_x, before_x + width))


def _radon_fbp_slice(
    mu: np.ndarray, angles_deg: np.ndarray, pixel_size_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.transform import iradon, radon

    padded, crop = _pad_square(mu)
    sino = radon(
        padded * pixel_size_mm, theta=angles_deg, circle=False, preserve_range=True
    )
    reconstruction = iradon(
        sino,
        theta=angles_deg,
        filter_name="ramp",
        circle=False,
        output_size=padded.shape[0],
        preserve_range=True,
    )
    reconstruction = reconstruction[crop] / pixel_size_mm
    return sino.astype(np.float32), reconstruction.astype(np.float32)


def synthetic_low_dose_slice(
    hu: np.ndarray,
    views: int,
    incident_photons: float,
    mu_water: float,
    rng: np.random.Generator,
    electronic_noise_std: float = 0.0,
    pixel_size_mm: float = 1.5,
) -> np.ndarray:
    from skimage.transform import iradon, radon

    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32)
    padded, crop = _pad_square(hu_to_mu(hu, mu_water))
    projection = radon(
        padded * pixel_size_mm, theta=angles, circle=False, preserve_range=True
    )
    expected = incident_photons * np.exp(-np.clip(projection, 0.0, 80.0))
    counts = rng.poisson(expected).astype(np.float32)
    if electronic_noise_std > 0:
        counts += rng.normal(0.0, electronic_noise_std, size=counts.shape)
    noisy_projection = -np.log(np.maximum(counts, 1.0) / incident_photons)
    reconstruction = iradon(
        noisy_projection,
        theta=angles,
        filter_name="ramp",
        circle=False,
        output_size=padded.shape[0],
        preserve_range=True,
    )
    reconstruction = reconstruction[crop] / pixel_size_mm
    return mu_to_hu(reconstruction, mu_water)


def sparse_view_slice(
    hu: np.ndarray, views: int, mu_water: float, pixel_size_mm: float = 1.5
) -> np.ndarray:
    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32)
    _, reconstruction = _radon_fbp_slice(
        hu_to_mu(hu, mu_water), angles, pixel_size_mm
    )
    return mu_to_hu(reconstruction, mu_water)
