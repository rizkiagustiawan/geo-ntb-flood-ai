"""
SAR Preprocessing for flood detection.
Implements Refined Lee speckle filter, incidence angle normalization,
and thermal noise removal per Twele et al. (2016) and Cao et al. (2024).
"""

import logging
import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def refined_lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Refined Lee speckle filter for SAR intensity/dB data.

    Uses local statistics (mean, variance) to adaptively filter speckle
    while preserving edges. Operates on dB values (additive noise model
    after log transform).

    Parameters
    ----------
    img : np.ndarray — SAR backscatter in dB (float32)
    window_size : int — Filter window size (must be odd, default 7)

    Returns
    -------
    np.ndarray — Filtered image (float32)
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    result = img.copy().astype(np.float32)
    valid = ~np.isnan(img) & ~np.isinf(img)

    img_clean = np.where(valid, img, 0.0).astype(np.float32)
    valid_float = valid.astype(np.float32)

    n_img = uniform_filter(img_clean, size=window_size, mode='constant')
    n_valid = uniform_filter(valid_float, size=window_size, mode='constant')
    n_valid = np.maximum(n_valid, 1e-10)

    local_mean = n_img / n_valid

    sq_img = np.where(valid, img_clean ** 2, 0.0)
    n_sq = uniform_filter(sq_img, size=window_size, mode='constant')
    local_var = (n_sq / n_valid) - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)

    overall_var = np.nanvar(img[valid]) if np.any(valid) else 1.0
    overall_var = max(overall_var, 1e-10)

    weight = np.clip(1.0 - (overall_var / (local_var + 1e-10)), 0.0, 1.0)

    filtered = local_mean + weight * (img_clean - local_mean)

    result[valid] = filtered[valid]
    result[~valid] = np.nan

    logger.info("Refined Lee filter applied: window=%d, input_var=%.4f, output_var=%.4f",
                window_size, overall_var, np.nanvar(result[valid]))
    return result


def normalize_incidence_angle(
    vv: np.ndarray,
    angle_map: np.ndarray,
    reference_angle: float = 37.5,
) -> np.ndarray:
    """Normalize SAR backscatter to a reference incidence angle.

    Applies empirical correction: σ⁰_norm = σ⁰ - β * (θ - θ_ref)
    where β ≈ 0.15 dB/degree for IW mode Sentinel-1 GRD over land.

    Parameters
    ----------
    vv : np.ndarray — VV backscatter in dB
    angle_map : np.ndarray — Local incidence angle in degrees
    reference_angle : float — Target angle (default 37.5°, midpoint of S1 IW)

    Returns
    -------
    np.ndarray — Angle-normalized backscatter (float32)
    """
    beta = 0.15  # dB/degree — empirical for S1 IW GRD over tropical land
    correction = beta * (angle_map - reference_angle)
    normalized = (vv - correction).astype(np.float32)

    logger.info("Incidence angle normalization: ref=%.1f°, β=%.2f dB/deg, range=[%.1f, %.1f]",
                reference_angle, beta, np.nanmin(angle_map), np.nanmax(angle_map))
    return normalized


def remove_thermal_noise(
    vv: np.ndarray,
    noise_level_db: float = 3.0,
    edge_fraction: float = 0.1,
) -> np.ndarray:
    """Remove thermal noise from SAR data, focusing on swath edges.

    Sentinel-1 GRD has elevated noise at swath edges. This function
    applies a linear noise taper that reduces backscatter at edges.

    Parameters
    ----------
    vv : np.ndarray — VV backscatter in dB
    noise_level_db : float — Maximum noise at edge (dB)
    edge_fraction : float — Fraction of width affected (0.1 = 10%)

    Returns
    -------
    np.ndarray — Noise-corrected backscatter (float32)
    """
    result = vv.copy().astype(np.float32)
    n_cols = vv.shape[1]
    edge_cols = max(int(n_cols * edge_fraction), 1)

    left_taper = np.linspace(noise_level_db, 0, edge_cols, dtype=np.float32)
    right_taper = np.linspace(0, noise_level_db, edge_cols, dtype=np.float32)

    result[:, :edge_cols] -= left_taper
    result[:, -edge_cols:] -= right_taper

    logger.info("Thermal noise removal: noise=%.1f dB, edge_fraction=%.2f",
                noise_level_db, edge_fraction)
    return result


def preprocess_sar(
    vv: np.ndarray,
    vh: np.ndarray,
    angle_map: np.ndarray = None,
    apply_lee: bool = True,
    lee_window: int = 7,
    normalize_angle: bool = True,
    remove_noise: bool = True,
) -> tuple:
    """Full SAR preprocessing pipeline.

    Parameters
    ----------
    vv, vh : np.ndarray — Raw VV/VH backscatter in dB
    angle_map : np.ndarray — Incidence angle map (degrees), optional
    apply_lee : bool — Apply Refined Lee filter
    lee_window : int — Lee filter window size
    normalize_angle : bool — Apply incidence angle normalization
    remove_noise : bool — Apply thermal noise removal

    Returns
    -------
    tuple[np.ndarray, np.ndarray] — Preprocessed (VV, VH) in dB
    """
    logger.info("=" * 40)
    logger.info("SAR PREPROCESSING PIPELINE")
    logger.info("=" * 40)

    vv_proc = vv.astype(np.float32)
    vh_proc = vh.astype(np.float32)

    if remove_noise:
        vv_proc = remove_thermal_noise(vv_proc)
        vh_proc = remove_thermal_noise(vh_proc, noise_level_db=2.0)

    if normalize_angle and angle_map is not None:
        vv_proc = normalize_incidence_angle(vv_proc, angle_map)
        vh_proc = normalize_incidence_angle(vh_proc, angle_map, reference_angle=37.5)
    elif normalize_angle and angle_map is None:
        logger.warning("Angle normalization requested but no angle_map provided — skipping")

    if apply_lee:
        vv_proc = refined_lee_filter(vv_proc, window_size=lee_window)
        vh_proc = refined_lee_filter(vh_proc, window_size=lee_window)

    logger.info("SAR preprocessing complete: VV [%.1f, %.1f], VH [%.1f, %.1f]",
                np.nanmin(vv_proc), np.nanmax(vv_proc),
                np.nanmin(vh_proc), np.nanmax(vh_proc))
    return vv_proc, vh_proc
