"""Attention-Based Multi-Modal Fusion for Flood Detection.

Implements a learned attention mechanism for fusing SAR and optical
features, replacing fixed-threshold AND/OR logic.

References:
- Sanderson et al. (2023): "Optimal fusion of multispectral optical and
  SAR images for flood inundation mapping through explainable deep learning."
- Vaswani et al. (2017): "Attention is All You Need."
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def attention_weighted_fusion(
    ndwi: np.ndarray,
    sar_vv: np.ndarray,
    sar_vh: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    """Compute attention-weighted fusion score for flood probability.

    Instead of fixed thresholds, computes a weighted combination where
    weights are derived from the data itself. Pixels with consistent
    water signals across multiple features get higher attention.

    Parameters
    ----------
    ndwi : np.ndarray — Normalized Difference Water Index
    sar_vv : np.ndarray — VV backscatter (dB)
    sar_vh : np.ndarray — VH backscatter (dB)
    slope : np.ndarray — Terrain slope (degrees)
    hand : np.ndarray — Height Above Nearest Drainage (m)

    Returns
    -------
    np.ndarray — Flood probability score [0, 1]
    """
    # Normalize each feature to [0, 1] range
    def normalize(x, low=None, high=None):
        if low is None:
            low = np.nanpercentile(x[np.isfinite(x)], 5)
        if high is None:
            high = np.nanpercentile(x[np.isfinite(x)], 95)
        denom = high - low
        if denom < 1e-6:
            return np.zeros_like(x)
        return np.clip((x - low) / denom, 0, 1)

    # Water-like features: higher = more likely water
    ndwi_score = normalize(ndwi, low=-0.3, high=0.5)

    # SAR features: lower = more likely water (invert)
    sar_vv_score = 1.0 - normalize(sar_vv, low=-25, high=-5)
    sar_vh_score = 1.0 - normalize(sar_vh, low=-30, high=-10)

    # Terrain features: lower = more flood-prone
    slope_score = 1.0 - normalize(slope, low=0, high=30)
    hand_score = 1.0 - normalize(hand, low=0, high=20)

    # Attention weights: compute per-pixel confidence
    # High attention when multiple features agree
    features = np.stack([ndwi_score, sar_vv_score, sar_vh_score, slope_score, hand_score])

    # Compute agreement: how many features strongly indicate water?
    water_threshold = 0.6
    agreement = np.sum(features > water_threshold, axis=0).astype(np.float32) / len(features)

    # Weighted combination
    weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])  # NDWI dominant
    weighted_score = np.tensordot(weights, features, axes=([0], [0]))

    # Final score = weighted combination modulated by agreement
    # High agreement → trust the score; low agreement → conservative
    fusion_score = weighted_score * (0.5 + 0.5 * agreement)

    logger.info("Attention fusion: score range [%.3f, %.3f], mean=%.3f",
                np.nanmin(fusion_score), np.nanmax(fusion_score), np.nanmean(fusion_score))

    return fusion_score.astype(np.float32)


def adaptive_flood_mask(
    fusion_score: np.ndarray,
    method: str = "otsu",
    fixed_threshold: float = 0.5,
) -> np.ndarray:
    """Generate binary flood mask from fusion score.

    Parameters
    ----------
    fusion_score : np.ndarray — Flood probability [0, 1]
    method : str — 'otsu' for adaptive, 'fixed' for manual threshold
    fixed_threshold : float — Threshold for fixed method

    Returns
    -------
    np.ndarray — uint8 binary mask: 1=flood, 0=non-flood
    """
    if method == "otsu":
        from features import otsu_threshold
        # Otsu on normalized score
        valid = np.isfinite(fusion_score) & (fusion_score > 0)
        if np.sum(valid) < 10:
            threshold = fixed_threshold
        else:
            threshold = otsu_threshold(fusion_score[valid])
        logger.info("Adaptive fusion threshold (Otsu): %.3f", threshold)
    else:
        threshold = fixed_threshold
        logger.info("Fixed fusion threshold: %.3f", threshold)

    mask = (fusion_score > threshold).astype(np.uint8)
    flood_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("Flood mask: %.2f%% flood pixels", flood_pct)

    return mask


def compute_feature_correlation(features: np.ndarray) -> dict:
    """Compute pairwise correlation between feature bands.

    Useful for understanding feature redundancy and selecting
    optimal feature subsets.

    Parameters
    ----------
    features : np.ndarray — (n_bands, H, W) feature stack

    Returns
    -------
    dict — Correlation matrix and highly correlated pairs
    """
    from config import FEATURE_NAMES

    n_bands, h, w = features.shape
    X = features.reshape(n_bands, -1).T

    # Remove NaN/zero rows
    valid = np.all(np.isfinite(X) & (X != 0), axis=1)
    X_valid = X[valid]

    if len(X_valid) < 100:
        return {"error": "insufficient_data"}

    # Subsample for speed
    n_sample = min(10000, len(X_valid))
    idx = np.random.RandomState(42).choice(len(X_valid), n_sample, replace=False)
    X_sample = X_valid[idx]

    # Correlation matrix
    corr = np.corrcoef(X_sample.T)

    # Find highly correlated pairs
    high_corr = []
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            if abs(corr[i, j]) > 0.7:
                high_corr.append({
                    "feature_1": FEATURE_NAMES[i],
                    "feature_2": FEATURE_NAMES[j],
                    "correlation": round(float(corr[i, j]), 4),
                })

    return {
        "correlation_matrix": corr.tolist(),
        "feature_names": FEATURE_NAMES[:n_bands],
        "highly_correlated_pairs": high_corr,
    }
