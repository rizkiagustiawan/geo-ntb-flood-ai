"""Change Detection for flood monitoring.

Implements pre-event vs during-event SAR change detection per
Schlaffer et al. (2015), Clement et al. (2018), and Liu et al. (2026).
Computes delta backscatter (ΔVV, ΔVH) between a pre-event baseline
and current observation.

The change signal is more robust than absolute thresholding because:
- It eliminates system bias (calibration, terrain)
- It isolates flood-induced backscatter changes from persistent water
- It works across different incidence angles and land cover types

Multi-temporal statistics (mean, std, CV) from time-series stacks
enable unsupervised flood detection without training data.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def compute_delta_sar(
    vv_current: np.ndarray,
    vh_current: np.ndarray,
    vv_baseline: np.ndarray,
    vh_baseline: np.ndarray,
) -> tuple:
    """Compute delta backscatter between current and baseline SAR.

    Per Campo et al. (2026): flood pixels show significant negative
    ΔVV (water replaces land → lower backscatter) and negative ΔVH
    (reduced volume scattering from submerged vegetation).

    Parameters
    ----------
    vv_current, vh_current : np.ndarray — Current SAR backscatter (dB)
    vv_baseline, vh_baseline : np.ndarray — Pre-event baseline (dB)

    Returns
    -------
    tuple[np.ndarray, np.ndarray] — (ΔVV, ΔVH) in dB
    """
    delta_vv = (vv_current - vv_baseline).astype(np.float32)
    delta_vh = (vh_current - vh_baseline).astype(np.float32)

    logger.info("Delta SAR computed: ΔVV [%.1f, %.1f], ΔVH [%.1f, %.1f]",
                np.nanmin(delta_vv), np.nanmax(delta_vv),
                np.nanmin(delta_vh), np.nanmax(delta_vh))
    return delta_vv, delta_vh


def detect_flood_by_change(
    delta_vv: np.ndarray,
    delta_vh: np.ndarray,
    vv_thresh: float = -3.0,
    vh_thresh: float = -3.0,
) -> np.ndarray:
    """Detect flood pixels via SAR change detection.

    Per Liu et al. (2026): significant negative change in both
    VV and VH indicates flooding. Threshold of -3 dB is typical
    for detecting water surface transitions.

    Parameters
    ----------
    delta_vv, delta_vh : np.ndarray — Delta backscatter (dB)
    vv_thresh : float — ΔVV threshold for flood (default -3 dB)
    vh_thresh : float — ΔVH threshold for flood (default -3 dB)

    Returns
    -------
    np.ndarray — uint8 mask: 1=flood, 0=non-flood
    """
    mask = ((delta_vv < vv_thresh) & (delta_vh < vh_thresh)).astype(np.uint8)
    flood_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("Change detection: ΔVV<%.1f & ΔVH<%.1f → %.2f%% flood",
                vv_thresh, vh_thresh, flood_pct)
    return mask


def fetch_baseline_sar(
    fetcher,
    bbox: list,
    current_date: str,
    baseline_days: int = 30,
) -> tuple | None:
    """Fetch pre-event baseline SAR from Sentinel Hub.

    Looks for the latest SAR scene before the flood event
    to establish a "dry" baseline for change detection.

    Parameters
    ----------
    fetcher : SentinelHubFetcher — Initialized fetcher instance
    bbox : list — [W, S, E, N] bounding box
    current_date : str — Current date (YYYY-MM-DD)
    baseline_days : int — Days before current date to search

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None — (VV, VH) baseline arrays
    """
    from sentinelhub import SentinelHubCatalog, BBox, CRS, DataCollection

    current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    search_start = (current_dt - timedelta(days=baseline_days)).strftime("%Y-%m-%dT00:00:00")
    search_end = (current_dt - timedelta(days=3)).strftime("%Y-%m-%dT23:59:59")

    catalog = SentinelHubCatalog(config=fetcher.config)
    bbox_obj = BBox(bbox, crs=CRS.WGS84)

    search_iterator = catalog.search(
        collection=DataCollection.SENTINEL1_IW,
        bbox=bbox_obj,
        time=(search_start, search_end),
        limit=1,
    )

    results = list(search_iterator)
    if not results:
        logger.warning("No baseline SAR found in %s to %s", search_start, search_end)
        return None

    baseline_date = results[0]["properties"]["datetime"][:10]
    logger.info("Baseline SAR found: %s", baseline_date)

    try:
        vv, vh = fetcher.fetch_sentinel1(bbox, baseline_date)
        return vv, vh
    except Exception as e:
        logger.error("Failed to fetch baseline SAR: %s", e)
        return None


def build_change_features(
    vv_current: np.ndarray,
    vh_current: np.ndarray,
    vv_baseline: np.ndarray,
    vh_baseline: np.ndarray,
) -> np.ndarray:
    """Build change detection feature bands for the ML pipeline.

    Returns a 2-band array: [ΔVV, ΔVH] suitable for appending
    to the existing 5-band feature stack.

    Parameters
    ----------
    vv_current, vh_current : np.ndarray — Current SAR
    vv_baseline, vh_baseline : np.ndarray — Baseline SAR

    Returns
    -------
    np.ndarray — (2, H, W) float32 array with [ΔVV, ΔVH]
    """
    delta_vv, delta_vh = compute_delta_sar(
        vv_current, vh_current, vv_baseline, vh_baseline
    )

    features = np.stack([delta_vv, delta_vh])
    logger.info("Change features built: shape=%s", features.shape)
    return features


def compute_temporal_statistics(
    sar_stack: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute multi-temporal statistics from a SAR time-series stack.

    Per Schlaffer et al. (2015): harmonic analysis and statistical
    moments of backscatter time series characterize land cover:
    - Permanent water: low mean, low variance
    - Dry land: high mean, moderate variance
    - Flooded: sudden drop → high CV, negative anomaly

    Parameters
    ----------
    sar_stack : np.ndarray — (n_times, H, W) SAR backscatter stack (dB)

    Returns
    -------
    dict with keys: 'mean', 'std', 'cv', 'min', 'max', 'range'
    """
    valid = np.isfinite(sar_stack)
    sar_clean = np.where(valid, sar_stack, np.nan)

    stats = {
        "mean": np.nanmean(sar_clean, axis=0).astype(np.float32),
        "std": np.nanstd(sar_clean, axis=0).astype(np.float32),
        "min": np.nanmin(sar_clean, axis=0).astype(np.float32),
        "max": np.nanmax(sar_clean, axis=0).astype(np.float32),
    }
    stats["range"] = (stats["max"] - stats["min"]).astype(np.float32)

    # Coefficient of Variation (CV) — normalized dispersion
    mean_abs = np.abs(stats["mean"])
    mean_safe = np.where(mean_abs > 1e-6, mean_abs, 1e-6)
    stats["cv"] = (stats["std"] / mean_safe).astype(np.float32)

    for k, v in stats.items():
        logger.info("Temporal %s: min=%.2f, max=%.2f, mean=%.2f",
                    k, np.nanmin(v), np.nanmax(v), np.nanmean(v))

    return stats


def detect_anomaly_flood(
    vv_current: np.ndarray,
    vv_mean: np.ndarray,
    vv_std: np.ndarray,
    n_sigma: float = 2.0,
) -> np.ndarray:
    """Detect flood via statistical anomaly detection.

    A pixel is flagged as flood if its current backscatter is more
    than n_sigma standard deviations below the temporal mean.
    This is an unsupervised approach that requires no training data.

    Parameters
    ----------
    vv_current : np.ndarray — Current VV backscatter (dB)
    vv_mean : np.ndarray — Temporal mean VV (dB)
    vv_std : np.ndarray — Temporal std VV (dB)
    n_sigma : float — Number of std deviations for anomaly threshold

    Returns
    -------
    np.ndarray — uint8 mask: 1=flood (anomaly), 0=normal
    """
    threshold = vv_mean - n_sigma * np.maximum(vv_std, 1.0)
    mask = (vv_current < threshold).astype(np.uint8)

    flood_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("Anomaly detection: n_sigma=%.1f → %.2f%% flood pixels",
                n_sigma, flood_pct)
    return mask


def detect_adaptive_change(
    delta_vv: np.ndarray,
    delta_vh: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """Detect flood via adaptive thresholding on change signal.

    Uses Otsu's method to automatically determine the optimal
    threshold for separating flood-induced changes from noise.

    Parameters
    ----------
    delta_vv, delta_vh : np.ndarray — Delta backscatter (dB)
    method : str — 'otsu' for adaptive, 'fixed' for legacy -3 dB

    Returns
    -------
    np.ndarray — uint8 mask: 1=flood, 0=non-flood
    """
    if method == "otsu":
        from features import otsu_threshold
        vv_thresh = otsu_threshold(delta_vv)
        vh_thresh = otsu_threshold(delta_vh)
        # Otsu on delta gives the separation point;
        # flood = negative change (below threshold)
        logger.info("Adaptive change thresholds: ΔVV=%.2f dB, ΔVH=%.2f dB",
                    vv_thresh, vh_thresh)
    else:
        vv_thresh = -3.0
        vh_thresh = -3.0
        logger.info("Fixed change thresholds: ΔVV=%.1f dB, ΔVH=%.1f dB",
                    vv_thresh, vh_thresh)

    mask = ((delta_vv < vv_thresh) & (delta_vh < vh_thresh)).astype(np.uint8)
    flood_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("Change detection (method=%s): %.2f%% flood", method, flood_pct)
    return mask
