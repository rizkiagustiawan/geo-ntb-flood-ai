"""Change Detection for flood monitoring.

Implements pre-event vs during-event SAR change detection per
Campo et al. (2026) and Liu et al. (2026). Computes delta backscatter
(ΔVV, ΔVH) between a pre-event baseline and current observation.

The change signal is more robust than absolute thresholding because:
- It eliminates system bias (calibration, terrain)
- It isolates flood-induced backscatter changes from persistent water
- It works across different incidence angles and land cover types
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
