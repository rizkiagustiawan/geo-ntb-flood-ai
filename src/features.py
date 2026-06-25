"""
features.py - Feature Engineering for NTB Flood Detection.
Computes NDWI, SAR threshold mask, DEM slope, and outputs multi-band feature stack.

NDWI computation uses the **Zero-Copy Pipeline**: Python passes file paths to the
Rust engine (flood_rs.compute_ndwi_io_rust), which reads bands via GDAL, computes
NDWI in parallel with Rayon, and writes the result — no NumPy intermediary needed.
"""

import gc
import sys
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def otsu_threshold(data: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu's optimal threshold for separating bimodal distribution.

    Operates on valid (non-NaN, non-zero) values only.

    Parameters
    ----------
    data : np.ndarray — 1D array of SAR backscatter values (dB)
    n_bins : int — Number of histogram bins

    Returns
    -------
    float — Optimal threshold value
    """
    valid = data[np.isfinite(data)]
    if len(valid) < 10:
        return float(np.nanmedian(valid)) if len(valid) > 0 else -15.0

    hist, bin_edges = np.histogram(valid, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        return float(np.nanmedian(valid))

    sum_total = np.dot(bin_centers, hist)

    sum_bg = 0.0
    w_bg = 0
    max_variance = 0.0
    threshold = bin_centers[0]

    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break

        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg

        variance = float(w_bg * w_fg * (mean_bg - mean_fg) ** 2)
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    logger.info("Otsu threshold computed: %.2f dB (n=%d, bins=%d)", threshold, len(valid), n_bins)
    return float(threshold)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def compute_ndwi(green_band, nir_band):
    """Compute Normalized Difference Water Index: (Green - NIR) / (Green + NIR).
    Returns float32 array in [-1, 1]. NoData where denominator is zero.

    Heavy pixel-wise arithmetic is delegated to the Rust engine (flood_rs)
    for parallel computation via Rayon. Falls back to NumPy if unavailable."""
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)

    try:
        import flood_rs
        # Rust expects 2-D arrays
        orig_shape = green.shape
        if green.ndim == 1:
            green = green.reshape(1, -1)
            nir = nir.reshape(1, -1)
        ndwi = flood_rs.calculate_ndwi(green, nir)
        ndwi = ndwi.reshape(orig_shape)
        valid_count = np.count_nonzero(~np.isnan(ndwi))
        if valid_count > 0:
            logger.info("NDWI computed via Rust engine: min=%.4f, max=%.4f, nan_count=%d",
                        np.nanmin(ndwi), np.nanmax(ndwi), np.count_nonzero(np.isnan(ndwi)))
        else:
            logger.warning("NDWI computed via Rust: all NaN (no valid pixels)")
    except ImportError:
        logger.warning("flood_rs not available, falling back to NumPy for NDWI")
        denom = green + nir
        ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
        valid_count = np.count_nonzero(~np.isnan(ndwi))
        if valid_count > 0:
            logger.info("NDWI computed (NumPy fallback): min=%.4f, max=%.4f, nan_count=%d",
                        np.nanmin(ndwi), np.nanmax(ndwi), np.count_nonzero(np.isnan(ndwi)))
        else:
            logger.warning("NDWI computed (NumPy): all NaN (no valid pixels)")
    return ndwi


def compute_ndwi_zero_copy(input_path: str, output_path: str):
    """Compute NDWI via the Zero-Copy Pipeline (Rust GDAL I/O).

    Delegates the entire read → compute → write cycle to flood_rs.compute_ndwi_io_rust,
    which opens the TIFF with GDAL in Rust, reads bands into Rust memory, computes
    NDWI in parallel (Rayon), and writes the result TIFF preserving GeoTransform + SRS.

    Falls back to rasterio + NumPy if flood_rs is not available.

    Parameters
    ----------
    input_path : str — Path to multi-band S2 TIFF (Band 1=Green, Band 2=NIR).
    output_path : str — Path for the output single-band NDWI TIFF.
    """
    try:
        import flood_rs
        if not hasattr(flood_rs, 'compute_ndwi_io_rust'):
            raise AttributeError("compute_ndwi_io_rust not available in flood_rs")
        logger.info("Zero-Copy Pipeline: delegating NDWI to Rust GDAL I/O")
        flood_rs.compute_ndwi_io_rust(input_path, output_path)
        logger.info("NDWI written via Rust zero-copy pipeline: %s", output_path)
    except (ImportError, AttributeError) as e:
        logger.warning("flood_rs zero-copy not available (%s) — falling back to rasterio + NumPy for NDWI", e)
        with rasterio.open(input_path) as ds:
            green = ds.read(1, out_dtype=np.float32)
            nir = ds.read(2, out_dtype=np.float32)
            profile = ds.profile.copy()

        denom = green + nir
        ndwi = np.where(denom != 0, (green - nir) / denom, np.nan).astype(np.float32)
        ndwi = np.nan_to_num(ndwi, nan=0.0)

        profile.update({"count": 1, "dtype": "float32"})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndwi, 1)
            dst.set_band_description(1, "NDWI")
        logger.info("NDWI written via rasterio fallback: %s", output_path)


def compute_sar_threshold(vv_band, vh_band, vv_thresh=-15.0, vh_thresh=-20.0, method='otsu'):
    """Compute SAR flood mask using adaptive (Otsu) or fixed thresholds.

    Parameters
    ----------
    vv_band, vh_band : np.ndarray — VV/VH backscatter in dB
    vv_thresh, vh_thresh : float — Fixed thresholds (used only when method='fixed')
    method : str — 'otsu' for adaptive, 'fixed' for legacy hardcoded

    Returns
    -------
    np.ndarray — uint8 mask: 1=water, 0=non-water
    """
    vv = vv_band.astype(np.float32)
    vh = vh_band.astype(np.float32)

    if method == 'otsu':
        vv_t = otsu_threshold(vv)
        vh_t = otsu_threshold(vh)
        logger.info("Adaptive SAR thresholds: VV=%.2f dB, VH=%.2f dB", vv_t, vh_t)
    else:
        vv_t = vv_thresh
        vh_t = vh_thresh
        logger.info("Fixed SAR thresholds: VV=%.1f dB, VH=%.1f dB", vv_t, vh_t)

    try:
        import flood_rs
        orig_shape = vv.shape
        if vv.ndim == 1:
            vv = vv.reshape(1, -1)
            vh = vh.reshape(1, -1)
        mask = flood_rs.calculate_sar_flood_mask(vv, vh, vv_t, vh_t)
        mask = mask.reshape(orig_shape)
    except ImportError:
        logger.warning("flood_rs not available, falling back to NumPy for SAR mask")
        mask = ((vv < vv_t) & (vh < vh_t)).astype(np.uint8)

    water_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("SAR mask (method=%s): VV<%.2f & VH<%.2f -> %.2f%% water",
                method, vv_t, vh_t, water_pct)
    return mask


def compute_slope(elevation, transform):
    """Compute slope from DEM elevation in degrees using numpy gradient.
    Uses pixel spacing derived from affine transform with cosine correction
    for EPSG:4326 at NTB latitude (~-8.5°S)."""
    import math
    elev = elevation.astype(np.float32)
    # Pixel spacing in metres (degrees → metres with cosine correction)
    cos_lat = math.cos(math.radians(-8.5))
    dx = abs(transform[0]) * 111320.0 * cos_lat  # longitude shrinks by cos(lat)
    dy = abs(transform[4]) * 111320.0

    grad_y, grad_x = np.gradient(elev, dy, dx)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    slope_deg = np.degrees(slope_rad)

    logger.info("Slope computed: min=%.2f, max=%.2f deg", np.nanmin(slope_deg), np.nanmax(slope_deg))
    return slope_deg


def compute_hand(elevation, transform, drainage_threshold_percentile=10):
    """Compute Height Above Nearest Drainage (HAND).

    For each pixel, computes the elevation difference between that pixel
    and the elevation of the nearest drainage channel pixel. Drainage
    channels are identified as pixels in the lowest percentile of elevation.

    Reference: Nobre et al. (2011), Tian et al. (2026)

    Parameters
    ----------
    elevation : np.ndarray — DEM elevation in metres (float32)
    transform : rasterio transform — Affine transform for pixel spacing
    drainage_threshold_percentile : float — Percentile of elevation to define drainage

    Returns
    -------
    np.ndarray — HAND values in metres (float32). Low values = near drainage = flood-prone.
    """
    from scipy.ndimage import distance_transform_edt

    elev = elevation.astype(np.float32)
    h, w = elev.shape

    # Identify drainage pixels: lowest percentile of elevation
    valid = np.isfinite(elev) & (elev > -9000)
    if not np.any(valid):
        logger.warning("No valid DEM pixels for HAND — returning zeros")
        return np.zeros_like(elev)

    threshold = np.percentile(elev[valid], drainage_threshold_percentile)
    drainage_mask = valid & (elev <= threshold)

    n_drainage = int(np.sum(drainage_mask))
    logger.info("HAND: drainage threshold=%.2fm (p%d), %d drainage pixels (%.1f%%)",
                threshold, drainage_threshold_percentile,
                n_drainage, 100.0 * n_drainage / (h * w))

    # Distance transform: for each non-drainage pixel, find nearest drainage pixel
    # We invert the mask so drainage=True → 0 distance
    binary = (~drainage_mask & valid).astype(np.uint8)

    # Use scipy distance_transform_edt to get nearest drainage indices
    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(binary, return_distances=True, return_indices=True)

    # HAND = elevation of current pixel - elevation of nearest drainage pixel
    # indices has shape (ndim, H, W) — use advanced indexing
    nearest_drainage_elev = elev[indices[0], indices[1]]
    hand = np.maximum(elev - nearest_drainage_elev, 0.0).astype(np.float32)

    # Mask invalid pixels
    hand[~valid] = np.nan

    logger.info("HAND computed: min=%.2f, max=%.2f, mean=%.2f m",
                np.nanmin(hand), np.nanmax(hand), np.nanmean(hand))
    return hand


def build_feature_stack():
    """Build multi-band feature stack from preprocessed rasters.
    Output bands: [NDWI, SAR_mask, Slope, VV, VH, HAND]
    Saved as feature_stack.tif in data/processed/.

    Applies SAR preprocessing (Refined Lee speckle filter, noise removal)
    before thresholding. Uses adaptive Otsu thresholding instead of fixed
    thresholds per Cao et al. (2024).

    HAND (Height Above Nearest Drainage) added per Tian et al. (2026).

    Memory-optimized: writes bands sequentially instead of np.stack()."""
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE STACK")
    logger.info("=" * 60)

    # --- Step 1: NDWI via Zero-Copy Pipeline (Rust GDAL I/O) ---
    s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
    ndwi_intermediate_path = PROCESSED_DIR / "ndwi_intermediate.tif"

    if not s2_path.exists():
        raise FileNotFoundError(f"Missing: {s2_path}")

    compute_ndwi_zero_copy(str(s2_path), str(ndwi_intermediate_path))

    # Read reference metadata from S2 for output profile
    with rasterio.open(s2_path) as ds:
        ref_profile = ds.profile.copy()
        ref_width = ds.width
        ref_height = ds.height
    logger.info("Loaded Sentinel-2 metadata: %dx%d", ref_width, ref_height)

    # Load Sentinel-1 (VV, VH)
    s1_path = PROCESSED_DIR / "sentinel1_reproj.tif"
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing: {s1_path}")
    with rasterio.open(s1_path) as ds:
        s1_shape = (ds.height, ds.width)
    logger.info("Loaded Sentinel-1 metadata")

    # Load DEM
    dem_path = PROCESSED_DIR / "dem_reproj.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"Missing: {dem_path}")
    with rasterio.open(dem_path) as ds:
        dem_shape = (ds.height, ds.width)
    logger.info("Loaded DEM metadata")

    # Validate dimensions match
    ref_shape = (ref_height, ref_width)
    shapes = {"s2": ref_shape, "s1": s1_shape, "dem": dem_shape}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        logger.error("Shape mismatch across rasters: %s", shapes)
        raise RuntimeError(f"Raster shape mismatch: {shapes}. Run preprocess.py first.")
    logger.info("All rasters aligned: %s", ref_shape)

    # Prepare output file
    out_path = PROCESSED_DIR / "feature_stack.tif"
    profile = ref_profile.copy()
    profile.update({
        "count": 6,
        "dtype": "float32",
        "compress": "lzw",
    })

    band_names = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB", "HAND_m"]

    with rasterio.open(out_path, "w", **profile) as dst:
        # Band 1: NDWI (read from zero-copy pipeline output)
        with rasterio.open(ndwi_intermediate_path) as ndwi_ds:
            ndwi = ndwi_ds.read(1, out_dtype=np.float32)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
        dst.write(ndwi, 1)
        del ndwi
        gc.collect()
        logger.info("Written band 1/6: NDWI (from Rust zero-copy pipeline)")

        # Band 2: SAR flood mask (WITH PREPROCESSING)
        with rasterio.open(s1_path) as s1_ds:
            vv_raw = s1_ds.read(1, out_dtype=np.float32)
            vh_raw = s1_ds.read(2, out_dtype=np.float32)

        # SAR preprocessing: speckle filter + noise removal
        from sar_preprocess import preprocess_sar
        vv_proc, vh_proc = preprocess_sar(
            vv_raw, vh_raw,
            apply_lee=True, lee_window=7,
            normalize_angle=False,  # no angle map available from GEE
            remove_noise=True,
        )

        # Adaptive thresholding (Otsu)
        sar_mask = compute_sar_threshold(vv_proc, vh_proc, method='otsu')
        dst.write(sar_mask.astype(np.float32), 2)
        del sar_mask, vv_raw, vh_raw
        gc.collect()
        logger.info("Written band 2/6: SAR_flood_mask (adaptive Otsu + speckle filtered)")

        # Band 3: Slope (load DEM, compute, keep elevation for HAND)
        with rasterio.open(dem_path) as dem_ds:
            elevation = dem_ds.read(1, out_dtype=np.float32)
            dem_transform = dem_ds.transform
        slope = compute_slope(elevation, dem_transform)
        slope = np.nan_to_num(slope, nan=0.0)
        dst.write(slope, 3)
        del slope
        gc.collect()
        logger.info("Written band 3/5: Slope_deg")

        # Band 4 & 5: PREPROCESSED VV and VH
        vv_proc = np.nan_to_num(vv_proc, nan=0.0)
        dst.write(vv_proc, 4)
        del vv_proc
        gc.collect()
        logger.info("Written band 4/6: VV_dB (preprocessed)")

        vh_proc = np.nan_to_num(vh_proc, nan=0.0)
        dst.write(vh_proc, 5)
        del vh_proc
        gc.collect()
        logger.info("Written band 5/6: VH_dB (preprocessed)")

        # Band 6: HAND (Height Above Nearest Drainage)
        hand = compute_hand(elevation, dem_transform)
        hand = np.nan_to_num(hand, nan=0.0)
        dst.write(hand, 6)
        del hand, elevation
        gc.collect()
        logger.info("Written band 6/6: HAND_m (Height Above Nearest Drainage)")

        # Set band descriptions
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)

    # Cleanup intermediate NDWI file
    try:
        ndwi_intermediate_path.unlink()
        logger.info("Cleaned up intermediate NDWI file: %s", ndwi_intermediate_path)
    except OSError:
        logger.warning("Could not remove intermediate NDWI file: %s", ndwi_intermediate_path)

    logger.info("Feature stack saved: %s (6 bands, %dx%d)", out_path, ref_height, ref_width)
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)

    return out_path


if __name__ == "__main__":
    try:
        build_feature_stack()
    except Exception as exc:
        logger.error("FEATURE ENGINEERING FAILED: %s", exc)
        sys.exit(1)
