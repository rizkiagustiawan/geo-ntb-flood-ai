"""
features.py - Feature Engineering for NTB Flood Detection.
Computes NDWI, SAR threshold mask, DEM slope, and outputs multi-band feature stack.
"""

import gc
import sys
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def compute_ndwi(green_band, nir_band):
    """Compute Normalized Difference Water Index: (Green - NIR) / (Green + NIR).
    Returns float32 array in [-1, 1]. NoData where denominator is zero."""
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
    logger.info("NDWI computed: min=%.4f, max=%.4f, nan_count=%d",
                np.nanmin(ndwi), np.nanmax(ndwi), np.count_nonzero(np.isnan(ndwi)))
    return ndwi


def compute_sar_threshold(vv_band, vh_band, vv_thresh=-15.0, vh_thresh=-20.0):
    """Compute SAR flood mask using dB thresholds.
    Pixels with VV < vv_thresh AND VH < vh_thresh are classified as water (1).
    Returns uint8 array: 1=water, 0=non-water."""
    vv = vv_band.astype(np.float32)
    vh = vh_band.astype(np.float32)
    mask = ((vv < vv_thresh) & (vh < vh_thresh)).astype(np.uint8)
    water_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("SAR threshold mask: VV<%.1f & VH<%.1f -> %.2f%% water pixels",
                vv_thresh, vh_thresh, water_pct)
    return mask


def compute_slope(elevation, transform):
    """Compute slope from DEM elevation in degrees using numpy gradient.
    Uses pixel spacing derived from affine transform."""
    elev = elevation.astype(np.float32)
    # Pixel spacing in metres (approximate from degrees)
    dx = abs(transform[0]) * 111320.0  # degrees to metres at equator
    dy = abs(transform[4]) * 111320.0

    grad_y, grad_x = np.gradient(elev, dy, dx)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    slope_deg = np.degrees(slope_rad)

    logger.info("Slope computed: min=%.2f, max=%.2f deg", np.nanmin(slope_deg), np.nanmax(slope_deg))
    return slope_deg


def build_feature_stack():
    """Build multi-band feature stack from preprocessed rasters.
    Output bands: [NDWI, SAR_mask, Slope, VV, VH]
    Saved as feature_stack.tif in data/processed/.
    Memory-optimized: writes bands sequentially instead of np.stack()."""
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE STACK")
    logger.info("=" * 60)

    # Load Sentinel-2 (B3=Green, B8=NIR)
    s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
    if not s2_path.exists():
        raise FileNotFoundError(f"Missing: {s2_path}")
    with rasterio.open(s2_path) as ds:
        green = ds.read(1, out_dtype=np.float32)
        nir = ds.read(2, out_dtype=np.float32)
        ref_profile = ds.profile.copy()
        ref_transform = ds.transform
        ref_width = ds.width
        ref_height = ds.height
        ref_crs = ds.crs
    logger.info("Loaded Sentinel-2: %dx%d", ref_width, ref_height)

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
    shapes = {"s2": green.shape, "s1": s1_shape, "dem": dem_shape}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        logger.error("Shape mismatch across rasters: %s", shapes)
        raise RuntimeError(f"Raster shape mismatch: {shapes}. Run preprocess.py first.")
    logger.info("All rasters aligned: %s", green.shape)

    # Prepare output file
    out_path = PROCESSED_DIR / "feature_stack.tif"
    profile = ref_profile.copy()
    profile.update({
        "count": 5,
        "dtype": "float32",
        "compress": "lzw",
    })

    band_names = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB"]

    with rasterio.open(out_path, "w", **profile) as dst:
        # Band 1: NDWI (computed from S2 green + nir, then free them)
        ndwi = compute_ndwi(green, nir)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
        dst.write(ndwi, 1)
        del green, nir, ndwi
        gc.collect()
        logger.info("Written band 1/5: NDWI (freed S2 bands)")

        # Band 2: SAR flood mask (load S1, compute, free)
        with rasterio.open(s1_path) as s1_ds:
            vv = s1_ds.read(1, out_dtype=np.float32)
            vh = s1_ds.read(2, out_dtype=np.float32)
        sar_mask = compute_sar_threshold(vv, vh)
        dst.write(sar_mask.astype(np.float32), 2)
        del sar_mask
        gc.collect()
        logger.info("Written band 2/5: SAR_flood_mask")

        # Band 3: Slope (load DEM, compute, free)
        with rasterio.open(dem_path) as dem_ds:
            elevation = dem_ds.read(1, out_dtype=np.float32)
            dem_transform = dem_ds.transform
        slope = compute_slope(elevation, dem_transform)
        slope = np.nan_to_num(slope, nan=0.0)
        dst.write(slope, 3)
        del elevation, slope
        gc.collect()
        logger.info("Written band 3/5: Slope_deg (freed DEM)")

        # Band 4 & 5: VV and VH (already loaded above)
        vv = np.nan_to_num(vv, nan=0.0)
        dst.write(vv, 4)
        del vv
        gc.collect()
        logger.info("Written band 4/5: VV_dB")

        vh = np.nan_to_num(vh, nan=0.0)
        dst.write(vh, 5)
        del vh
        gc.collect()
        logger.info("Written band 5/5: VH_dB")

        # Set band descriptions
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)

    logger.info("Feature stack saved: %s (5 bands, %dx%d)", out_path, ref_height, ref_width)
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
