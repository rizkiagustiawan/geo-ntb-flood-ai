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
        logger.info("NDWI computed via Rust engine: min=%.4f, max=%.4f, nan_count=%d",
                    np.nanmin(ndwi), np.nanmax(ndwi), np.count_nonzero(np.isnan(ndwi)))
    except ImportError:
        logger.warning("flood_rs not available, falling back to NumPy for NDWI")
        denom = green + nir
        ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
        logger.info("NDWI computed (NumPy fallback): min=%.4f, max=%.4f, nan_count=%d",
                    np.nanmin(ndwi), np.nanmax(ndwi), np.count_nonzero(np.isnan(ndwi)))
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
        logger.info("Zero-Copy Pipeline: delegating NDWI to Rust GDAL I/O")
        flood_rs.compute_ndwi_io_rust(input_path, output_path)
        logger.info("NDWI written via Rust zero-copy pipeline: %s", output_path)
    except ImportError:
        logger.warning("flood_rs not available — falling back to rasterio + NumPy for NDWI")
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


def compute_sar_threshold(vv_band, vh_band, vv_thresh=-15.0, vh_thresh=-20.0):
    """Compute SAR flood mask using dB thresholds.
    Pixels with VV < vv_thresh AND VH < vh_thresh are classified as water (1).
    Returns uint8 array: 1=water, 0=non-water.

    Heavy pixel-wise arithmetic is delegated to the Rust engine (flood_rs)
    for parallel computation via Rayon. Falls back to NumPy if unavailable."""
    vv = vv_band.astype(np.float32)
    vh = vh_band.astype(np.float32)

    try:
        import flood_rs
        orig_shape = vv.shape
        if vv.ndim == 1:
            vv = vv.reshape(1, -1)
            vh = vh.reshape(1, -1)
        mask = flood_rs.calculate_sar_flood_mask(vv, vh, vv_thresh, vh_thresh)
        mask = mask.reshape(orig_shape)
        water_pct = 100.0 * np.sum(mask) / mask.size
        logger.info("SAR threshold mask via Rust: VV<%.1f & VH<%.1f -> %.2f%% water pixels",
                    vv_thresh, vh_thresh, water_pct)
    except ImportError:
        logger.warning("flood_rs not available, falling back to NumPy for SAR mask")
        mask = ((vv < vv_thresh) & (vh < vh_thresh)).astype(np.uint8)
        water_pct = 100.0 * np.sum(mask) / mask.size
        logger.info("SAR threshold mask (NumPy fallback): VV<%.1f & VH<%.1f -> %.2f%% water pixels",
                    vv_thresh, vh_thresh, water_pct)
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


def build_feature_stack():
    """Build multi-band feature stack from preprocessed rasters.
    Output bands: [NDWI, SAR_mask, Slope, VV, VH]
    Saved as feature_stack.tif in data/processed/.

    NDWI is computed via the Zero-Copy Pipeline: Rust reads the S2 TIFF with
    GDAL, computes NDWI in parallel, and writes a standalone NDWI TIFF.
    Python then reads the result back for the final feature stack assembly.

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
        ref_transform = ds.transform
        ref_width = ds.width
        ref_height = ds.height
        ref_crs = ds.crs
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
        "count": 5,
        "dtype": "float32",
        "compress": "lzw",
    })

    band_names = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB"]

    with rasterio.open(out_path, "w", **profile) as dst:
        # Band 1: NDWI (read from zero-copy pipeline output)
        with rasterio.open(ndwi_intermediate_path) as ndwi_ds:
            ndwi = ndwi_ds.read(1, out_dtype=np.float32)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
        dst.write(ndwi, 1)
        del ndwi
        gc.collect()
        logger.info("Written band 1/5: NDWI (from Rust zero-copy pipeline)")

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

    # Cleanup intermediate NDWI file
    try:
        ndwi_intermediate_path.unlink()
        logger.info("Cleaned up intermediate NDWI file: %s", ndwi_intermediate_path)
    except OSError:
        logger.warning("Could not remove intermediate NDWI file: %s", ndwi_intermediate_path)

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
