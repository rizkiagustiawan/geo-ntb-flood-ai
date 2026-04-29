"""
preprocess.py - Preprocessing for NTB Flood Detection.
CRS validation, reprojection, resampling, and tiling of raw rasters.
"""

import gc
import sys
import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_CRS = "EPSG:4326"
TARGET_RES = 10  # metres
TILE_SIZE = 256  # pixels

EXPECTED_FILES = {
    "sentinel1": "sentinel1_vv_vh.tif",
    "sentinel2": "sentinel2_green_nir.tif",
    "dem": "dem_srtm_30m.tif",
}


def validate_raw_files():
    """Check all expected raw files exist."""
    missing = []
    for name, fname in EXPECTED_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        for m in missing:
            logger.error("Missing raw file: %s", m)
        raise FileNotFoundError(f"Missing raw files: {missing}")
    logger.info("All %d raw files validated", len(EXPECTED_FILES))


def check_crs(filepath):
    """Check CRS of a raster. Returns CRS string."""
    with rasterio.open(filepath) as ds:
        crs = ds.crs
        if crs is None:
            raise RuntimeError(f"No CRS defined for {filepath}. STOP.")
        logger.info("CRS check: %s -> %s", filepath.name, crs)
        return str(crs)


def reproject_raster(src_path, dst_path, target_crs=TARGET_CRS, target_res=None):
    """Reproject raster to target CRS and optionally resample resolution."""
    with rasterio.open(src_path) as src:
        if str(src.crs) == target_crs and target_res is None:
            logger.info("Already in %s, copying: %s", target_crs, src_path.name)
            import shutil
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return dst_path

        if target_res:
            # Convert metres to degrees (approximate at equator)
            res_deg = target_res / 111320.0
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=res_deg
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

        # Force Float32 to halve memory (Float64 is overkill for satellite data)
        dst_dtype = src.dtypes[0]
        if dst_dtype == 'float64':
            dst_dtype = 'float32'
            logger.info("Downcast %s: float64 -> float32 (memory optimization)", src_path.name)

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": dst_dtype,
        })

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

        logger.info("Reprojected %s -> %s (%dx%d)", src_path.name, target_crs, width, height)
        return dst_path


def resample_to_match(src_path, ref_path, dst_path):
    """Resample src_path to match ref_path spatial extent and resolution."""
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = ref.crs

    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": ref_crs,
            "transform": ref_transform,
            "width": ref_width,
            "height": ref_height,
        })

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )

        logger.info("Resampled %s to match %s (%dx%d)", src_path.name, ref_path.name, ref_width, ref_height)
        return dst_path


def tile_raster(src_path, tile_dir, tile_size=TILE_SIZE):
    """Tile raster into tile_size x tile_size blocks. Returns list of tile paths."""
    tile_dir = Path(tile_dir)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tiles = []

    with rasterio.open(src_path) as src:
        n_cols = math.ceil(src.width / tile_size)
        n_rows = math.ceil(src.height / tile_size)
        logger.info("Tiling %s -> %d x %d = %d tiles", src_path.name, n_cols, n_rows, n_cols * n_rows)

        for row in range(n_rows):
            for col in range(n_cols):
                x_off = col * tile_size
                y_off = row * tile_size
                w = min(tile_size, src.width - x_off)
                h = min(tile_size, src.height - y_off)

                window = Window(x_off, y_off, w, h)
                transform = src.window_transform(window)
                data = src.read(window=window)

                # Skip empty tiles
                if np.all(data == 0) or np.all(np.isnan(data)):
                    continue

                tile_name = f"{src_path.stem}_tile_{row:03d}_{col:03d}.tif"
                tile_path = tile_dir / tile_name

                meta = src.meta.copy()
                meta.update({"width": w, "height": h, "transform": transform})

                with rasterio.open(tile_path, "w", **meta) as dst:
                    dst.write(data)

                tiles.append(tile_path)

        logger.info("Created %d non-empty tiles from %s", len(tiles), src_path.name)

    return tiles


def run_preprocessing():
    """Run full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING - NTB FLOOD DETECTION")
    logger.info("=" * 60)

    # 1. Validate raw files
    validate_raw_files()

    # 2. Check CRS for all files
    crs_map = {}
    for name, fname in EXPECTED_FILES.items():
        path = RAW_DIR / fname
        crs_map[name] = check_crs(path)

    # 3. Reproject all to target CRS
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reprojected = {}

    # Sentinel-1 as reference (10m)
    s1_reproj = PROCESSED_DIR / "sentinel1_reproj.tif"
    reproject_raster(RAW_DIR / EXPECTED_FILES["sentinel1"], s1_reproj, TARGET_CRS)
    reprojected["sentinel1"] = s1_reproj

    # Sentinel-2
    s2_reproj = PROCESSED_DIR / "sentinel2_reproj.tif"
    reproject_raster(RAW_DIR / EXPECTED_FILES["sentinel2"], s2_reproj, TARGET_CRS)
    reprojected["sentinel2"] = s2_reproj

    # DEM - reproject then resample to match Sentinel-1 grid
    dem_reproj_tmp = PROCESSED_DIR / "dem_reproj_tmp.tif"
    reproject_raster(RAW_DIR / EXPECTED_FILES["dem"], dem_reproj_tmp, TARGET_CRS)
    dem_resampled = PROCESSED_DIR / "dem_reproj.tif"
    resample_to_match(dem_reproj_tmp, s1_reproj, dem_resampled)
    reprojected["dem"] = dem_resampled

    # Clean temp
    if dem_reproj_tmp.exists() and dem_reproj_tmp != dem_resampled:
        dem_reproj_tmp.unlink()

    # 4. Validate CRS consistency
    ref_crs = None
    for name, path in reprojected.items():
        crs = check_crs(path)
        if ref_crs is None:
            ref_crs = crs
        elif crs != ref_crs:
            raise RuntimeError(f"CRS MISMATCH after reprojection: {name}={crs}, expected={ref_crs}. STOP.")

    logger.info("All rasters aligned to %s", ref_crs)

    # 5. Tile all reprojected rasters
    tile_base = PROCESSED_DIR / "tiles"
    all_tiles = {}
    for name, path in reprojected.items():
        tile_dir = tile_base / name
        tiles = tile_raster(path, tile_dir, TILE_SIZE)
        all_tiles[name] = tiles
        logger.info("Tiled %s: %d tiles", name, len(tiles))
        gc.collect()

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return {"reprojected": reprojected, "tiles": all_tiles}


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as exc:
        logger.error("PREPROCESSING FAILED: %s", exc)
        sys.exit(1)
