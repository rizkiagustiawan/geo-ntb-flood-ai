"""Mock Data Generator for A.E.C.O Pipeline Testing.

Generates synthetic satellite rasters that mimic real Sentinel-1/2
and DEM data for NTB region, enabling full pipeline testing without
actual satellite downloads.

Usage:
    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --size 512 --with-flood
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
GSW_DIR = DATA_DIR / "gsw"

# NTB bounding box (Sumbawa)
BBOX = [116.6, -9.1, 119.2, -8.1]


def make_geotiff(path: Path, data: np.ndarray, bbox: list, crs: str = "EPSG:4326"):
    """Write a GeoTIFF with proper CRS and transform."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        count = 1
    else:
        count = data.shape[0]

    h, w = data.shape[-2], data.shape[-1]
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(str(path), "w", **profile) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("  Created: %s (%d bands, %dx%d, %.2f MB)", path.name, count, h, w, size_mb)


def generate_dem(size: int, with_topography: bool = True) -> np.ndarray:
    """Generate synthetic DEM with realistic topography.

    Creates elevation with:
    - Mountain range in the center (like Sumbawa's spine)
    - Coastal lowlands
    - River valleys (drainage channels)
    """
    np.random.seed(42)
    h, w = size, size

    # Base elevation: mountain spine in center
    y, x = np.mgrid[0:h, 0:w]
    center_y = h // 2
    center_x = w // 2

    # Mountain ridge (east-west)
    ridge = 2000 * np.exp(-((y - center_y) ** 2) / (2 * (h * 0.15) ** 2))

    # Some random terrain
    terrain = np.random.randn(h, w).astype(np.float32) * 100
    from scipy.ndimage import gaussian_filter
    terrain = gaussian_filter(terrain, sigma=size // 20)

    # Coastal gradient (lower towards edges)
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    coastal = np.clip(dist_from_center / (size * 0.4), 0, 1) * -500

    # River valleys (V-shaped depressions)
    river_y = center_y + np.sin(x / (w * 0.1)) * (h * 0.1)
    river_mask = np.exp(-((y - river_y) ** 2) / (2 * (h * 0.02) ** 2))
    river = river_mask * -200

    elevation = (ridge + terrain + coastal + river + 500).astype(np.float32)
    elevation = np.clip(elevation, -10, 3000)

    return elevation


def generate_sentinel1(size: int, elevation: np.ndarray, with_flood: bool = True) -> tuple:
    """Generate synthetic Sentinel-1 SAR VV/VH bands.

    Realistic SAR backscatter:
    - Water: VV < -18 dB, VH < -25 dB (dark)
    - Land: VV ~ -10 dB, VH ~ -17 dB
    - Urban: VV ~ -5 dB, VH ~ -12 dB (bright)
    - Forest: VV ~ -12 dB, VH ~ -20 dB
    """
    np.random.seed(123)
    h, w = size, size

    # Base land backscatter
    vv = np.random.randn(h, w).astype(np.float32) * 2 - 10
    vh = np.random.randn(h, w).astype(np.float32) * 2 - 17

    # Water bodies: low backscatter in valleys/rivers
    center_y = h // 2
    x_grid = np.arange(w)
    river_y = center_y + np.sin(x_grid / (w * 0.1)) * (h * 0.1)
    y_grid = np.arange(h)[:, None]
    river_mask = np.exp(-((y_grid - river_y) ** 2) / (2 * (h * 0.02) ** 2))

    # Coastal water (edges)
    dist_from_edge = np.minimum(
        np.minimum(np.arange(h)[:, None], h - 1 - np.arange(h)[:, None]),
        np.minimum(np.arange(w)[None, :], w - 1 - np.arange(w)[None, :])
    )
    coastal_water = (dist_from_edge < size * 0.05).astype(np.float32)

    # Permanent water
    water_mask = ((river_mask > 0.7) | (coastal_water > 0)).astype(np.float32)

    # Apply water backscatter
    vv = vv * (1 - water_mask * 0.8) - water_mask * 10
    vh = vh * (1 - water_mask * 0.7) - water_mask * 12

    # Add speckle noise (multiplicative in linear, additive in dB)
    speckle_vv = np.random.randn(h, w).astype(np.float32) * 1.5
    speckle_vh = np.random.randn(h, w).astype(np.float32) * 1.0
    vv += speckle_vv
    vh += speckle_vh

    # Flood: additional low-backscatter areas in lowlands and along rivers
    flood_mask = np.zeros((h, w), dtype=np.float32)
    if with_flood:
        # Flood zone 1: Low-elevation coastal areas
        coastal_flood = (elevation < 50).astype(np.float32)

        # Flood zone 2: Along river valleys in lowlands
        lowland = (elevation < 300).astype(np.float32)
        river_flood = lowland * (river_mask > 0.3).astype(np.float32)

        # Flood zone 3: Random scattered flood patches
        np.random.seed(999)
        scatter = np.random.rand(h, w).astype(np.float32)
        scatter_flood = (scatter > 0.98).astype(np.float32) * (elevation < 500).astype(np.float32)

        # Combine flood zones
        flood_mask = np.clip(coastal_flood + river_flood + scatter_flood, 0, 1).astype(np.float32)

        # Apply flood backscatter
        vv = vv * (1 - flood_mask * 0.7) - flood_mask * 8
        vh = vh * (1 - flood_mask * 0.6) - flood_mask * 10

    return vv.astype(np.float32), vh.astype(np.float32), flood_mask


def generate_sentinel2(size: int, flood_mask: np.ndarray) -> tuple:
    """Generate synthetic Sentinel-2 Green/NIR bands.

    Reflectance values:
    - Water: Green ~0.08, NIR ~0.02 (high NDWI)
    - Vegetation: Green ~0.05, NIR ~0.30 (low NDWI)
    - Bare soil: Green ~0.15, NIR ~0.20
    """
    np.random.seed(456)
    h, w = size, size

    # Vegetation reflectance
    green = np.random.randn(h, w).astype(np.float32) * 0.02 + 0.05
    nir = np.random.randn(h, w).astype(np.float32) * 0.05 + 0.25

    # Water areas: high green, low NIR
    water_mask = flood_mask > 0
    green[water_mask] = np.random.randn(int(np.sum(water_mask))).astype(np.float32) * 0.01 + 0.08
    nir[water_mask] = np.random.randn(int(np.sum(water_mask))).astype(np.float32) * 0.005 + 0.02

    # Clip to valid reflectance range
    green = np.clip(green, 0, 1).astype(np.float32)
    nir = np.clip(nir, 0, 1).astype(np.float32)

    return green, nir


def generate_gsw_occurrence(size: int, permanent_water_mask: np.ndarray) -> np.ndarray:
    """Generate synthetic GSW occurrence data.

    Occurrence = percentage of time water was present (1984-2021).
    Permanent water: >80%, Seasonal: 20-80%, Dry: <20%
    """
    np.random.seed(789)
    h, w = size, size

    # Start with permanent water from river/coast
    occurrence = permanent_water_mask.astype(np.float32) * 90

    # Add some noise for seasonal variation
    noise = np.random.randn(h, w).astype(np.float32) * 10
    occurrence = np.clip(occurrence + noise, 0, 100).astype(np.uint8)

    return occurrence


def generate_mock_data(size: int = 256, with_flood: bool = True):
    """Generate complete mock dataset for pipeline testing.

    Parameters
    ----------
    size : int — Raster dimensions (size × size pixels)
    with_flood : bool — Whether to include flood pixels
    """
    logger.info("=" * 60)
    logger.info("GENERATING MOCK DATA (size=%d, with_flood=%s)", size, with_flood)
    logger.info("=" * 60)

    # Create directories
    for d in [PROCESSED_DIR, LABELS_DIR, MODELS_DIR, PREDICTIONS_DIR, GSW_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. DEM
    logger.info("Generating DEM...")
    elevation = generate_dem(size)

    # 2. Sentinel-1 SAR
    logger.info("Generating Sentinel-1 SAR...")
    vv, vh, flood_mask = generate_sentinel1(size, elevation, with_flood)

    # 3. Sentinel-2 Optical
    logger.info("Generating Sentinel-2...")
    green, nir = generate_sentinel2(size, flood_mask)

    # 4. GSW Occurrence
    logger.info("Generating GSW occurrence...")
    # Permanent water = river + coastal (where elevation is very low)
    river_y = size // 2 + np.sin(np.arange(size) / (size * 0.1)) * (size * 0.1)
    y_grid = np.arange(size)[:, None]
    river_mask = np.exp(-((y_grid - river_y) ** 2) / (2 * (size * 0.02) ** 2))
    permanent_water = ((river_mask > 0.7) | (elevation < 0)).astype(np.float32)
    gsw_occurrence = generate_gsw_occurrence(size, permanent_water)

    # 5. Labels (for training — based on flood_mask)
    labels = (flood_mask > 0).astype(np.uint8)

    # Write all rasters
    logger.info("Writing rasters...")

    # DEM (single band)
    make_geotiff(PROCESSED_DIR / "dem_reproj.tif", elevation, BBOX)

    # Sentinel-1 (2 bands: VV, VH)
    s1_data = np.stack([vv, vh])
    make_geotiff(PROCESSED_DIR / "sentinel1_reproj.tif", s1_data, BBOX)

    # Sentinel-2 (2 bands: Green, NIR)
    s2_data = np.stack([green, nir])
    make_geotiff(PROCESSED_DIR / "sentinel2_reproj.tif", s2_data, BBOX)

    # GSW Occurrence
    make_geotiff(GSW_DIR / "gsw_occurrence_ntb.tif", gsw_occurrence, BBOX)

    # Labels
    make_geotiff(LABELS_DIR / "flood_labels.tif", labels, BBOX)

    # Summary
    n_flood = int(np.sum(labels))
    n_total = labels.size
    logger.info("-" * 60)
    logger.info("MOCK DATA SUMMARY:")
    logger.info("  Size: %dx%d pixels", size, size)
    logger.info("  Flood pixels: %d (%.2f%%)", n_flood, 100 * n_flood / n_total)
    logger.info("  Elevation range: %.0f - %.0f m", np.min(elevation), np.max(elevation))
    logger.info("  VV range: %.1f - %.1f dB", np.min(vv), np.max(vv))
    logger.info("  NDWI range: %.3f - %.3f", np.nanmin((green - nir) / (green + nir)),
                np.nanmax((green - nir) / (green + nir)))
    logger.info("-" * 60)
    logger.info("FILES CREATED:")
    logger.info("  data/processed/dem_reproj.tif")
    logger.info("  data/processed/sentinel1_reproj.tif (VV + VH)")
    logger.info("  data/processed/sentinel2_reproj.tif (Green + NIR)")
    logger.info("  data/gsw/gsw_occurrence_ntb.tif")
    logger.info("  data/labels/flood_labels.tif")
    logger.info("=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("  1. python src/features.py          # Build feature stack")
    logger.info("  2. python src/model.py              # Train models")
    logger.info("  3. python src/predict.py            # Generate flood map")
    logger.info("  4. python src/ground_truth.py --validate  # Validate against GSW")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock data for A.E.C.O testing")
    parser.add_argument("--size", type=int, default=256, help="Raster size (pixels)")
    parser.add_argument("--with-flood", action="store_true", default=True, help="Include flood pixels")
    parser.add_argument("--no-flood", action="store_true", help="No flood pixels (dry scene)")
    args = parser.parse_args()

    with_flood = not args.no_flood
    generate_mock_data(size=args.size, with_flood=with_flood)
