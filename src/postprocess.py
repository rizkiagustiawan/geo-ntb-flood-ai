"""
postprocess.py - Ocean Masking for NTB Flood Detection.
Masks out ocean/coastal pixels (DEM <= 2m or NoData) from flood prediction,
saves cleaned map and auto-generates preview PNG.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

OCEAN_ELEV_THRESHOLD = 2.0  # metres — mask pixels at or below this


def run_postprocess(
    flood_path=None,
    dem_path=None,
    output_path=None,
    elev_threshold=OCEAN_ELEV_THRESHOLD,
):
    """Mask ocean from flood map using DEM elevation."""
    logger.info("=" * 60)
    logger.info("STARTING POSTPROCESS — OCEAN MASKING")
    logger.info("=" * 60)

    flood_in = Path(flood_path) if flood_path else PREDICTIONS_DIR / "flood_map.tif"
    dem_in = Path(dem_path) if dem_path else PROCESSED_DIR / "dem_reproj.tif"
    out_path = Path(output_path) if output_path else PREDICTIONS_DIR / "final_flood_map.tif"

    if not flood_in.exists():
        raise FileNotFoundError(f"Flood map not found: {flood_in}")
    if not dem_in.exists():
        raise FileNotFoundError(f"DEM not found: {dem_in}")

    # Load flood prediction
    with rasterio.open(flood_in) as ds:
        flood = ds.read(1)
        profile = ds.profile.copy()
        flood_shape = ds.shape
    logger.info("Loaded flood map: %s shape=%s", flood_in.name, flood_shape)

    # Load DEM
    with rasterio.open(dem_in) as ds:
        elev = ds.read(1).astype(np.float32)
        dem_nodata = ds.nodata
        dem_shape = ds.shape
    logger.info("Loaded DEM: %s shape=%s nodata=%s", dem_in.name, dem_shape, dem_nodata)

    # Validate shapes match
    if flood_shape != dem_shape:
        raise RuntimeError(
            f"Shape mismatch: flood={flood_shape}, DEM={dem_shape}. "
            "Ensure preprocess.py resampled DEM to match."
        )

    # Build ocean mask: True where pixel should be zeroed out
    ocean_mask = elev <= elev_threshold
    if dem_nodata is not None:
        ocean_mask |= elev == dem_nodata
    ocean_mask |= np.isnan(elev)

    n_masked = int(np.sum(ocean_mask))
    n_total = flood.size
    logger.info(
        "Ocean mask: %d pixels (%.2f%%) at elevation <= %.1fm",
        n_masked, 100.0 * n_masked / n_total, elev_threshold,
    )

    # Count flood pixels before masking
    flood_before = int(np.sum(flood == 1))

    # Apply mask
    cleaned = flood.copy()
    cleaned[ocean_mask] = 0

    flood_after = int(np.sum(cleaned == 1))
    removed = flood_before - flood_after
    logger.info(
        "Flood pixels: %d -> %d (removed %d ocean false-positives)",
        flood_before, flood_after, removed,
    )

    # Save cleaned flood map
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": 255})

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(cleaned[np.newaxis, :, :])
        dst.set_band_description(1, "flood_prediction_masked")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved: %s (%.2f MB)", out_path, size_mb)

    # Auto-generate preview PNG via visualize.py
    preview_path = out_path.with_name("final_flood_map_preview.png")
    try:
        from visualize import visualize_flood_map

        visualize_flood_map(input_path=str(out_path), output_path=str(preview_path))
        logger.info("Preview saved: %s", preview_path)
    except Exception as exc:
        logger.warning("Preview generation failed (non-fatal): %s", exc)

    logger.info("=" * 60)
    logger.info("POSTPROCESS COMPLETE -> %s", out_path)
    logger.info("=" * 60)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTB Flood - Ocean Mask Postprocess")
    parser.add_argument("--flood", type=str, default=None, help="Path to flood_map.tif")
    parser.add_argument("--dem", type=str, default=None, help="Path to dem_reproj.tif")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--threshold", type=float, default=OCEAN_ELEV_THRESHOLD,
        help="Elevation threshold in metres (default: 2.0)",
    )
    args = parser.parse_args()
    try:
        result = run_postprocess(args.flood, args.dem, args.output, args.threshold)
        print(f"  final_flood_map: {result}")
    except Exception as exc:
        logger.error("POSTPROCESS FAILED: %s", exc)
        sys.exit(1)
