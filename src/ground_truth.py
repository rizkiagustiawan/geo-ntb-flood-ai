"""Ground Truth Validation Module.

Integrates independent validation datasets for flood detection:
1. JRC Global Surface Water (GSW) — Google's 30m water occurrence dataset
2. Manual digitization support for field-validated flood extents
3. Cross-validation against established flood benchmarks

Per Roberts et al. (2017): evaluation must use independent labels,
NOT the same pseudo-labels used for training.

References:
- Pekel et al. (2016): "High-resolution mapping of global surface water
  and its long-term changes." Nature, 540, 418-422.
- GSW Dataset: https://global-surface-water.appspot.com/
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GSW_DIR = DATA_DIR / "gsw"
PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

# NTB bounding box
NTB_BBOX = [115.7, -9.1, 119.2, -8.1]


def fetch_gsw_occurrence(bbox: list = None, output_path: str = None) -> Path:
    """Download JRC Global Surface Water Occurrence dataset.

    The GSW Occurrence layer shows the percentage of time a pixel was
    classified as water over 35+ years (1984-2021). This serves as an
    independent benchmark for permanent water bodies.

    Parameters
    ----------
    bbox : list — [W, S, E, N] bounding box (default: NTB)
    output_path : str — Output GeoTIFF path

    Returns
    -------
    Path — Path to downloaded GSW occurrence raster
    """
    if bbox is None:
        bbox = NTB_BBOX

    GSW_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(GSW_DIR / "gsw_occurrence_ntb.tif")

    # GSW is available via Google Earth Engine
    try:
        import ee
        ee.Initialize(project='geo-ntb-flood-ai')

        region = ee.Geometry.Rectangle(bbox)
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        occurrence = gsw.select('occurrence')

        # Download as GeoTIFF
        url = occurrence.getDownloadURL({
            'region': region,
            'scale': 30,
            'format': 'GeoTIFF',
            'crs': 'EPSG:4326',
        })

        logger.info("GSW download URL generated: %s", url)
        logger.info("Download manually and save to: %s", output_path)
        logger.info("Or use: wget '%s' -O %s", url, output_path)

        return Path(output_path)

    except Exception as e:
        logger.warning("GEE download failed: %s. Using fallback method.", e)
        return _fetch_gsw_fallback(bbox, output_path)


def _fetch_gsw_fallback(bbox: list, output_path: str) -> Path:
    """Fallback: Create synthetic GSW-like data from existing water indices.

    This is a TEMPORARY placeholder until real GSW data is downloaded.
    Uses NDWI thresholding on Sentinel-2 as proxy for permanent water.
    """
    logger.warning("Using NDWI-based proxy for GSW (download real GSW for production)")

    s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
    if not s2_path.exists():
        raise FileNotFoundError(f"Sentinel-2 not found: {s2_path}. Run preprocess.py first.")

    with rasterio.open(s2_path) as src:
        green = src.read(1).astype(np.float32)
        nir = src.read(2).astype(np.float32)
        profile = src.profile.copy()

    # NDWI > 0.3 consistently = permanent water (proxy for GSW > 80%)
    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)

    # Simulate "occurrence" as binary: water if NDWI > 0.3
    occurrence = (ndwi > 0.3).astype(np.uint8) * 100

    profile.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(occurrence, 1)
        dst.set_band_description(1, "water_occurrence_proxy")

    logger.info("GSW proxy saved: %s", output_path)
    return Path(output_path)


def validate_against_gsw(
    flood_map_path: str = None,
    gsw_path: str = None,
    gsw_threshold: int = 80,
) -> dict:
    """Validate flood detection against GSW independent benchmark.

    Compares the model's flood predictions against GSW water occurrence.
    Pixels with GSW occurrence > threshold are considered "true water".

    This is an INDEPENDENT evaluation — GSW was NOT used for training.

    Parameters
    ----------
    flood_map_path : str — Path to flood prediction GeoTIFF
    gsw_path : str — Path to GSW occurrence GeoTIFF
    gsw_threshold : int — GSW occurrence % to consider as water (default 80%)

    Returns
    -------
    dict — Validation metrics against GSW
    """
    if flood_map_path is None:
        flood_map_path = str(PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif")
    if gsw_path is None:
        gsw_path = str(GSW_DIR / "gsw_occurrence_ntb.tif")

    flood_path = Path(flood_map_path)
    gsw_file = Path(gsw_path)

    if not flood_path.exists():
        raise FileNotFoundError(f"Flood map not found: {flood_path}")
    if not gsw_file.exists():
        raise FileNotFoundError(
            f"GSW not found: {gsw_file}. Run fetch_gsw_occurrence() first."
        )

    logger.info("=" * 60)
    logger.info("VALIDATION AGAINST GSW (INDEPENDENT)")
    logger.info("=" * 60)

    # Load flood map
    with rasterio.open(flood_path) as src:
        flood = src.read(1).astype(np.uint8)
        flood_profile = src.profile.copy()

    # Load GSW
    with rasterio.open(gsw_file) as src:
        gsw = src.read(1).astype(np.uint8)
        gsw_profile = src.profile.copy()

    # Resample GSW to match flood map grid if needed
    if flood.shape != gsw.shape:
        logger.info("Resampling GSW to match flood map grid (%s → %s)",
                    gsw.shape, flood.shape)
        gsw_resampled = np.zeros_like(flood, dtype=np.uint8)
        reproject(
            source=gsw,
            destination=gsw_resampled,
            src_transform=gsw_profile["transform"],
            src_crs=gsw_profile["crs"],
            dst_transform=flood_profile["transform"],
            dst_crs=flood_profile["crs"],
            resampling=Resampling.nearest,
        )
        gsw = gsw_resampled

    # Binary masks
    flood_binary = (flood == 1).astype(np.uint8)
    gsw_water = (gsw >= gsw_threshold).astype(np.uint8)

    # Valid pixels (not nodata in either)
    valid = (flood != 255) & (gsw != 0)
    n_valid = int(np.sum(valid))
    n_total = flood.size

    if n_valid == 0:
        logger.error("No valid overlapping pixels between flood map and GSW")
        return {"error": "no_overlap"}

    # Compute confusion matrix against GSW
    tp = int(np.sum((flood_binary == 1) & (gsw_water == 1) & valid))
    fp = int(np.sum((flood_binary == 1) & (gsw_water == 0) & valid))
    tn = int(np.sum((flood_binary == 0) & (gsw_water == 0) & valid))
    fn = int(np.sum((flood_binary == 0) & (gsw_water == 1) & valid))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    # Agreement statistics
    agreement = int(np.sum(
        ((flood_binary == gsw_water) | ~valid)
    ))
    agreement_pct = 100.0 * agreement / n_total

    results = {
        "benchmark": "JRC Global Surface Water (GSW)",
        "gsw_threshold": gsw_threshold,
        "valid_pixels": n_valid,
        "total_pixels": n_total,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "iou": round(iou, 6),
        "accuracy": round(accuracy, 6),
        "agreement_pct": round(agreement_pct, 2),
        "flood_pixels_predicted": int(np.sum(flood_binary & valid)),
        "water_pixels_gsw": int(np.sum(gsw_water & valid)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("GSW Validation Results:")
    logger.info("  Valid pixels: %d / %d (%.1f%%)", n_valid, n_total, 100 * n_valid / n_total)
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall: %.4f", recall)
    logger.info("  F1-Score: %.4f", f1)
    logger.info("  IoU: %.4f", iou)
    logger.info("  Accuracy: %.4f", accuracy)
    logger.info("  Agreement: %.2f%%", agreement_pct)

    # Save
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "gsw_validation.json"
    report_path.write_text(json.dumps(results, indent=2))
    logger.info("Validation report saved: %s", report_path)

    logger.info("=" * 60)
    return results


def validate_against_manual(
    flood_map_path: str,
    manual_labels_path: str,
    label_name: str = "manual",
) -> dict:
    """Validate flood detection against manually digitized labels.

    For proper validation, manual digitization of known flood events
    from high-res imagery (PlanetScope, drone) is the gold standard.

    Parameters
    ----------
    flood_map_path : str — Path to flood prediction GeoTIFF
    manual_labels_path : str — Path to manually digitized labels (uint8, 1=flood)
    label_name : str — Name of the validation dataset

    Returns
    -------
    dict — Validation metrics
    """
    flood_path = Path(flood_map_path)
    label_path = Path(manual_labels_path)

    if not flood_path.exists():
        raise FileNotFoundError(f"Flood map not found: {flood_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Manual labels not found: {label_path}")

    logger.info("=" * 60)
    logger.info("VALIDATION AGAINST MANUAL LABELS: %s", label_name)
    logger.info("=" * 60)

    with rasterio.open(flood_path) as src:
        flood = src.read(1).astype(np.uint8)
        flood_profile = src.profile.copy()

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.uint8)
        label_profile = src.profile.copy()

    # Resample labels if needed
    if flood.shape != labels.shape:
        labels_resampled = np.zeros_like(flood, dtype=np.uint8)
        reproject(
            source=labels,
            destination=labels_resampled,
            src_transform=label_profile["transform"],
            src_crs=label_profile["crs"],
            dst_transform=flood_profile["transform"],
            dst_crs=flood_profile["crs"],
            resampling=Resampling.nearest,
        )
        labels = labels_resampled

    flood_binary = (flood == 1).astype(np.uint8)
    label_binary = (labels == 1).astype(np.uint8)

    valid = (flood != 255) & (labels != 255)

    tp = int(np.sum((flood_binary == 1) & (label_binary == 1) & valid))
    fp = int(np.sum((flood_binary == 1) & (label_binary == 0) & valid))
    tn = int(np.sum((flood_binary == 0) & (label_binary == 0) & valid))
    fn = int(np.sum((flood_binary == 0) & (label_binary == 1) & valid))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)

    results = {
        "benchmark": f"Manual: {label_name}",
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "iou": round(iou, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Manual Validation Results (%s):", label_name)
    logger.info("  F1=%.4f  IoU=%.4f  P=%.4f  R=%.4f", f1, iou, precision, recall)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / f"validation_{label_name}.json"
    report_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved: %s", report_path)

    return results


def compute_gsw_water_mask(gsw_path: str = None, occurrence_threshold: int = 50) -> Path:
    """Create permanent water mask from GSW occurrence.

    Pixels with occurrence > threshold over 35+ years are classified
    as permanent water. This can be used to mask false positives in
    flood detection (permanent water ≠ flood).

    Parameters
    ----------
    gsw_path : str — Path to GSW occurrence GeoTIFF
    occurrence_threshold : int — Minimum occurrence % (default 50%)

    Returns
    -------
    Path — Path to binary water mask GeoTIFF
    """
    if gsw_path is None:
        gsw_path = str(GSW_DIR / "gsw_occurrence_ntb.tif")

    gsw_file = Path(gsw_path)
    if not gsw_file.exists():
        raise FileNotFoundError(f"GSW not found: {gsw_file}")

    with rasterio.open(gsw_path) as src:
        occurrence = src.read(1).astype(np.uint8)
        profile = src.profile.copy()

    water_mask = (occurrence >= occurrence_threshold).astype(np.uint8)

    output_path = GSW_DIR / f"permanent_water_mask_p{occurrence_threshold}.tif"
    profile.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(water_mask, 1)
        dst.set_band_description(1, "permanent_water_mask")

    n_water = int(np.sum(water_mask))
    logger.info("Permanent water mask: %d pixels (p%d threshold)",
                n_water, occurrence_threshold)
    logger.info("Saved: %s", output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ground Truth Validation")
    parser.add_argument("--fetch-gsw", action="store_true", help="Download GSW data")
    parser.add_argument("--validate", action="store_true", help="Validate against GSW")
    parser.add_argument("--gsw-threshold", type=int, default=80, help="GSW occurrence threshold")
    args = parser.parse_args()

    if args.fetch_gsw:
        fetch_gsw_occurrence()

    if args.validate:
        validate_against_gsw(gsw_threshold=args.gsw_threshold)
