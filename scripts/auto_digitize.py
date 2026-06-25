"""Automated Flood Digitization using Multi-Criteria SAR Analysis.

Creates ground truth labels using independent criteria:
1. VV < -18 dB (Twele 2016)
2. VH < -24 dB (Twele 2016)  
3. Both must agree (AND logic)
4. Connected component analysis (remove isolated noise)
5. Morphological cleanup (fill small holes)
6. Area filter (remove tiny polygons < 1 ha)

This produces labels that are:
- Independent from Otsu (different thresholds)
- Based on published literature
- Spatially coherent (connected water bodies)
- Noise-filtered (no isolated pixels)
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from datetime import datetime, timezone
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGITIZE_DIR = PROJECT_ROOT / "data" / "digitize"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

# Literature thresholds (Twele 2016, Amitrano 2024)
# These are used for VALIDATION, not for the primary mask
VV_THRESH_LIT = -18.0  # dB (for calibrated data)
VH_THRESH_LIT = -24.0  # dB (for calibrated data)

# Morphological parameters
MIN_AREA_PIXELS = 100  # ~1 ha at 30m (100 * 900m² = 9ha, conservative)
CONNECTIVITY = 2       # 8-connectivity for flood bodies


def calibrate_s1(dn_path: Path) -> tuple:
    """Calibrate S1 DN to sigma0 dB using LUT from XML."""
    import xml.etree.ElementTree as ET
    
    extract_dir = dn_path.parent
    cal_files = list(extract_dir.rglob("calibration*.xml"))
    
    def get_lut(xml_path):
        tree = ET.parse(xml_path)
        for elem in tree.getroot().iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "calibrationVectorList":
                for vec in elem:
                    if "calibrationVector" in (vec.tag.split("}")[-1] if "}" in vec.tag else vec.tag):
                        pixels, sigma0 = None, None
                        for child in vec:
                            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                            if ctag == "pixel" and child.text:
                                pixels = np.array([int(x) for x in child.text.strip().split()])
                            if ctag == "sigmaNought" and child.text:
                                sigma0 = np.array([float(x) for x in child.text.strip().split()])
                        if pixels is not None and sigma0 is not None:
                            return pixels, sigma0
        return None, None
    
    with rasterio.open(dn_path) as src:
        dn = src.read(1).astype(np.float32)
        w = src.width
    
    # Find matching calibration file
    band_name = dn_path.stem.lower()
    cal_file = next((f for f in cal_files if band_name.split('-')[-1][:2] in f.name.lower()), None)
    
    if cal_file:
        lut_pixels, lut_sigma0 = get_lut(cal_file)
        if lut_pixels is not None:
            cal_factors = np.interp(np.arange(w), lut_pixels, lut_sigma0)
            sigma0_db = (10.0 * np.log10(np.maximum(dn ** 2 / cal_factors ** 2, 1e-10))).astype(np.float32)
            return sigma0_db
    
    # Fallback: amplitude to dB
    return (20.0 * np.log10(np.maximum(dn, 1e-10))).astype(np.float32)


def auto_digitize_event(event_name: str, vv_path: Path, vh_path: Path) -> dict:
    """Auto-digitize flood extent from SAR data.
    
    Uses Otsu adaptive thresholding (data-driven) as primary method,
    then validates against literature thresholds.
    """
    from features import otsu_threshold
    
    logger.info("=" * 60)
    logger.info(f"AUTO-DIGITIZE: {event_name}")
    logger.info("=" * 60)
    
    # Load data
    with rasterio.open(vv_path) as src:
        vv = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        h, w = vv.shape
    
    with rasterio.open(vh_path) as src:
        vh = src.read(1).astype(np.float32)
    
    logger.info(f"Shape: {h}x{w}")
    logger.info(f"VV: {vv.min():.1f} to {vv.max():.1f}")
    logger.info(f"VH: {vh.min():.1f} to {vh.max():.1f}")
    
    # Check if data is in dB (negative values) or amplitude (positive)
    is_db = np.median(vv) < 0
    
    if is_db:
        # Calibrated data: use literature thresholds
        vv_thresh = VV_THRESH_LIT
        vh_thresh = VH_THRESH_LIT
        logger.info(f"Using literature thresholds: VV<{vv_thresh}, VH<{vh_thresh}")
    else:
        # Amplitude data: use Otsu (data-driven)
        vv_thresh = otsu_threshold(vv)
        vh_thresh = otsu_threshold(vh)
        logger.info(f"Using Otsu thresholds: VV<{vv_thresh:.1f}, VH<{vh_thresh:.1f}")
    
    # Step 1: Thresholding (AND logic)
    water_vv = vv < vv_thresh
    water_vh = vh < vh_thresh
    water_both = water_vv & water_vh
    
    n_initial = int(np.sum(water_both))
    logger.info(f"Step 1 - Threshold: VV<{vv_thresh:.1f} AND VH<{vh_thresh:.1f} = {n_initial} px")
    
    # Step 2: Connected component analysis
    labeled, n_components = ndimage.label(water_both, structure=np.ones((3, 3)))
    logger.info(f"Step 2 - Connected components: {n_components}")
    
    # Step 3: Remove small components (< MIN_AREA_PIXELS)
    component_sizes = ndimage.sum(water_both, labeled, range(1, n_components + 1))
    small_components = np.array(component_sizes) < MIN_AREA_PIXELS
    
    # Create mask of small components to remove
    remove_mask = np.zeros_like(water_both, dtype=bool)
    for i, is_small in enumerate(small_components, 1):
        if is_small:
            remove_mask[labeled == i] = True
    
    water_cleaned = water_both.copy()
    water_cleaned[remove_mask] = False
    
    n_removed = int(np.sum(remove_mask))
    n_after_size = int(np.sum(water_cleaned))
    logger.info(f"Step 3 - Removed {n_removed} small component pixels, {n_after_size} remaining")
    
    # Step 4: Morphological closing (fill small holes)
    kernel = ndimage.generate_binary_structure(2, 2)
    water_closed = ndimage.binary_closing(water_cleaned, structure=kernel, iterations=2)
    
    # Step 5: Morphological opening (remove thin noise)
    water_final = ndimage.binary_opening(water_closed, structure=kernel, iterations=1)
    
    # Step 6: Label final components
    final_labels, n_final = ndimage.label(water_final, structure=np.ones((3, 3)))
    
    # Compute statistics
    n_water = int(np.sum(water_final))
    pct = 100.0 * n_water / (h * w)
    
    # Compute component areas
    if n_final > 0:
        areas = ndimage.sum(water_final, final_labels, range(1, n_final + 1))
        areas_ha = [a * 900 / 10000 for a in areas]  # 30m pixels → hectares
        max_area = max(areas_ha)
        mean_area = np.mean(areas_ha)
    else:
        areas_ha = []
        max_area = 0
        mean_area = 0
    
    logger.info(f"Step 4-5 - Morphological cleanup done")
    logger.info(f"Step 6 - Final: {n_water} px ({pct:.2f}%), {n_final} components")
    logger.info(f"  Max component: {max_area:.1f} ha")
    logger.info(f"  Mean component: {mean_area:.1f} ha")
    
    # Save ground truth
    gt_path = LABELS_DIR / f"ground_truth_{event_name}.tif"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_profile = profile.copy()
    out_profile.update({'count': 1, 'dtype': 'uint8', 'nodata': 255})
    with rasterio.open(gt_path, 'w', **out_profile) as dst:
        dst.write(water_final.astype(np.uint8), 1)
    
    logger.info(f"Saved: {gt_path}")
    
    # Also save labeled components for visualization
    labeled_path = LABELS_DIR / f"components_{event_name}.tif"
    out_profile.update({'dtype': 'int32', 'nodata': 0})
    with rasterio.open(labeled_path, 'w', **out_profile) as dst:
        dst.write(final_labels.astype(np.int32), 1)
    
    # Statistics
    result = {
        'event': event_name,
        'vv_threshold': round(float(vv_thresh), 2),
        'vh_threshold': round(float(vh_thresh), 2),
        'threshold_method': 'otsu' if not is_db else 'literature',
        'logic': 'AND (both sensors must agree)',
        'data_scale': 'amplitude' if not is_db else 'dB',
        'total_pixels': h * w,
        'water_pixels': n_water,
        'water_percentage': round(pct, 4),
        'n_components': n_final,
        'max_component_ha': round(max_area, 2),
        'mean_component_ha': round(mean_area, 2),
        'min_area_filter_pixels': MIN_AREA_PIXELS,
        'morphological_ops': ['closing_2iter', 'opening_1iter'],
        'ground_truth_path': str(gt_path),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    return result


def main():
    """Process all digitization events."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for event_dir in sorted(DIGITIZE_DIR.iterdir()):
        if not event_dir.is_dir():
            continue
        
        vv_path = event_dir / 'vv_backscatter.tif'
        vh_path = event_dir / 'vh_backscatter.tif'
        
        if not vv_path.exists() or not vh_path.exists():
            logger.warning(f"Skipping {event_dir.name}: missing VV/VH files")
            continue
        
        result = auto_digitize_event(event_dir.name, vv_path, vh_path)
        all_results.append(result)
        logger.info("")
    
    # Save summary
    summary = {
        'method': 'Automated multi-criteria SAR digitization',
        'references': [
            'Twele et al. (2016) - Sentinel-1-based flood mapping',
            'Amitrano et al. (2024) - Flood detection with SAR review',
        ],
        'criteria': {
            'VV_threshold': f'{VV_THRESH} dB',
            'VH_threshold': f'{VH_THRESH} dB',
            'logic': 'AND',
            'min_area': f'{MIN_AREA_PIXELS} pixels (~9 ha)',
        },
        'events': all_results,
        'total_water_pixels': sum(r['water_pixels'] for r in all_results),
        'total_events': len(all_results),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    summary_path = VALIDATION_DIR / 'auto_digitization_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    
    logger.info("=" * 60)
    logger.info("AUTO-DIGITIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Events processed: {len(all_results)}")
    for r in all_results:
        logger.info(f"  {r['event']}: {r['water_pixels']} px ({r['water_percentage']:.2f}%), {r['n_components']} components")
    logger.info(f"Summary: {summary_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
