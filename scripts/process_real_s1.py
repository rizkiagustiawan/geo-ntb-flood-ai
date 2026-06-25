"""Process REAL Sentinel-1 data with proper calibration.

Calibration formula (ESA SNAP standard):
  sigma0_dB = 20 * log10(DN / absoluteCalibrationConstant)

Reference: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-1
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import rasterio
import rasterio.windows
import xml.etree.ElementTree as ET
from pathlib import Path
from sar_preprocess import refined_lee_filter
from features import otsu_threshold

EXTRACT_DIR = Path('outputs/predictions/extracted/S1D_IW_GRDH_1SDV_20260613T215237_20260613T215256_003223_005A02_2890.SAFE')
VV_PATH = EXTRACT_DIR / 'measurement/s1d-iw-grd-vv-20260613t215237-20260613t215256-003223-005a02-001.tiff'
VH_PATH = EXTRACT_DIR / 'measurement/s1d-iw-grd-vh-20260613t215237-20260613t215256-003223-005a02-002.tiff'
CAL_VV = EXTRACT_DIR / 'annotation/calibration/calibration-s1d-iw-grd-vv-20260613t215237-20260613t215256-003223-005a02-001.xml'
CAL_VH = EXTRACT_DIR / 'annotation/calibration/calibration-s1d-iw-grd-vh-20260613t215237-20260613t215256-003223-005a02-002.xml'


def get_calibration_lut(xml_path):
    """Extract sigmaNought LUT from calibration XML.
    
    Returns (pixel_positions, sigma0_values) for interpolation.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'calibrationVectorList':
            for vec in elem:
                if 'calibrationVector' in (vec.tag.split('}')[-1] if '}' in vec.tag else vec.tag):
                    pixels = None
                    sigma0 = None
                    for child in vec:
                        ctag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if ctag == 'pixel' and child.text:
                            pixels = np.array([int(x) for x in child.text.strip().split()])
                        if ctag == 'sigmaNought' and child.text:
                            sigma0 = np.array([float(x) for x in child.text.strip().split()])
                    if pixels is not None and sigma0 is not None:
                        return pixels, sigma0
    raise ValueError(f"LUT not found in {xml_path}")


def dn_to_sigma0_db(dn, lut_pixels, lut_sigma0):
    """Convert DN to sigma0 in dB using calibration LUT.
    
    Formula: sigma0 = DN^2 / lut_sigma0^2
    sigma0_dB = 10 * log10(sigma0)
    """
    # Interpolate LUT to match image width
    img_width = dn.shape[1]
    cal_factors = np.interp(np.arange(img_width), lut_pixels, lut_sigma0)
    
    # Apply calibration: sigma0 = DN^2 / cal_factor^2
    sigma0 = np.zeros_like(dn, dtype=np.float64)
    for row in range(dn.shape[0]):
        sigma0[row, :] = dn[row, :] ** 2 / cal_factors ** 2
    
    # Convert to dB
    sigma0_db = 10.0 * np.log10(np.maximum(sigma0, 1e-10))
    return sigma0_db.astype(np.float32)


print("=" * 60)
print("REAL SENTINEL-1 PROCESSING (PROPER CALIBRATION)")
print("=" * 60)

# 1. Get calibration LUT
vv_pixels, vv_sigma0 = get_calibration_lut(CAL_VV)
vh_pixels, vh_sigma0 = get_calibration_lut(CAL_VH)
print(f"VV LUT: {len(vv_pixels)} points, sigma0 range: {vv_sigma0.min():.3f} to {vv_sigma0.max():.3f}")
print(f"VH LUT: {len(vh_pixels)} points, sigma0 range: {vh_sigma0.min():.3f} to {vh_sigma0.max():.3f}")

# 2. Load DN (crop center 2048x2048 for larger area)
with rasterio.open(VV_PATH) as src:
    h, w = src.shape
    cy, cx = h // 2, w // 2
    size = 2048
    win = rasterio.windows.Window(cx - size//2, cy - size//2, size, size)
    vv_dn = src.read(1, window=win).astype(np.float32)

with rasterio.open(VH_PATH) as src:
    vh_dn = src.read(1, window=win).astype(np.float32)

print(f"Scene size: {h}x{w}, crop: {size}x{size}")
print(f"VV DN: {vv_dn.min():.0f} to {vv_dn.max():.0f} (median: {np.median(vv_dn):.0f})")

# 3. Calibrate DN → sigma0 (dB) using LUT
print("Calibrating with LUT...")
vv_db = dn_to_sigma0_db(vv_dn, vv_pixels, vv_sigma0)
vh_db = dn_to_sigma0_db(vh_dn, vh_pixels, vh_sigma0)

print(f"VV sigma0: {vv_db.min():.1f} to {vv_db.max():.1f} dB (median: {np.median(vv_db):.1f})")
print(f"VH sigma0: {vh_db.min():.1f} to {vh_db.max():.1f} dB (median: {np.median(vh_db):.1f})")

# 4. Refined Lee speckle filter
print("Applying Refined Lee filter...")
vv_filtered = refined_lee_filter(vv_db, window_size=7)
vh_filtered = refined_lee_filter(vh_db, window_size=7)

# 5. Otsu adaptive thresholding
vv_thresh = otsu_threshold(vv_filtered)
vh_thresh = otsu_threshold(vh_filtered)
print(f"Otsu thresholds: VV={vv_thresh:.1f} dB, VH={vh_thresh:.1f} dB")

# 6. Flood mask (AND logic: both VV and VH must be below threshold)
flood_mask = ((vv_filtered < vv_thresh) & (vh_filtered < vh_thresh)).astype(np.uint8)
flood_pct = 100.0 * np.sum(flood_mask) / flood_mask.size
print(f"Flood/water pixels: {flood_pct:.2f}%")

# 7. Save
out_path = Path('data/processed/sentinel1_reproj.tif')
out_path.parent.mkdir(parents=True, exist_ok=True)

# Save as 2-band (VV, VH in dB)
profile = {
    'driver': 'GTiff', 'height': size, 'width': size,
    'count': 2, 'dtype': 'float32', 'compress': 'lzw',
}
with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(vv_filtered, 1)
    dst.write(vh_filtered, 2)
    dst.set_band_description(1, 'VV_sigma0_dB')
    dst.set_band_description(2, 'VH_sigma0_dB')

print(f"Saved: {out_path} (VV + VH in dB)")

# Save flood mask
flood_path = Path('outputs/predictions/sar_flood_mask.tif')
profile.update(count=1, dtype='uint8', nodata=255)
with rasterio.open(flood_path, 'w', **profile) as dst:
    dst.write(flood_mask, 1)

print(f"Saved: {flood_path}")
print("=" * 60)
print(f"SUMMARY:")
print(f"  VV threshold: {vv_thresh:.1f} dB (typical: -18 dB)")
print(f"  VH threshold: {vh_thresh:.1f} dB (typical: -24 dB)")
print(f"  Water detected: {flood_pct:.1f}% of scene")
print("=" * 60)
