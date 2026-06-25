"""Process real Sentinel-1 data from ASF download with proper calibration."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import rasterio
import rasterio.windows
import xml.etree.ElementTree as ET
from pathlib import Path

VV_PATH = 'outputs/predictions/extracted/S1D_IW_GRDH_1SDV_20260613T215237_20260613T215256_003223_005A02_2890.SAFE/measurement/s1d-iw-grd-vv-20260613t215237-20260613t215256-003223-005a02-001.tiff'
VH_PATH = 'outputs/predictions/extracted/S1D_IW_GRDH_1SDV_20260613T215237_20260613T215256_003223_005A02_2890.SAFE/measurement/s1d-iw-grd-vh-20260613t215237-20260613t215256-003223-005a02-002.tiff'
CAL_VV = 'outputs/predictions/extracted/S1D_IW_GRDH_1SDV_20260613T215237_20260613T215256_003223_005A02_2890.SAFE/annotation/calibration/calibration-s1d-iw-grd-vv-20260613t215237-20260613t215256-003223-005a02-001.xml'
CAL_VH = 'outputs/predictions/extracted/S1D_IW_GRDH_1SDV_20260613T215237_20260613T215256_003223_005A02_2890.SAFE/annotation/calibration/calibration-s1d-iw-grd-vh-20260613t215237-20260613t215256-003223-005a02-002.xml'


def load_calibration_lut(xml_path):
    """Load sigma0 calibration LUT from Sentinel-1 calibration XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find calibrationVectorList
    for elem in root.iter():
        if 'calibrationVectorList' in elem.tag:
            # Get first calibration vector
            for vec in elem:
                if 'calibrationVector' in vec.tag:
                    # Get pixel and sigma0 values
                    pixels = None
                    sigma0 = None
                    for child in vec:
                        if 'pixel' in child.tag:
                            pixels = [int(x) for x in child.text.strip().split()]
                        if 'sigmaNought' in child.tag:
                            sigma0 = [float(x) for x in child.text.strip().split()]
                    if pixels and sigma0:
                        return np.array(pixels), np.array(sigma0)
    return None, None


def calibrate_dn(dn, cal_pixels, cal_sigma0, crop_slice=None):
    """Apply calibration: sigma0 = DN^2 * cal_factor.
    
    Parameters
    ----------
    dn : np.ndarray — DN values (H, W)
    cal_pixels : np.ndarray — Calibration pixel positions
    cal_sigma0 : np.ndarray — Calibration sigma0 values
    crop_slice : slice — Column slice for cropping (optional)
    """
    img_width = dn.shape[1]
    
    # Interpolate calibration LUT to match cropped width
    if crop_slice is not None:
        # Map crop positions back to full-image pixel positions
        pixel_positions = np.arange(crop_slice.start, crop_slice.stop)
    else:
        pixel_positions = np.arange(img_width)
    
    cal_factors = np.interp(pixel_positions, cal_pixels, cal_sigma0)
    
    # Apply: sigma0 = DN^2 / cal_factor (not multiply!)
    sigma0 = np.zeros_like(dn, dtype=np.float32)
    for row in range(dn.shape[0]):
        sigma0[row, :] = dn[row, :] ** 2 / cal_factors
    
    # Convert to dB
    sigma0_db = 10.0 * np.log10(np.maximum(sigma0, 1e-10))
    return sigma0_db.astype(np.float32)


print("=" * 60)
print("PROCESSING REAL SENTINEL-1 DATA (WITH CALIBRATION)")
print("=" * 60)

# 1. Load calibration LUT
print("Loading calibration...")
vv_pixels, vv_sigma0 = load_calibration_lut(CAL_VV)
vh_pixels, vh_sigma0 = load_calibration_lut(CAL_VH)
print(f"VV calibration: {len(vv_pixels)} points, sigma0 range: {vv_sigma0.min():.6f} to {vv_sigma0.max():.6f}")
print(f"VH calibration: {len(vh_pixels)} points, sigma0 range: {vh_sigma0.min():.6f} to {vh_sigma0.max():.6f}")

# 2. Load DN and crop
with rasterio.open(VV_PATH) as src:
    h, w = src.shape
    cy, cx = h // 2, w // 2
    size = 1024
    win = rasterio.windows.Window(cx - size//2, cy - size//2, size, size)
    vv_dn = src.read(1, window=win).astype(np.float32)
    full_w = w

with rasterio.open(VH_PATH) as src:
    vh_dn = src.read(1, window=win).astype(np.float32)

print(f"Cropped: {vv_dn.shape}")
print(f"VV DN: {vv_dn.min():.0f} to {vv_dn.max():.0f}")

# 3. Calibrate DN → sigma0 (dB)
print("Calibrating...")
crop_slice = slice(cx - size//2, cx + size//2)
vv_db = calibrate_dn(vv_dn, vv_pixels, vv_sigma0, crop_slice)
vh_db = calibrate_dn(vh_dn, vh_pixels, vh_sigma0, crop_slice)

print(f"VV sigma0 dB: {vv_db.min():.1f} to {vv_db.max():.1f} (median: {np.median(vv_db):.1f})")
print(f"VH sigma0 dB: {vh_db.min():.1f} to {vh_db.max():.1f} (median: {np.median(vh_db):.1f})")

# 4. SAR preprocessing
from sar_preprocess import preprocess_sar
vv_proc, vh_proc = preprocess_sar(vv_db, vh_db, apply_lee=True, lee_window=7, remove_noise=True)

# 5. Otsu thresholding
from features import compute_sar_threshold
sar_mask = compute_sar_threshold(vv_proc, vh_proc, method='otsu')

flood_pct = 100.0 * np.sum(sar_mask) / sar_mask.size
print(f"SAR flood mask: {flood_pct:.2f}% water pixels")

# 6. Save
out_path = Path('outputs/predictions/real_flood_map.tif')
profile = {'driver': 'GTiff', 'height': size, 'width': size, 'count': 1, 'dtype': 'uint8', 'compress': 'lzw'}
with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(sar_mask, 1)

print(f"Saved: {out_path}")
print("=" * 60)
