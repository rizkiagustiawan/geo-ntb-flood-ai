"""Semi-Automated Flood Digitization Tool.

Downloads Sentinel-1/2 for known flood events in NTB,
computes NDWI + SAR mask, and creates a base map for
manual digitization in QGIS.

Usage:
    python scripts/digitize_flood.py --event taliwang_2023
    python scripts/digitize_flood.py --event bima_2024
    python scripts/digitize_flood.py --lat -8.7 --lon 116.8 --date 2023-02-15
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIGITIZE_DIR = PROJECT_ROOT / "data" / "digitize"
DIGITIZE_DIR.mkdir(parents=True, exist_ok=True)

# Known NTB flood events
NTB_EVENTS = {
    "taliwang_2023": {
        "name": "Taliwang, Sumbawa Barat - Feb 2023",
        "bbox": [116.7, -8.85, 116.95, -8.65],
        "date_start": "2023-02-05",
        "date_end": "2023-02-20",
        "s1_dates": ["2023-02-08", "2023-02-20"],
        "description": "Banjir bandang Taliwang, luas >500 ha",
    },
    "bima_2024": {
        "name": "Bima, Dompu - Jan 2024",
        "bbox": [118.5, -8.6, 118.8, -8.4],
        "date_start": "2024-01-10",
        "date_end": "2024-01-25",
        "s1_dates": ["2024-01-12", "2024-01-24"],
        "description": "Banjir Sungai Bima, >1000 ha terendam",
    },
    "sumbawa_2025": {
        "name": "Sumbawa - Feb 2025",
        "bbox": [117.3, -8.8, 117.6, -8.5],
        "date_start": "2025-02-01",
        "date_end": "2025-02-15",
        "s1_dates": ["2025-02-03", "2025-02-15"],
        "description": "Banjir Sumbawa tengah",
    },
    "lombok_2025": {
        "name": "Lombok Tengah - Mar 2025",
        "bbox": [116.2, -8.8, 116.5, -8.5],
        "date_start": "2025-03-10",
        "date_end": "2025-03-25",
        "s1_dates": ["2025-03-12", "2025-03-24"],
        "description": "Banjir Lombok Tengah",
    },
}


def download_s1_for_event(event: dict) -> Path:
    """Download Sentinel-1 data for a flood event."""
    import asf_search as asf

    bbox = event["bbox"]
    wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

    logger.info("Searching Sentinel-1 for %s...", event["name"])

    # Search for post-flood scene
    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=wkt,
        start=event["date_start"],
        end=event["date_end"],
        processingLevel="GRD_HD",
        maxResults=5,
    )

    if not results:
        logger.warning("No S1 scenes found for %s", event["date_start"])
        return None

    # Download smallest scene
    scenes = [(r, r.properties.get("bytes", 0)) for r in results]
    scenes.sort(key=lambda x: x[1])
    scene, size = scenes[0]

    scene_name = scene.properties["sceneName"]
    zip_path = DIGITIZE_DIR / f"{scene_name}.zip"

    if zip_path.exists():
        logger.info("Already downloaded: %s", scene_name)
    else:
        logger.info("Downloading: %s (%.0f MB)", scene_name, size / 1024 / 1024)
        import os
        session = asf.ASFSession().auth_with_creds(
            os.environ.get("EARTHDATA_USER", ""),
            os.environ.get("EARTHDATA_PASS", ""),
        )
        scene.download(path=str(DIGITIZE_DIR), session=session)

    return zip_path


def extract_s1_bands(zip_path: Path) -> tuple:
    """Extract VV/VH from Sentinel-1 ZIP."""
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as z:
        tiff_files = [f for f in z.namelist() if f.endswith(".tiff") and "measurement" in f]
        z.extractall(DIGITIZE_DIR / "extracted")

    extract_dir = DIGITIZE_DIR / "extracted"
    vv_path = next(str(extract_dir / f) for f in tiff_files if "-vv-" in f.lower())
    vh_path = next(str(extract_dir / f) for f in tiff_files if "-vh-" in f.lower())

    return vv_path, vh_path


def calibrate_s1(vv_path: str, vh_path: str, event: dict) -> tuple:
    """Calibrate S1 DN to sigma0 dB."""
    import xml.etree.ElementTree as ET

    # Find calibration XML
    extract_dir = DIGITIZE_DIR / "extracted"
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

    # Find VV and VH calibration files
    vv_cal = next((f for f in cal_files if "vv" in f.name.lower()), None)
    vh_cal = next((f for f in cal_files if "vh" in f.name.lower()), None)

    with rasterio.open(vv_path) as src:
        vv_dn = src.read(1).astype(np.float32)
        w = src.width

    with rasterio.open(vh_path) as src:
        vh_dn = src.read(1).astype(np.float32)

    # Calibrate
    if vv_cal:
        vv_pixels, vv_sigma0 = get_lut(vv_cal)
        cal_factors = np.interp(np.arange(w), vv_pixels, vv_sigma0)
        vv_db = (10.0 * np.log10(np.maximum(vv_dn ** 2 / cal_factors ** 2, 1e-10))).astype(np.float32)
    else:
        vv_db = (20.0 * np.log10(np.maximum(vv_dn, 1e-10))).astype(np.float32)

    if vh_cal:
        vh_pixels, vh_sigma0 = get_lut(vh_cal)
        cal_factors = np.interp(np.arange(w), vh_pixels, vh_sigma0)
        vh_db = (10.0 * np.log10(np.maximum(vh_dn ** 2 / cal_factors ** 2, 1e-10))).astype(np.float32)
    else:
        vh_db = (20.0 * np.log10(np.maximum(vh_dn, 1e-10))).astype(np.float32)

    return vv_db, vh_db


def create_digitization_base(event_name: str, vv_db: np.ndarray, vh_db: np.ndarray):
    """Create base maps for manual digitization in QGIS."""
    from features import otsu_threshold, compute_sar_threshold
    from sar_preprocess import refined_lee_filter

    event_dir = DIGITIZE_DIR / event_name
    event_dir.mkdir(parents=True, exist_ok=True)

    # Speckle filter
    vv_filtered = refined_lee_filter(vv_db, window_size=7)
    vh_filtered = refined_lee_filter(vh_db, window_size=7)

    # Otsu threshold
    sar_mask = compute_sar_threshold(vv_filtered, vh_filtered, method="otsu")

    # Save for QGIS
    h, w = vv_filtered.shape
    profile = {
        "driver": "GTiff", "height": h, "width": w,
        "count": 1, "dtype": "float32", "compress": "lzw",
    }

    # VV backscatter (for visual inspection)
    with rasterio.open(event_dir / "vv_backscatter.tif", "w", **profile) as dst:
        dst.write(vv_filtered, 1)

    # SAR flood mask (auto-detection)
    profile.update(dtype="uint8", nodata=255)
    with rasterio.open(event_dir / "sar_flood_mask.tif", "w", **profile) as dst:
        dst.write(sar_mask, 1)

    # Create QGIS project file template
    qml_content = """<!DOCTYPE qgis>
<qgis version="3.0">
  <pipe>
    <rasterrenderer opacity="1" type="paletted">
      <colorramp type="random"/>
      <colorEntry value="0" color="255,255,255,255" label="Land"/>
      <colorEntry value="1" color="0,0,255,200" label="Water/Flood"/>
      <colorEntry value="255" color="128,128,128,100" label="NoData"/>
    </rasterrenderer>
  </pipe>
</qgis>"""
    (event_dir / "flood_style.qml").write_text(qml_content)

    logger.info("Base maps saved to: %s", event_dir)
    logger.info("  vv_backscatter.tif — buka di QGIS untuk visualisasi")
    logger.info("  sar_flood_mask.tif — auto-detection (edit manual)")
    logger.info("  flood_style.qml — style untuk QGIS")

    # Instructions
    instructions = f"""
============================================================
PANDUAN DIGITIZATION: {event_name}
============================================================

1. Buka QGIS
2. Load: {event_dir}/vv_backscatter.tif
3. Load: {event_dir}/sar_flood_mask.tif
4. Apply style: flood_style.qml
5. Buat shapefile baru (Polygon) untuk ground truth
6. Draw polygon di area yang terdeteksi banjir
7. Edit: hapus false positive (gedung, jembatan)
8. Edit: tambah banjir yang terlewat
9. Save shapefile
10. Convert ke raster:

    python scripts/shapefile_to_raster.py \\
        --shapefile flood_polygons.shp \\
        --reference vv_backscatter.tif \\
        --output data/labels/ground_truth_{event_name}.tif

============================================================
"""
    (event_dir / "INSTRUCTIONS.txt").write_text(instructions)
    print(instructions)

    return event_dir


def main():
    parser = argparse.ArgumentParser(description="Flood Digitization Tool")
    parser.add_argument("--event", type=str, choices=list(NTB_EVENTS.keys()),
                        help="Known flood event name")
    parser.add_argument("--lat", type=float, help="Center latitude")
    parser.add_argument("--lon", type=float, help="Center longitude")
    parser.add_argument("--date", type=str, help="Flood date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List known events")
    args = parser.parse_args()

    if args.list:
        print("Known NTB Flood Events:")
        print("=" * 60)
        for key, event in NTB_EVENTS.items():
            print(f"  {key}: {event['name']}")
            print(f"    Date: {event['date_start']} to {event['date_end']}")
            print(f"    Area: {event['bbox']}")
            print(f"    Desc: {event['description']}")
            print()
        return

    if args.event:
        event = NTB_EVENTS[args.event]
    elif args.lat and args.lon and args.date:
        event = {
            "name": f"Custom ({args.lat}, {args.lon})",
            "bbox": [args.lon - 0.15, args.lat - 0.15, args.lon + 0.15, args.lat + 0.15],
            "date_start": args.date,
            "date_end": args.date,
        }
    else:
        parser.print_help()
        return

    # Download S1
    zip_path = download_s1_for_event(event)
    if not zip_path:
        logger.error("No data available for this event")
        return

    # Extract and calibrate
    vv_path, vh_path = extract_s1_bands(zip_path)
    vv_db, vh_db = calibrate_s1(vv_path, vh_path, event)

    # Create digitization base
    event_name = args.event or "custom"
    create_digitization_base(event_name, vv_db, vh_db)


if __name__ == "__main__":
    main()
