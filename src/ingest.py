"""
ingest.py - Data Ingestion for NTB Flood Detection.
Downloads Sentinel-1, Sentinel-2 via GEE, DEM via SRTM, BMKG rainfall via XML.

ROI: Plampang, Sumbawa (centre -8.78, 117.78) with 20km buffer.
At ~8.78°S: 20km ≈ 0.18° lat, ≈ 0.183° lon.
Result: ~4000x4000px at 10m → well under GEE 32768px and 50MB limits.
"""

import os
import sys
import json
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta, timezone

import ee
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ROI: Plampang, Sumbawa — 20km buffer around centre point
ROI_CENTRE = {"lat": -8.78, "lon": 117.78}
ROI_BUFFER_KM = 20
# Convert km to degrees (approximate at latitude -8.78)
_LAT_DEG = ROI_BUFFER_KM / 111.32
_LON_DEG = ROI_BUFFER_KM / (111.32 * np.cos(np.radians(ROI_CENTRE["lat"])))
ROI_BBOX = {
    "west":  ROI_CENTRE["lon"] - _LON_DEG,
    "south": ROI_CENTRE["lat"] - _LAT_DEG,
    "east":  ROI_CENTRE["lon"] + _LON_DEG,
    "north": ROI_CENTRE["lat"] + _LAT_DEG,
}
logger.info("ROI BBOX: W=%.4f S=%.4f E=%.4f N=%.4f (~%dkm buffer)",
            ROI_BBOX["west"], ROI_BBOX["south"], ROI_BBOX["east"], ROI_BBOX["north"], ROI_BUFFER_KM)

DEFAULT_END = datetime.now(timezone.utc)
DEFAULT_START = DEFAULT_END - timedelta(days=30)
BMKG_DEFAULT = "https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-NTB.xml"
S1_SCALE = 10
S2_SCALE = 10


def validate_environment():
    """Validate GEE_KEY and BMKG_ENDPOINT. Raises EnvironmentError if GEE_KEY missing."""
    gee_key_raw = os.environ.get("GEE_KEY")
    if gee_key_raw is None:
        key_path = PROJECT_ROOT.parent / "gee-key.json"
        if key_path.exists():
            logger.info("Using key file at %s", key_path)
            gee_key_raw = key_path.read_text()
        else:
            raise EnvironmentError("GEE_KEY not set and no gee-key.json found.")

    if os.path.isfile(gee_key_raw):
        with open(gee_key_raw) as f:
            gee_key_data = json.load(f)
    else:
        gee_key_data = json.loads(gee_key_raw)

    # CRITICAL: Handle escaped newlines in private_key
    if "private_key" in gee_key_data:
        gee_key_data["private_key"] = gee_key_data["private_key"].replace("\\n", "\n")

    return {"gee_key": gee_key_data, "bmkg_endpoint": os.environ.get("BMKG_ENDPOINT", BMKG_DEFAULT)}


def authenticate_gee(sa_info):
    """Authenticate GEE headless with service account."""
    try:
        creds = ee.ServiceAccountCredentials(email=sa_info["client_email"], key_data=json.dumps(sa_info))
        ee.Initialize(credentials=creds)
        logger.info("GEE authenticated as %s", sa_info["client_email"])
    except Exception as exc:
        logger.error("GEE auth failed: %s", exc)
        raise


def _get_roi_region():
    """Return ee.Geometry.Rectangle for the Plampang ROI."""
    return ee.Geometry.Rectangle([ROI_BBOX["west"], ROI_BBOX["south"], ROI_BBOX["east"], ROI_BBOX["north"]])


def _download_gee_image(image, bands, filename, region, scale):
    """Download GEE image as GeoTIFF to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / filename
    logger.info("Downloading %s (bands=%s, scale=%dm)", filename, bands, scale)

    url = image.select(bands).getDownloadURL({
        "region": region, "scale": scale, "format": "GEO_TIFF", "crs": "EPSG:4326"
    })
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.2f MB)", out_path, size_mb)

    with rasterio.open(out_path) as ds:
        logger.info("Validated %s: shape=%s crs=%s", filename, ds.shape, ds.crs)
    return out_path


def ingest_sentinel1(start_date=None, end_date=None):
    """Download Sentinel-1 GRD (VV, VH) median composite for Plampang ROI."""
    start = start_date or DEFAULT_START.strftime("%Y-%m-%d")
    end = end_date or DEFAULT_END.strftime("%Y-%m-%d")
    logger.info("Ingesting Sentinel-1 [%s -> %s]", start, end)

    region = _get_roi_region()
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .select(["VV", "VH"]))

    count = col.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Sentinel-1 images between {start} and {end}")
    logger.info("Found %d Sentinel-1 scenes", count)

    return _download_gee_image(col.median().clip(region), ["VV", "VH"], "sentinel1_vv_vh.tif", region, S1_SCALE)


def ingest_sentinel2(start_date=None, end_date=None):
    """Download Sentinel-2 SR (B3 Green, B8 NIR) median composite for Plampang ROI."""
    start = start_date or DEFAULT_START.strftime("%Y-%m-%d")
    end = end_date or DEFAULT_END.strftime("%Y-%m-%d")
    logger.info("Ingesting Sentinel-2 [%s -> %s]", start, end)

    region = _get_roi_region()
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .select(["B3", "B8"]))

    count = col.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Sentinel-2 images between {start} and {end}")
    logger.info("Found %d Sentinel-2 scenes", count)

    return _download_gee_image(col.median().clip(region), ["B3", "B8"], "sentinel2_green_nir.tif", region, S2_SCALE)


def ingest_dem():
    """Download DEM (SRTM 30m) for Plampang ROI."""
    logger.info("Ingesting DEM (SRTM 30m)")
    region = _get_roi_region()
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(region)
    return _download_gee_image(dem, ["elevation"], "dem_srtm_30m.tif", region, 30)


def ingest_bmkg_rainfall(endpoint=None):
    """Fetch BMKG rainfall XML, parse to JSON, save to data/raw/."""
    url = endpoint or BMKG_DEFAULT
    logger.info("Fetching BMKG rainfall from %s", url)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "bmkg_rainfall.json"

    xml_content = None
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        xml_content = resp.text
    except requests.RequestException as exc:
        logger.error("BMKG primary failed: %s", exc)
        try:
            fallback = "https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml"
            logger.info("Trying fallback: %s", fallback)
            resp = requests.get(fallback, timeout=60)
            resp.raise_for_status()
            xml_content = resp.text
        except requests.RequestException as exc2:
            logger.error("BMKG fallback failed: %s", exc2)
            synthetic = {"source": "synthetic_fallback",
                         "timestamp": datetime.now(timezone.utc).isoformat(),
                         "region": "NTB", "note": "BMKG API unreachable", "stations": []}
            out_path.write_text(json.dumps(synthetic, indent=2))
            logger.warning("Saved synthetic rainfall placeholder")
            return out_path

    records = []
    try:
        root = ET.fromstring(xml_content)
        for area in root.iter("area"):
            aid = area.get("id", "unknown")
            adesc = area.get("description", "")
            alat = area.get("latitude", "")
            alon = area.get("longitude", "")
            for param in area.iter("parameter"):
                pid = param.get("id", "")
                if pid not in ("hu", "weather", "t", "ws"):
                    continue
                for tr in param.iter("timerange"):
                    for val in tr.iter("value"):
                        records.append({"area_id": aid, "area_desc": adesc, "latitude": alat,
                                        "longitude": alon, "parameter": pid,
                                        "timerange_type": tr.get("type", ""),
                                        "datetime": tr.get("datetime", ""),
                                        "unit": val.get("unit", ""), "value": val.text})
    except ET.ParseError as exc:
        logger.error("BMKG XML parse failed: %s", exc)
        raise RuntimeError(f"BMKG XML parsing failed: {exc}") from exc

    output = {"source": "BMKG", "endpoint": url,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "region": "NTB", "record_count": len(records), "records": records}
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Saved BMKG rainfall -> %s (%d records)", out_path, len(records))
    return out_path


def run_ingestion(start_date=None, end_date=None):
    """Run full ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION - PLAMPANG, SUMBAWA (%dkm ROI)", ROI_BUFFER_KM)
    logger.info("=" * 60)

    env = validate_environment()
    authenticate_gee(env["gee_key"])

    outputs = {}
    outputs["sentinel1"] = ingest_sentinel1(start_date, end_date)
    outputs["sentinel2"] = ingest_sentinel2(start_date, end_date)
    outputs["dem"] = ingest_dem()
    outputs["bmkg"] = ingest_bmkg_rainfall(env["bmkg_endpoint"])

    for name, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing: {name} -> {path}")
        logger.info("OK %s -> %s", name, path)

    logger.info("INGESTION COMPLETE - %d files to %s", len(outputs), RAW_DIR)
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NTB Flood - Data Ingestion (Plampang ROI)")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()
    try:
        results = run_ingestion(args.start_date, args.end_date)
        for n, p in results.items():
            print(f"  {n}: {p}")
    except Exception as exc:
        logger.error("INGESTION FAILED: %s", exc)
        sys.exit(1)
