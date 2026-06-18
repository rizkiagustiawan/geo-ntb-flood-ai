"""Sentinel Hub API Client for A.E.C.O Flood Monitoring.

Automated satellite data fetching for NTB region using
Sentinel Hub Process API.

Usage:
    fetcher = SentinelHubFetcher()
    rasters = fetcher.fetch_all()
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger("aeco-satellite")

# NTB bounding boxes [West, South, East, North]
NTB_BBOX = [115.7, -9.1, 119.2, -8.1]
LOMBOK_BBOX = [115.7, -8.9, 116.6, -8.1]
SUMBAWA_BBOX = [116.6, -9.1, 119.2, -8.1]

# Data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STATUS_FILE = DATA_DIR / "satellite_status.json"


class SentinelHubFetcher:
    """Fetches Sentinel-1/2 data from Sentinel Hub Process API."""

    def __init__(self):
        from sentinelhub import SHConfig

        self.config = SHConfig()
        self.config.sh_client_id = os.environ.get("SH_CLIENT_ID", "")
        self.config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET", "")
        self.config.sh_instance_id = os.environ.get("SH_INSTANCE_ID", "")

        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError(
                "Sentinel Hub credentials not set. "
                "Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env"
            )

    def check_new_scenes(self, bbox: list, days_back: int = 7) -> str | None:
        """Check if new Sentinel-1 scenes are available.

        Args:
            bbox: [W, S, E, N] bounding box
            days_back: How many days to look back

        Returns:
            ISO date string of latest scene, or None if no new data.
        """
        from sentinelhub import SentinelHubCatalog, BBox, CRS, DataCollection

        catalog = SentinelHubCatalog(config=self.config)
        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (
            (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00"),
            datetime.utcnow().strftime("%Y-%m-%dT23:59:59"),
        )

        search_iterator = catalog.search(
            collection=DataCollection.SENTINEL1_IW,
            bbox=bbox_obj,
            time=time_interval,
            limit=1,
        )

        results = list(search_iterator)
        if not results:
            return None

        return results[0]["properties"]["datetime"]

    def fetch_sentinel1(self, bbox: list, date: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch Sentinel-1 SAR VV/VH bands via Process API.

        Args:
            bbox: [W, S, E, N] bounding box
            date: ISO date string (YYYY-MM-DD)

        Returns:
            Tuple of (VV, VH) numpy arrays in dB.
        """
        from sentinelhub import (
            SentinelHubRequest,
            BBox,
            CRS,
            DataCollection,
            MimeType,
        )

        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (f"{date}T00:00:00", f"{date}T23:59:59")

        evalscript_s1 = """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "VH"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.VV, sample.VH];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript_s1,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW.define_from(
                        "s1iw", service_url=self.config.sh_base_url
                    ),
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=[512, 512],
            config=self.config,
        )

        data = request.get_data()[0]
        vv = data[:, :, 0].astype(np.float32)
        vh = data[:, :, 1].astype(np.float32)
        return vv, vh

    def fetch_sentinel2(self, bbox: list, date: str) -> tuple[np.ndarray, np.ndarray]:
        """Fetch Sentinel-2 Green/NIR bands via Process API.

        Args:
            bbox: [W, S, E, N] bounding box
            date: ISO date string (YYYY-MM-DD)

        Returns:
            Tuple of (Green, NIR) numpy arrays as reflectance.
        """
        from sentinelhub import (
            SentinelHubRequest,
            BBox,
            CRS,
            DataCollection,
            MimeType,
        )

        bbox_obj = BBox(bbox, crs=CRS.WGS84)
        time_interval = (f"{date}T00:00:00", f"{date}T23:59:59")

        evalscript_s2 = """
        //VERSION=3
        function setup() {
          return {
            input: ["B03", "B08"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B03, sample.B08];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript_s2,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_obj,
            size=[512, 512],
            config=self.config,
        )

        data = request.get_data()[0]
        green = data[:, :, 0].astype(np.float32)
        nir = data[:, :, 1].astype(np.float32)
        return green, nir

    def save_rasters(self, rasters: dict) -> None:
        """Save fetched rasters to data/processed/ as GeoTIFF.

        Overwrites existing files so pipeline always uses latest data.

        Args:
            rasters: Dict with s1_vv, s1_vh, s2_green, s2_nir arrays + metadata.
        """
        import rasterio
        from rasterio.transform import from_bounds

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        bbox = rasters.get("bbox", NTB_BBOX)
        transform = from_bounds(
            bbox[0], bbox[1], bbox[2], bbox[3],
            rasters["s1_vv"].shape[1],
            rasters["s1_vv"].shape[0],
        )

        # Save Sentinel-1 (VV + VH as 2-band)
        s1_path = PROCESSED_DIR / "sentinel1_reproj.tif"
        with rasterio.open(
            str(s1_path), "w", driver="GTiff",
            height=rasters["s1_vv"].shape[0],
            width=rasters["s1_vv"].shape[1],
            count=2, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(rasters["s1_vv"], 1)
            dst.write(rasters["s1_vh"], 2)
        logger.info(f"Saved S1: {s1_path}")

        # Save Sentinel-2 (Green + NIR as 2-band)
        s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
        with rasterio.open(
            str(s2_path), "w", driver="GTiff",
            height=rasters["s2_green"].shape[0],
            width=rasters["s2_green"].shape[1],
            count=2, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(rasters["s2_green"], 1)
            dst.write(rasters["s2_nir"], 2)
        logger.info(f"Saved S2: {s2_path}")

        # Save sync status
        self.save_status(rasters)

    def save_status(self, rasters: dict) -> None:
        """Save sync status to JSON for dashboard."""
        import json

        status = {
            "last_sync": datetime.utcnow().isoformat() + "Z",
            "sentinel1": {
                "date": rasters["date"],
                "status": "ok",
            },
            "sentinel2": {
                "date": rasters["date"],
                "cloud": rasters.get("cloud_pct", 0),
                "status": "ok",
            },
            "bbox": rasters.get("bbox", NTB_BBOX),
        }

        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        logger.info(f"Saved status: {STATUS_FILE}")

    def fetch_all(self, bbox: list | None = None) -> dict | None:
        """Full fetch cycle: check scenes → fetch S1 + S2 → return rasters.

        Args:
            bbox: Override bounding box. Defaults to NTB_BBOX.

        Returns:
            Dict with raster arrays + metadata, or None if no new data.
        """
        if bbox is None:
            bbox = NTB_BBOX

        # 1. Check for new scenes
        logger.info(f"Checking new scenes for bbox={bbox}")
        latest_date = self.check_new_scenes(bbox)

        if not latest_date:
            logger.info("No new Sentinel-1 scenes found")
            return None

        date_str = latest_date[:10]  # YYYY-MM-DD
        logger.info(f"New scene found: {date_str}")

        # 2. Fetch Sentinel-1 SAR
        logger.info(f"Fetching Sentinel-1 for {date_str}")
        vv, vh = self.fetch_sentinel1(bbox, date_str)

        # 3. Fetch Sentinel-2 Optical
        logger.info(f"Fetching Sentinel-2 for {date_str}")
        green, nir = self.fetch_sentinel2(bbox, date_str)

        return {
            "s1_vv": vv,
            "s1_vh": vh,
            "s2_green": green,
            "s2_nir": nir,
            "date": date_str,
            "bbox": bbox,
            "cloud_pct": 0,  # TODO: extract from S2 metadata
        }
