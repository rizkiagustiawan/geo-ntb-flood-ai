import signal
import shutil
import subprocess
import sys
import time
import os
import logging
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime

# Anchor all paths to this file's location (geo-ntb-flood-ai/)
PROJECT_ROOT = Path(__file__).resolve().parent

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] Oracle: %(message)s"
)
logger = logging.getLogger("FloodOracle")


class FloodOracle:
    def __init__(self):
        self.name = "Sumbawa-A.E.C.O"  # Autonomous ESG Compliance Oracle

        # RTK binary: env var override, then PATH lookup, then hardcoded fallback
        self.rtk_bin = (
            os.environ.get("RTK_BIN")
            or shutil.which("rtk")
            or "/home/awan/.cargo/bin/rtk"
        )
        if not os.path.isfile(self.rtk_bin):
            logger.warning(
                "⚠️ RTK binary not found at %s — pipeline will fail", self.rtk_bin
            )

        # All paths anchored to PROJECT_ROOT, never relative to current working directory
        self.pipeline_script = str(PROJECT_ROOT / "scripts" / "launch.sh")
        self.final_map = str(
            PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif"
        )

    def cleanup_temp_files(self):
        """Clearing temporary artifacts to prevent storage bloat and data corruption."""
        temp_files = [
            PROJECT_ROOT / "data" / "processed" / "feature_stack.tif",
            PROJECT_ROOT / "data" / "processed" / "dem_reproj_tmp.tif",
        ]
        logger.info("🧹 Cleaning up temporary artifacts...")
        for f in temp_files:
            if f.exists():
                f.unlink()
                logger.info("🗑️ Removed: %s", f)

    def calculate_impact(self):
        """Calculates total flood extent in Hectares.
        Reads actual pixel resolution from the GeoTIFF transform (EPSG:4326)
        and converts degrees to metric area for precision."""
        if not os.path.exists(self.final_map):
            return 0

        with rasterio.open(self.final_map) as src:
            data = src.read(1)
            # Flood pixels are identified by value 1
            flood_pixels = int(np.sum(data == 1))

            # Get pixel dimensions (in degrees for EPSG:4326)
            px_x_deg, px_y_deg = src.res

            # Convert degrees → meters (approximate for NTB region ~ -8°S)
            # 1 degree lat ≈ 111,320m
            px_x_m = px_x_deg * 111320.0
            px_y_m = px_y_deg * 111320.0
            pixel_area_m2 = px_x_m * px_y_m

            # 1 Hectare = 10,000 m²
            hectares = (flood_pixels * pixel_area_m2) / 10_000
            return hectares

    def run_cycle(self):
        """Executes a full monitoring cycle."""
        start_time = datetime.now()
        logger.info(
            f"🚀 [{self.name}] Cycle started at {start_time.strftime('%H:%M:%S')}"
        )

        try:
            # Ensure stale artifacts are removed to avoid runtime conflicts
            feature_stack = PROJECT_ROOT / "data" / "processed" / "feature_stack.tif"
            if feature_stack.exists():
                feature_stack.unlink()

            # Execute Pipeline via RTK Binary for clean, summarized logs
            subprocess.run(
                [self.rtk_bin, "sh", self.pipeline_script],
                check=True,
                cwd=str(PROJECT_ROOT),
            )

            # Impact Assessment
            area_ha = self.calculate_impact()

            logger.info("=" * 40)
            logger.info(f"✅ CYCLE COMPLETE: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"📊 ESTIMATED IMPACT: {area_ha:.2f} Hectares")
            logger.info("=" * 40)

            # Cleanup heavy intermediate files
            self.cleanup_temp_files()

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Pipeline failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected Oracle Error: {e}")


def _shutdown(signum, frame):
    """Graceful shutdown handler for SIGINT/SIGTERM."""
    logger.info("🛑 Shutdown signal received. Exiting cleanly.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    oracle = FloodOracle()

    print(f"""
    🛰️  {oracle.name} v1.1 ONLINE
    -------------------------------------------
    Target: Sumbawa Island, NTB
    Mode: Autonomous / RTK-Wrapped
    Engine: Caveman-Optimized (i7-8550U Safe)
    -------------------------------------------
    """)

    while True:
        oracle.run_cycle()

        # 1-hour interval between patrols to prevent CPU thermal throttling
        logger.info("💤 Sleeping for 1 hour. Thermal protection active.")
        time.sleep(3600)
