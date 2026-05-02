"""GeoESG A.E.C.O — Celery Task Queue for Heavy Geoprocessing.

Offloads long-running geospatial tasks (AOI flood stats, PDF generation)
to a background Celery worker, preventing HTTP 502 timeouts on the
FastAPI web process.

Broker: Redis (redis://redis:6379/0)
Backend: Redis (redis://redis:6379/1)
"""

import logging
import os
import sys
from pathlib import Path

from celery import Celery

# Ensure api/ is importable for report_generator and notifier
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger("aeco-worker")

# ---------------------------------------------------------------------------
# Celery App Configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery(
    "aeco_tasks",
    broker=REDIS_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # One task at a time (memory-heavy geo ops)
)


# ---------------------------------------------------------------------------
# Task: Compute AOI Flood Statistics (Vectorized v2.0)
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="aeco.compute_aoi_stats")
def task_compute_aoi_stats(self, feature_dict: dict) -> dict:
    """Background task: Clip flood map to AOI and compute vectorized stats.

    Args:
        feature_dict: A GeoJSON Feature dict (serializable).

    Returns:
        Dict with flood statistics (JSON-serializable).
    """
    from report_generator import compute_aoi_flood_stats
    from notifier import send_flood_alert

    self.update_state(state="PROCESSING", meta={"step": "clipping_aoi"})

    try:
        stats = compute_aoi_flood_stats(feature_dict)
    except Exception as exc:
        logger.error(f"AOI stats failed: {exc}")
        return {"error": str(exc)}

    self.update_state(state="PROCESSING", meta={"step": "checking_alerts"})

    # --- Telegram EWS Trigger ---
    if stats.get("flooded_area_ha", 0) > 1.0:
        geom = feature_dict.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon" and coords:
            ring = coords[0]
        elif geom.get("type") == "MultiPolygon" and coords:
            ring = coords[0][0]
        else:
            ring = []

        if ring:
            c_lat = sum(c[1] for c in ring) / len(ring)
            c_lon = sum(c[0] for c in ring) / len(ring)
        else:
            c_lat, c_lon = -8.5, 116.8

        send_flood_alert(
            area_ha=stats["flooded_area_ha"],
            lat=c_lat,
            lon=c_lon,
            timestamp=stats.get("timestamp", ""),
        )

    return stats


# ---------------------------------------------------------------------------
# Task: Generate ESG PDF Report
# ---------------------------------------------------------------------------
@celery_app.task(bind=True, name="aeco.generate_report")
def task_generate_report(self, feature_dict: dict) -> dict:
    """Background task: Compute stats + generate PDF.

    Returns:
        Dict with stats and the path to the generated PDF.
    """
    from report_generator import compute_aoi_flood_stats, generate_esg_pdf

    self.update_state(state="PROCESSING", meta={"step": "computing_stats"})

    try:
        stats = compute_aoi_flood_stats(feature_dict)
    except Exception as exc:
        logger.error(f"Report stats failed: {exc}")
        return {"error": str(exc)}

    self.update_state(state="PROCESSING", meta={"step": "generating_pdf"})

    try:
        pdf_path = generate_esg_pdf(stats)
    except Exception as exc:
        logger.error(f"PDF generation failed: {exc}")
        return {"error": str(exc)}

    return {**stats, "pdf_path": pdf_path}
