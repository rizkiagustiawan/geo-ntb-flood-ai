"""
api/main.py - Consolidated FastAPI for NTB Flood Detection.
Endpoints: /health, /predict (POST), /predict/at (RTK point query),
/tiles/{z}/{x}/{y}.png (tile server), / (dashboard).
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
WEB_DIR = PROJECT_ROOT / "outputs" / "web"

FINAL_MAP = PREDICTIONS_DIR / "final_flood_map.tif"
RAW_MAP = PREDICTIONS_DIR / "flood_map.tif"

app = FastAPI(
    title="Sumbawa Flood AI",
    description="NTB Flood Detection API — Plampang, Sumbawa",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    lat: float
    lon: float


class BboxRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_map_path():
    """Return best available flood map path."""
    if FINAL_MAP.exists():
        return FINAL_MAP
    if RAW_MAP.exists():
        return RAW_MAP
    return None


def _point_query(map_path, lon, lat):
    """Query flood status at a single coordinate. Returns (value, valid)."""
    with rasterio.open(map_path) as src:
        row, col = src.index(lon, lat)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return None, False
        val = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        return int(val), True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    map_path = _get_map_path()
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flood_map_available": map_path is not None,
        "flood_map": str(map_path) if map_path else None,
        "metrics_available": metrics_path.exists(),
    }


@app.get("/predict/at")
def predict_at(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    """RTK point query: check flood status at exact coordinate."""
    map_path = _get_map_path()
    if map_path is None:
        raise HTTPException(404, "No flood map available. Run pipeline first.")

    val, valid = _point_query(map_path, lon, lat)
    if not valid:
        raise HTTPException(400, f"Coordinate ({lat}, {lon}) outside map bounds.")

    return {
        "lat": lat,
        "lon": lon,
        "flood": val == 1,
        "value": val,
        "map_source": map_path.name,
    }


@app.post("/predict")
def predict_post(req: PredictRequest):
    """POST variant of RTK point query."""
    map_path = _get_map_path()
    if map_path is None:
        raise HTTPException(404, "No flood map available.")

    val, valid = _point_query(map_path, req.lon, req.lat)
    if not valid:
        raise HTTPException(400, f"Coordinate ({req.lat}, {req.lon}) outside bounds.")

    return {
        "lat": req.lat,
        "lon": req.lon,
        "flood": val == 1,
        "value": val,
    }


@app.get("/metrics")
def get_metrics():
    """Return evaluation metrics JSON."""
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    if not metrics_path.exists():
        # Try xgboost metrics
        metrics_path = MODELS_DIR / "xgboost_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "No metrics available. Run evaluate.py.")
    return json.loads(metrics_path.read_text())


@app.get("/stats")
def get_stats():
    """Return flood map statistics."""
    map_path = _get_map_path()
    if map_path is None:
        raise HTTPException(404, "No flood map available.")

    with rasterio.open(map_path) as src:
        data = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        crs = str(src.crs)

    valid = data[data != (nodata or 255)]
    n_flood = int(np.sum(valid == 1))
    n_dry = int(np.sum(valid == 0))
    total = n_flood + n_dry

    return {
        "map": map_path.name,
        "crs": crs,
        "bounds": {"west": bounds.left, "south": bounds.bottom,
                    "east": bounds.right, "north": bounds.top},
        "shape": list(data.shape),
        "total_pixels": total,
        "flood_pixels": n_flood,
        "non_flood_pixels": n_dry,
        "flood_percentage": round(100.0 * n_flood / max(total, 1), 2),
    }


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Serve XYZ map tiles from flood GeoTIFF with cyan-blue flood overlay."""
    map_path = _get_map_path()
    if map_path is None:
        raise HTTPException(404, "No flood map available.")

    try:
        from rio_tiler.io import Reader
        from rio_tiler.errors import TileOutsideBounds
        from rio_tiler.utils import render
    except ImportError:
        raise HTTPException(500, "rio-tiler not installed. pip install rio-tiler")

    with Reader(str(map_path)) as dst:
        try:
            img = dst.tile(x, y, z)
        except TileOutsideBounds:
            raise HTTPException(404, "Tile outside bounds")

        data = img.data[0]
        alpha = (data == 1).astype(np.uint8) * 255
        rgb = np.zeros((3, img.height, img.width), dtype=np.uint8)
        rgb[0] = 0    # R
        rgb[1] = 180  # G
        rgb[2] = 216  # B
        content = render(rgb, alpha, img_format="PNG")

    return Response(content=content, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve Leaflet dashboard."""
    html_path = WEB_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Sumbawa Flood AI</h1><p>Dashboard not found. Place index.html in outputs/web/</p>")
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
