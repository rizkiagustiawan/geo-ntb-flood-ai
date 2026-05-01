import os
import json
import logging
import base64
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import rasterio
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- PATH CONFIG (Docker Optimized) ---
# Di dalam kontainer, kita pakai /app sebagai basis
PROJECT_ROOT = Path("/app")
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
FINAL_MAP = PREDICTIONS_DIR / "final_flood_map.tif"

app = FastAPI(title="GeoESG A.E.C.O API")

# --- 1. MOUNTING STATIC FILES ---
# Pastikan folder ada sebelum mount untuk menghindari startup error
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
    logger.info(f"Assets directory mounted from {ASSETS_DIR}")
else:
    logger.warning(f"Assets directory NOT FOUND at {ASSETS_DIR}")

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# --- 2. BYPASS LOG NOISE ---
TRANSPARENT_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Mencegah error 404 favicon di log browser"""
    return Response(content=TRANSPARENT_PNG, media_type="image/png")

# --- 3. CORE ENDPOINTS ---

@app.get("/health")
def health():
    """Endpoint untuk dashboard UI mengecek status koneksi"""
    return {
        "status": "LIVE",  # Trigger warna hijau di UI
        "state": "online",
        "engine": "FastAPI + Rust",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/stats")
def get_stats():
    """Menghitung statistik banjir dari file GeoTIFF terbaru"""
    if not FINAL_MAP.exists():
        logger.error(f"Map file not found at {FINAL_MAP}")
        raise HTTPException(404, "Flood map not found")
        
    with rasterio.open(FINAL_MAP) as src:
        data = src.read(1)
        b = src.bounds
        # Masking nodata (biasanya 255 atau sesuai metadata)
        valid = data[data != (src.nodata or 255)]
        n_flood = int(np.sum(valid == 1))
        total = len(valid)
        
    return {
        "map": FINAL_MAP.name,
        "flood_pixels": n_flood,
        "flood_percentage": round(100.0 * n_flood / max(total, 1), 2),
        "total_pixels": total,
        "bounds": {
            "west": b.left,
            "south": b.bottom,
            "east": b.right,
            "north": b.top
        }
    }

@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Render GeoTIFF menjadi tile PNG secara on-the-fly"""
    if not FINAL_MAP.exists():
        return Response(content=TRANSPARENT_PNG, media_type="image/png")
    try:
        from rio_tiler.io import Reader
        from rio_tiler.utils import render
        with Reader(str(FINAL_MAP)) as dst:
            img = dst.tile(x, y, z)
            # Masking: Tampilkan warna hanya untuk piksel banjir (nilai 1)
            alpha = (img.data[0] == 1).astype(np.uint8) * 255
            rgb = np.zeros((3, img.height, img.width), dtype=np.uint8)
            # Warna Cyan (R:0, G:180, B:216) untuk area banjir
            rgb[1], rgb[2] = 180, 216 
            return Response(content=render(rgb, alpha, img_format="PNG"), media_type="image/png")
    except Exception as e:
        logger.error(f"Tile rendering error: {e}")
        return Response(content=TRANSPARENT_PNG, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Endpoint utama untuk melayani Dashboard UI"""
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Error: index.html not found in /app</h1>")
    return HTMLResponse(html_path.read_text())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
