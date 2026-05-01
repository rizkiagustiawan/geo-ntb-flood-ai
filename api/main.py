"""GeoESG A.E.C.O API — FastAPI + Rust Zero-Copy Flood Engine.

Endpoints:
    GET  /health          → liveness probe
    GET  /stats           → flood statistics from latest GeoTIFF
    GET  /predict/at      → RTK point-query: flood status at (lat, lon)
    POST /predict/area    → polygon zonal statistics: flooded area in hectares
    GET  /tiles/{z}/{x}/{y}.png → on-the-fly tile rendering
    GET  /                → dashboard HTML
"""

import base64
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import rasterio.mask
import rasterio.windows
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Rust Engine ---
try:
    import flood_rs

    RUST_READY = True
except ImportError:
    RUST_READY = False

import sys
sys.path.append(str(Path(__file__).parent))
from report_generator import generate_esg_pdf

# --- Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("aeco-api")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
FINAL_MAP = PREDICTIONS_DIR / "final_flood_map.tif"

# Co-registered rasters (same CRS, shape, bounds)
S1_RASTER = DATA_DIR / "processed" / "sentinel1_reproj.tif"  # Band1=VV, Band2=VH
S2_RASTER = DATA_DIR / "processed" / "sentinel2_reproj.tif"  # Band1=Green, Band2=NIR

# Thresholds (matching flood_agent.py)
NDWI_THRESH = 0.3
SAR_VV_THRESH = -15.0

# --- Pydantic Response Models ---


class FloodPrediction(BaseModel):
    """Structured response for /predict/at RTK validation endpoint."""

    lat: float = Field(..., description="Query latitude (WGS84)")
    lon: float = Field(..., description="Query longitude (WGS84)")
    flood: Literal[0, 1] = Field(..., description="0=safe, 1=flood")
    ndwi: float | None = Field(None, description="NDWI value at point")
    sar_vv: float | None = Field(None, description="SAR VV backscatter (dB)")
    method: str = Field(..., description="Compute method used")
    crs: str = Field("EPSG:4326", description="Coordinate reference system")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")


class HealthResponse(BaseModel):
    status: str
    engine: str
    rust_ready: bool
    timestamp: str


class StatsResponse(BaseModel):
    map: str
    flood_pixels: int
    flood_percentage: float
    total_pixels: int
    bounds: dict


class GeoJSONFeature(BaseModel):
    """Input: GeoJSON Feature with Polygon/MultiPolygon geometry."""

    type: Literal["Feature"] = Field(..., description="Must be 'Feature'")
    geometry: dict[str, Any] = Field(
        ..., description="GeoJSON geometry (Polygon or MultiPolygon)"
    )
    properties: dict[str, Any] | None = Field(
        default=None, description="Optional feature properties"
    )


class AreaReport(BaseModel):
    """Audit-ready ESG flood area report for a polygon."""

    total_pixels: int = Field(..., description="Total valid pixels in polygon")
    flooded_pixels: int = Field(..., description="Pixels classified as flood")
    total_area_ha: float = Field(..., description="Total polygon area (ha)")
    flooded_area_ha: float = Field(..., description="Flooded area (ha)")
    flood_percentage: float = Field(..., description="% of polygon flooded")
    pixel_resolution_m: float = Field(
        ..., description="Approx pixel edge length in meters"
    )
    method: str = Field(..., description="Compute method")
    crs: str = Field("EPSG:4326")
    geometry_type: str = Field(..., description="Input geometry type")
    timestamp: str = Field(..., description="ISO-8601 UTC")


# --- App ---
app = FastAPI(
    title="GeoESG A.E.C.O API",
    version="0.3.0",
    description="Zero-copy Rust-accelerated flood detection API for NTB, Indonesia.",
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Transparent 1x1 PNG for fallback
TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=TRANSPARENT_PNG, media_type="image/png")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="LIVE",
        engine="FastAPI + Rust (PyO3/Rayon)",
        rust_ready=RUST_READY,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if not FINAL_MAP.exists():
        raise HTTPException(404, "Flood map not found")
    with rasterio.open(FINAL_MAP) as src:
        data = src.read(1)
        b = src.bounds
        valid = data[data != (src.nodata or 255)]
        n_flood = int(np.sum(valid == 1))
        total = len(valid)
    return StatsResponse(
        map=FINAL_MAP.name,
        flood_pixels=n_flood,
        flood_percentage=round(100.0 * n_flood / max(total, 1), 2),
        total_pixels=total,
        bounds={
            "west": b.left,
            "south": b.bottom,
            "east": b.right,
            "north": b.top,
        },
    )


# ---------------------------------------------------------------------------
# /predict/at — RTK Point Validation (Core Stage 5 Endpoint)
# ---------------------------------------------------------------------------
@app.get("/predict/at", response_model=FloodPrediction)
def predict_at(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude (WGS84)"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude (WGS84)"),
):
    """Query flood status at exact RTK coordinates.

    Reads a 1×1 pixel window from co-registered Sentinel-1 (VV) and
    Sentinel-2 (Green, NIR) rasters, passes them to
    `flood_rs.compute_ndwi_and_mask` via zero-copy, and returns the
    binary flood classification (0=safe, 1=flood).
    """
    if not RUST_READY:
        raise HTTPException(503, "Rust engine (flood_rs) not available")

    # --- Validate raster availability ---
    if not S1_RASTER.exists():
        raise HTTPException(
            404, f"Sentinel-1 raster not found: {S1_RASTER.name}"
        )

    # --- Read SAR VV at point ---
    with rasterio.open(S1_RASTER) as src_s1:
        if not _point_in_bounds(lat, lon, src_s1.bounds):
            raise HTTPException(
                422,
                f"Point ({lat}, {lon}) outside raster bounds: "
                f"W={src_s1.bounds.left:.4f} S={src_s1.bounds.bottom:.4f} "
                f"E={src_s1.bounds.right:.4f} N={src_s1.bounds.top:.4f}",
            )
        row, col = src_s1.index(lon, lat)
        win = rasterio.windows.Window(col, row, 1, 1)
        vv_pixel = src_s1.read(1, window=win).astype(np.float32)  # (1,1)

    # --- Read Sentinel-2 Green + NIR at point (if available) ---
    if S2_RASTER.exists():
        with rasterio.open(S2_RASTER) as src_s2:
            row2, col2 = src_s2.index(lon, lat)
            win2 = rasterio.windows.Window(col2, row2, 1, 1)
            green_pixel = src_s2.read(1, window=win2).astype(np.float32)
            nir_pixel = src_s2.read(2, window=win2).astype(np.float32)

        # --- Fused Rust compute: NDWI + SAR → mask ---
        mask = flood_rs.compute_ndwi_and_mask(
            green_pixel, nir_pixel, vv_pixel, NDWI_THRESH, SAR_VV_THRESH
        )

        # Extract scalar NDWI for response metadata
        g, n = float(green_pixel[0, 0]), float(nir_pixel[0, 0])
        denom = g + n
        ndwi_val = (g - n) / denom if denom != 0.0 else None

        return FloodPrediction(
            lat=lat,
            lon=lon,
            flood=int(mask[0, 0]),
            ndwi=round(ndwi_val, 4) if ndwi_val is not None else None,
            sar_vv=round(float(vv_pixel[0, 0]), 2),
            method="fused_ndwi_sar (compute_ndwi_and_mask)",
            crs="EPSG:4326",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    else:
        # --- Fallback: SAR-only ---
        with rasterio.open(S1_RASTER) as src_s1:
            vh_pixel = src_s1.read(2, window=win).astype(np.float32)
        mask = flood_rs.calculate_sar_flood_mask(
            vv_pixel, vh_pixel, -18.0, -24.0
        )
        return FloodPrediction(
            lat=lat,
            lon=lon,
            flood=int(mask[0, 0]),
            ndwi=None,
            sar_vv=round(float(vv_pixel[0, 0]), 2),
            method="sar_only (calculate_sar_flood_mask)",
            crs="EPSG:4326",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _point_in_bounds(lat: float, lon: float, bounds) -> bool:
    """Check if (lat, lon) falls within rasterio BoundingBox."""
    return (
        bounds.left <= lon <= bounds.right
        and bounds.bottom <= lat <= bounds.top
    )


# ---------------------------------------------------------------------------
# /predict/area — Polygon Zonal Statistics (Stage 7 Endpoint)
# ---------------------------------------------------------------------------
@app.post("/predict/area", response_model=AreaReport)
def predict_area(feature: GeoJSONFeature):
    """Compute flooded area within a GeoJSON polygon.

    Crops co-registered S1/S2 rasters to the polygon via rasterio.mask,
    runs flood_rs.compute_ndwi_and_mask on the masked arrays, and
    returns an audit-ready ESG area report in hectares.
    """
    if not RUST_READY:
        raise HTTPException(503, "Rust engine (flood_rs) not available")

    geom = feature.geometry
    geom_type = geom.get("type", "")
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise HTTPException(
            422, f"Geometry must be Polygon or MultiPolygon, got '{geom_type}'"
        )

    if not S1_RASTER.exists() or not S2_RASTER.exists():
        raise HTTPException(404, "Co-registered S1/S2 rasters not found")

    shapes = [geom]

    try:
        # --- Crop S1 (VV) to polygon ---
        with rasterio.open(S1_RASTER) as src_s1:
            vv_masked, vv_transform = rasterio.mask.mask(
                src_s1, shapes, indexes=[1],
                crop=True, filled=True, nodata=np.nan,
            )
        # --- Crop S2 (Green, NIR) to polygon ---
        with rasterio.open(S2_RASTER) as src_s2:
            s2_masked, s2_transform = rasterio.mask.mask(
                src_s2, shapes, indexes=[1, 2],
                crop=True, filled=True, nodata=np.nan,
            )
    except ValueError:
        raise HTTPException(
            422, "Polygon does not overlap with raster bounds"
        )

    # Extract 2D arrays — (bands, H, W) → (H, W)
    vv_2d = vv_masked[0].astype(np.float32)
    green_2d = s2_masked[0].astype(np.float32)
    nir_2d = s2_masked[1].astype(np.float32)

    # --- Rust fused compute ---
    mask = flood_rs.compute_ndwi_and_mask(
        green_2d, nir_2d, vv_2d, NDWI_THRESH, SAR_VV_THRESH
    )

    # --- Zonal statistics ---
    # Valid pixels = not NaN in any input band
    valid_mask = (
        ~np.isnan(vv_2d) & ~np.isnan(green_2d) & ~np.isnan(nir_2d)
    )
    total_valid = int(np.sum(valid_mask))
    flooded = int(np.sum((mask == 1) & valid_mask))

    # --- Pixel area calculation (degrees → meters → hectares) ---
    # Pixel resolution in degrees
    dx_deg = abs(vv_transform[0])
    dy_deg = abs(vv_transform[4])

    # Approximate center latitude from the cropped window
    center_lat = _geom_centroid_lat(geom)
    cos_lat = math.cos(math.radians(center_lat))

    # Degree → meter conversion (WGS84 approximation)
    dx_m = dx_deg * 111320.0 * cos_lat
    dy_m = dy_deg * 111320.0
    pixel_area_sqm = dx_m * dy_m
    pixel_area_ha = pixel_area_sqm / 10000.0
    pixel_res_m = round((dx_m + dy_m) / 2.0, 2)

    total_area_ha = round(total_valid * pixel_area_ha, 4)
    flooded_area_ha = round(flooded * pixel_area_ha, 4)
    pct = round(100.0 * flooded / max(total_valid, 1), 2)

    return AreaReport(
        total_pixels=total_valid,
        flooded_pixels=flooded,
        total_area_ha=total_area_ha,
        flooded_area_ha=flooded_area_ha,
        flood_percentage=pct,
        pixel_resolution_m=pixel_res_m,
        method="fused_ndwi_sar (compute_ndwi_and_mask)",
        crs="EPSG:4326",
        geometry_type=geom_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# /predict/report — Generate ESG PDF Report
# ---------------------------------------------------------------------------
@app.post("/predict/report")
def predict_report(feature: GeoJSONFeature):
    report = predict_area(feature)
    report_dict = report.dict() if hasattr(report, "dict") else report.model_dump()
    pdf_path = generate_esg_pdf(report_dict)
    return FileResponse(pdf_path, media_type="application/pdf", filename="esg_report.pdf")


def _geom_centroid_lat(geom: dict) -> float:
    """Extract approximate centroid latitude from GeoJSON geometry."""
    coords = geom.get("coordinates", [])
    if geom["type"] == "MultiPolygon":
        # Flatten first polygon's exterior ring
        coords = coords[0][0] if coords else []
    elif geom["type"] == "Polygon":
        coords = coords[0] if coords else []
    if not coords:
        return -8.5  # Sumbawa fallback
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats)


# ---------------------------------------------------------------------------
# /tiles/{z}/{x}/{y}.png
# ---------------------------------------------------------------------------
@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    if not FINAL_MAP.exists():
        return Response(content=TRANSPARENT_PNG, media_type="image/png")
    try:
        from rio_tiler.io import Reader
        from rio_tiler.utils import render

        with Reader(str(FINAL_MAP)) as dst:
            img = dst.tile(x, y, z)
            alpha = (img.data[0] == 1).astype(np.uint8) * 255
            rgb = np.zeros((3, img.height, img.width), dtype=np.uint8)
            rgb[1], rgb[2] = 180, 216
            return Response(
                content=render(rgb, alpha, img_format="PNG"),
                media_type="image/png",
            )
    except Exception as e:
        logger.error(f"Tile rendering error: {e}")
        return Response(content=TRANSPARENT_PNG, media_type="image/png")


# ---------------------------------------------------------------------------
# / (Dashboard)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Error: index.html not found</h1>")
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
