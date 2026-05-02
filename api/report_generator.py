"""GeoESG A.E.C.O — AOI-Clipped Flood Report Generator.

Performs spatial clipping of the final flood map using an AOI GeoJSON
boundary, computes pixel-level flood statistics strictly within the AOI,
and generates an audit-ready ESG PDF report.

Key design decisions:
    - CRS is NEVER hardcoded. Both the GeoJSON and raster CRS are read
      dynamically; if they mismatch, the GeoJSON geometry is reprojected
      in-memory to match the raster's CRS via pyproj.
    - Pixel area is computed using WGS-84 latitude-corrected approximation
      when the raster CRS is geographic, or directly from the transform
      when the CRS is projected (meters).
"""

import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.mask
from fpdf import FPDF
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

logger = logging.getLogger("aeco-report")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MAP = PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif"


# ---------------------------------------------------------------------------
# CRS Utilities
# ---------------------------------------------------------------------------
def _reproject_geojson_geometry(
    geom_dict: dict[str, Any],
    src_crs: CRS,
    dst_crs: CRS,
) -> dict[str, Any]:
    """Reproject a GeoJSON geometry dict from src_crs to dst_crs in-memory.

    Uses pyproj.Transformer for thread-safe, PROJ-pipeline-based
    coordinate transformation. Returns a new GeoJSON geometry dict.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geom_shapely = shape(geom_dict)
    reprojected = shapely_transform(transformer.transform, geom_shapely)
    return mapping(reprojected)


def _resolve_geojson_crs(feature: dict[str, Any]) -> CRS:
    """Detect CRS from a GeoJSON Feature.

    Per RFC-7946, GeoJSON coordinates are WGS-84 (EPSG:4326) unless
    the Feature explicitly carries a 'crs' property (legacy spec).
    """
    crs_prop = (feature.get("properties") or {}).get("crs")
    if crs_prop:
        try:
            return CRS.from_user_input(crs_prop)
        except Exception:
            pass
    return CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# AOI Clipping + Statistics
# ---------------------------------------------------------------------------
def compute_aoi_flood_stats(feature: dict[str, Any]) -> dict[str, Any]:
    """Clip final_flood_map.tif to AOI and compute flood statistics.

    Args:
        feature: A GeoJSON Feature dict with Polygon/MultiPolygon geometry.

    Returns:
        Dict with keys: total_area_ha, flooded_area_ha, flood_percentage,
        total_pixels, flooded_pixels, pixel_resolution_m, raster_crs,
        geometry_type, timestamp.

    Raises:
        FileNotFoundError: If final_flood_map.tif does not exist.
        ValueError: If polygon does not overlap raster bounds.
    """
    if not FINAL_MAP.exists():
        raise FileNotFoundError(f"Flood map not found: {FINAL_MAP}")

    geom = feature["geometry"]
    geojson_crs = _resolve_geojson_crs(feature)

    with rasterio.open(FINAL_MAP) as src:
        raster_crs = CRS.from_user_input(src.crs)

        # --- Dynamic CRS alignment ---
        if not geojson_crs.equals(raster_crs):
            logger.info(
                "CRS mismatch detected: GeoJSON=%s, Raster=%s. "
                "Reprojecting GeoJSON to raster CRS.",
                geojson_crs.to_epsg() or geojson_crs.to_wkt(),
                raster_crs.to_epsg() or raster_crs.to_wkt(),
            )
            geom = _reproject_geojson_geometry(geom, geojson_crs, raster_crs)

        # --- Spatial clipping via rasterio.mask ---
        clipped, clipped_transform = rasterio.mask.mask(
            src,
            [geom],
            crop=True,
            filled=True,
            nodata=255,
        )

        nodata_val = src.nodata if src.nodata is not None else 255
        raster_transform = src.transform  # for pixel-area from full raster

    # --- Pixel-level flood statistics ---
    band = clipped[0]
    valid_mask = band != nodata_val
    total_valid = int(np.sum(valid_mask))
    flooded = int(np.sum((band == 1) & valid_mask))

    # --- Pixel area computation ---
    dx = abs(clipped_transform[0])
    dy = abs(clipped_transform[4])

    if raster_crs.is_geographic:
        # CRS is in degrees → latitude-corrected meter conversion
        center_lat = _geom_centroid_lat(feature["geometry"])
        cos_lat = math.cos(math.radians(center_lat))
        dx_m = dx * 111320.0 * cos_lat
        dy_m = dy * 111320.0
    else:
        # CRS is projected (meters/feet) — transform values are already metric
        linear_unit = raster_crs.axis_info[0].unit_name
        if "metre" in linear_unit or "meter" in linear_unit:
            dx_m = dx
            dy_m = dy
        else:
            # Fallback: assume meters (most projected CRS are metric)
            dx_m = dx
            dy_m = dy
            logger.warning(
                "Projected CRS unit '%s' not explicitly metric; "
                "assuming meters for pixel area calculation.",
                linear_unit,
            )

    pixel_area_ha = (dx_m * dy_m) / 10000.0
    pixel_res_m = round((dx_m + dy_m) / 2.0, 2)

    total_area_ha = round(total_valid * pixel_area_ha, 4)
    flooded_area_ha = round(flooded * pixel_area_ha, 4)
    pct = round(100.0 * flooded / max(total_valid, 1), 2)

    return {
        "total_area_ha": total_area_ha,
        "flooded_area_ha": flooded_area_ha,
        "flood_percentage": pct,
        "total_pixels": total_valid,
        "flooded_pixels": flooded,
        "pixel_resolution_m": pixel_res_m,
        "raster_crs": str(raster_crs.to_epsg() or raster_crs.to_wkt()),
        "geometry_type": feature["geometry"].get("type", "Unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _geom_centroid_lat(geom: dict) -> float:
    """Extract approximate centroid latitude from GeoJSON geometry."""
    coords = geom.get("coordinates", [])
    if geom["type"] == "MultiPolygon":
        coords = coords[0][0] if coords else []
    elif geom["type"] == "Polygon":
        coords = coords[0] if coords else []
    if not coords:
        return -8.5  # Sumbawa fallback
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats)


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------
def generate_esg_pdf(report_data: dict[str, Any]) -> str:
    """Generate an audit-ready ESG PDF from AOI-clipped flood statistics.

    Args:
        report_data: Dict from compute_aoi_flood_stats() or compatible.

    Returns:
        Absolute path string to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()

    # --- Header ---
    pdf.set_font("helvetica", "B", 18)
    pdf.cell(
        0, 12, "GeoESG A.E.C.O Audit Report",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.set_font("helvetica", "", 10)
    pdf.cell(
        0, 8,
        f"Generated: {report_data.get('timestamp', datetime.now(timezone.utc).isoformat())}",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(5)

    # --- AOI Metadata ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Area of Interest (AOI)", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 11)
    pdf.cell(
        0, 8,
        f"Geometry Type: {report_data.get('geometry_type', 'N/A')}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.cell(
        0, 8,
        f"Raster CRS: EPSG:{report_data.get('raster_crs', 'N/A')}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.cell(
        0, 8,
        f"Pixel Resolution: {report_data.get('pixel_resolution_m', 'N/A')} m",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(5)

    # --- Flood Statistics (AOI-Clipped) ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(
        0, 10, "Flood Statistics (AOI-Clipped)",
        new_x="LMARGIN", new_y="NEXT",
    )

    pdf.set_font("helvetica", "", 12)
    stats_rows = [
        ("Total AOI Area", f"{report_data.get('total_area_ha', 0)} Ha"),
        ("Flooded AOI Area", f"{report_data.get('flooded_area_ha', 0)} Ha"),
        ("Flood Percentage", f"{report_data.get('flood_percentage', 0)}%"),
        ("Total Pixels (Valid)", str(report_data.get("total_pixels", 0))),
        ("Flooded Pixels", str(report_data.get("flooded_pixels", 0))),
    ]
    for label, value in stats_rows:
        pdf.cell(0, 9, f"{label}: {value}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)

    # --- Methodology Note ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    methodology_text = (
        "Flood detection via multisensor fusion: Sentinel-1 SAR backscatter "
        "thresholding combined with Sentinel-2 NDWI verification. "
        "The flood map was spatially clipped to the provided AOI boundary "
        "using rasterio.mask before computing zonal statistics. "
        "CRS alignment between the GeoJSON AOI and the raster was validated "
        "dynamically; any mismatch triggers automatic in-memory reprojection "
        "via pyproj."
    )
    pdf.multi_cell(0, 6, methodology_text)

    pdf.ln(5)
    pdf.set_font("helvetica", "I", 9)
    pdf.cell(
        0, 8,
        "This report is auto-generated by the GeoESG A.E.C.O pipeline. "
        "Statistics are computed per-pixel and may differ from vector-based area calculations.",
        new_x="LMARGIN", new_y="NEXT",
    )

    # --- Write PDF ---
    out_dir = Path("/tmp/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"esg_report_{uuid.uuid4().hex[:8]}.pdf"

    pdf.output(str(out_path))
    logger.info("PDF generated: %s", out_path)
    return str(out_path)
