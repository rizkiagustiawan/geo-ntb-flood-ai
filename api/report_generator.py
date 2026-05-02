"""GeoESG A.E.C.O v2.0 — AOI-Clipped Flood Report Generator.

Performs spatial clipping of the final flood map using an AOI GeoJSON
boundary, vectorizes flood pixels into legal-grade polygons via
rasterio.features.shapes, applies Douglas-Peucker simplification,
and computes geodesic area using pyproj for audit-ready ESG reporting.

Key design decisions:
    - CRS is NEVER hardcoded. Both the GeoJSON and raster CRS are read
      dynamically; if they mismatch, the GeoJSON geometry is reprojected
      in-memory to match the raster's CRS via pyproj.
    - Flood area is computed via high-precision vector boundary extraction
      (rasterio.features.shapes) followed by Douglas-Peucker simplification
      and geodesic area calculation, replacing raw pixel counting.
    - Pixel-level counts are retained as secondary metadata for QA.
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
import rasterio.features
from rasterio.warp import reproject, Resampling
from fpdf import FPDF
from pyproj import CRS, Geod, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform, unary_union

import os
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

logger = logging.getLogger("aeco-report")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_MAP = PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif"
DEM_PATH = PROJECT_ROOT / "data" / "dem_sumbawa.tif"

# Douglas-Peucker tolerance in the raster's native CRS units.
# For WGS-84 (degrees): ~0.0001° ≈ 11m at the equator.
# For projected CRS (meters): 10m is a reasonable smoothing threshold.
DP_TOLERANCE_DEG = 0.0001
DP_TOLERANCE_M = 10.0


# ---------------------------------------------------------------------------
# CRS Utilities
# ---------------------------------------------------------------------------
def _reproject_geojson_geometry(
    geom_dict: dict[str, Any],
    src_crs: CRS,
    dst_crs: CRS,
) -> dict[str, Any]:
    """Reproject a GeoJSON geometry dict from src_crs to dst_crs in-memory."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geom_shapely = shape(geom_dict)
    reprojected = shapely_transform(transformer.transform, geom_shapely)
    return mapping(reprojected)


def _resolve_geojson_crs(feature: dict[str, Any]) -> CRS:
    """Detect CRS from a GeoJSON Feature."""
    crs_prop = (feature.get("properties") or {}).get("crs")
    if crs_prop:
        try:
            return CRS.from_user_input(crs_prop)
        except Exception:
            pass
    return CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Vectorized Flood Area Calculation
# ---------------------------------------------------------------------------
def _vectorize_flood_area(
    flood_band: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    raster_crs: CRS,
) -> dict[str, Any]:
    """Vectorize flood pixels into polygons and compute geodesic area."""
    binary = np.where((flood_band == 1) & valid_mask, 1, 0).astype(np.uint8)

    flood_shapes = list(rasterio.features.shapes(
        binary, mask=(binary == 1), transform=transform
    ))

    if not flood_shapes:
        return {
            "flooded_area_ha": 0.0,
            "flood_polygon_geojson": None,
            "num_polygons": 0,
        }

    polygons = [shape(geom) for geom, val in flood_shapes if val == 1]
    merged = unary_union(polygons)

    tolerance = DP_TOLERANCE_DEG if raster_crs.is_geographic else DP_TOLERANCE_M
    simplified = merged.simplify(tolerance, preserve_topology=True)

    wgs84 = CRS.from_epsg(4326)
    if raster_crs.is_geographic and raster_crs.to_epsg() == 4326:
        geom_wgs84 = simplified
    else:
        transformer = Transformer.from_crs(raster_crs, wgs84, always_xy=True)
        geom_wgs84 = shapely_transform(transformer.transform, simplified)

    geod = Geod(ellps="WGS84")
    if geom_wgs84.geom_type == "Polygon":
        area_m2 = abs(geod.geometry_area_perimeter(geom_wgs84)[0])
    elif geom_wgs84.geom_type == "MultiPolygon":
        area_m2 = sum(
            abs(geod.geometry_area_perimeter(p)[0]) for p in geom_wgs84.geoms
        )
    else:
        area_m2 = 0.0

    area_ha = round(area_m2 / 10000.0, 4)

    if simplified.geom_type == "MultiPolygon":
        n_polys = len(list(simplified.geoms))
    else:
        n_polys = 1

    return {
        "flooded_area_ha": area_ha,
        "flood_polygon_geojson": mapping(geom_wgs84),
        "num_polygons": n_polys,
    }


# ---------------------------------------------------------------------------
# AOI Clipping + Statistics
# ---------------------------------------------------------------------------
def compute_aoi_flood_stats(feature: dict[str, Any]) -> dict[str, Any]:
    if not FINAL_MAP.exists():
        raise FileNotFoundError(f"Flood map not found: {FINAL_MAP}")

    geom = feature["geometry"]
    geojson_crs = _resolve_geojson_crs(feature)

    with rasterio.open(FINAL_MAP) as src:
        raster_crs = CRS.from_user_input(src.crs)

        if not geojson_crs.equals(raster_crs):
            logger.info("CRS mismatch. Reprojecting GeoJSON to raster CRS.")
            geom = _reproject_geojson_geometry(geom, geojson_crs, raster_crs)

        clipped, clipped_transform = rasterio.mask.mask(
            src, [geom], crop=True, filled=True, nodata=255,
        )
        nodata_val = src.nodata if src.nodata is not None else 255

    band = clipped[0]
    valid_mask = band != nodata_val

    dem_mask = None
    if DEM_PATH.exists():
        try:
            with rasterio.open(DEM_PATH) as dem_src:
                dem_crs = CRS.from_user_input(dem_src.crs)
                if not dem_crs.equals(geojson_crs):
                    dem_geom = _reproject_geojson_geometry(
                        feature["geometry"], geojson_crs, dem_crs
                    )
                else:
                    dem_geom = feature["geometry"]

                dem_clipped, dem_transform = rasterio.mask.mask(
                    dem_src, [dem_geom], crop=True, filled=True,
                    nodata=dem_src.nodata or -9999,
                )

                if (dem_clipped.shape[1:] != band.shape
                        or not dem_crs.equals(raster_crs)):
                    resampled_dem = np.zeros(
                        (1, band.shape[0], band.shape[1]), dtype=np.float32
                    )
                    reproject(
                        source=dem_clipped, destination=resampled_dem,
                        src_transform=dem_transform, src_crs=dem_crs,
                        dst_transform=clipped_transform, dst_crs=raster_crs,
                        resampling=Resampling.bilinear,
                    )
                    dem_mask = resampled_dem[0] > 0
                else:
                    dem_mask = dem_clipped[0] > 0
        except Exception as e:
            logger.warning(f"Failed to apply DEM mask: {e}")
            dem_mask = None

    if dem_mask is not None:
        valid_mask = valid_mask & dem_mask

    total_valid = int(np.sum(valid_mask))
    flooded_pixels = int(np.sum((band == 1) & valid_mask))

    dx = abs(clipped_transform[0])
    dy = abs(clipped_transform[4])

    if raster_crs.is_geographic:
        center_lat = _geom_centroid_lat(feature["geometry"])
        cos_lat = math.cos(math.radians(center_lat))
        dx_m = dx * 111320.0 * cos_lat
        dy_m = dy * 111320.0
    else:
        dx_m = dx
        dy_m = dy

    pixel_area_ha = (dx_m * dy_m) / 10000.0
    pixel_res_m = round((dx_m + dy_m) / 2.0, 2)
    total_area_ha = round(total_valid * pixel_area_ha, 4)

    vector_result = _vectorize_flood_area(band, valid_mask, clipped_transform, raster_crs)
    flooded_area_ha = vector_result["flooded_area_ha"]

    pct = round(100.0 * flooded_area_ha / max(total_area_ha, 0.0001), 2)

    return {
        "total_area_ha": total_area_ha,
        "flooded_area_ha": flooded_area_ha,
        "flood_percentage": pct,
        "total_pixels": total_valid,
        "flooded_pixels": flooded_pixels,
        "pixel_resolution_m": pixel_res_m,
        "raster_crs": str(raster_crs.to_epsg() or raster_crs.to_wkt()),
        "geometry_type": feature["geometry"].get("type", "Unknown"),
        "num_flood_polygons": vector_result["num_polygons"],
        "method": "vectorized_douglas_peucker_geodesic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # INI YANG TADI KETINGGALAN
        "aoi_geometry": feature["geometry"],
        "flood_polygon_geojson": vector_result["flood_polygon_geojson"]
    }


def _geom_centroid_lat(geom: dict) -> float:
    coords = geom.get("coordinates", [])
    if geom["type"] == "MultiPolygon":
        coords = coords[0][0] if coords else []
    elif geom["type"] == "Polygon":
        coords = coords[0] if coords else []
    if not coords:
        return -8.5
    lats = [c[1] for c in coords]
    return sum(lats) / len(lats)


# ---------------------------------------------------------------------------
# INI FUNGSI PETA YANG TADI KETINGGALAN
# ---------------------------------------------------------------------------
def _generate_map_plot(aoi_geom: dict, flood_geom: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    aoi_shape = shape(aoi_geom)
    gdf_aoi = gpd.GeoDataFrame({'geometry': [aoi_shape]}, crs="EPSG:4326")
    gdf_aoi_wm = gdf_aoi.to_crs(epsg=3857)
    gdf_aoi_wm.plot(ax=ax, facecolor="none", edgecolor="#00FF00", linewidth=3, label="Area of Interest (AOI)")

    if flood_geom and flood_geom.get('coordinates'):
        flood_shape = shape(flood_geom)
        gdf_flood = gpd.GeoDataFrame({'geometry': [flood_shape]}, crs="EPSG:4326")
        gdf_flood_wm = gdf_flood.to_crs(epsg=3857)
        gdf_flood_wm.plot(ax=ax, facecolor="#FF0000", alpha=0.6, edgecolor="none", label="Flooded Area")

    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    except Exception as e:
        logger.warning(f"Basemap gagal dimuat: {e}")

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------
def generate_esg_pdf(report_data: dict[str, Any]) -> str:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("helvetica", "B", 18)
    pdf.cell(0, 12, "GeoESG A.E.C.O Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {report_data.get('timestamp', datetime.now(timezone.utc).isoformat())}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Area of Interest (AOI)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, f"Geometry Type: {report_data.get('geometry_type', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Raster CRS: EPSG:{report_data.get('raster_crs', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Pixel Resolution: {report_data.get('pixel_resolution_m', 'N/A')} m", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Flood Statistics (AOI-Clipped)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 12)
    stats_rows = [
        ("Total AOI Area", f"{report_data.get('total_area_ha', 0)} Ha"),
        ("Flooded AOI Area", f"{report_data.get('flooded_area_ha', 0)} Ha"),
        ("Flood Percentage", f"{report_data.get('flood_percentage', 0)}%"),
        ("Total Pixels (Valid)", str(report_data.get("total_pixels", 0))),
        ("Flooded Pixels", str(report_data.get("flooded_pixels", 0))),
        ("Flood Polygons", str(report_data.get("num_flood_polygons", 0))),
    ]
    for label, value in stats_rows:
        pdf.cell(0, 9, f"{label}: {value}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ---------------------------------------------------------------------------
    # INI INJEKSI PDF YANG TADI KETINGGALAN
    # ---------------------------------------------------------------------------
    aoi_geom = report_data.get("aoi_geometry")
    flood_geom = report_data.get("flood_polygon_geojson")

    if aoi_geom:
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Spatial Overlay (AOI vs Flood Map)", new_x="LMARGIN", new_y="NEXT")
        
        map_path = f"/tmp/reports/map_temp_{uuid.uuid4().hex[:8]}.png"
        try:
            _generate_map_plot(aoi_geom, flood_geom, map_path)
            pdf.image(map_path, x=20, w=170)
            pdf.ln(5)
            if os.path.exists(map_path):
                os.remove(map_path)
        except Exception as e:
            logger.error(f"Gagal render peta di PDF: {e}")
            pdf.set_font("helvetica", "I", 10)
            pdf.cell(0, 10, "Map rendering unavailable.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    methodology_text = (
        "Flood detection via multisensor fusion: Sentinel-1 SAR backscatter "
        "thresholding combined with Sentinel-2 NDWI verification. "
        "Area is calculated using high-precision vector boundary extraction "
        "(rasterio.features.shapes) and Douglas-Peucker simplification, "
        "with geodesic area computed on the WGS-84 ellipsoid via pyproj.Geod. "
        "The flood map was spatially clipped to the provided AOI boundary "
        "using rasterio.mask before vectorization. "
        "CRS alignment between the GeoJSON AOI and the raster was validated "
        "dynamically; any mismatch triggers automatic in-memory reprojection "
        "via pyproj."
    )
    pdf.multi_cell(0, 6, methodology_text)
    pdf.ln(5)
    pdf.set_font("helvetica", "I", 8)
    pdf.multi_cell(
        180, 5,
        "This report is auto-generated by the GeoESG A.E.C.O pipeline. "
        "Statistics are computed via vectorized polygon extraction and may "
        "differ from raw pixel-count or vector-based area calculations."
    )

    out_dir = Path("/tmp/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"esg_report_{uuid.uuid4().hex[:8]}.pdf"

    pdf.output(str(out_path))
    logger.info("PDF generated: %s", out_path)
    return str(out_path)
