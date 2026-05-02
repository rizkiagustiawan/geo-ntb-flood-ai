
with open('api/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add pyproj
if "from pyproj import Transformer, CRS" not in content:
    content = content.replace("from fastapi import", "from pyproj import Transformer, CRS\nfrom fastapi import")

# Add DEM_PATH
if "DEM_PATH =" not in content:
    content = content.replace(
        'S2_RASTER = DATA_DIR / "processed" / "sentinel2_reproj.tif"  # Band1=Green, Band2=NIR\n',
        'S2_RASTER = DATA_DIR / "processed" / "sentinel2_reproj.tif"  # Band1=Green, Band2=NIR\nDEM_PATH = DATA_DIR / "dem_sumbawa.tif"\n'
    )

# Add status to FloodPrediction
old_model = """class FloodPrediction(BaseModel):
    \"\"\"Structured response for /predict/at RTK validation endpoint.\"\"\"

    lat: float = Field(..., description="Query latitude (WGS84)")
    lon: float = Field(..., description="Query longitude (WGS84)")
    flood: Literal[0, 1] = Field(..., description="0=safe, 1=flood")
    ndwi: float | None = Field(None, description="NDWI value at point")
    sar_vv: float | None = Field(None, description="SAR VV backscatter (dB)")
    method: str = Field(..., description="Compute method used")
    crs: str = Field("EPSG:4326", description="Coordinate reference system")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")"""

new_model = """class FloodPrediction(BaseModel):
    \"\"\"Structured response for /predict/at RTK validation endpoint.\"\"\"

    lat: float = Field(..., description="Query latitude (WGS84)")
    lon: float = Field(..., description="Query longitude (WGS84)")
    flood: Literal[0, 1] = Field(..., description="0=safe, 1=flood")
    status: str | None = Field(None, description="Status classification (e.g., 'permanent_water', 'flood_detected', 'safe')")
    ndwi: float | None = Field(None, description="NDWI value at point")
    sar_vv: float | None = Field(None, description="SAR VV backscatter (dB)")
    method: str = Field(..., description="Compute method used")
    crs: str = Field("EPSG:4326", description="Coordinate reference system")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")"""

content = content.replace(old_model, new_model)

# Update predict_at logic
old_predict_logic = """        # --- Fused Rust compute: NDWI + SAR → mask ---
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
        )"""

new_predict_logic = """        # --- Fused Rust compute: NDWI + SAR → mask ---
        mask = flood_rs.compute_ndwi_and_mask(
            green_pixel, nir_pixel, vv_pixel, NDWI_THRESH, SAR_VV_THRESH
        )

        # Extract scalar NDWI for response metadata
        g, n = float(green_pixel[0, 0]), float(nir_pixel[0, 0])
        denom = g + n
        ndwi_val = (g - n) / denom if denom != 0.0 else None
        
        flood_val = int(mask[0, 0])
        status_label = "flood_detected" if flood_val == 1 else "safe"
        
        # --- DEM Ocean Masking ---
        if DEM_PATH.exists():
            with rasterio.open(DEM_PATH) as dem_src:
                dem_crs = CRS.from_user_input(dem_src.crs)
                wgs84_crs = CRS.from_epsg(4326)
                
                query_lon, query_lat = lon, lat
                if not dem_crs.equals(wgs84_crs):
                    transformer = Transformer.from_crs(wgs84_crs, dem_crs, always_xy=True)
                    query_lon, query_lat = transformer.transform(lon, lat)
                
                try:
                    d_row, d_col = dem_src.index(query_lon, query_lat)
                    d_win = rasterio.windows.Window(d_col, d_row, 1, 1)
                    dem_val = dem_src.read(1, window=d_win)[0, 0]
                    if dem_val <= 0 and flood_val == 1:
                        flood_val = 0
                        status_label = "permanent_water"
                except Exception as e:
                    # Ignore out-of-bounds DEM queries gracefully
                    pass

        return FloodPrediction(
            lat=lat,
            lon=lon,
            flood=flood_val,
            status=status_label,
            ndwi=round(ndwi_val, 4) if ndwi_val is not None else None,
            sar_vv=round(float(vv_pixel[0, 0]), 2),
            method="fused_ndwi_sar (compute_ndwi_and_mask)",
            crs="EPSG:4326",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )"""

content = content.replace(old_predict_logic, new_predict_logic)

with open('api/main.py', 'w', encoding='utf-8') as f:
    f.write(content)
