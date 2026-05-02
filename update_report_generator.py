
with open('api/report_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Resampling and reproject
import_stmt = "from rasterio.warp import reproject, Resampling\n"
if "from rasterio.warp" not in content:
    content = content.replace("import rasterio.mask\n", "import rasterio.mask\nfrom rasterio.warp import reproject, Resampling\n")

# Add DEM_PATH
if "DEM_PATH" not in content:
    content = content.replace(
        'FINAL_MAP = PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif"\n',
        'FINAL_MAP = PROJECT_ROOT / "outputs" / "predictions" / "final_flood_map.tif"\nDEM_PATH = PROJECT_ROOT / "data" / "dem_sumbawa.tif"\n'
    )

# Replace the statistics logic block
old_stats_logic = """    # --- Pixel-level flood statistics ---
    band = clipped[0]
    valid_mask = band != nodata_val
    total_valid = int(np.sum(valid_mask))
    flooded = int(np.sum((band == 1) & valid_mask))"""

new_stats_logic = """    # --- Pixel-level flood statistics ---
    band = clipped[0]
    valid_mask = band != nodata_val

    # --- DEM Elevation Masking ---
    dem_mask = None
    if DEM_PATH.exists():
        try:
            with rasterio.open(DEM_PATH) as dem_src:
                dem_crs = CRS.from_user_input(dem_src.crs)
                dem_geom = _reproject_geojson_geometry(feature["geometry"], geojson_crs, dem_crs) if not dem_crs.equals(geojson_crs) else feature["geometry"]
                
                dem_clipped, dem_transform = rasterio.mask.mask(
                    dem_src, [dem_geom], crop=True, filled=True, nodata=dem_src.nodata or -9999
                )
                
                # Check if we need to resample/reproject the DEM to match the final flood map's grid
                if dem_clipped.shape[1:] != band.shape or not dem_crs.equals(raster_crs):
                    resampled_dem = np.zeros((1, band.shape[0], band.shape[1]), dtype=np.float32)
                    reproject(
                        source=dem_clipped,
                        destination=resampled_dem,
                        src_transform=dem_transform,
                        src_crs=dem_crs,
                        dst_transform=clipped_transform,
                        dst_crs=raster_crs,
                        resampling=Resampling.bilinear
                    )
                    dem_mask = resampled_dem[0] > 0
                else:
                    dem_mask = dem_clipped[0] > 0
                    
        except Exception as e:
            logger.warning(f"Failed to apply DEM mask: {e}")
            dem_mask = None

    if dem_mask is not None:
        # Valid land pixels are those that are valid in the raster AND have elevation > 0
        valid_mask = valid_mask & dem_mask
        logger.info("DEM mask applied successfully for AOI clipping.")

    total_valid = int(np.sum(valid_mask))
    flooded = int(np.sum((band == 1) & valid_mask))"""

content = content.replace(old_stats_logic, new_stats_logic)

with open('api/report_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)
