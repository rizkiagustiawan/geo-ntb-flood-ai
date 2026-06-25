"""Convert digitized shapefile to raster ground truth.

Usage:
    python scripts/shapefile_to_raster.py \
        --shapefile flood_polygons.shp \
        --reference sentinel1_reproj.tif \
        --output data/labels/ground_truth.tif
"""

import argparse
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Shapefile to Raster")
    parser.add_argument("--shapefile", required=True, help="Input shapefile/GeoJSON")
    parser.add_argument("--reference", required=True, help="Reference raster for grid")
    parser.add_argument("--output", required=True, help="Output GeoTIFF")
    args = parser.parse_args()

    # Load polygon
    print(f"Loading: {args.shapefile}")
    gdf = gpd.read_file(args.shapefile)
    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")

    # Load reference grid
    with rasterio.open(args.reference) as src:
        profile = src.profile.copy()
        transform = src.transform
        out_shape = src.shape
        ref_crs = src.crs

    # Reproject if needed
    if gdf.crs != ref_crs:
        print(f"  Reprojecting: {gdf.crs} → {ref_crs}")
        gdf = gdf.to_crs(ref_crs)

    # Rasterize
    print(f"Rasterizing to {out_shape}...")
    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(shapes, out_shape=out_shape, transform=transform)

    n_flood = int(np.sum(mask))
    print(f"  Flood pixels: {n_flood} ({100*n_flood/mask.size:.2f}%)")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
