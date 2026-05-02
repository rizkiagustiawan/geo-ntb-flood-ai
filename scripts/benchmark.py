"""Sumbawa-A.E.C.O — Performance Benchmark Script.

Measures actual execution time for AOI flood statistics computation
and PDF report generation. Results are printed to stdout for use in
README documentation.

Usage:
    python scripts/benchmark.py
"""

import json
import sys
import time
from pathlib import Path

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "api"))


def run_benchmark():
    from report_generator import compute_aoi_flood_stats, generate_esg_pdf

    geojson_path = PROJECT_ROOT / "data" / "sumbawa_island.geojson"
    if not geojson_path.exists():
        print(f"ERROR: GeoJSON not found at {geojson_path}")
        sys.exit(1)

    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        print("ERROR: No features in GeoJSON")
        sys.exit(1)

    print("=" * 60)
    print("Sumbawa-A.E.C.O — Performance Benchmark")
    print("=" * 60)
    print(f"GeoJSON features: {len(features)}")
    print()

    # --- Benchmark: First feature (largest polygon = Bima regency) ---
    feature = features[0]
    name = feature.get("properties", {}).get("NAME_2", "Unknown")
    print(f"[1/2] AOI Stats — {name}")

    t0 = time.perf_counter()
    stats = compute_aoi_flood_stats(feature)
    t1 = time.perf_counter()
    stats_time = t1 - t0

    print(f"  Total Area      : {stats['total_area_ha']} Ha")
    print(f"  Flooded Area    : {stats['flooded_area_ha']} Ha")
    print(f"  Flood Polygons  : {stats['num_flood_polygons']}")
    print(f"  Flood %         : {stats['flood_percentage']}%")
    print(f"  Pixel Resolution: {stats['pixel_resolution_m']} m")
    print(f"  ⏱  Time         : {stats_time:.2f}s")
    print()

    # --- Benchmark: PDF Generation ---
    print("[2/2] PDF Report Generation")
    t2 = time.perf_counter()
    pdf_path = generate_esg_pdf(stats)
    t3 = time.perf_counter()
    pdf_time = t3 - t2

    pdf_size = Path(pdf_path).stat().st_size / 1024
    print(f"  PDF Path        : {pdf_path}")
    print(f"  PDF Size        : {pdf_size:.1f} KB")
    print(f"  ⏱  Time         : {pdf_time:.2f}s")
    print()

    # --- Benchmark: Smallest feature (for comparison) ---
    # Find Sumbawa Barat (Taliwang area)
    ksb_feature = None
    for feat in features:
        if feat.get("properties", {}).get("NAME_2") == "SumbawaBarat":
            ksb_feature = feat
            break

    if ksb_feature:
        name2 = ksb_feature.get("properties", {}).get("NAME_2", "Unknown")
        print(f"[BONUS] AOI Stats — {name2}")
        t4 = time.perf_counter()
        stats2 = compute_aoi_flood_stats(ksb_feature)
        t5 = time.perf_counter()
        stats2_time = t5 - t4

        print(f"  Total Area      : {stats2['total_area_ha']} Ha")
        print(f"  Flooded Area    : {stats2['flooded_area_ha']} Ha")
        print(f"  Flood Polygons  : {stats2['num_flood_polygons']}")
        print(f"  ⏱  Time         : {stats2_time:.2f}s")
        print()

    # --- Summary ---
    total = stats_time + pdf_time
    print("=" * 60)
    print("SUMMARY (Copy these numbers to README.md)")
    print("=" * 60)
    print(f"AOI Stats ({name}): {stats_time:.2f}s | {stats['total_area_ha']} Ha | {stats['num_flood_polygons']} polygons")
    print(f"PDF Generation        : {pdf_time:.2f}s | {pdf_size:.1f} KB")
    print(f"Total Pipeline        : {total:.2f}s")
    if ksb_feature:
        print(f"AOI Stats ({name2})   : {stats2_time:.2f}s | {stats2['total_area_ha']} Ha | {stats2['num_flood_polygons']} polygons")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
