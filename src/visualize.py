"""
visualize.py - Flood Map Visualization for NTB Flood Detection.
Reads flood prediction GeoTIFF, renders with custom colormap, saves PNG.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"


def visualize_flood_map(input_path=None, output_path=None):
    """Read flood map GeoTIFF and save as styled PNG."""
    in_path = Path(input_path) if input_path else PREDICTIONS_DIR / "flood_map.tif"
    out_path = (
        Path(output_path) if output_path else PREDICTIONS_DIR / "flood_map_preview.png"
    )

    if not in_path.exists():
        raise FileNotFoundError(
            f"Flood map not found: {in_path}. Run predict.py first."
        )

    # Read raster
    with rasterio.open(in_path) as ds:
        flood = ds.read(1)
        bounds = ds.bounds
        crs = ds.crs
        nodata = ds.nodata
    logger.info("Loaded %s: shape=%s, crs=%s", in_path.name, flood.shape, crs)

    # Mask nodata pixels
    if nodata is not None:
        flood_masked = np.ma.masked_equal(flood, int(nodata))
    else:
        flood_masked = np.ma.array(flood)

    # Stats
    total = flood_masked.count()
    n_flood = int(np.sum(flood_masked == 1))
    n_dry = int(np.sum(flood_masked == 0))
    pct = 100.0 * n_flood / max(total, 1)
    logger.info("Flood: %d pixels (%.2f%%), Non-flood: %d pixels", n_flood, pct, n_dry)

    # Custom colormap: 0=light grey, 1=bright cyan-blue
    cmap = ListedColormap(["#D9D9D9", "#00B4D8"])
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=100)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    ax.imshow(
        flood_masked,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    ax.set_title(
        "Flood Detection — Sumbawa Island (XGBoost)",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Legend
    legend_patches = [
        Patch(
            facecolor="#D9D9D9",
            edgecolor="black",
            linewidth=0.5,
            label=f"Non-Flood ({n_dry:,} px)",
        ),
        Patch(
            facecolor="#00B4D8",
            edgecolor="black",
            linewidth=0.5,
            label=f"Flood ({n_flood:,} px — {pct:.1f}%)",
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="grey",
    )

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax.tick_params(labelsize=9)

    # CRS annotation
    ax.annotate(
        f"CRS: {crs}",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=7,
        color="grey",
        alpha=0.7,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved preview: %s (%.2f MB)", out_path, size_mb)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTB Flood - Visualize Prediction")
    parser.add_argument("--input", type=str, default=None, help="Path to flood_map.tif")
    parser.add_argument("--output", type=str, default=None, help="Path for output PNG")
    args = parser.parse_args()
    try:
        result = visualize_flood_map(args.input, args.output)
        print(f"  preview: {result}")
    except Exception as exc:
        logger.error("VISUALIZATION FAILED: %s", exc)
        sys.exit(1)
