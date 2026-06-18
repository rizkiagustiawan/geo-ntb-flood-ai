"""Dataset Builder for Deep Learning.

Slices the large feature_stack.tif and flood_labels.tif into smaller 
256x256 patches suitable for Convolutional Neural Networks (U-Net).
Filters out patches that are mostly NoData or empty.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"

PATCH_SIZE = 256
STRIDE = 128  # Overlap by 50% to generate more training data
MAX_NODATA_PCT = 0.5  # Max 50% NoData allowed in a patch


def build_patches():
    """Extracts 256x256 patches from features and labels."""
    logger.info("=" * 60)
    logger.info("STARTING DATASET BUILDER (U-NET PATCHES)")
    logger.info("=" * 60)

    feature_path = PROCESSED_DIR / "feature_stack.tif"
    label_path = LABELS_DIR / "flood_labels.tif"

    if not feature_path.exists() or not label_path.exists():
        logger.error("Missing feature_stack.tif or flood_labels.tif. Run model.py first.")
        sys.exit(1)

    features_out = PATCHES_DIR / "features"
    labels_out = PATCHES_DIR / "labels"
    features_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_path) as src_feat, rasterio.open(label_path) as src_label:
        height, width = src_feat.shape


        if src_label.shape != (height, width):
            raise ValueError("Feature and label dimensions do not match!")

        patch_count = 0
        skipped_nodata = 0
        skipped_empty = 0

        for row_off in range(0, height - PATCH_SIZE + 1, STRIDE):
            for col_off in range(0, width - PATCH_SIZE + 1, STRIDE):
                window = Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE)

                feat_patch = src_feat.read(window=window)  # (Bands, 256, 256)
                label_patch = src_label.read(1, window=window)  # (256, 256)

                # Validation 1: NoData threshold (check NaN or 0 across all bands)
                nodata_mask = np.isnan(feat_patch[0]) | (np.sum(feat_patch, axis=0) == 0)
                nodata_pct = np.sum(nodata_mask) / (PATCH_SIZE * PATCH_SIZE)
                
                if nodata_pct > MAX_NODATA_PCT:
                    skipped_nodata += 1
                    continue

                # Validation 2: Skip patches where labels are entirely 0, 
                # but keep a small fraction (e.g. 10%) of them as negative examples
                if np.sum(label_patch) == 0:
                    if np.random.rand() > 0.1:  # Drop 90% of empty (non-flood) patches
                        skipped_empty += 1
                        continue

                # Replace NaNs with 0 for neural network input
                feat_patch = np.nan_to_num(feat_patch, nan=0.0).astype(np.float32)
                label_patch = label_patch.astype(np.uint8)

                # Save patches
                patch_name = f"patch_{row_off}_{col_off}"
                np.save(features_out / f"{patch_name}.npy", feat_patch)
                np.save(labels_out / f"{patch_name}.npy", label_patch)
                patch_count += 1

    logger.info("Dataset Building Complete:")
    logger.info(f"  Saved Patches : {patch_count}")
    logger.info(f"  Skipped (NoData): {skipped_nodata}")
    logger.info(f"  Skipped (Empty) : {skipped_empty}")
    logger.info(f"  Output Dir    : {PATCHES_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_patches()
