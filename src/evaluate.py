"""
evaluate.py - Metrics for NTB Flood Detection.
Computes IoU, F1, Precision, Recall, Confusion Matrix.
Outputs metrics JSON to outputs/models/.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def load_raster_band(path, band=1):
    """Load single band from GeoTIFF as uint8."""
    with rasterio.open(path) as ds:
        data = ds.read(band).astype(np.uint8)
    return data


def compute_confusion(y_true, y_pred):
    """Compute TP, FP, TN, FN from flat arrays."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_metrics(tp, fp, tn, fn):
    """Derive IoU, F1, Precision, Recall from confusion counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "iou": round(iou, 6),
        "f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "accuracy": round(accuracy, 6),
    }


def run_evaluation(pred_path=None, label_path=None):
    """Run full evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 60)

    pred_file = Path(pred_path) if pred_path else PREDICTIONS_DIR / "final_flood_map.tif"
    label_file = Path(label_path) if label_path else LABELS_DIR / "flood_labels.tif"

    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"Labels not found: {label_file}")

    y_pred = load_raster_band(pred_file).flatten()
    y_true = load_raster_band(label_file).flatten()

    if y_pred.shape != y_true.shape:
        raise RuntimeError(f"Shape mismatch: pred={y_pred.shape} labels={y_true.shape}")

    # Filter nodata (255)
    valid = (y_pred != 255) & (y_true != 255)
    y_pred = y_pred[valid]
    y_true = y_true[valid]
    logger.info("Valid pixels: %d", len(y_pred))

    tp, fp, tn, fn = compute_confusion(y_true, y_pred)
    logger.info("Confusion: TP=%d FP=%d TN=%d FN=%d", tp, fp, tn, fn)

    metrics = compute_metrics(tp, fp, tn, fn)
    metrics["confusion_matrix"] = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    metrics["total_pixels"] = len(y_pred)

    for k, v in metrics.items():
        if k not in ("confusion_matrix", "total_pixels"):
            logger.info("  %s: %.4f", k, v)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "evaluation_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved: %s", out_path)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NTB Flood - Evaluate")
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    args = parser.parse_args()
    try:
        run_evaluation(args.pred, args.labels)
    except Exception as exc:
        logger.error("EVALUATION FAILED: %s", exc)
        sys.exit(1)
