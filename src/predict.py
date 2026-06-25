"""
predict.py - Flood Prediction for NTB Flood Detection.
Loads trained XGBoost model + feature stack, runs pixel-wise inference,
outputs flood map GeoTIFF (uint8: 0=non-flood, 1=flood).
"""

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

FEATURE_NAMES = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB", "HAND_m"]


def load_models():
    """Load trained ensemble models (XGBoost, RF, LightGBM)."""
    models = {}
    for name, filename in [("xgb", "xgboost.pkl"), ("rf", "random_forest.pkl"), ("lgb", "lightgbm.pkl")]:
        path = MODELS_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info("Loaded %s model", name)
        else:
            logger.warning("Model missing: %s", path)
    if not models:
        raise FileNotFoundError("No trained models found. Run model.py first.")
    return models


def load_feature_stack():
    """Load feature stack GeoTIFF. Returns (data, profile).
    data shape: (n_bands, H, W)."""
    path = PROCESSED_DIR / "feature_stack.tif"
    if not path.exists():
        raise FileNotFoundError(f"Feature stack not found: {path}. Run features.py first.")
    with rasterio.open(path) as ds:
        data = ds.read()  # (bands, H, W)
        profile = ds.profile.copy()
    logger.info("Loaded feature stack: %d bands, %dx%d", data.shape[0], data.shape[1], data.shape[2])
    return data, profile


def predict_flood(models, features, chunk_rows=1000):
    """Run pixel-wise ensemble inference on feature stack in row-chunks.
    features: (n_bands, H, W) float32 array.
    chunk_rows: number of rows to process at once (memory-bound).
    Returns: (H, W) uint8 array — 0=non-flood, 1=flood."""
    from model import predict_ensemble
    n_bands, height, width = features.shape
    n_pixels = height * width

    logger.info("Predicting %d pixels (%dx%d) in chunks of %d rows using %d ensemble models", 
                n_pixels, height, width, chunk_rows, len(models))

    flood_map = np.zeros((height, width), dtype=np.uint8)
    n_valid_total = 0
    n_flood_total = 0

    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        chunk = features[:, row_start:row_end, :]  # (n_bands, chunk_h, W)
        chunk_h = row_end - row_start
        n_chunk = chunk_h * width

        # Flatten chunk: (chunk_h * W, n_bands)
        X_flat = chunk.reshape(n_bands, -1).T

        # Identify valid pixels
        valid_mask = ~(np.all(X_flat == 0, axis=1) | np.any(np.isnan(X_flat), axis=1))
        n_valid = np.sum(valid_mask)
        n_valid_total += n_valid

        if n_valid == 0:
            continue

        X_valid = np.nan_to_num(X_flat[valid_mask], nan=0.0)
        
        # Ensemble prediction (majority vote)
        y_pred = predict_ensemble(models, X_valid)

        chunk_pred = np.zeros(n_chunk, dtype=np.uint8)
        chunk_pred[valid_mask] = y_pred
        flood_map[row_start:row_end, :] = chunk_pred.reshape(chunk_h, width)
        n_flood_total += np.sum(y_pred == 1)

    flood_pct = 100.0 * n_flood_total / n_pixels
    logger.info("Prediction complete: %d valid, %d flood pixels (%.2f%%)",
                n_valid_total, n_flood_total, flood_pct)

    return flood_map


def predict_unet(features, onnx_model_path, patch_size=256, stride=128):
    """Run overlapping sliding-window inference using ONNX U-Net in Rust."""
    import flood_rs

    n_bands, height, width = features.shape
    if n_bands != 6:
        raise ValueError(f"U-Net expects 6 bands, got {n_bands}")

    logger.info("Predicting %dx%d using U-Net (ONNX via Rust) with %dx%d patches and stride %d", 
                height, width, patch_size, patch_size, stride)

    # Accumulators for overlapping predictions
    prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # 2D Gaussian weight window to prioritize center pixels over edges
    y, x = np.ogrid[-patch_size/2:patch_size/2, -patch_size/2:patch_size/2]
    window_weight = np.exp(-(x*x + y*y) / (2.0 * (patch_size/4)**2)).astype(np.float32)

    batch_size = 16
    batch_patches = []
    batch_coords = []

    def process_batch(patches, coords):
        batch_array = np.stack(patches).astype(np.float32)
        preds = flood_rs.predict_unet_chunk(batch_array, onnx_model_path)
        
        for i, (r, c, h, w) in enumerate(coords):
            pred_patch = preds[i, 0, :h, :w]
            weight_patch = window_weight[:h, :w]
            prob_map[r:r+h, c:c+w] += pred_patch * weight_patch
            weight_map[r:r+h, c:c+w] += weight_patch

    for row in range(0, height, stride):
        for col in range(0, width, stride):
            r_end = min(row + patch_size, height)
            c_end = min(col + patch_size, width)
            h = r_end - row
            w = c_end - col

            patch = features[:, row:r_end, col:c_end]
            
            if h < patch_size or w < patch_size:
                padded = np.zeros((n_bands, patch_size, patch_size), dtype=np.float32)
                padded[:, :h, :w] = patch
                patch = padded

            batch_patches.append(patch)
            batch_coords.append((row, col, h, w))

            if len(batch_patches) >= batch_size:
                process_batch(batch_patches, batch_coords)
                batch_patches = []
                batch_coords = []

    if batch_patches:
        process_batch(batch_patches, batch_coords)

    weight_map = np.maximum(weight_map, 1e-6)
    final_prob = prob_map / weight_map
    
    flood_map = (final_prob > 0.5).astype(np.uint8)
    
    flood_pct = 100.0 * np.sum(flood_map) / flood_map.size
    logger.info("U-Net Prediction complete: %.2f%% flood pixels", flood_pct)

    return flood_map


def save_flood_map(flood_map, profile, output_path=None):
    """Save flood map as GeoTIFF with CRS/transform from feature stack profile.
    dtype uint8, values 0 (non-flood) / 1 (flood)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path) if output_path else PREDICTIONS_DIR / "flood_map.tif"

    out_profile = profile.copy()
    out_profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 255,
    })

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(flood_map[np.newaxis, :, :])
        dst.set_band_description(1, "flood_prediction")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Flood map saved: %s (%.2f MB)", out_path, size_mb)

    with rasterio.open(out_path) as ds:
        logger.info("Verified: shape=%s crs=%s dtype=%s", ds.shape, ds.crs, ds.dtypes)

    return out_path


def run_prediction(model_type="ensemble", model_path=None):
    """Run full prediction pipeline using ensemble models or U-Net."""
    logger.info("=" * 60)
    logger.info("STARTING FLOOD PREDICTION (%s)", model_type.upper())
    logger.info("=" * 60)

    features, profile = load_feature_stack()
    
    if model_type == "unet":
        path = str(Path(model_path) if model_path else MODELS_DIR / "unet_flood.onnx")
        if not Path(path).exists():
            raise FileNotFoundError(f"ONNX model missing: {path}. Run unet_model.py to generate it.")
        flood_map = predict_unet(features, path)
    else:
        models = load_models()
        flood_map = predict_flood(models, features)
        
    out_path = save_flood_map(flood_map, profile)

    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE -> %s", out_path)
    logger.info("=" * 60)

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NTB Flood - Prediction")
    parser.add_argument("--type", type=str, choices=["ensemble", "unet"], default="ensemble", 
                        help="Model type to use: 'ensemble' (default) or 'unet'")
    parser.add_argument("--model", type=str, default=None, help="Path to model (Pickle or ONNX)")
    args = parser.parse_args()
    try:
        result = run_prediction(model_type=args.type, model_path=args.model)
        print(f"  flood_map: {result}")
    except Exception as exc:
        logger.error("PREDICTION FAILED: %s", exc)
        sys.exit(1)
