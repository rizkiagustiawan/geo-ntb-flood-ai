"""
model.py - Model Training for NTB Flood Detection.
Baseline thresholding, RandomForest, and XGBoost classifiers.
Saves trained models to outputs/models/.
"""

import gc
import sys
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

FEATURE_NAMES = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB"]
RANDOM_STATE = 42


def load_feature_stack():
    """Load feature stack from processed directory. Returns (bands, profile)."""
    path = PROCESSED_DIR / "feature_stack.tif"
    if not path.exists():
        raise FileNotFoundError(f"Feature stack missing: {path}. Run features.py first.")
    with rasterio.open(path) as ds:
        data = ds.read()  # (bands, H, W)
        profile = ds.profile.copy()
    logger.info("Loaded feature stack: %d bands, %dx%d", data.shape[0], data.shape[1], data.shape[2])
    return data, profile


def generate_unsupervised_labels(features: np.ndarray) -> np.ndarray:
    """Generate flood labels using unsupervised clustering (no rule circularity).

    Uses KMeans on the feature space to separate water-like vs land-like pixels,
    then assigns the cluster with higher NDWI / lower VV as "flood".

    This avoids circularity because labels are NOT derived from the same
    threshold rules that the ML model would learn.

    Parameters
    ----------
    features : np.ndarray — (n_bands, H, W) feature stack

    Returns
    -------
    np.ndarray — uint8 labels (H, W): 1=flood, 0=non-flood
    """
    from sklearn.cluster import MiniBatchKMeans

    n_bands, h, w = features.shape
    X = features.reshape(n_bands, -1).T.astype(np.float32)

    valid = ~np.all(X == 0, axis=1) & np.all(np.isfinite(X), axis=1)
    X_valid = X[valid]

    if len(X_valid) < 100:
        logger.warning("Too few valid pixels (%d) for clustering — falling back to threshold", len(X_valid))
        ndwi = features[0]
        sar_mask = features[1]
        slope = features[2]
        return ((ndwi > 0.1) & (sar_mask == 1) & (slope < 10.0)).astype(np.uint8)

    sample_size = min(50000, len(X_valid))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_valid), sample_size, replace=False)
    X_sample = X_valid[idx]

    kmeans = MiniBatchKMeans(n_clusters=2, random_state=RANDOM_STATE, batch_size=1000)
    kmeans.fit(X_sample)
    all_labels = kmeans.predict(X_valid)

    centers = kmeans.cluster_centers_
    ndwi_col = 0  # NDWI is band 0
    flood_cluster = int(np.argmax(centers[:, ndwi_col]))

    result = np.zeros(h * w, dtype=np.uint8)
    result[valid] = (all_labels == flood_cluster).astype(np.uint8)
    result = result.reshape(h, w)

    flood_pct = 100.0 * np.sum(result) / result.size
    logger.info("Unsupervised labels: KMeans k=2, flood_cluster=%d, %.2f%% flood",
                flood_cluster, flood_pct)
    return result


def load_labels(features=None):
    """Load flood labels raster. If not available, generate unsupervised labels.

    Uses KMeans clustering instead of rule-based thresholding to avoid
    circularity (model learning its own label rules).
    """
    label_path = LABELS_DIR / "flood_labels.tif"

    if label_path.exists():
        with rasterio.open(label_path) as ds:
            labels = ds.read(1).astype(np.uint8)
        logger.info("Loaded labels from %s: %d flood, %d non-flood",
                     label_path, np.sum(labels == 1), np.sum(labels == 0))
        return labels

    logger.warning("No label file found. Generating unsupervised labels (KMeans).")
    if features is None:
        data, _ = load_feature_stack()
    else:
        data = features

    labels = generate_unsupervised_labels(data)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    stack_path = PROCESSED_DIR / "feature_stack.tif"
    with rasterio.open(stack_path) as ref:
        profile = ref.profile.copy()
        profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
        with rasterio.open(label_path, "w", **profile) as dst:
            dst.write(labels[np.newaxis, :, :])
    logger.info("Saved unsupervised labels to %s", label_path)

    return labels


def prepare_training_data(features, labels, sample_frac=0.05):
    """Flatten and sample pixel data for training.
    Returns X_train, X_test, y_train, y_test."""
    n_bands, h, w = features.shape
    X_flat = features.reshape(n_bands, -1).T  # (N_pixels, n_bands)
    y_flat = labels.flatten()                  # (N_pixels,)

    # Remove nodata pixels (where all features are 0)
    valid_mask = ~np.all(X_flat == 0, axis=1)
    X_valid = X_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    logger.info("Valid pixels: %d / %d", len(X_valid), len(X_flat))

    # Sample to reduce memory
    n_total = len(X_valid)
    n_sample = max(int(n_total * sample_frac), min(50000, n_total))
    if n_sample < n_total:
        idx = np.random.RandomState(RANDOM_STATE).choice(n_total, n_sample, replace=False)
        X_sampled = X_valid[idx]
        y_sampled = y_valid[idx]
        logger.info("Sampled %d pixels (%.1f%%)", n_sample, 100.0 * n_sample / n_total)
    else:
        X_sampled = X_valid
        y_sampled = y_valid

    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sampled
    )
    logger.info("Train: %d, Test: %d (flood ratio: train=%.3f, test=%.3f)",
                len(X_train), len(X_test),
                np.mean(y_train), np.mean(y_test))

    return X_train, X_test, y_train, y_test


def baseline_threshold_model(features):
    """Simple rule-based flood detection: NDWI > 0.1 AND SAR_mask == 1 AND slope < 10.
    Returns uint8 prediction array (H, W)."""
    logger.info("Running baseline threshold model")
    ndwi = features[0]
    sar_mask = features[1]
    slope = features[2]

    prediction = ((ndwi > 0.1) & (sar_mask == 1) & (slope < 10.0)).astype(np.uint8)
    flood_pct = 100.0 * np.sum(prediction) / prediction.size
    logger.info("Baseline: %.2f%% flood pixels", flood_pct)

    # Save baseline prediction
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / "baseline_threshold.json"
    meta = {
        "model": "baseline_threshold",
        "rules": {"NDWI": ">0.1", "SAR_mask": "==1", "Slope": "<10.0"},
        "flood_percentage": round(flood_pct, 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Baseline metadata saved: %s", meta_path)

    return prediction


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train RandomForest classifier. Returns trained model."""
    logger.info("Training RandomForest classifier")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["non-flood", "flood"], output_dict=True)
    logger.info("RandomForest test accuracy: %.4f", report["accuracy"])
    logger.info("RandomForest F1 (flood): %.4f", report["flood"]["f1-score"])

    # Feature importance
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))
    logger.info("Feature importances: %s", importances)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "random_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    logger.info("RandomForest saved: %s", model_path)

    # Save metrics
    metrics_path = MODELS_DIR / "random_forest_metrics.json"
    metrics = {
        "model": "RandomForest",
        "n_estimators": 200,
        "accuracy": report["accuracy"],
        "classification_report": report,
        "feature_importances": importances,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return rf


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost classifier. Returns trained model."""
    logger.info("Training XGBoost classifier")

    # Handle class imbalance
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["non-flood", "flood"], output_dict=True)
    logger.info("XGBoost test accuracy: %.4f", report["accuracy"])
    logger.info("XGBoost F1 (flood): %.4f", report["flood"]["f1-score"])

    # Feature importance
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    logger.info("Feature importances: %s", importances)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("XGBoost saved: %s", model_path)

    # Also save native xgb format
    xgb_path = MODELS_DIR / "xgboost.json"
    model.save_model(str(xgb_path))
    logger.info("XGBoost native saved: %s", xgb_path)

    # Save metrics
    metrics_path = MODELS_DIR / "xgboost_metrics.json"
    metrics = {
        "model": "XGBoost",
        "n_estimators": 300,
        "accuracy": report["accuracy"],
        "classification_report": report,
        "feature_importances": importances,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return model


def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM classifier. Returns trained model.

    Per Mallick et al. (2026): LightGBM with histogram-based splitting
    is faster than XGBoost for large-scale flood mapping and handles
    class imbalance via is_unbalance parameter.
    """
    import lightgbm as lgb

    logger.info("Training LightGBM classifier")

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["non-flood", "flood"], output_dict=True)
    logger.info("LightGBM test accuracy: %.4f", report["accuracy"])
    logger.info("LightGBM F1 (flood): %.4f", report["flood"]["f1-score"])

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    logger.info("Feature importances: %s", importances)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lightgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("LightGBM saved: %s", model_path)

    metrics_path = MODELS_DIR / "lightgbm_metrics.json"
    metrics = {
        "model": "LightGBM",
        "n_estimators": 300,
        "accuracy": report["accuracy"],
        "classification_report": report,
        "feature_importances": importances,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return model


def predict_ensemble(models: dict, X: np.ndarray) -> np.ndarray:
    """Ensemble prediction via majority vote across RF, XGBoost, LightGBM.

    Per Mallick et al. (2026): ensemble methods reduce individual model
    bias and improve generalization across heterogeneous flood events.

    Parameters
    ----------
    models : dict — {"rf": model, "xgb": model, "lgb": model}
    X : np.ndarray — Feature matrix (N_pixels, n_features)

    Returns
    -------
    np.ndarray — uint8 predictions via majority vote
    """
    votes = np.zeros((len(X), len(models)), dtype=np.uint8)
    for i, (name, model) in enumerate(models.items()):
        votes[:, i] = model.predict(X).astype(np.uint8)

    # Majority vote: flood if >= 2/3 models agree
    threshold = len(models) / 2.0
    prediction = (np.sum(votes, axis=1) >= threshold).astype(np.uint8)
    logger.info("Ensemble prediction: %d models, threshold=%.1f, %.2f%% flood",
                len(models), threshold, 100.0 * np.sum(prediction) / len(prediction))
    return prediction


def run_training():
    """Run full model training pipeline (RF + XGBoost + LightGBM ensemble)."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING - NTB FLOOD DETECTION")
    logger.info("=" * 60)

    # 1. Load data
    features, profile = load_feature_stack()
    labels = load_labels(features=features)

    # 2. Baseline threshold
    baseline_pred = baseline_threshold_model(features)

    # 3. Prepare ML training data
    X_train, X_test, y_train, y_test = prepare_training_data(features, labels)

    del features, labels, baseline_pred
    gc.collect()
    logger.info("Freed full rasters from memory")

    # 4. RandomForest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    gc.collect()

    # 5. XGBoost
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
    gc.collect()

    # 6. LightGBM (Mallick et al. 2026 — ensemble ML for flood resilience)
    lgb_model = train_lightgbm(X_train, X_test, y_train, y_test)
    gc.collect()

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("  Baseline: outputs/models/baseline_threshold.json")
    logger.info("  RandomForest: outputs/models/random_forest.pkl")
    logger.info("  XGBoost: outputs/models/xgboost.pkl")
    logger.info("  LightGBM: outputs/models/lightgbm.pkl")
    logger.info("=" * 60)

    return {"rf": rf_model, "xgb": xgb_model, "lgb": lgb_model}


if __name__ == "__main__":
    try:
        run_training()
    except Exception as exc:
        logger.error("MODEL TRAINING FAILED: %s", exc)
        sys.exit(1)
