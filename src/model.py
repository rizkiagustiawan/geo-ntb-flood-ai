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

FEATURE_NAMES = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB", "HAND_m"]
RANDOM_STATE = 42


def load_feature_stack():
    """Load feature stack from processed directory. Returns (bands, profile)."""
    path = PROCESSED_DIR / "feature_stack.tif"
    if not path.exists():
        raise FileNotFoundError(f"Feature stack missing: {path}. Run features.py first.")
    with rasterio.open(path) as ds:
        data = ds.read()  # (bands, H, W)
        profile = ds.profile.copy()
        # Detect feature names from band descriptions
        global FEATURE_NAMES
        detected = [ds.descriptions[i] or f"band_{i+1}" for i in range(ds.count)]
        if detected and detected[0] is not None:
            FEATURE_NAMES = detected
    logger.info("Loaded feature stack: %d bands, %dx%d", data.shape[0], data.shape[1], data.shape[2])
    logger.info("Feature names: %s", FEATURE_NAMES)
    return data, profile


def generate_unsupervised_labels(features: np.ndarray) -> np.ndarray:
    """Generate flood labels using SAR backscatter (data-driven, not rule-based).

    Uses Otsu adaptive thresholding on VV band to separate water from land.
    This is NOT circular because Otsu is an unsupervised method that finds
    the optimal threshold from the data distribution itself.

    For SAR-only stacks (no NDWI), uses VV threshold directly.
    For stacks with NDWI, uses KMeans clustering.

    Parameters
    ----------
    features : np.ndarray — (n_bands, H, W) feature stack

    Returns
    -------
    np.ndarray — uint8 labels (H, W): 1=flood, 0=non-flood
    """
    n_bands, h, w = features.shape

    # Check if we have NDWI (optical data)
    if FEATURE_NAMES[0] == "NDWI":
        # Use KMeans for multi-sensor fusion
        return _generate_kmeans_labels(features)

    # SAR-only: use Otsu on VV band (unsupervised, data-driven)
    if "VV_dB" in FEATURE_NAMES:
        vv_col = FEATURE_NAMES.index("VV_dB")
        vv = features[vv_col]
    else:
        vv = features[0]

    from features import otsu_threshold
    vv_flat = vv.flatten()
    valid = np.isfinite(vv_flat) & (vv_flat != 0)
    if np.sum(valid) < 100:
        logger.warning("Too few valid pixels for Otsu — using median")
        threshold = float(np.nanmedian(vv_flat[valid]))
    else:
        threshold = otsu_threshold(vv_flat[valid])

    # Water = VV below threshold (more negative = darker)
    labels = (vv < threshold).astype(np.uint8)
    flood_pct = 100.0 * np.sum(labels) / labels.size
    logger.info("SAR labels: Otsu threshold=%.2f dB, %.2f%% flood", threshold, flood_pct)
    return labels


def _generate_kmeans_labels(features: np.ndarray) -> np.ndarray:
    """Generate labels using KMeans (for multi-sensor data with NDWI)."""
    from sklearn.cluster import MiniBatchKMeans

    n_bands, h, w = features.shape
    X = features.reshape(n_bands, -1).T.astype(np.float32)

    valid = ~np.all(X == 0, axis=1) & np.all(np.isfinite(X), axis=1)
    X_valid = X[valid]

    if len(X_valid) < 100:
        logger.warning("Too few valid pixels for clustering")
        return np.zeros(h * w, dtype=np.uint8).reshape(h, w)

    sample_size = min(50000, len(X_valid))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_valid), sample_size, replace=False)
    X_sample = X_valid[idx]

    kmeans = MiniBatchKMeans(n_clusters=2, random_state=RANDOM_STATE, batch_size=1000)
    kmeans.fit(X_sample)
    all_labels = kmeans.predict(X_valid)

    # Pick flood cluster by actual NDWI values
    ndwi_col = FEATURE_NAMES.index("NDWI")
    ndwi_at = [np.median(X_valid[all_labels == c, ndwi_col]) for c in range(2)]
    flood_cluster = int(np.argmax(ndwi_at))

    result = np.zeros(h * w, dtype=np.uint8)
    result[valid] = (all_labels == flood_cluster).astype(np.uint8)
    result = result.reshape(h, w)

    flood_pct = 100.0 * np.sum(result) / result.size
    logger.info("KMeans labels: flood_cluster=%d, %.2f%% flood", flood_cluster, flood_pct)
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


def spatial_cross_validate(features, labels, n_folds=5, model_type="xgboost"):
    """Spatial Cross-Validation with tile-based blocking (Roberts et al. 2017).

    Divides the raster into n_folds x n_folds spatial tiles to avoid
    spatial autocorrelation leakage in pixel-based random splits.

    Parameters
    ----------
    features : np.ndarray — (n_bands, H, W) feature stack
    labels : np.ndarray — (H, W) binary labels
    n_folds : int — Number of spatial folds per axis (total folds = n_folds^2)
    model_type : str — 'xgboost', 'random_forest', or 'lightgbm'

    Returns
    -------
    dict — Per-fold metrics and summary statistics (mean ± std)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    n_bands, h, w = features.shape
    tile_h = h // n_folds
    tile_w = w // n_folds

    logger.info("=" * 60)
    logger.info("SPATIAL CROSS-VALIDATION (%dx%d tiles, model=%s)", n_folds, n_folds, model_type)
    logger.info("=" * 60)

    # Create tile indices for each pixel
    tile_rows = np.arange(h) // tile_h
    tile_cols = np.arange(w) // tile_w
    tile_rows = np.clip(tile_rows, 0, n_folds - 1)
    tile_cols = np.clip(tile_cols, 0, n_folds - 1)

    # Assign each pixel to a tile
    tile_map = np.zeros((h, w), dtype=np.int32)
    for i in range(n_folds):
        for j in range(n_folds):
            mask = (tile_rows[:, None] == i) & (tile_cols[None, :] == j)
            tile_map[mask] = i * n_folds + j

    total_folds = n_folds * n_folds
    fold_metrics = []

    for fold_idx in range(total_folds):
        # Test pixels = pixels in this tile
        test_mask = tile_map.flatten() == fold_idx
        train_mask = ~test_mask

        # Flatten features
        X_flat = features.reshape(n_bands, -1).T
        y_flat = labels.flatten()

        # Filter valid pixels
        valid_train = train_mask & ~np.all(X_flat == 0, axis=1) & np.all(np.isfinite(X_flat), axis=1)
        valid_test = test_mask & ~np.all(X_flat == 0, axis=1) & np.all(np.isfinite(X_flat), axis=1)

        X_train = X_flat[valid_train]
        y_train = y_flat[valid_train]
        X_test = X_flat[valid_test]
        y_test = y_flat[valid_test]

        if len(X_train) < 100 or len(X_test) < 100:
            logger.warning("Fold %d: too few samples (train=%d, test=%d) — skipping",
                          fold_idx, len(X_train), len(X_test))
            continue

        # Sample training data if too large
        max_train = 100000
        if len(X_train) > max_train:
            idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), max_train, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Train model
        if model_type == "xgboost":
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
            )
        elif model_type == "lightgbm":
            import lightgbm as lgb
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbose=-1,
            )
        else:  # random_forest
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)
        fold_prec = precision_score(y_test, y_pred, zero_division=0)
        fold_rec = recall_score(y_test, y_pred, zero_division=0)
        fold_acc = accuracy_score(y_test, y_pred)

        fold_metrics.append({
            "fold": fold_idx,
            "tile_row": fold_idx // n_folds,
            "tile_col": fold_idx % n_folds,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "flood_ratio_train": round(float(np.mean(y_train)), 6),
            "flood_ratio_test": round(float(np.mean(y_test)), 6),
            "accuracy": round(fold_acc, 6),
            "precision": round(fold_prec, 6),
            "recall": round(fold_rec, 6),
            "f1": round(fold_f1, 6),
        })

        logger.info("  Fold %2d: F1=%.4f P=%.4f R=%.4f (train=%d, test=%d, flood=%.2f%%)",
                    fold_idx, fold_f1, fold_prec, fold_rec,
                    len(X_train), len(X_test), 100 * np.mean(y_test))

    # Summary statistics
    if not fold_metrics:
        logger.error("No valid folds — spatial CV failed")
        return {"error": "no_valid_folds"}

    f1s = [m["f1"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]

    summary = {
        "model": model_type,
        "n_folds": len(fold_metrics),
        "grid": f"{n_folds}x{n_folds}",
        "f1_mean": round(float(np.mean(f1s)), 6),
        "f1_std": round(float(np.std(f1s)), 6),
        "precision_mean": round(float(np.mean(precs)), 6),
        "precision_std": round(float(np.std(precs)), 6),
        "recall_mean": round(float(np.mean(recs)), 6),
        "recall_std": round(float(np.std(recs)), 6),
        "accuracy_mean": round(float(np.mean(accs)), 6),
        "accuracy_std": round(float(np.std(accs)), 6),
        "per_fold": fold_metrics,
    }

    logger.info("-" * 60)
    logger.info("SCV Summary: F1=%.4f±%.4f  P=%.4f±%.4f  R=%.4f±%.4f  Acc=%.4f±%.4f",
                summary["f1_mean"], summary["f1_std"],
                summary["precision_mean"], summary["precision_std"],
                summary["recall_mean"], summary["recall_std"],
                summary["accuracy_mean"], summary["accuracy_std"])
    logger.info("=" * 60)

    # Save
    scv_path = MODELS_DIR / f"spatial_cv_{model_type}.json"
    scv_path.write_text(json.dumps(summary, indent=2))
    logger.info("SCV results saved: %s", scv_path)

    return summary


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


def train_stacking_ensemble(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> object:
    """Train a stacking meta-learner on top of base model predictions.

    Per Wolpert (1992): Stacking uses a meta-learner to combine base
    model predictions, learning optimal weights instead of simple voting.
    Typically +2-3% F1 over majority voting.

    Parameters
    ----------
    models : dict — {"rf": model, "xgb": model, "lgb": model}
    X_train, y_train — Training data
    X_test, y_test — Test data

    Returns
    -------
    object — Trained meta-learner (LogisticRegression)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    logger.info("=" * 40)
    logger.info("STACKING ENSEMBLE (Meta-Learner)")
    logger.info("=" * 40)

    # Generate base model predictions (probabilities)
    def get_probas(models_dict, X):
        probas = []
        for name, model in models_dict.items():
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)[:, 1]
            else:
                p = model.predict(X).astype(float)
            probas.append(p)
        return np.column_stack(probas)

    X_train_meta = get_probas(models, X_train)
    X_test_meta = get_probas(models, X_test)

    logger.info("Meta-features shape: train=%s, test=%s",
                X_train_meta.shape, X_test_meta.shape)

    # Train meta-learner
    meta_model = LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
    )
    meta_model.fit(X_train_meta, y_train)

    # Evaluate
    y_pred_base = predict_ensemble(models, X_test)
    y_pred_meta = meta_model.predict(X_test_meta).astype(np.uint8)

    f1_base = f1_score(y_test, y_pred_base, zero_division=0)
    f1_meta = f1_score(y_test, y_pred_meta, zero_division=0)

    logger.info("Base ensemble F1: %.4f", f1_base)
    logger.info("Stacking F1:      %.4f", f1_meta)
    logger.info("Improvement:      %+.4f", f1_meta - f1_base)
    logger.info("Meta-learner weights: %s", dict(zip(models.keys(), meta_model.coef_[0].tolist())))
    logger.info("=" * 40)

    # Save
    meta_path = MODELS_DIR / "stacking_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta_model, f)
    logger.info("Saved: %s", meta_path)

    return meta_model


def predict_stacking(models: dict, meta_model, X: np.ndarray) -> np.ndarray:
    """Predict using stacking ensemble.

    Parameters
    ----------
    models : dict — Base models
    meta_model — Trained meta-learner
    X : np.ndarray — Feature matrix

    Returns
    -------
    np.ndarray — uint8 predictions
    """
    probas = []
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)[:, 1]
        else:
            p = model.predict(X).astype(float)
        probas.append(p)

    X_meta = np.column_stack(probas)
    prediction = meta_model.predict(X_meta).astype(np.uint8)

    logger.info("Stacking prediction: %.2f%% flood",
                100.0 * np.sum(prediction) / len(prediction))
    return prediction


def explain_model_shap(model, X_sample: np.ndarray, model_name: str = "xgboost"):
    """Generate SHAP explanations for a trained model.

    Per Aydin et al. (2023): SHAP values provide interpretable feature
    importance that explains WHY the model classifies a pixel as flood.

    Parameters
    ----------
    model : trained sklearn/xgboost model
    X_sample : np.ndarray — Sample feature matrix (N, n_features)
    model_name : str — Model name for logging
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — run: pip install shap")
        return None

    logger.info("=" * 40)
    logger.info("SHAP EXPLAINABILITY (%s)", model_name.upper())
    logger.info("=" * 40)

    # Use a subsample for speed
    n_sample = min(1000, len(X_sample))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_sample), n_sample, replace=False)
    X_explain = X_sample[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Handle different output shapes
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (flood)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = dict(zip(FEATURE_NAMES, mean_abs_shap.tolist()))

    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("Feature importance (SHAP):")
    for name, importance in sorted_features:
        logger.info("  %s: %.4f", name, importance)

    # Save
    shap_path = MODELS_DIR / f"shap_{model_name}.json"
    import json
    shap_path.write_text(json.dumps({
        "model": model_name,
        "n_samples": n_sample,
        "feature_importance": dict(sorted_features),
        "mean_abs_shap": feature_importance,
    }, indent=2))
    logger.info("SHAP results saved: %s", shap_path)

    return feature_importance


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

    # 7. SHAP Explainability (Aydin et al. 2023)
    try:
        explain_model_shap(xgb_model, X_test, model_name="xgboost")
    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)
    gc.collect()

    # 8. Spatial Cross-Validation (Roberts et al. 2017)
    # Reload features for SCV (we deleted them above)
    features_scv, _ = load_feature_stack()
    labels_scv = load_labels(features=features_scv)
    for mtype in ["xgboost", "random_forest", "lightgbm"]:
        try:
            spatial_cross_validate(features_scv, labels_scv, n_folds=5, model_type=mtype)
        except Exception as exc:
            logger.warning("SCV failed for %s: %s", mtype, exc)
    del features_scv, labels_scv
    gc.collect()

    return {"rf": rf_model, "xgb": xgb_model, "lgb": lgb_model}


if __name__ == "__main__":
    try:
        run_training()
    except Exception as exc:
        logger.error("MODEL TRAINING FAILED: %s", exc)
        sys.exit(1)
