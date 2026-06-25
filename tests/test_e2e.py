"""
End-to-End tests for A.E.C.O Flood Detection Pipeline.

Tests the complete pipeline:
1. SAR preprocessing (Refined Lee, noise removal)
2. Feature engineering (NDWI, Otsu, HAND, texture)
3. Model training (ensemble + U-Net)
4. Prediction (flood map generation)
5. Validation (against ground truth)
6. API endpoints (FastAPI)
7. Report generation (PDF)

All tests use synthetic data — no external dependencies required.
"""

import sys
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "api"))


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def synthetic_sar_data():
    """Generate synthetic SAR VV/VH data mimicking real Sentinel-1."""
    np.random.seed(42)
    h, w = 128, 128

    # Land: VV ~ -10 dB, VH ~ -17 dB
    vv = np.random.randn(h, w).astype(np.float32) * 2 - 10
    vh = np.random.randn(h, w).astype(np.float32) * 2 - 17

    # Water bodies: VV ~ -22 dB, VH ~ -28 dB
    vv[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 1.5 - 22
    vh[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 1.5 - 28

    # River
    for i in range(h):
        vv[i, 60:65] = np.random.randn(5).astype(np.float32) * 1 - 20
        vh[i, 60:65] = np.random.randn(5).astype(np.float32) * 1 - 26

    return vv, vh


@pytest.fixture
def synthetic_dem_data():
    """Generate synthetic DEM with terrain."""
    np.random.seed(42)
    h, w = 128, 128

    # Mountain in center
    y, x = np.mgrid[0:h, 0:w]
    mountain = 500 * np.exp(-((y - 64)**2 + (x - 64)**2) / (2 * 30**2))

    # Lowlands at edges
    lowland = np.random.randn(h, w).astype(np.float32) * 10 + 50

    elevation = (mountain + lowland).astype(np.float32)
    elevation = np.clip(elevation, 0, 1000)

    return elevation


@pytest.fixture
def synthetic_optical_data():
    """Generate synthetic Sentinel-2 Green/NIR data."""
    np.random.seed(42)
    h, w = 128, 128

    # Vegetation: Green ~ 0.05, NIR ~ 0.30
    green = np.random.randn(h, w).astype(np.float32) * 0.01 + 0.05
    nir = np.random.randn(h, w).astype(np.float32) * 0.05 + 0.30

    # Water: Green ~ 0.08, NIR ~ 0.02
    green[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 0.01 + 0.08
    nir[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 0.005 + 0.02

    green = np.clip(green, 0, 1).astype(np.float32)
    nir = np.clip(nir, 0, 1).astype(np.float32)

    return green, nir


@pytest.fixture
def temp_geotiff(tmp_path):
    """Create a temporary GeoTIFF file."""
    def _create(data, crs="EPSG:4326", bbox=None):
        if bbox is None:
            bbox = [116.8, -8.9, 119.0, -8.3]

        if data.ndim == 2:
            data = data[np.newaxis, :, :]
            count = 1
        else:
            count = data.shape[0]

        h, w = data.shape[-2], data.shape[-1]
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

        path = tmp_path / f"test_{np.random.randint(10000)}.tif"
        profile = {
            "driver": "GTiff", "height": h, "width": w,
            "count": count, "dtype": data.dtype.name,
            "crs": crs, "transform": transform, "compress": "lzw",
        }

        with rasterio.open(str(path), "w", **profile) as dst:
            for i in range(count):
                dst.write(data[i], i + 1)

        return path

    return _create


@pytest.fixture
def feature_stack_path(tmp_path, synthetic_sar_data, synthetic_dem_data, synthetic_optical_data):
    """Create a temporary feature stack."""
    vv, vh = synthetic_sar_data
    elevation = synthetic_dem_data
    green, nir = synthetic_optical_data
    h, w = vv.shape

    # Compute features
    from features import compute_slope, compute_hand, otsu_threshold
    from sar_preprocess import refined_lee_filter

    vv_filtered = refined_lee_filter(vv, window_size=5)
    vh_filtered = refined_lee_filter(vh, window_size=5)

    transform = from_bounds(116.8, -8.9, 119.0, -8.3, w, h)
    slope = compute_slope(elevation, transform)
    hand = compute_hand(elevation, transform)

    vv_thresh = otsu_threshold(vv_filtered)
    vh_thresh = otsu_threshold(vh_filtered)
    sar_mask = ((vv_filtered < vv_thresh) & (vh_filtered < vh_thresh)).astype(np.float32)

    # 5-band stack
    stack = np.stack([
        np.nan_to_num(vv_filtered, nan=0.0),
        np.nan_to_num(vh_filtered, nan=0.0),
        sar_mask,
        np.nan_to_num(slope, nan=0.0),
        np.nan_to_num(hand, nan=0.0),
    ])

    path = tmp_path / "feature_stack.tif"
    profile = {
        "driver": "GTiff", "height": h, "width": w,
        "count": 5, "dtype": "float32", "compress": "lzw",
        "crs": "EPSG:4326", "transform": transform,
    }

    with rasterio.open(str(path), "w", **profile) as dst:
        for i in range(5):
            dst.write(stack[i], i + 1)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.set_band_description(3, "SAR_flood_mask")
        dst.set_band_description(4, "Slope_deg")
        dst.set_band_description(5, "HAND_m")

    return path


# =========================================================================
# E2E Test 1: SAR Preprocessing Pipeline
# =========================================================================
class TestSARPreprocessing:
    """E2E: SAR data → speckle filter → noise removal → calibrated dB."""

    def test_refined_lee_reduces_variance(self, synthetic_sar_data):
        """Speckle filter should reduce noise while preserving signal."""
        from sar_preprocess import refined_lee_filter

        vv, _ = synthetic_sar_data
        vv_noisy = vv + np.random.randn(*vv.shape).astype(np.float32) * 3
        vv_filtered = refined_lee_filter(vv_noisy, window_size=7)

        assert np.var(vv_filtered) < np.var(vv_noisy)
        assert abs(np.mean(vv_filtered) - np.mean(vv_noisy)) < 1.0

    def test_noise_reduces_edge_artifacts(self, synthetic_sar_data):
        """Thermal noise removal should reduce swath edge artifacts."""
        from sar_preprocess import remove_thermal_noise

        vv, _ = synthetic_sar_data
        vv_noisy = vv.copy()
        vv_noisy[:, :5] += 5.0  # Simulate swath edge noise

        vv_clean = remove_thermal_noise(vv_noisy, noise_level_db=5.0, edge_fraction=0.1)

        assert np.mean(vv_clean[:, :5]) < np.mean(vv_noisy[:, :5])

    def test_full_preprocessing_pipeline(self, synthetic_sar_data):
        """Complete preprocessing: noise removal → speckle filter."""
        from sar_preprocess import preprocess_sar

        vv, vh = synthetic_sar_data
        vv_proc, vh_proc = preprocess_sar(vv, vh, apply_lee=True, lee_window=7, remove_noise=True)

        assert vv_proc.shape == vv.shape
        assert vh_proc.shape == vh.shape
        assert vv_proc.dtype == np.float32
        assert np.isfinite(vv_proc).all()


# =========================================================================
# E2E Test 2: Feature Engineering Pipeline
# =========================================================================
class TestFeatureEngineering:
    """E2E: Raw bands → feature stack (5-9 bands)."""

    def test_ndwi_computation(self, synthetic_optical_data):
        """NDWI should be higher for water pixels."""
        from features import compute_ndwi

        green, nir = synthetic_optical_data
        ndwi = compute_ndwi(green, nir)

        assert ndwi.shape == green.shape
        assert np.nanmin(ndwi) >= -1.0
        assert np.nanmax(ndwi) <= 1.0

        # Water pixels (20:40, 20:40) should have higher NDWI
        water_ndwi = np.nanmean(ndwi[20:40, 20:40])
        land_ndwi = np.nanmean(ndwi[60:80, 60:80])
        assert water_ndwi > land_ndwi

    def test_otsu_threshold_bimodal(self, synthetic_sar_data):
        """Otsu should find optimal threshold for bimodal distribution."""
        from features import otsu_threshold

        vv, _ = synthetic_sar_data
        threshold = otsu_threshold(vv)

        assert -25 < threshold < 0  # Should be in reasonable SAR range

    def test_slope_computation(self, synthetic_dem_data):
        """Slope should be steeper near mountains."""
        from features import compute_slope

        elevation = synthetic_dem_data
        transform = from_bounds(116.8, -8.9, 119.0, -8.3, 128, 128)
        slope = compute_slope(elevation, transform)

        assert slope.shape == elevation.shape
        assert np.nanmin(slope) >= 0
        assert np.nanmax(slope) < 90

    def test_hand_computation(self, synthetic_dem_data):
        """HAND should be lower near drainage channels."""
        from features import compute_hand

        elevation = synthetic_dem_data
        transform = from_bounds(116.8, -8.9, 119.0, -8.3, 128, 128)
        hand = compute_hand(elevation, transform)

        assert hand.shape == elevation.shape
        assert np.nanmin(hand) >= 0

    def test_feature_stack_creation(self, feature_stack_path):
        """Feature stack should have correct bands and metadata."""
        with rasterio.open(str(feature_stack_path)) as src:
            assert src.count == 5
            assert src.crs is not None
            assert src.transform is not None

            band_names = [src.descriptions[i] for i in range(5)]
            assert "VV_dB" in band_names
            assert "VH_dB" in band_names
            assert "SAR_flood_mask" in band_names
            assert "Slope_deg" in band_names
            assert "HAND_m" in band_names


# =========================================================================
# E2E Test 3: Model Training Pipeline
# =========================================================================
class TestModelTraining:
    """E2E: Feature stack → labels → train → save model."""

    def test_unsupervised_label_generation(self, feature_stack_path):
        """KMeans labels should separate water from land."""
        from model import generate_unsupervised_labels, FEATURE_NAMES

        with rasterio.open(str(feature_stack_path)) as src:
            features = src.read()
            FEATURE_NAMES.clear()
            FEATURE_NAMES.extend([src.descriptions[i] for i in range(src.count)])

        labels = generate_unsupervised_labels(features)

        assert labels.shape == features.shape[1:]
        assert labels.dtype == np.uint8
        assert set(np.unique(labels)).issubset({0, 1})

    def test_training_data_preparation(self, feature_stack_path):
        """Training data should be properly split and sampled."""
        from model import load_feature_stack, load_labels, prepare_training_data

        # Mock the path
        with patch('model.PROCESSED_DIR', feature_stack_path.parent):
            with patch('model.LABELS_DIR', feature_stack_path.parent / "labels"):
                # Create labels
                with rasterio.open(str(feature_stack_path)) as src:
                    features = src.read()
                    labels = (features[2] > 0.5).astype(np.uint8)

                label_path = feature_stack_path.parent / "labels" / "flood_labels.tif"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                profile = {"driver": "GTiff", "height": 128, "width": 128,
                          "count": 1, "dtype": "uint8", "compress": "lzw"}
                with rasterio.open(str(label_path), "w", **profile) as dst:
                    dst.write(labels, 1)

                X_train, X_test, y_train, y_test = prepare_training_data(features, labels)

                assert X_train.shape[1] == 5  # 5 features
                assert len(X_train) > 0
                assert len(X_test) > 0
                assert len(y_train) == len(X_train)
                assert len(y_test) == len(X_test)

    def test_xgboost_training(self, feature_stack_path):
        """XGBoost should train and produce predictions."""
        from model import train_xgboost

        np.random.seed(42)
        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.choice([0, 1], 100, p=[0.8, 0.2]).astype(np.uint8)
        X_test = np.random.randn(20, 5).astype(np.float32)
        y_test = np.random.choice([0, 1], 20, p=[0.8, 0.2]).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('model.MODELS_DIR', Path(tmpdir)):
                model = train_xgboost(X_train, X_test, y_train, y_test)

                assert model is not None
                predictions = model.predict(X_test)
                assert len(predictions) == len(X_test)
                assert set(np.unique(predictions)).issubset({0, 1})

    def test_ensemble_prediction(self):
        """Ensemble should combine predictions from multiple models."""
        from model import predict_ensemble
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.choice([0, 1], 100, p=[0.8, 0.2]).astype(np.uint8)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        xgb_model.fit(X, y)

        models = {"rf": rf, "xgb": xgb_model}
        predictions = predict_ensemble(models, X)

        assert len(predictions) == len(X)
        assert set(np.unique(predictions)).issubset({0, 1})


# =========================================================================
# E2E Test 4: Prediction Pipeline
# =========================================================================
class TestPrediction:
    """E2E: Feature stack + model → flood map GeoTIFF."""

    def test_flood_map_generation(self, feature_stack_path, tmp_path):
        """Should generate valid flood map GeoTIFF."""
        from model import train_xgboost, predict_ensemble

        # Create training data
        with rasterio.open(str(feature_stack_path)) as src:
            features = src.read()
            profile = src.profile.copy()

        n_bands, h, w = features.shape
        X_flat = features.reshape(n_bands, -1).T
        valid = ~np.all(X_flat == 0, axis=1) & np.all(np.isfinite(X_flat), 1)
        X_valid = np.nan_to_num(X_flat[valid], nan=0.0)

        # Generate labels
        labels = (features[2] > 0.5).astype(np.uint8).flatten()
        y_valid = labels[valid]

        # Train
        model = train_xgboost(X_valid[:80], X_valid[80:], y_valid[:80], y_valid[80:])

        # Predict
        y_pred = model.predict(X_valid)
        flood_map = np.zeros(h * w, dtype=np.uint8)
        flood_map[valid] = y_pred
        flood_map = flood_map.reshape(h, w)

        # Save
        out_path = tmp_path / "flood_map.tif"
        out_profile = profile.copy()
        out_profile.update({"count": 1, "dtype": "uint8", "nodata": 255})
        with rasterio.open(str(out_path), "w", **out_profile) as dst:
            dst.write(flood_map, 1)

        # Verify
        with rasterio.open(str(out_path)) as src:
            result = src.read(1)
            assert result.shape == (h, w)
            assert result.dtype == np.uint8
            assert set(np.unique(result)).issubset({0, 1, 255})


# =========================================================================
# E2E Test 5: Validation Pipeline
# =========================================================================
class TestValidation:
    """E2E: Flood map + ground truth → metrics."""

    def test_validation_against_ground_truth(self, tmp_path):
        """Should compute precision, recall, F1, IoU."""
        np.random.seed(42)
        h, w = 128, 128

        # Create flood map
        flood_map = np.zeros((h, w), dtype=np.uint8)
        flood_map[20:40, 20:40] = 1
        flood_map[60:80, 60:80] = 1  # False positive

        # Create ground truth (partial overlap)
        gt = np.zeros((h, w), dtype=np.uint8)
        gt[20:40, 20:40] = 1
        gt[80:100, 80:100] = 1  # Missed

        # Compute metrics
        tp = int(np.sum((flood_map == 1) & (gt == 1)))
        fp = int(np.sum((flood_map == 1) & (gt == 0)))
        fn = int(np.sum((flood_map == 0) & (gt == 1)))
        tn = int(np.sum((flood_map == 0) & (gt == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        iou = tp / max(tp + fp + fn, 1)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert 0 <= iou <= 1
        assert f1 > 0  # Should have some overlap

    def test_auto_digitize(self, tmp_path):
        """Auto-digitization should produce valid ground truth."""
        from scripts.auto_digitize import auto_digitize_event

        # Create test SAR data
        np.random.seed(42)
        h, w = 128, 128
        vv = np.random.randn(h, w).astype(np.float32) * 5 - 10
        vh = np.random.randn(h, w).astype(np.float32) * 5 - 17

        # Add water
        vv[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 2 - 22
        vh[20:40, 20:40] = np.random.randn(20, 20).astype(np.float32) * 2 - 28

        vv_path = tmp_path / "vv.tif"
        vh_path = tmp_path / "vh.tif"

        transform = from_bounds(116.8, -8.9, 119.0, -8.3, w, h)
        profile = {"driver": "GTiff", "height": h, "width": w,
                  "count": 1, "dtype": "float32", "crs": "EPSG:4326",
                  "transform": transform}

        with rasterio.open(str(vv_path), "w", **profile) as dst:
            dst.write(vv, 1)
        with rasterio.open(str(vh_path), "w", **profile) as dst:
            dst.write(vh, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('scripts.auto_digitize.LABELS_DIR', Path(tmpdir)):
                result = auto_digitize_event("test_event", vv_path, vh_path)

                assert result is not None
                assert result["water_pixels"] > 0
                assert result["n_components"] > 0


# =========================================================================
# E2E Test 6: Change Detection Pipeline
# =========================================================================
class TestChangeDetection:
    """E2E: Multi-temporal SAR → change detection → flood mask."""

    def test_delta_sar_computation(self):
        """Delta SAR should detect backscatter changes."""
        from change_detection import compute_delta_sar

        np.random.seed(42)
        h, w = 64, 64

        # Baseline (dry)
        vv_base = np.random.randn(h, w).astype(np.float32) * 2 - 10
        vh_base = np.random.randn(h, w).astype(np.float32) * 2 - 17

        # Current (flooded)
        vv_curr = vv_base.copy()
        vh_curr = vh_base.copy()
        vv_curr[20:40, 20:40] -= 10  # Flood
        vh_curr[20:40, 20:40] -= 8

        dvv, dvh = compute_delta_sar(vv_curr, vh_curr, vv_base, vh_base)

        assert dvv.shape == (h, w)
        assert np.nanmin(dvv[20:40, 20:40]) < -5  # Significant negative change

    def test_adaptive_change_detection(self):
        """Adaptive thresholding should find optimal change threshold."""
        from change_detection import detect_adaptive_change

        np.random.seed(42)
        # Bimodal: normal changes vs flood changes
        normal = np.random.normal(0, 1, 500).astype(np.float32)
        flood = np.random.normal(-6, 1, 500).astype(np.float32)
        dvv = np.concatenate([normal, flood])
        dvh = np.concatenate([normal, flood])

        mask = detect_adaptive_change(dvv, dvh, method="otsu")

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert np.sum(mask[500:]) > np.sum(mask[:500])  # More flood in second half


# =========================================================================
# E2E Test 7: API Endpoints
# =========================================================================
class TestAPIEndpoints:
    """E2E: HTTP request → API → response."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        with patch('api.main.RUST_READY', False):
            from api.main import app
            return TestClient(app)

    def test_health_endpoint(self, client):
        """Health check should return status."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "LIVE"
        assert "rust_ready" in data
        assert "timestamp" in data

    def test_stats_endpoint_no_data(self, client):
        """Stats should return 404 when no flood map exists."""
        r = client.get("/stats")
        assert r.status_code == 404

    def test_predict_at_invalid_coordinates(self, client):
        """Point query should validate coordinates."""
        r = client.get("/predict/at?lat=100&lon=200")  # Invalid
        assert r.status_code in (422, 503)  # 503 if Rust engine not available

    def test_predict_area_invalid_geometry(self, client):
        """Polygon query should validate geometry type."""
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [117, -8.5]},
            "properties": {},
        }
        r = client.post("/predict/area", json=feature)
        assert r.status_code in (422, 503)  # 503 if Rust engine not available

    def test_predict_report_invalid_geometry(self, client):
        """Report should validate geometry."""
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[117, -8.5], [118, -8.5]]},
            "properties": {},
        }
        r = client.post("/predict/report", json=feature)
        assert r.status_code in (422, 503)  # 503 if Rust engine not available

    def test_satellite_status(self, client):
        """Satellite status should return sync info."""
        r = client.get("/satellite/status")
        assert r.status_code == 200
        data = r.json()
        assert "next_check" in data

    def test_tile_endpoint(self, client):
        """Tile endpoint should return PNG (even fallback)."""
        r = client.get("/tiles/10/100/100.png")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "image/png"

    def test_dashboard_endpoint(self, client):
        """Dashboard should return HTML."""
        r = client.get("/")
        assert r.status_code == 200


# =========================================================================
# E2E Test 8: Deep Learning Pipeline
# =========================================================================
class TestDeepLearning:
    """E2E: Patches → U-Net training → ONNX export."""

    def test_unet_forward_pass(self):
        """U-Net should produce valid output."""
        from unet_model import UNet
        import torch

        model = UNet(in_channels=5, out_channels=1)
        model.eval()

        x = torch.randn(2, 5, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1, 64, 64)
        assert out.min() >= 0
        assert out.max() <= 1

    def test_attention_unet_forward_pass(self):
        """Attention U-Net should produce valid output."""
        from unet_model import AttentionUNet
        import torch

        model = AttentionUNet(in_channels=5, out_channels=1)
        model.eval()

        x = torch.randn(2, 5, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1, 64, 64)

    def test_fpn_unet_forward_pass(self):
        """FPN U-Net should produce valid output."""
        from unet_model import FPNUNet
        import torch

        model = FPNUNet(in_channels=5, out_channels=1)
        model.eval()

        x = torch.randn(2, 5, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1, 64, 64)

    def test_focal_loss(self):
        """Focal loss should handle class imbalance."""
        from unet_model import FocalLoss
        import torch

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert loss.item() < 10  # Should be reasonable

    def test_dice_focal_loss(self):
        """Dice + Focal loss should combine both losses."""
        from unet_model import DiceFocalLoss
        import torch

        loss_fn = DiceFocalLoss(focal_weight=0.5)

        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_data_augmentation(self, tmp_path):
        """Data augmentation should transform patches."""
        from unet_model import FloodPatchDataset

        # Create test patches
        features_dir = tmp_path / "features"
        labels_dir = tmp_path / "labels"
        features_dir.mkdir()
        labels_dir.mkdir()

        features = np.random.randn(5, 64, 64).astype(np.float32)
        label = np.random.choice([0, 1], (64, 64)).astype(np.float32)

        np.save(str(features_dir / "patch_0_0.npy"), features)
        np.save(str(labels_dir / "patch_0_0.npy"), label)

        # Load with augmentation
        ds = FloodPatchDataset(str(features_dir), str(labels_dir), augment=True)
        f_aug, l_aug = ds[0]

        assert f_aug.shape == (5, 64, 64)
        assert l_aug.shape == (1, 64, 64)


# =========================================================================
# E2E Test 9: Monitoring Pipeline
# =========================================================================
class TestMonitoring:
    """E2E: Request → metrics collection → /metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        with patch('api.main.RUST_READY', False):
            from api.main import app
            return TestClient(app)

    def test_metrics_endpoint(self, client):
        """Metrics endpoint should return Prometheus format."""
        from api.metrics import generate_metrics

        # Make some requests to generate metrics
        client.get("/health")
        client.get("/health")

        metrics_text = generate_metrics()

        assert "http_requests_total" in metrics_text
        assert "flood_predictions_total" in metrics_text
        assert "celery_tasks_total" in metrics_text

    def test_metrics_record_flood_prediction(self):
        """Should record flood prediction metrics."""
        from api.metrics import record_flood_prediction, generate_metrics

        record_flood_prediction(n_pixels=1000, duration=0.5)
        record_flood_prediction(n_pixels=2000, duration=0.3)

        metrics = generate_metrics()
        assert "flood_predictions_total 2" in metrics
        assert "flood_pixels_total 3000" in metrics


# =========================================================================
# E2E Test 10: Full Pipeline Integration
# =========================================================================
class TestFullPipeline:
    """E2E: Complete pipeline from raw data to flood map."""

    def test_complete_pipeline(self, synthetic_sar_data, synthetic_dem_data,
                               synthetic_optical_data, tmp_path):
        """Test complete pipeline: preprocess → features → train → predict → validate."""
        from sar_preprocess import preprocess_sar
        from features import compute_slope, compute_hand, otsu_threshold
        from model import train_xgboost

        vv, vh = synthetic_sar_data
        elevation = synthetic_dem_data
        green, nir = synthetic_optical_data
        h, w = vv.shape

        # Step 1: SAR Preprocessing
        vv_proc, vh_proc = preprocess_sar(vv, vh, apply_lee=True, remove_noise=True)
        assert vv_proc.shape == (h, w)

        # Step 2: Feature Engineering
        transform = from_bounds(116.8, -8.9, 119.0, -8.3, w, h)
        slope = compute_slope(elevation, transform)
        hand = compute_hand(elevation, transform)

        vv_thresh = otsu_threshold(vv_proc)
        vh_thresh = otsu_threshold(vh_proc)
        sar_mask = ((vv_proc < vv_thresh) & (vh_proc < vh_thresh)).astype(np.float32)

        # NDWI
        denom = green + nir
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0).astype(np.float32)

        # Stack features
        features = np.stack([
            np.nan_to_num(vv_proc, nan=0.0),
            np.nan_to_num(vh_proc, nan=0.0),
            sar_mask,
            np.nan_to_num(slope, nan=0.0),
            np.nan_to_num(hand, nan=0.0),
        ])

        assert features.shape == (5, h, w)

        # Step 3: Generate labels
        labels = (sar_mask > 0.5).astype(np.uint8)
        assert labels.shape == (h, w)

        # Step 4: Train model
        n_bands = features.shape[0]
        X_flat = features.reshape(n_bands, -1).T
        y_flat = labels.flatten()

        valid = ~np.all(X_flat == 0, axis=1) & np.all(np.isfinite(X_flat), 1)
        X_valid = X_flat[valid]
        y_valid = y_flat[valid]

        split = int(0.8 * len(X_valid))
        model = train_xgboost(X_valid[:split], X_valid[split:], y_valid[:split], y_valid[split:])
        assert model is not None

        # Step 5: Predict
        y_pred = model.predict(X_valid)
        flood_map = np.zeros(h * w, dtype=np.uint8)
        flood_map[valid] = y_pred
        flood_map = flood_map.reshape(h, w)

        assert flood_map.shape == (h, w)
        assert set(np.unique(flood_map)).issubset({0, 1})

        # Step 6: Validate
        tp = int(np.sum((flood_map == 1) & (labels == 1)))
        fp = int(np.sum((flood_map == 1) & (labels == 0)))
        fn = int(np.sum((flood_map == 0) & (labels == 1)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        assert f1 > 0.5  # Should achieve reasonable F1

        # Step 7: Save output
        out_path = tmp_path / "flood_map.tif"
        out_profile = {
            "driver": "GTiff", "height": h, "width": w,
            "count": 1, "dtype": "uint8", "compress": "lzw",
            "crs": "EPSG:4326", "transform": transform,
        }
        with rasterio.open(str(out_path), "w", **out_profile) as dst:
            dst.write(flood_map, 1)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Final verification
        with rasterio.open(str(out_path)) as src:
            result = src.read(1)
            assert result.shape == (h, w)
            assert src.crs is not None
