"""
tests/test_pipeline.py - Integration and unit tests for NTB Flood Detection.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def predictions_dir(project_root):
    return project_root / "outputs" / "predictions"


@pytest.fixture
def models_dir(project_root):
    return project_root / "outputs" / "models"


@pytest.fixture
def processed_dir(project_root):
    return project_root / "data" / "processed"


# =========================================================================
# Test: File Existence
# =========================================================================
class TestFileExistence:
    def test_feature_stack_exists(self, processed_dir):
        assert (processed_dir / "feature_stack.tif").exists()

    def test_flood_map_exists(self, predictions_dir):
        assert (predictions_dir / "flood_map.tif").exists()

    def test_final_flood_map_exists(self, predictions_dir):
        assert (predictions_dir / "final_flood_map.tif").exists()

    def test_xgboost_model_exists(self, models_dir):
        assert (models_dir / "xgboost.pkl").exists()

    def test_random_forest_model_exists(self, models_dir):
        assert (models_dir / "random_forest.pkl").exists()

    def test_preview_png_exists(self, predictions_dir):
        assert (predictions_dir / "final_flood_map_preview.png").exists()


# =========================================================================
# Test: GeoTIFF Validity
# =========================================================================
class TestGeoTIFFValidity:
    def test_flood_map_is_valid_geotiff(self, predictions_dir):
        path = predictions_dir / "final_flood_map.tif"
        with rasterio.open(path) as ds:
            assert ds.crs is not None
            assert ds.crs.to_epsg() == 4326
            assert ds.count == 1
            assert ds.dtypes[0] == "uint8"
            data = ds.read(1)
            unique = set(np.unique(data))
            assert unique <= {0, 1, 255}, f"Unexpected values: {unique}"

    def test_feature_stack_bands(self, processed_dir):
        path = processed_dir / "feature_stack.tif"
        with rasterio.open(path) as ds:
            assert ds.count == 5, f"Expected 5 bands, got {ds.count}"
            assert ds.dtypes[0] == "float32"

    def test_flood_map_shape_matches_features(self, processed_dir, predictions_dir):
        with rasterio.open(processed_dir / "feature_stack.tif") as fs:
            fs_shape = fs.shape
        with rasterio.open(predictions_dir / "final_flood_map.tif") as fm:
            fm_shape = fm.shape
        assert fs_shape == fm_shape, f"Shape mismatch: features={fs_shape} flood={fm_shape}"


# =========================================================================
# Test: Model Validity
# =========================================================================
class TestModelValidity:
    def test_xgboost_can_predict(self, models_dir):
        with open(models_dir / "xgboost.pkl", "rb") as f:
            model = pickle.load(f)
        X = np.random.rand(10, 5).astype(np.float32)
        y = model.predict(X)
        assert y.shape == (10,)
        assert set(y).issubset({0, 1})

    def test_random_forest_can_predict(self, models_dir):
        with open(models_dir / "random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        X = np.random.rand(10, 5).astype(np.float32)
        y = model.predict(X)
        assert y.shape == (10,)
        assert set(y).issubset({0, 1})

    def test_xgboost_metrics_json_valid(self, models_dir):
        path = models_dir / "xgboost_metrics.json"
        data = json.loads(path.read_text())
        assert "accuracy" in data
        assert 0 <= data["accuracy"] <= 1


# =========================================================================
# Test: Evaluate Module
# =========================================================================
class TestEvaluate:
    def test_compute_metrics(self):
        from evaluate import compute_metrics
        m = compute_metrics(tp=80, fp=10, tn=90, fn=20)
        assert 0 < m["iou"] < 1
        assert 0 < m["f1"] < 1
        assert 0 < m["precision"] <= 1
        assert 0 < m["recall"] <= 1

    def test_perfect_score(self):
        from evaluate import compute_metrics
        m = compute_metrics(tp=100, fp=0, tn=100, fn=0)
        assert m["iou"] == 1.0
        assert m["f1"] == 1.0

    def test_confusion_matrix(self):
        from evaluate import compute_confusion
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        tp, fp, tn, fn = compute_confusion(y_true, y_pred)
        assert tp == 2
        assert fp == 1
        assert tn == 1
        assert fn == 1


# =========================================================================
# Test: Features Module
# =========================================================================
class TestFeatures:
    def test_ndwi_range(self):
        from features import compute_ndwi
        green = np.array([100, 200, 50], dtype=np.float32)
        nir = np.array([200, 100, 50], dtype=np.float32)
        ndwi = compute_ndwi(green, nir)
        assert np.all((ndwi >= -1) & (ndwi <= 1))

    def test_ndwi_zero_denom(self):
        from features import compute_ndwi
        green = np.array([0], dtype=np.float32)
        nir = np.array([0], dtype=np.float32)
        ndwi = compute_ndwi(green, nir)
        assert np.isnan(ndwi[0])

    def test_sar_threshold(self):
        from features import compute_sar_threshold
        vv = np.array([-20, -10, -16], dtype=np.float32)
        vh = np.array([-25, -15, -21], dtype=np.float32)
        mask = compute_sar_threshold(vv, vh)
        assert mask[0] == 1  # both below thresh
        assert mask[1] == 0  # VV above
        assert mask[2] == 1  # both below


# =========================================================================
# Test: API (requires httpx + running server, use TestClient)
# =========================================================================
class TestAPI:
    @pytest.fixture
    def client(self):
        sys.path.insert(0, str(PROJECT_ROOT / "api"))
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "operational"
        assert "flood_map_available" in data

    def test_predict_at_valid(self, client):
        # Plampang centre point
        r = client.get("/predict/at", params={"lat": -8.78, "lon": 117.78})
        if r.status_code == 200:
            data = r.json()
            assert "flood" in data
            assert isinstance(data["flood"], bool)

    def test_predict_at_out_of_bounds(self, client):
        r = client.get("/predict/at", params={"lat": 0, "lon": 0})
        assert r.status_code in (400, 404)

    def test_predict_post(self, client):
        r = client.post("/predict", json={"lat": -8.78, "lon": 117.78})
        if r.status_code == 200:
            data = r.json()
            assert "flood" in data

    def test_stats(self, client):
        r = client.get("/stats")
        if r.status_code == 200:
            data = r.json()
            assert "flood_pixels" in data
            assert "flood_percentage" in data

    def test_metrics(self, client):
        r = client.get("/metrics")
        # May 404 if evaluate.py hasn't run
        assert r.status_code in (200, 404)

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Sumbawa" in r.text or "Flood" in r.text
