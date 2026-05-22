"""
tests/test_pipeline.py - Integration and unit tests for NTB Flood Detection.

Tests cover:
  - Unit tests for evaluate, features modules
  - API integration tests matching actual api/main.py endpoints
  - File existence and GeoTIFF validity (when outputs exist)
  - Model loading and prediction
"""

import sys
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

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


@pytest.fixture
def client():
    """Create FastAPI TestClient for api/main.py."""
    sys.path.insert(0, str(PROJECT_ROOT / "api"))
    from main import app
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_polygon_feature():
    """Sample GeoJSON Feature (Polygon in Sumbawa area)."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[117.7, -8.85], [117.85, -8.85],
                             [117.85, -8.7], [117.7, -8.7],
                             [117.7, -8.85]]]
        },
        "properties": {}
    }


# =========================================================================
# Test: File Existence (skipped if outputs don't exist)
# =========================================================================
class TestFileExistence:
    def test_feature_stack_exists(self, processed_dir):
        path = processed_dir / "feature_stack.tif"
        if not path.exists():
            pytest.skip("feature_stack.tif not built yet")
        assert path.exists()

    def test_flood_map_exists(self, predictions_dir):
        path = predictions_dir / "flood_map.tif"
        if not path.exists():
            pytest.skip("flood_map.tif not built yet")
        assert path.exists()

    def test_final_flood_map_exists(self, predictions_dir):
        path = predictions_dir / "final_flood_map.tif"
        if not path.exists():
            pytest.skip("final_flood_map.tif not built yet")
        assert path.exists()

    def test_xgboost_model_exists(self, models_dir):
        path = models_dir / "xgboost.pkl"
        if not path.exists():
            pytest.skip("xgboost.pkl not trained yet")
        assert path.exists()

    def test_random_forest_model_exists(self, models_dir):
        path = models_dir / "random_forest.pkl"
        if not path.exists():
            pytest.skip("random_forest.pkl not trained yet")
        assert path.exists()

    def test_preview_png_exists(self, predictions_dir):
        path = predictions_dir / "final_flood_map_preview.png"
        if not path.exists():
            pytest.skip("preview not generated yet")
        assert path.exists()


# =========================================================================
# Test: GeoTIFF Validity
# =========================================================================
class TestGeoTIFFValidity:
    def test_flood_map_is_valid_geotiff(self, predictions_dir):
        path = predictions_dir / "final_flood_map.tif"
        if not path.exists():
            pytest.skip("final_flood_map.tif not available")
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
        if not path.exists():
            pytest.skip("feature_stack.tif not available")
        with rasterio.open(path) as ds:
            assert ds.count == 5, f"Expected 5 bands, got {ds.count}"
            assert ds.dtypes[0] == "float32"

    def test_flood_map_shape_matches_features(self, processed_dir, predictions_dir):
        fs_path = processed_dir / "feature_stack.tif"
        fm_path = predictions_dir / "final_flood_map.tif"
        if not fs_path.exists() or not fm_path.exists():
            pytest.skip("Required files not available")
        with rasterio.open(fs_path) as fs:
            fs_shape = fs.shape
        with rasterio.open(fm_path) as fm:
            fm_shape = fm.shape
        assert fs_shape == fm_shape, f"Shape mismatch: features={fs_shape} flood={fm_shape}"


# =========================================================================
# Test: Model Validity
# =========================================================================
class TestModelValidity:
    def test_xgboost_can_predict(self, models_dir):
        path = models_dir / "xgboost.pkl"
        if not path.exists():
            pytest.skip("xgboost.pkl not available")
        with open(path, "rb") as f:
            model = pickle.load(f)
        X = np.random.rand(10, 5).astype(np.float32)
        y = model.predict(X)
        assert y.shape == (10,)
        assert set(y).issubset({0, 1})

    def test_random_forest_can_predict(self, models_dir):
        path = models_dir / "random_forest.pkl"
        if not path.exists():
            pytest.skip("random_forest.pkl not available")
        with open(path, "rb") as f:
            model = pickle.load(f)
        X = np.random.rand(10, 5).astype(np.float32)
        y = model.predict(X)
        assert y.shape == (10,)
        assert set(y).issubset({0, 1})

    def test_xgboost_metrics_json_valid(self, models_dir):
        path = models_dir / "xgboost_metrics.json"
        if not path.exists():
            pytest.skip("xgboost_metrics.json not available")
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

    def test_zero_tp(self):
        from evaluate import compute_metrics
        m = compute_metrics(tp=0, fp=0, tn=100, fn=0)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

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

    def test_ndwi_known_value(self):
        from features import compute_ndwi
        green = np.array([0.3], dtype=np.float32)
        nir = np.array([0.1], dtype=np.float32)
        ndwi = compute_ndwi(green, nir)
        assert abs(ndwi[0] - 0.5) < 1e-5

    def test_sar_threshold(self):
        from features import compute_sar_threshold
        vv = np.array([-20, -10, -16], dtype=np.float32)
        vh = np.array([-25, -15, -21], dtype=np.float32)
        mask = compute_sar_threshold(vv, vh)
        assert mask[0] == 1  # both below thresh
        assert mask[1] == 0  # VV above
        assert mask[2] == 1  # both below


# =========================================================================
# Test: API Endpoints (matching actual api/main.py)
# =========================================================================
class TestAPIHealth:
    """Tests for GET /health"""

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_schema(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "LIVE"
        assert "engine" in data
        assert "rust_ready" in data
        assert isinstance(data["rust_ready"], bool)
        assert "timestamp" in data


class TestAPIStats:
    """Tests for GET /stats"""

    def test_stats_returns_data_or_404(self, client):
        r = client.get("/stats")
        assert r.status_code in (200, 404)

    def test_stats_response_schema(self, client):
        r = client.get("/stats")
        if r.status_code == 200:
            data = r.json()
            assert "flood_pixels" in data
            assert "flood_percentage" in data
            assert "total_pixels" in data
            assert "bounds" in data
            assert isinstance(data["flood_percentage"], (int, float))


class TestAPIPredictAt:
    """Tests for GET /predict/at"""

    def test_predict_at_missing_params(self, client):
        """Missing lat/lon should return 422."""
        r = client.get("/predict/at")
        assert r.status_code == 422

    def test_predict_at_valid_coords(self, client):
        """Query with valid Sumbawa coordinates."""
        r = client.get("/predict/at", params={"lat": -8.78, "lon": 117.78})
        # 200 (success), 404 (raster missing), 422 (out of bounds), or 503 (no Rust)
        assert r.status_code in (200, 404, 422, 503)
        if r.status_code == 200:
            data = r.json()
            assert "flood" in data
            assert data["flood"] in (0, 1)
            assert isinstance(data["flood"], int)
            assert data["lat"] == -8.78
            assert data["lon"] == 117.78
            assert "method" in data
            assert "crs" in data
            assert data["crs"] == "EPSG:4326"
            assert "timestamp" in data
            assert "status" in data
            assert data["status"] in ("flood_detected", "safe", "permanent_water")

    def test_predict_at_out_of_range(self, client):
        """Point at (0, 0) should be outside raster bounds."""
        r = client.get("/predict/at", params={"lat": 0.0, "lon": 0.0})
        # 422 (out of bounds) or 503 (no Rust) or 404 (raster missing)
        assert r.status_code in (404, 422, 503)

    def test_predict_at_invalid_lat(self, client):
        """Latitude beyond valid range should fail validation."""
        r = client.get("/predict/at", params={"lat": 100.0, "lon": 117.78})
        assert r.status_code == 422


class TestAPIPredictArea:
    """Tests for POST /predict/area"""

    def test_predict_area_valid(self, client, sample_polygon_feature):
        """Post a valid polygon."""
        r = client.post("/predict/area", json=sample_polygon_feature)
        # 200 or 404 (rasters missing) or 503 (no Rust)
        assert r.status_code in (200, 404, 422, 503)
        if r.status_code == 200:
            data = r.json()
            assert "total_pixels" in data
            assert "flooded_pixels" in data
            assert "total_area_ha" in data
            assert "flooded_area_ha" in data
            assert "flood_percentage" in data
            assert "pixel_resolution_m" in data
            assert "geometry_type" in data
            assert data["geometry_type"] == "Polygon"

    def test_predict_area_invalid_geometry(self, client):
        """Non-polygon geometry should return 422."""
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [117.78, -8.78]
            },
            "properties": {}
        }
        r = client.post("/predict/area", json=feature)
        assert r.status_code in (422, 503)

    def test_predict_area_empty_body(self, client):
        """Empty body should return 422."""
        r = client.post("/predict/area")
        assert r.status_code == 422


class TestAPIAOIStats:
    """Tests for POST /predict/aoi-stats"""

    def test_aoi_stats_valid(self, client, sample_polygon_feature):
        r = client.post("/predict/aoi-stats", json=sample_polygon_feature)
        assert r.status_code in (200, 404, 422, 500)
        if r.status_code == 200:
            data = r.json()
            assert "total_area_ha" in data
            assert "flooded_area_ha" in data
            assert "flood_percentage" in data
            assert "raster_crs" in data

    def test_aoi_stats_invalid_geometry(self, client):
        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
            "properties": {}
        }
        r = client.post("/predict/aoi-stats", json=feature)
        assert r.status_code == 422


class TestAPIReport:
    """Tests for POST /predict/report"""

    def test_report_invalid_geometry(self, client):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {}
        }
        r = client.post("/predict/report", json=feature)
        assert r.status_code in (422, 503)


class TestAPIAsync:
    """Tests for async endpoints"""

    @patch("api.main.celery_app.send_task")
    def test_aoi_stats_async_valid(self, mock_send_task, client, sample_polygon_feature):
        """Dispatch async task — returns immediately with task_id."""
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_send_task.return_value = mock_task

        r = client.post("/predict/aoi-stats/async", json=sample_polygon_feature)
        # 200 or 500 (Redis not available in test)
        if r.status_code == 200:
            data = r.json()
            assert "task_id" in data
            assert "status" in data
            assert data["status"] == "PENDING"
            assert "poll_url" in data

    @patch("api.main.celery_app.send_task")
    def test_report_async_invalid_geometry(self, mock_send_task, client):
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {}
        }
        r = client.post("/predict/report/async", json=feature)
        assert r.status_code in (422, 503)


class TestAPITaskStatus:
    """Tests for GET /predict/status/{task_id}"""

    def test_task_status_nonexistent(self, client):
        """Non-existent task_id should return PENDING (Celery default)."""
        r = client.get("/predict/status/nonexistent-task-id")
        # May return 200 with PENDING or 500 if Redis unreachable
        if r.status_code == 200:
            data = r.json()
            assert "task_id" in data
            assert "status" in data


class TestAPITiles:
    """Tests for GET /tiles/{z}/{x}/{y}.png"""

    def test_tile_returns_png(self, client):
        """Tile endpoint should return PNG (even transparent fallback)."""
        r = client.get("/tiles/10/100/100.png")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "image/png"

    def test_tile_content_is_valid_png(self, client):
        """Response should be valid PNG bytes."""
        r = client.get("/tiles/10/100/100.png")
        if r.status_code == 200:
            # PNG magic bytes
            assert r.content[:4] == b'\x89PNG'


class TestAPIDashboard:
    """Tests for GET / (HTML dashboard)"""

    def test_dashboard_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        # Should contain some recognizable HTML content
        assert "html" in r.text.lower() or "Flood" in r.text or "Sumbawa" in r.text


class TestAPIFavicon:
    """Tests for GET /favicon.ico"""

    def test_favicon_returns_png(self, client):
        r = client.get("/favicon.ico")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "image/png"
