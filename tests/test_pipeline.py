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
        # Use fixed method for deterministic test
        mask = compute_sar_threshold(vv, vh, method='fixed')
        assert mask[0] == 1  # both below thresh
        assert mask[1] == 0  # VV above
        assert mask[2] == 1  # both below

    def test_sar_threshold_adaptive(self):
        from features import compute_sar_threshold
        np.random.seed(42)
        # Create bimodal distribution
        dark_vv = np.random.normal(-20, 2, 1000).astype(np.float32)
        bright_vv = np.random.normal(-8, 2, 1000).astype(np.float32)
        vv = np.concatenate([dark_vv, bright_vv])
        dark_vh = np.random.normal(-25, 2, 1000).astype(np.float32)
        bright_vh = np.random.normal(-15, 2, 1000).astype(np.float32)
        vh = np.concatenate([dark_vh, bright_vh])
        mask = compute_sar_threshold(vv, vh, method='otsu')
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


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


class TestSpatialCrossValidation:
    """Tests for spatial cross-validation (Roberts et al. 2017)."""

    def test_scv_returns_metrics(self):
        """SCV should return per-fold metrics and summary statistics."""
        from model import spatial_cross_validate
        np.random.seed(42)
        n_bands, h, w = 5, 30, 30
        features = np.random.randn(n_bands, h, w).astype(np.float32)
        labels = np.random.choice([0, 1], (h, w), p=[0.9, 0.1]).astype(np.uint8)

        result = spatial_cross_validate(features, labels, n_folds=2, model_type="random_forest")

        assert "f1_mean" in result
        assert "f1_std" in result
        assert "per_fold" in result
        assert len(result["per_fold"]) > 0
        for fold in result["per_fold"]:
            assert "f1" in fold
            assert "precision" in fold
            assert "recall" in fold

    def test_scv_different_models(self):
        """SCV should work with different model types."""
        from model import spatial_cross_validate
        np.random.seed(42)
        features = np.random.randn(5, 30, 30).astype(np.float32)
        labels = np.random.choice([0, 1], (30, 30), p=[0.9, 0.1]).astype(np.uint8)

        result = spatial_cross_validate(features, labels, n_folds=2, model_type="random_forest")
        assert "f1_mean" in result


class TestChangeDetection:
    """Tests for multi-temporal change detection (Schlaffer 2015, Clement 2018)."""

    def test_compute_delta_sar(self):
        """Delta SAR should compute difference between current and baseline."""
        from change_detection import compute_delta_sar
        np.random.seed(42)
        vv_curr = np.full((10, 10), -10.0, dtype=np.float32)
        vh_curr = np.full((10, 10), -18.0, dtype=np.float32)
        vv_base = np.full((10, 10), -8.0, dtype=np.float32)
        vh_base = np.full((10, 10), -15.0, dtype=np.float32)

        dvv, dvh = compute_delta_sar(vv_curr, vh_curr, vv_base, vh_base)
        assert dvv.shape == (10, 10)
        np.testing.assert_allclose(dvv, -2.0, atol=0.01)
        np.testing.assert_allclose(dvh, -3.0, atol=0.01)

    def test_detect_flood_by_change(self):
        """Change detection should flag negative anomalies as flood."""
        from change_detection import detect_flood_by_change
        dvv = np.array([[-5.0, -1.0, -4.0], [0.0, -3.5, -0.5]], dtype=np.float32)
        dvh = np.array([[-4.0, -1.0, -3.5], [0.0, -4.0, -0.5]], dtype=np.float32)

        mask = detect_flood_by_change(dvv, dvh, vv_thresh=-3.0, vh_thresh=-3.0)
        assert mask[0, 0] == 1   # both < -3
        assert mask[0, 1] == 0   # both > -3
        assert mask[1, 0] == 0   # both > -3
        assert mask[1, 2] == 0   # both > -3

    def test_temporal_statistics(self):
        """Temporal stats should compute mean, std, cv correctly."""
        from change_detection import compute_temporal_statistics
        np.random.seed(42)
        stack = np.random.randn(5, 10, 10).astype(np.float32) * 2 - 10

        stats = compute_temporal_statistics(stack)
        assert "mean" in stats
        assert "std" in stats
        assert "cv" in stats
        assert stats["mean"].shape == (10, 10)

    def test_adaptive_change_detection(self):
        """Adaptive change detection should use Otsu on delta signal."""
        from change_detection import detect_adaptive_change
        np.random.seed(42)
        # Simulate clear bimodal distribution
        normal = np.random.normal(0, 1, 500).astype(np.float32)
        flood = np.random.normal(-6, 1, 500).astype(np.float32)
        dvv = np.concatenate([normal, flood])
        dvh = np.concatenate([normal, flood])

        mask = detect_adaptive_change(dvv, dvh, method="otsu")
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


class TestGroundTruth:
    """Tests for ground truth validation module."""

    def test_validate_against_gsw_metrics(self):
        """GSW validation should compute precision, recall, F1, IoU."""
        from ground_truth import validate_against_gsw
        import rasterio
        import tempfile

        np.random.seed(42)
        h, w = 50, 50

        # Create synthetic flood map
        flood = np.zeros((h, w), dtype=np.uint8)
        flood[10:20, 10:20] = 1  # flood in one area

        # Create synthetic GSW (some overlap, some not)
        gsw = np.zeros((h, w), dtype=np.uint8)
        gsw[10:20, 10:20] = 100  # same area = perfect overlap
        gsw[30:40, 30:40] = 100  # additional permanent water

        # Write to temp files
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            flood_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            gsw_path = f.name

        transform = rasterio.transform.from_bounds(116, -9, 119, -8, w, h)
        profile = {
            "driver": "GTiff", "height": h, "width": w,
            "count": 1, "dtype": "uint8", "crs": "EPSG:4326",
            "transform": transform, "nodata": 255,
        }

        with rasterio.open(flood_path, "w", **profile) as dst:
            dst.write(flood, 1)
        with rasterio.open(gsw_path, "w", **profile) as dst:
            dst.write(gsw, 1)

        result = validate_against_gsw(flood_path, gsw_path, gsw_threshold=50)

        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "iou" in result
        assert result["f1_score"] > 0  # should have some overlap
        assert result["precision"] > 0

        # Cleanup
        Path(flood_path).unlink()
        Path(gsw_path).unlink()


class TestAttentionFusion:
    """Tests for attention-based multi-modal fusion."""

    def test_attention_fusion_returns_score(self):
        """Attention fusion should return probability score [0, 1]."""
        from attention_fusion import attention_weighted_fusion
        np.random.seed(42)
        h, w = 20, 20
        ndwi = np.random.randn(h, w).astype(np.float32) * 0.3
        sar_vv = np.random.randn(h, w).astype(np.float32) * 5 - 15
        sar_vh = np.random.randn(h, w).astype(np.float32) * 5 - 22
        slope = np.random.rand(h, w).astype(np.float32) * 10
        hand = np.random.rand(h, w).astype(np.float32) * 5

        score = attention_weighted_fusion(ndwi, sar_vv, sar_vh, slope, hand)
        assert score.shape == (h, w)
        assert score.dtype == np.float32
        assert np.all(score >= 0)
        assert np.all(score <= 1)

    def test_adaptive_flood_mask(self):
        """Adaptive flood mask should produce binary output."""
        from attention_fusion import adaptive_flood_mask
        np.random.seed(42)
        score = np.concatenate([
            np.random.uniform(0.7, 1.0, 100),  # flood-like
            np.random.uniform(0.0, 0.3, 900),  # non-flood
        ]).astype(np.float32)

        mask = adaptive_flood_mask(score, method="otsu")
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
