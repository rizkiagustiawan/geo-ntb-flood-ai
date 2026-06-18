"""
Tests for SAR preprocessing module.
Validates speckle filter, incidence angle normalization, and thermal noise removal.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sar_preprocess import refined_lee_filter, normalize_incidence_angle, remove_thermal_noise


class TestRefinedLeeFilter:
    def test_reduces_speckle_variance(self):
        np.random.seed(42)
        clean = np.full((100, 100), -10.0, dtype=np.float32)
        speckled = clean + np.random.normal(0, 3.0, clean.shape).astype(np.float32)
        filtered = refined_lee_filter(speckled, window_size=7)
        assert np.var(filtered) < np.var(speckled)

    def test_preserves_mean_backscatter(self):
        np.random.seed(42)
        clean = np.full((100, 100), -10.0, dtype=np.float32)
        speckled = clean + np.random.normal(0, 2.0, clean.shape).astype(np.float32)
        filtered = refined_lee_filter(speckled, window_size=7)
        assert abs(np.mean(filtered) - np.mean(speckled)) < 0.5

    def test_preserves_edge_contrast(self):
        img = np.zeros((100, 100), dtype=np.float32)
        img[:50, :] = -15.0
        img[50:, :] = -5.0
        filtered = refined_lee_filter(img, window_size=7)
        edge_diff = abs(np.mean(filtered[45:50, :]) - np.mean(filtered[50:55, :]))
        assert edge_diff > 5.0

    def test_handles_nan(self):
        img = np.full((50, 50), -10.0, dtype=np.float32)
        img[20:30, 20:30] = np.nan
        filtered = refined_lee_filter(img, window_size=5)
        # Non-NaN regions should still have valid values
        assert not np.any(np.isnan(filtered[:18, :18]))
        # NaN region may remain NaN if entire window is NaN — that's acceptable
        # The key test is that the filter doesn't crash on NaN input

    def test_window_size_must_be_odd(self):
        img = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError):
            refined_lee_filter(img, window_size=6)


class TestNormalizeIncidenceAngle:
    def test_flattens_backscatter_variation(self):
        np.random.seed(42)
        n_rows, n_cols = 100, 100
        vv = np.full((n_rows, n_cols), -10.0, dtype=np.float32)
        angle_map = np.linspace(29, 46, n_cols).reshape(1, -1).repeat(n_rows, axis=0).astype(np.float32)
        vv += (angle_map - 37.5) * 0.15
        normalized = normalize_incidence_angle(vv, angle_map, reference_angle=37.5)
        assert np.std(normalized) < np.std(vv)

    def test_identity_at_reference_angle(self):
        vv = np.full((50, 50), -10.0, dtype=np.float32)
        angle_map = np.full((50, 50), 37.5, dtype=np.float32)
        normalized = normalize_incidence_angle(vv, angle_map, reference_angle=37.5)
        np.testing.assert_allclose(normalized, vv, atol=0.01)


class TestRemoveThermalNoise:
    def test_reduces_noise_at_swath_edges(self):
        vv = np.full((100, 100), -10.0, dtype=np.float32)
        noise = np.zeros((100, 100), dtype=np.float32)
        noise[:, :10] = 5.0
        noisy = vv + noise
        cleaned = remove_thermal_noise(noisy, noise_level_db=5.0, edge_fraction=0.1)
        assert np.mean(cleaned[:, :10]) < np.mean(noisy[:, :10])

    def test_center_unchanged(self):
        vv = np.full((100, 100), -10.0, dtype=np.float32)
        cleaned = remove_thermal_noise(vv, noise_level_db=5.0, edge_fraction=0.1)
        np.testing.assert_allclose(cleaned[:, 20:-20], vv[:, 20:-20], atol=0.01)


class TestPreprocessSAR:
    def test_full_pipeline(self):
        from sar_preprocess import preprocess_sar
        np.random.seed(42)
        vv = np.random.normal(-10, 3, (50, 50)).astype(np.float32)
        vh = np.random.normal(-18, 3, (50, 50)).astype(np.float32)
        vv_p, vh_p = preprocess_sar(vv, vh, normalize_angle=False)
        assert vv_p.shape == vv.shape
        assert vh_p.shape == vh.shape
        assert np.var(vv_p) < np.var(vv)
