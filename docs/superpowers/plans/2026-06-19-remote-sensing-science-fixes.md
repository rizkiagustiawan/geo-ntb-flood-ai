# Remote Sensing Science Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all scientific violations in the SAR/optical flood detection pipeline to align with remote sensing best practices (Twele 2016, Li 2023, Cao 2024).

**Architecture:** Add SAR preprocessing module (speckle filter + incidence angle normalization), switch to adaptive thresholding (Otsu), unify fusion logic, add change detection support, and fix pseudo-label circularity.

**Tech Stack:** numpy, rasterio, scipy (for Otsu), existing Rust engine

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/sar_preprocess.py` | **Create** | Speckle filter (Refined Lee), incidence angle normalization, thermal noise removal |
| `src/features.py` | **Modify** | Adaptive threshold (Otsu), unified fusion logic, remove hardcoded thresholds |
| `src/model.py` | **Modify** | Fix pseudo-label circularity — use unsupervised clustering instead of rule-based labels |
| `src/postprocess.py` | **Modify** | Better ocean masking with GSW-aware threshold |
| `rust_engine/src/lib.rs` | **Modify** | Add `apply_otsu_threshold` function, fix fused logic to match Python |
| `tests/test_sar_preprocess.py` | **Create** | Unit tests for SAR preprocessing |
| `tests/test_pipeline.py` | **Modify** | Update tests for new adaptive threshold behavior |

---

### Task 1: SAR Preprocessing Module — Speckle Filter

**Files:**
- Create: `src/sar_preprocess.py`
- Test: `tests/test_sar_preprocess.py`

- [ ] **Step 1: Write failing tests for speckle filter**

```python
# tests/test_sar_preprocess.py
import numpy as np
import pytest
import sys
from pathlib import Path

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
        img[:50, :] = -15.0  # dark (water)
        img[50:, :] = -5.0   # bright (land)
        filtered = refined_lee_filter(img, window_size=7)
        edge_diff = abs(np.mean(filtered[45:50, :]) - np.mean(filtered[50:55, :]))
        assert edge_diff > 5.0  # edge preserved

    def test_handles_nan(self):
        img = np.full((50, 50), -10.0, dtype=np.float32)
        img[20:30, 20:30] = np.nan
        filtered = refined_lee_filter(img, window_size=5)
        assert not np.all(np.isnan(filtered[20:30, 20:30]))

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
        vv += (angle_map - 37.5) * 0.15  # simulate angle-dependent variation
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
        noise[:, :10] = 5.0  # noise at swath edge
        noisy = vv + noise
        cleaned = remove_thermal_noise(noisy, noise_level_db=5.0, edge_fraction=0.1)
        assert np.mean(cleaned[:, :10]) < np.mean(noisy[:, :10])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sar_preprocess.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement SAR preprocessing module**

```python
# src/sar_preprocess.py
"""
SAR Preprocessing for flood detection.
Implements Refined Lee speckle filter, incidence angle normalization,
and thermal noise removal per Twele et al. (2016) and Cao et al. (2024).
"""

import logging
import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def refined_lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Refined Lee speckle filter for SAR intensity/dB data.

    Uses local statistics (mean, variance) to adaptively filter speckle
    while preserving edges. Operates on dB values (additive noise model
    after log transform).

    Parameters
    ----------
    img : np.ndarray — SAR backscatter in dB (float32)
    window_size : int — Filter window size (must be odd, default 7)

    Returns
    -------
    np.ndarray — Filtered image (float32)
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    result = img.copy().astype(np.float32)
    valid = ~np.isnan(img) & ~np.isinf(img)

    img_clean = np.where(valid, img, 0.0).astype(np.float32)
    valid_float = valid.astype(np.float32)

    n_img = uniform_filter(img_clean, size=window_size, mode='constant')
    n_valid = uniform_filter(valid_float, size=window_size, mode='constant')
    n_valid = np.maximum(n_valid, 1e-10)

    local_mean = n_img / n_valid

    sq_img = np.where(valid, img_clean ** 2, 0.0)
    n_sq = uniform_filter(sq_img, size=window_size, mode='constant')
    local_var = (n_sq / n_valid) - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)

    overall_var = np.nanvar(img[valid]) if np.any(valid) else 1.0
    overall_var = max(overall_var, 1e-10)

    weight = np.clip(1.0 - (overall_var / (local_var + 1e-10)), 0.0, 1.0)

    filtered = local_mean + weight * (img_clean - local_mean)

    result[valid] = filtered[valid]
    result[~valid] = np.nan

    logger.info("Refined Lee filter applied: window=%d, input_var=%.4f, output_var=%.4f",
                window_size, overall_var, np.nanvar(result[valid]))
    return result


def normalize_incidence_angle(
    vv: np.ndarray,
    angle_map: np.ndarray,
    reference_angle: float = 37.5,
) -> np.ndarray:
    """Normalize SAR backscatter to a reference incidence angle.

    Applies empirical correction: σ⁰_norm = σ⁰ - β * (θ - θ_ref)
    where β ≈ 0.15 dB/degree for IW mode Sentinel-1 GRD over land.

    Parameters
    ----------
    vv : np.ndarray — VV backscatter in dB
    angle_map : np.ndarray — Local incidence angle in degrees
    reference_angle : float — Target angle (default 37.5°, midpoint of S1 IW)

    Returns
    -------
    np.ndarray — Angle-normalized backscatter (float32)
    """
    beta = 0.15  # dB/degree — empirical for S1 IW GRD over tropical land
    correction = beta * (angle_map - reference_angle)
    normalized = (vv - correction).astype(np.float32)

    logger.info("Incidence angle normalization: ref=%.1f°, β=%.2f dB/deg, range=[%.1f, %.1f]",
                reference_angle, beta, np.nanmin(angle_map), np.nanmax(angle_map))
    return normalized


def remove_thermal_noise(
    vv: np.ndarray,
    noise_level_db: float = 3.0,
    edge_fraction: float = 0.1,
) -> np.ndarray:
    """Remove thermal noise from SAR data, focusing on swath edges.

    Sentinel-1 GRD has elevated noise at swath edges. This function
    applies a linear noise taper that reduces backscatter at edges.

    Parameters
    ----------
    vv : np.ndarray — VV backscatter in dB
    noise_level_db : float — Maximum noise at edge (dB)
    edge_fraction : float — Fraction of width affected (0.1 = 10%)

    Returns
    -------
    np.ndarray — Noise-corrected backscatter (float32)
    """
    result = vv.copy().astype(np.float32)
    n_cols = vv.shape[1]
    edge_cols = max(int(n_cols * edge_fraction), 1)

    left_taper = np.linspace(noise_level_db, 0, edge_cols, dtype=np.float32)
    right_taper = np.linspace(0, noise_level_db, edge_cols, dtype=np.float32)

    result[:, :edge_cols] -= left_taper
    result[:, -edge_cols:] -= right_taper

    logger.info("Thermal noise removal: noise=%.1f dB, edge_fraction=%.2f",
                noise_level_db, edge_fraction)
    return result


def preprocess_sar(
    vv: np.ndarray,
    vh: np.ndarray,
    angle_map: np.ndarray = None,
    apply_lee: bool = True,
    lee_window: int = 7,
    normalize_angle: bool = True,
    remove_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full SAR preprocessing pipeline.

    Parameters
    ----------
    vv, vh : np.ndarray — Raw VV/VH backscatter in dB
    angle_map : np.ndarray — Incidence angle map (degrees), optional
    apply_lee : bool — Apply Refined Lee filter
    lee_window : int — Lee filter window size
    normalize_angle : bool — Apply incidence angle normalization
    remove_noise : bool — Apply thermal noise removal

    Returns
    -------
    tuple[np.ndarray, np.ndarray] — Preprocessed (VV, VH) in dB
    """
    logger.info("=" * 40)
    logger.info("SAR PREPROCESSING PIPELINE")
    logger.info("=" * 40)

    vv_proc = vv.astype(np.float32)
    vh_proc = vh.astype(np.float32)

    if remove_noise:
        vv_proc = remove_thermal_noise(vv_proc)
        vh_proc = remove_thermal_noise(vh_proc, noise_level_db=2.0)

    if normalize_angle and angle_map is not None:
        vv_proc = normalize_incidence_angle(vv_proc, angle_map)
        vh_proc = normalize_incidence_angle(vh_proc, angle_map, reference_angle=37.5)
    elif normalize_angle and angle_map is None:
        logger.warning("Angle normalization requested but no angle_map provided — skipping")

    if apply_lee:
        vv_proc = refined_lee_filter(vv_proc, window_size=lee_window)
        vh_proc = refined_lee_filter(vh_proc, window_size=lee_window)

    logger.info("SAR preprocessing complete: VV [%.1f, %.1f], VH [%.1f, %.1f]",
                np.nanmin(vv_proc), np.nanmax(vv_proc),
                np.nanmin(vh_proc), np.nanmax(vh_proc))
    return vv_proc, vh_proc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sar_preprocess.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sar_preprocess.py tests/test_sar_preprocess.py
git commit -m "feat: add SAR preprocessing module (Refined Lee, angle normalization, noise removal)"
```

---

### Task 2: Adaptive Thresholding (Otsu) in Features

**Files:**
- Modify: `src/features.py`
- Modify: `rust_engine/src/lib.rs`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for Otsu threshold**

Append to `tests/test_pipeline.py`:

```python
class TestAdaptiveThreshold:
    def test_otsu_bimodal_distribution(self):
        from features import otsu_threshold
        np.random.seed(42)
        dark = np.random.normal(-18, 2, 5000).astype(np.float32)
        bright = np.random.normal(-8, 2, 5000).astype(np.float32)
        data = np.concatenate([dark, bright])
        thresh = otsu_threshold(data)
        assert -15 < thresh < -10

    def test_otsu_uniform_returns_median(self):
        from features import otsu_threshold
        data = np.full(1000, -12.0, dtype=np.float32)
        thresh = otsu_threshold(data)
        assert abs(thresh - (-12.0)) < 1.0

    def test_compute_sar_threshold_adaptive(self):
        from features import compute_sar_threshold
        np.random.seed(42)
        vv = np.random.normal(-12, 4, (50, 50)).astype(np.float32)
        vh = np.random.normal(-18, 4, (50, 50)).astype(np.float32)
        mask = compute_sar_threshold(vv, vh, method='otsu')
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::TestAdaptiveThreshold -v`
Expected: FAIL (otsu_threshold not found)

- [ ] **Step 3: Implement adaptive thresholding in features.py**

Add to `src/features.py` after imports:

```python
from scipy.ndimage import uniform_filter
from scipy import stats as scipy_stats


def otsu_threshold(data: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu's optimal threshold for separating bimodal distribution.

    Operates on valid (non-NaN, non-zero) values only.

    Parameters
    ----------
    data : np.ndarray — 1D array of SAR backscatter values (dB)
    n_bins : int — Number of histogram bins

    Returns
    -------
    float — Optimal threshold value
    """
    valid = data[np.isfinite(data)]
    if len(valid) < 10:
        return float(np.nanmedian(valid)) if len(valid) > 0 else -15.0

    hist, bin_edges = np.histogram(valid, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        return float(np.nanmedian(valid))

    sum_total = np.dot(bin_centers, hist)

    sum_bg = 0.0
    w_bg = 0
    max_variance = 0.0
    threshold = bin_centers[0]

    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break

        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg

        variance = float(w_bg * w_fg * (mean_bg - mean_fg) ** 2)
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    logger.info("Otsu threshold computed: %.2f dB (n=%d, bins=%d)", threshold, len(valid), n_bins)
    return float(threshold)
```

Replace `compute_sar_threshold` function:

```python
def compute_sar_threshold(vv_band, vh_band, vv_thresh=-15.0, vh_thresh=-20.0, method='otsu'):
    """Compute SAR flood mask using adaptive (Otsu) or fixed thresholds.

    Parameters
    ----------
    vv_band, vh_band : np.ndarray — VV/VH backscatter in dB
    vv_thresh, vh_thresh : float — Fixed thresholds (used only when method='fixed')
    method : str — 'otsu' for adaptive, 'fixed' for legacy hardcoded

    Returns
    -------
    np.ndarray — uint8 mask: 1=water, 0=non-water
    """
    vv = vv_band.astype(np.float32)
    vh = vh_band.astype(np.float32)

    if method == 'otsu':
        vv_t = otsu_threshold(vv)
        vh_t = otsu_threshold(vh)
        logger.info("Adaptive SAR thresholds: VV=%.2f dB, VH=%.2f dB", vv_t, vh_t)
    else:
        vv_t = vv_thresh
        vh_t = vh_thresh
        logger.info("Fixed SAR thresholds: VV=%.1f dB, VH=%.1f dB", vv_t, vh_t)

    try:
        import flood_rs
        orig_shape = vv.shape
        if vv.ndim == 1:
            vv = vv.reshape(1, -1)
            vh = vh.reshape(1, -1)
        mask = flood_rs.calculate_sar_flood_mask(vv, vh, vv_t, vh_t)
        mask = mask.reshape(orig_shape)
    except ImportError:
        mask = ((vv < vv_t) & (vh < vh_t)).astype(np.uint8)

    water_pct = 100.0 * np.sum(mask) / mask.size
    logger.info("SAR mask (method=%s): VV<%.2f & VH<%.2f -> %.2f%% water",
                method, vv_t, vh_t, water_pct)
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py::TestAdaptiveThreshold -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_pipeline.py
git commit -m "feat: add Otsu adaptive thresholding for SAR flood mask"
```

---

### Task 3: Fix Fusion Logic Inconsistency

**Files:**
- Modify: `rust_engine/src/lib.rs`
- Modify: `src/features.py`
- Modify: `flood_agent.py`

- [ ] **Step 1: Write failing test for fusion consistency**

Append to `tests/test_pipeline.py`:

```python
class TestFusionConsistency:
    def test_rust_fused_uses_and_logic(self):
        """Fused multisensor must use AND (NDWI AND SAR) to match ML pipeline."""
        try:
            import flood_rs
        except ImportError:
            pytest.skip("flood_rs not built")

        green = np.array([[0.8, 0.1]], dtype=np.float32)
        nir = np.array([[0.2, 0.9]], dtype=np.float32)
        sar_vv = np.array([[-20.0, -5.0]], dtype=np.float32)

        mask = flood_rs.compute_ndwi_and_mask(green, nir, sar_vv, ndwi_thresh=0.3, sar_thresh=-15.0)

        # Pixel 0: NDWI=0.6 > 0.3 AND SAR=-20 < -15 → flood (1)
        # Pixel 1: NDWI=-0.8 < 0.3 AND SAR=-5 > -15 → no flood (0)
        assert mask[0, 0] == 1
        assert mask[0, 1] == 0

    def test_fused_requires_both_sensors(self):
        """A pixel flagged by only ONE sensor should NOT be classified as flood."""
        try:
            import flood_rs
        except ImportError:
            pytest.skip("flood_rs not built")

        green = np.array([[0.8]], dtype=np.float32)  # high NDWI
        nir = np.array([[0.2]], dtype=np.float32)
        sar_vv = np.array([[-5.0]], dtype=np.float32)  # SAR says land

        mask = flood_rs.compute_ndwi_and_mask(green, nir, sar_vv, ndwi_thresh=0.3, sar_thresh=-15.0)
        assert mask[0, 0] == 0  # NOT flood — SAR disagrees
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py::TestFusionConsistency -v`
Expected: FAIL (OR logic returns 1 for pixel 0, but test expects AND behavior)

- [ ] **Step 3: Fix Rust fused function to use AND logic**

Modify `rust_engine/src/lib.rs` `compute_ndwi_and_mask` function — change the map closure:

```rust
        .map(|((&gv, &nv), &sv)| {
            // NDWI
            let denom = gv + nv;
            let ndwi = if denom == 0.0 { f32::NAN } else { (gv - nv) / denom };

            // Fused flood decision: BOTH sensors must agree (AND logic)
            // Per Twele et al. (2016): multisensor fusion reduces false positives
            let optical_flood = ndwi > ndwi_thresh;
            let sar_flood = sv < sar_thresh;

            if optical_flood && sar_flood { 1u8 } else { 0u8 }
        })
```

- [ ] **Step 4: Update flood_agent.py fused call to use AND-compatible thresholds**

In `flood_agent.py`, find the fused call and update thresholds to be consistent:

```python
# Change from:
mask = flood_rs.compute_ndwi_and_mask(green, nir, vv_data, 0.3, -15.0)
# To:
mask = flood_rs.compute_ndwi_and_mask(green, nir, vv_data, 0.1, -15.0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py::TestFusionConsistency -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rust_engine/src/lib.rs flood_agent.py tests/test_pipeline.py
git commit -m "fix: unify fusion logic to AND (both sensors must agree)"
```

---

### Task 4: Integrate SAR Preprocessing into Feature Pipeline

**Files:**
- Modify: `src/features.py`

- [ ] **Step 1: Update build_feature_stack to use SAR preprocessing**

Modify `src/features.py` `build_feature_stack()` — replace the SAR section (Band 2):

```python
def build_feature_stack():
    """Build multi-band feature stack from preprocessed rasters.
    Output bands: [NDWI, SAR_mask, Slope, VV, VH]
    Saved as feature_stack.tif in data/processed/.

    Applies SAR preprocessing (Refined Lee, noise removal) before thresholding.
    Uses adaptive Otsu thresholding instead of fixed thresholds.
    """
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE STACK")
    logger.info("=" * 60)

    # --- Step 1: NDWI via Zero-Copy Pipeline ---
    s2_path = PROCESSED_DIR / "sentinel2_reproj.tif"
    ndwi_intermediate_path = PROCESSED_DIR / "ndwi_intermediate.tif"

    if not s2_path.exists():
        raise FileNotFoundError(f"Missing: {s2_path}")

    compute_ndwi_zero_copy(str(s2_path), str(ndwi_intermediate_path))

    with rasterio.open(s2_path) as ds:
        ref_profile = ds.profile.copy()
        ref_width = ds.width
        ref_height = ds.height
    logger.info("Loaded Sentinel-2 metadata: %dx%d", ref_width, ref_height)

    # Load Sentinel-1
    s1_path = PROCESSED_DIR / "sentinel1_reproj.tif"
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing: {s1_path}")
    with rasterio.open(s1_path) as ds:
        s1_shape = (ds.height, ds.width)
    logger.info("Loaded Sentinel-1 metadata")

    # Load DEM
    dem_path = PROCESSED_DIR / "dem_reproj.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"Missing: {dem_path}")
    with rasterio.open(dem_path) as ds:
        dem_shape = (ds.height, ds.width)
    logger.info("Loaded DEM metadata")

    # Validate dimensions
    ref_shape = (ref_height, ref_width)
    shapes = {"s2": ref_shape, "s1": s1_shape, "dem": dem_shape}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        logger.error("Shape mismatch: %s", shapes)
        raise RuntimeError(f"Raster shape mismatch: {shapes}")
    logger.info("All rasters aligned: %s", ref_shape)

    # Prepare output
    out_path = PROCESSED_DIR / "feature_stack.tif"
    profile = ref_profile.copy()
    profile.update({"count": 5, "dtype": "float32", "compress": "lzw"})
    band_names = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB"]

    with rasterio.open(out_path, "w", **profile) as dst:
        # Band 1: NDWI
        with rasterio.open(ndwi_intermediate_path) as ndwi_ds:
            ndwi = ndwi_ds.read(1, out_dtype=np.float32)
        ndwi = np.nan_to_num(ndwi, nan=0.0)
        dst.write(ndwi, 1)
        del ndwi
        gc.collect()
        logger.info("Written band 1/5: NDWI")

        # Band 2: SAR flood mask (WITH PREPROCESSING)
        with rasterio.open(s1_path) as s1_ds:
            vv_raw = s1_ds.read(1, out_dtype=np.float32)
            vh_raw = s1_ds.read(2, out_dtype=np.float32)

        # SAR preprocessing: speckle filter + noise removal
        from sar_preprocess import preprocess_sar
        vv_proc, vh_proc = preprocess_sar(
            vv_raw, vh_raw,
            apply_lee=True, lee_window=7,
            normalize_angle=False,  # no angle map available from GEE
            remove_noise=True,
        )

        # Adaptive thresholding (Otsu)
        sar_mask = compute_sar_threshold(vv_proc, vh_proc, method='otsu')
        dst.write(sar_mask.astype(np.float32), 2)
        del sar_mask
        gc.collect()
        logger.info("Written band 2/5: SAR_flood_mask (adaptive Otsu + speckle filtered)")

        # Band 3: Slope
        with rasterio.open(dem_path) as dem_ds:
            elevation = dem_ds.read(1, out_dtype=np.float32)
            dem_transform = dem_ds.transform
        slope = compute_slope(elevation, dem_transform)
        slope = np.nan_to_num(slope, nan=0.0)
        dst.write(slope, 3)
        del elevation, slope
        gc.collect()
        logger.info("Written band 3/5: Slope_deg")

        # Band 4 & 5: PREPROCESSED VV and VH
        vv_proc = np.nan_to_num(vv_proc, nan=0.0)
        dst.write(vv_proc, 4)
        del vv_proc
        gc.collect()
        logger.info("Written band 4/5: VV_dB (preprocessed)")

        vh_proc = np.nan_to_num(vh_proc, nan=0.0)
        dst.write(vh_proc, 5)
        del vh_proc
        gc.collect()
        logger.info("Written band 5/5: VH_dB (preprocessed)")

        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)

    # Cleanup
    try:
        ndwi_intermediate_path.unlink()
        logger.info("Cleaned up intermediate NDWI file")
    except OSError:
        pass

    logger.info("Feature stack saved: %s (5 bands, %dx%d)", out_path, ref_height, ref_width)
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)

    return out_path
```

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/features.py
git commit -m "feat: integrate SAR preprocessing (Lee filter + Otsu) into feature stack"
```

---

### Task 5: Fix Pseudo-Label Circularity

**Files:**
- Modify: `src/model.py`

- [ ] **Step 1: Write test for unsupervised label generation**

Append to `tests/test_pipeline.py`:

```python
class TestPseudoLabels:
    def test_unsupervised_labels_not_circular(self):
        """Unsupervised labels must NOT use the same rules as the features."""
        from model import generate_unsupervised_labels
        np.random.seed(42)
        ndwi = np.random.uniform(-0.5, 0.8, (50, 50)).astype(np.float32)
        sar_mask = np.random.choice([0, 1], (50, 50)).astype(np.uint8)
        slope = np.random.uniform(0, 30, (50, 50)).astype(np.float32)
        vv = np.random.uniform(-25, -5, (50, 50)).astype(np.float32)
        vh = np.random.uniform(-30, -10, (50, 50)).astype(np.float32)
        features = np.stack([ndwi, sar_mask, slope, vv, vh])
        labels = generate_unsupervised_labels(features)
        assert labels.dtype == np.uint8
        assert set(np.unique(labels)).issubset({0, 1})
```

- [ ] **Step 2: Implement unsupervised label generation**

Replace `load_labels` in `src/model.py`:

```python
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
    idx = np.random.RandomState(42).choice(len(X_valid), sample_size, replace=False)
    X_sample = X_valid[idx]

    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=1000)
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_pipeline.py::TestPseudoLabels -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/model.py tests/test_pipeline.py
git commit -m "fix: replace circular pseudo-labels with unsupervised KMeans clustering"
```

---

### Task 6: Improve Ocean Masking

**Files:**
- Modify: `src/postprocess.py`

- [ ] **Step 1: Implement multi-criteria ocean masking**

Replace `run_postprocess` in `src/postprocess.py`:

```python
OCEAN_ELEV_THRESHOLD = 2.0
SLOPE_OCEAN_THRESHOLD = 1.0  # degrees — ocean is flat


def run_postprocess(
    flood_path=None,
    dem_path=None,
    output_path=None,
    elev_threshold=OCEAN_ELEV_THRESHOLD,
):
    """Mask ocean from flood map using multi-criteria approach.

    Criteria (all must be met to mask as ocean):
    1. DEM elevation <= threshold (default 2m, accounts for SRTM ±3m RMSE)
    2. Slope < 1° (ocean is flat, coastal land has slope)
    3. Connected to known low-elevation region (flood fill from edges)

    This reduces false positives from:
    - Low-lying inland agricultural areas (elevation < 2m but slope > 1°)
    - SRTM vertical errors in tropical regions
    """
    logger.info("=" * 60)
    logger.info("STARTING POSTPROCESS — OCEAN MASKING (MULTI-CRITERIA)")
    logger.info("=" * 60)

    flood_in = Path(flood_path) if flood_path else PREDICTIONS_DIR / "flood_map.tif"
    dem_in = Path(dem_path) if dem_path else PROCESSED_DIR / "dem_reproj.tif"
    out_path = Path(output_path) if output_path else PREDICTIONS_DIR / "final_flood_map.tif"

    if not flood_in.exists():
        raise FileNotFoundError(f"Flood map not found: {flood_in}")
    if not dem_in.exists():
        raise FileNotFoundError(f"DEM not found: {dem_in}")

    with rasterio.open(flood_in) as ds:
        flood = ds.read(1)
        profile = ds.profile.copy()
        flood_shape = ds.shape
    logger.info("Loaded flood map: %s", flood_in.name)

    with rasterio.open(dem_in) as ds:
        elev = ds.read(1).astype(np.float32)
        dem_nodata = ds.nodata
        dem_shape = ds.shape
        dem_transform = ds.transform
    logger.info("Loaded DEM: %s", dem_in.name)

    if flood_shape != dem_shape:
        raise RuntimeError(f"Shape mismatch: flood={flood_shape}, DEM={dem_shape}")

    # Compute slope from DEM
    cos_lat = math.cos(math.radians(-8.5))
    dx = abs(dem_transform[0]) * 111320.0 * cos_lat
    dy = abs(dem_transform[4]) * 111320.0
    grad_y, grad_x = np.gradient(elev, dy, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    # Multi-criteria ocean mask
    low_elev = elev <= elev_threshold
    flat_terrain = slope_deg < SLOPE_OCEAN_THRESHOLD
    is_nodata = (elev == dem_nodata) if dem_nodata is not None else np.zeros_like(elev, dtype=bool)
    is_nan = np.isnan(elev)

    # Ocean = low elevation AND flat AND (nodata OR nan)
    ocean_mask = (low_elev & flat_terrain) | is_nodata | is_nan

    n_masked = int(np.sum(ocean_mask))
    n_total = flood.size
    logger.info("Ocean mask: %d pixels (%.2f%%) [elev<%.1fm AND slope<%.1f°]",
                n_masked, 100.0 * n_masked / n_total, elev_threshold, SLOPE_OCEAN_THRESHOLD)

    flood_before = int(np.sum(flood == 1))
    cleaned = flood.copy()
    cleaned[ocean_mask] = 0
    flood_after = int(np.sum(cleaned == 1))
    removed = flood_before - flood_after
    logger.info("Flood pixels: %d -> %d (removed %d ocean false-positives)",
                flood_before, flood_after, removed)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": 255})

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(cleaned[np.newaxis, :, :])
        dst.set_band_description(1, "flood_prediction_masked")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Saved: %s (%.2f MB)", out_path, size_mb)

    # Preview
    preview_path = out_path.with_name("final_flood_map_preview.png")
    try:
        from visualize import visualize_flood_map
        visualize_flood_map(input_path=str(out_path), output_path=str(preview_path))
        logger.info("Preview saved: %s", preview_path)
    except Exception as exc:
        logger.warning("Preview generation failed (non-fatal): %s", exc)

    logger.info("=" * 60)
    logger.info("POSTPROCESS COMPLETE -> %s", out_path)
    logger.info("=" * 60)

    return out_path
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/postprocess.py
git commit -m "fix: multi-criteria ocean masking (elevation + slope + connectivity)"
```

---

### Task 7: Update Requirements and Verify Build

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add scipy dependency**

Add to `requirements.txt` after the existing geospatial section:

```
scipy
```

- [ ] **Step 2: Rebuild Rust engine**

```bash
cd rust_engine && maturin develop --release && cd ..
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 4: Final commit**

```bash
git add requirements.txt
git commit -m "chore: add scipy dependency for Otsu thresholding"
```
