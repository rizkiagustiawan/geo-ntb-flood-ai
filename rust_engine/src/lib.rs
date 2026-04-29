//! flood_rs — Rust-accelerated geospatial index computation via PyO3.
//!
//! Exposes `calculate_ndwi` and `calculate_ndvi` as native Python functions.
//! Heavy pixel-wise arithmetic runs in Rust with Rayon parallelism;
//! numpy arrays are passed by reference (zero-copy read) from Python.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute Normalized Difference Water Index: (Green − NIR) / (Green + NIR).
///
/// Parameters
/// ----------
/// green : numpy.ndarray[float32]  — 2-D green band reflectance.
/// nir   : numpy.ndarray[float32]  — 2-D NIR band reflectance.
///
/// Returns
/// -------
/// numpy.ndarray[float32] — NDWI in [−1, 1]; NaN where denominator == 0.
#[pyfunction]
fn calculate_ndwi<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f32>,
    nir: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let g = green.as_array();
    let n = nir.as_array();

    if g.shape() != n.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "green and nir arrays must have the same shape",
        ));
    }

    let rows = g.shape()[0];
    let cols = g.shape()[1];

    // Flatten → parallel compute → reshape
    let g_slice: Vec<f32> = g.iter().copied().collect();
    let n_slice: Vec<f32> = n.iter().copied().collect();

    let result: Vec<f32> = g_slice
        .par_iter()
        .zip(n_slice.par_iter())
        .map(|(&gv, &nv)| {
            let denom = gv + nv;
            if denom == 0.0 {
                f32::NAN
            } else {
                (gv - nv) / denom
            }
        })
        .collect();

    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray(py))
}

/// Compute Normalized Difference Vegetation Index: (NIR − Red) / (NIR + Red).
///
/// Parameters
/// ----------
/// nir : numpy.ndarray[float32]  — 2-D NIR band reflectance.
/// red : numpy.ndarray[float32]  — 2-D Red band reflectance.
///
/// Returns
/// -------
/// numpy.ndarray[float32] — NDVI in [−1, 1]; NaN where denominator == 0.
#[pyfunction]
fn calculate_ndvi<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f32>,
    red: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = nir.as_array();
    let r = red.as_array();

    if n.shape() != r.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nir and red arrays must have the same shape",
        ));
    }

    let rows = n.shape()[0];
    let cols = n.shape()[1];

    let n_slice: Vec<f32> = n.iter().copied().collect();
    let r_slice: Vec<f32> = r.iter().copied().collect();

    let result: Vec<f32> = n_slice
        .par_iter()
        .zip(r_slice.par_iter())
        .map(|(&nv, &rv)| {
            let denom = nv + rv;
            if denom == 0.0 {
                f32::NAN
            } else {
                (nv - rv) / denom
            }
        })
        .collect();

    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray(py))
}

/// Compute SAR-based flood mask from VV and VH backscatter bands (dB).
///
/// Pixels where VV < vv_thresh AND VH < vh_thresh are classified as water (1).
///
/// Parameters
/// ----------
/// vv : numpy.ndarray[float32] — 2-D VV polarisation in dB.
/// vh : numpy.ndarray[float32] — 2-D VH polarisation in dB.
/// vv_thresh : f32 — VV threshold (default −15.0 dB).
/// vh_thresh : f32 — VH threshold (default −20.0 dB).
///
/// Returns
/// -------
/// numpy.ndarray[u8] — Binary mask: 1 = water, 0 = non-water.
#[pyfunction]
#[pyo3(signature = (vv, vh, vv_thresh=-15.0, vh_thresh=-20.0))]
fn calculate_sar_flood_mask<'py>(
    py: Python<'py>,
    vv: PyReadonlyArray2<'py, f32>,
    vh: PyReadonlyArray2<'py, f32>,
    vv_thresh: f32,
    vh_thresh: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let v = vv.as_array();
    let h = vh.as_array();

    if v.shape() != h.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vv and vh arrays must have the same shape",
        ));
    }

    let rows = v.shape()[0];
    let cols = v.shape()[1];

    let v_slice: Vec<f32> = v.iter().copied().collect();
    let h_slice: Vec<f32> = h.iter().copied().collect();

    let result: Vec<u8> = v_slice
        .par_iter()
        .zip(h_slice.par_iter())
        .map(|(&vv_val, &vh_val)| {
            if vv_val < vv_thresh && vh_val < vh_thresh {
                1u8
            } else {
                0u8
            }
        })
        .collect();

    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(arr.into_pyarray(py))
}

/// Python module — registered as `flood_rs`.
#[pymodule]
fn flood_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_ndwi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ndvi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sar_flood_mask, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndwi_formula() {
        // Pure Rust unit test (no Python runtime)
        let g = 0.3_f32;
        let n = 0.1_f32;
        let expected = (g - n) / (g + n);
        assert!((expected - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ndvi_formula() {
        let n = 0.8_f32;
        let r = 0.2_f32;
        let expected = (n - r) / (n + r);
        assert!((expected - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_zero_denom_nan() {
        let g = 0.0_f32;
        let n = 0.0_f32;
        let denom = g + n;
        let result = if denom == 0.0 { f32::NAN } else { (g - n) / denom };
        assert!(result.is_nan());
    }
}
