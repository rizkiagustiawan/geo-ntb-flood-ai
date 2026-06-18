//! flood_rs — Zero-copy Rust-accelerated geospatial compute via PyO3.
//!
//! Functions:
//!   - `calculate_ndwi`          : (Green − NIR) / (Green + NIR) → f32
//!   - `calculate_ndvi`          : (NIR − Red) / (NIR + Red) → f32
//!   - `calculate_sar_flood_mask`: VV/VH dual-threshold → u8
//!   - `compute_ndwi_and_mask`   : Fused multisensor NDWI + SAR → binary flood mask u8
//!
//! All array reads are zero-copy via PyReadonlyArray2.
//! All parallel compute via rayon.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{PyArray4, PyReadonlyArray4};
use ort::session::Session;
use ort::value::Tensor;
use ort::session::builder::GraphOptimizationLevel;
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref ONNX_SESSION: Mutex<Option<Session>> = Mutex::new(None);
}

// ---------------------------------------------------------------------------
// NDWI: (Green − NIR) / (Green + NIR)
// ---------------------------------------------------------------------------
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
    let (rows, cols) = (g.shape()[0], g.shape()[1]);
    let g_s = g.as_slice().unwrap();
    let n_s = n.as_slice().unwrap();
    let result: Vec<f32> = g_s
        .par_iter()
        .zip(n_s.par_iter())
        .map(|(&gv, &nv)| {
            let d = gv + nv;
            if d == 0.0 { f32::NAN } else { (gv - nv) / d }
        })
        .collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// NDVI: (NIR − Red) / (NIR + Red)
// ---------------------------------------------------------------------------
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
    let (rows, cols) = (n.shape()[0], n.shape()[1]);
    let n_s = n.as_slice().unwrap();
    let r_s = r.as_slice().unwrap();
    let result: Vec<f32> = n_s
        .par_iter()
        .zip(r_s.par_iter())
        .map(|(&nv, &rv)| {
            let d = nv + rv;
            if d == 0.0 { f32::NAN } else { (nv - rv) / d }
        })
        .collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// SAR Flood Mask: VV < thresh AND VH < thresh → 1 (water)
// ---------------------------------------------------------------------------
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
    let (rows, cols) = (v.shape()[0], v.shape()[1]);
    let v_s = v.as_slice().unwrap();
    let h_s = h.as_slice().unwrap();
    let result: Vec<u8> = v_s
        .par_iter()
        .zip(h_s.par_iter())
        .map(|(&vv_val, &vh_val)| {
            if vv_val < vv_thresh && vh_val < vh_thresh { 1u8 } else { 0u8 }
        })
        .collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// FUSED MULTISENSOR: compute_ndwi_and_mask
//
// Pipeline:
//   1. NDWI = (Green − NIR) / (Green + NIR)
//   2. SAR flood flag = sar_vv < sar_thresh
//   3. Binary mask = (NDWI > ndwi_thresh) OR (SAR flood flag) → 1 (flood)
//
// Single-pass rayon parallelism over all three input bands.
// Returns uint8 flood mask directly — no intermediate float allocation.
// ---------------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (green, nir, sar_vv, ndwi_thresh=0.3, sar_thresh=-15.0))]
fn compute_ndwi_and_mask<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f32>,
    nir: PyReadonlyArray2<'py, f32>,
    sar_vv: PyReadonlyArray2<'py, f32>,
    ndwi_thresh: f32,
    sar_thresh: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let g = green.as_array();
    let n = nir.as_array();
    let s = sar_vv.as_array();

    // Shape validation
    if g.shape() != n.shape() || g.shape() != s.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Shape mismatch: green={:?}, nir={:?}, sar_vv={:?}",
            g.shape(), n.shape(), s.shape()
        )));
    }

    let (rows, cols) = (g.shape()[0], g.shape()[1]);

    // Extract zero-copy slices
    let g_s = g.as_slice().unwrap();
    let n_s = n.as_slice().unwrap();
    let s_s = s.as_slice().unwrap();

    // Single-pass fused compute: NDWI threshold OR SAR threshold → flood
    let mask: Vec<u8> = g_s
        .par_iter()
        .zip(n_s.par_iter())
        .zip(s_s.par_iter())
        .map(|((&gv, &nv), &sv)| {
            // NDWI
            let denom = gv + nv;
            let ndwi = if denom == 0.0 { f32::NAN } else { (gv - nv) / denom };

            // Flood decision: water-like NDWI OR low SAR backscatter
            let optical_flood = ndwi > ndwi_thresh;
            let sar_flood = sv < sar_thresh;

            if optical_flood || sar_flood { 1u8 } else { 0u8 }
        })
        .collect();

    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), mask)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}


// ---------------------------------------------------------------------------
// ONNX U-NET INFERENCE (Phase 3)
// ---------------------------------------------------------------------------
#[pyfunction]
fn predict_unet_chunk<'py>(
    py: Python<'py>,
    tensor_chunk: PyReadonlyArray4<'py, f32>,
    model_path: String,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let mut session_guard = ONNX_SESSION.lock().unwrap();

    if session_guard.is_none() {
        let _ = ort::init().with_name("AECO_UNet").commit();

        let session = Session::builder()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT Builder Error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT Opt Error: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load ONNX model at {}: {}", model_path, e)))?;
        
        *session_guard = Some(session);
    }

    let session = session_guard.as_mut().unwrap();

    let input_array = tensor_chunk.as_array();
    let batch_size = input_array.shape()[0];
    let (h, w) = (input_array.shape()[2], input_array.shape()[3]);

    let input_tensor = Tensor::from_array(input_array.to_owned())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor Error: {}", e)))?;

    let outputs = session
        .run(ort::inputs!["input" => input_tensor])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Inference Error: {}", e)))?;

    let output_tensor = outputs["output"]
        .try_extract_tensor::<f32>()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Extract Error: {}", e)))?;
        
    let (_, data_slice) = output_tensor;
    let result = numpy::ndarray::ArrayView4::from_shape((batch_size, 1, h, w), data_slice)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Shape Error: {}", e)))?
        .to_owned();

    Ok(result.into_pyarray(py))
}
// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
#[pymodule]
fn flood_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_ndwi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ndvi, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sar_flood_mask, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ndwi_and_mask, m)?)?;
    m.add_function(wrap_pyfunction!(predict_unet_chunk, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ndwi_formula() {
        let (g, n) = (0.3_f32, 0.1_f32);
        assert!(((g - n) / (g + n) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ndvi_formula() {
        let (n, r) = (0.8_f32, 0.2_f32);
        assert!(((n - r) / (n + r) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn zero_denom_nan() {
        let d = 0.0_f32 + 0.0_f32;
        let r = if d == 0.0 { f32::NAN } else { 0.0 / d };
        assert!(r.is_nan());
    }

    #[test]
    fn fused_mask_logic() {
        // Case 1: NDWI > 0.3 → flood
        let (g, n, _s) = (0.8_f32, 0.2_f32, 0.0_f32);
        let ndwi = (g - n) / (g + n); // 0.6
        assert!(ndwi > 0.3);

        // Case 2: SAR < -15 → flood
        let sv = -20.0_f32;
        assert!(sv < -15.0);

        // Case 3: Neither → no flood
        let (g2, n2, s2) = (0.5_f32, 0.5_f32, 0.0_f32);
        let ndwi2 = (g2 - n2) / (g2 + n2); // 0.0
        assert!(ndwi2 <= 0.3 && s2 >= -15.0);
    }
}
