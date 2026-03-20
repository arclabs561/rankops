//! Python bindings for rankops (Rust) using PyO3.
//!
//! Provides a Python API that mirrors the Rust API, enabling seamless
//! integration with Python RAG/search stacks.
//!
//! # Usage
//!
//! ```python
//! import rankops
//!
//! bm25 = [("d1", 12.5), ("d2", 11.0)]
//! dense = [("d2", 0.9), ("d3", 0.8)]
//!
//! fused = rankops.rrf(bm25, dense, k=60)
//! # [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
//! ```

// Note: allow(deprecated) needed for pyo3 0.24 compatibility
// TODO: Remove when upgrading to pyo3 0.25+ which uses IntoPyObject
// Impact: This suppresses deprecation warnings for pyo3's IntoPy trait methods.
// Action: Check pyo3 changelog when upgrading to 0.25+ to see if IntoPyObject migration is needed.
#![allow(deprecated)]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use ::rankops::explain::{
    combmnz_explain, combsum_explain, dbsf_explain, rrf_explain, ConsensusReport, Explanation,
    FusedResult, RetrieverId, RetrieverStats, SourceContribution,
};
use ::rankops::validate::{
    validate, validate_bounds, validate_finite_scores, validate_no_duplicates,
    validate_non_negative_scores, validate_sorted, ValidationResult,
};
use ::rankops::{
    additive_multi_task_with_config, borda_multi, borda_with_config, combmnz_multi,
    combmnz_with_config, combsum_multi, combsum_with_config, dbsf_multi, dbsf_with_config,
    isr_multi, isr_with_config, rrf_multi, rrf_with_config, standardized_multi,
    standardized_with_config, weighted, AdditiveMultiTaskConfig, FusionConfig, Normalization,
    RrfConfig, StandardizedConfig, WeightedConfig,
};

/// Helper to create (id, score) tuples for Python - PyO3 0.27 compatible
fn make_result_tuple<'py>(
    py: Python<'py>,
    id: String,
    score: f32,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            id.into_pyobject(py)?.into_any(),
            score.into_pyobject(py)?.into_any(),
        ],
    )
}

// TODO: Remove allow(deprecated) when upgrading to pyo3 0.25+ which uses IntoPyObject
/// Registers the module functions and classes.
/// Publicly exposed to allow other crates to include this module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Rank-based fusion
    m.add_function(wrap_pyfunction!(rrf_py, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(isr_py, m)?)?;
    m.add_function(wrap_pyfunction!(isr_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(borda_py, m)?)?;
    m.add_function(wrap_pyfunction!(borda_multi_py, m)?)?;

    // Score-based fusion
    m.add_function(wrap_pyfunction!(combsum_py, m)?)?;
    m.add_function(wrap_pyfunction!(combsum_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(combmnz_py, m)?)?;
    m.add_function(wrap_pyfunction!(combmnz_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_py, m)?)?;
    m.add_function(wrap_pyfunction!(dbsf_py, m)?)?;
    m.add_function(wrap_pyfunction!(dbsf_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(standardized_py, m)?)?;
    m.add_function(wrap_pyfunction!(standardized_multi_py, m)?)?;
    m.add_function(wrap_pyfunction!(additive_multi_task_py, m)?)?;

    // Explainability
    m.add_function(wrap_pyfunction!(rrf_explain_py, m)?)?;
    m.add_function(wrap_pyfunction!(combsum_explain_py, m)?)?;
    m.add_function(wrap_pyfunction!(combmnz_explain_py, m)?)?;
    m.add_function(wrap_pyfunction!(dbsf_explain_py, m)?)?;

    // Configuration classes
    m.add_class::<RrfConfigPy>()?;
    m.add_class::<FusionConfigPy>()?;
    m.add_class::<WeightedConfigPy>()?;
    m.add_class::<StandardizedConfigPy>()?;
    m.add_class::<AdditiveMultiTaskConfigPy>()?;

    // Explainability classes
    m.add_class::<FusedResultPy>()?;
    m.add_class::<ExplanationPy>()?;
    m.add_class::<SourceContributionPy>()?;
    m.add_class::<RetrieverIdPy>()?;
    m.add_class::<ConsensusReportPy>()?;
    m.add_class::<RetrieverStatsPy>()?;

    // Validation functions
    m.add_function(wrap_pyfunction!(validate_sorted_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_no_duplicates_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_finite_scores_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_non_negative_scores_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_bounds_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_py, m)?)?;
    m.add_class::<ValidationResultPy>()?;

    Ok(())
}

/// Python module for rank fusion.
#[allow(deprecated)]
#[pymodule]
fn rankops(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(py, m)
}

/// RRF fusion for two ranked lists.
///
/// # Arguments
/// * `results_a`: List of (id, score) tuples from first retriever
/// * `results_b`: List of (id, score) tuples from second retriever
/// * `k`: Smoothing constant (default: 60)
/// * `top_k`: Maximum number of results to return (default: None = all)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[allow(deprecated)]
#[pyfunction(name = "rrf")]
#[pyo3(signature = (results_a, results_b, k = 60, top_k = None))]
fn rrf_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    k: u32,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    // Convert Python lists to Rust Vec<(String, f32)>
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;

    // Call Rust function
    let mut config = RrfConfig::new(k);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = rrf_with_config(&a, &b, config);

    // Convert back to Python list
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// RRF fusion for multiple ranked lists.
///
/// # Arguments
/// * `lists`: List of lists, each containing (id, score) tuples
/// * `k`: Smoothing constant (default: 60)
/// * `top_k`: Maximum number of results to return (default: None = all)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[allow(deprecated)]
#[pyfunction(name = "rrf_multi")]
#[pyo3(signature = (lists, k = 60, top_k = None))]
fn rrf_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    k: u32,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    // Convert Python lists to Rust Vec<Vec<(String, f32)>>
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }

    // Convert to slice of slices for Rust API
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();

    // Call Rust function
    let mut config = RrfConfig::new(k);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = rrf_multi(&slices, config);

    // Convert back to Python list
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Helper to convert Python list of (id, score) tuples to Rust Vec.
fn py_list_to_ranked(py_list: &Bound<'_, PyList>) -> PyResult<Vec<(String, f32)>> {
    let mut result = Vec::new();
    for item in py_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each item must be a (id, score) tuple",
            ));
        }
        let id = tuple.get_item(0)?.extract::<String>()?;
        let score = tuple.get_item(1)?.extract::<f32>()?;
        result.push((id, score));
    }
    Ok(result)
}

/// ISR (Inverse Square Rank) fusion for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "isr")]
#[pyo3(signature = (results_a, results_b, k = 1, top_k = None))]
fn isr_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    k: u32,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = RrfConfig::new(k);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = isr_with_config(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// ISR fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "isr_multi")]
#[pyo3(signature = (lists, k = 1, top_k = None))]
fn isr_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    k: u32,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = RrfConfig::new(k);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = isr_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// CombSUM fusion for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "combsum")]
#[pyo3(signature = (results_a, results_b, top_k = None))]
fn combsum_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = combsum_with_config(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// CombSUM fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "combsum_multi")]
#[pyo3(signature = (lists, top_k = None))]
fn combsum_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = combsum_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// CombMNZ fusion for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "combmnz")]
#[pyo3(signature = (results_a, results_b, top_k = None))]
fn combmnz_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = combmnz_with_config(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// CombMNZ fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "combmnz_multi")]
#[pyo3(signature = (lists, top_k = None))]
fn combmnz_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = combmnz_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Borda count fusion for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "borda")]
#[pyo3(signature = (results_a, results_b, top_k = None))]
fn borda_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = borda_with_config(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Borda count fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "borda_multi")]
#[pyo3(signature = (lists, top_k = None))]
fn borda_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = borda_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Weighted fusion for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "weighted")]
#[pyo3(signature = (results_a, results_b, weight_a, weight_b, normalize = true, top_k = None))]
fn weighted_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    weight_a: f32,
    weight_b: f32,
    normalize: bool,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if !weight_a.is_finite() || !weight_b.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must be finite numbers",
        ));
    }
    if (weight_a.abs() < 1e-9) && (weight_b.abs() < 1e-9) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights cannot both be zero",
        ));
    }
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = WeightedConfig::new(weight_a, weight_b).with_normalize(normalize);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = weighted(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// DBSF (Distribution-Based Score Fusion) for two ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "dbsf")]
#[pyo3(signature = (results_a, results_b, top_k = None))]
fn dbsf_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = dbsf_with_config(&a, &b, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// DBSF fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "dbsf_multi")]
#[pyo3(signature = (lists, top_k = None))]
fn dbsf_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = dbsf_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Standardized fusion for multiple ranked lists.
#[allow(deprecated)]
#[pyfunction(name = "standardized_multi")]
#[pyo3(signature = (lists, clip_range = (-3.0, 3.0), top_k = None))]
fn standardized_multi_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    clip_range: (f32, f32),
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let config = StandardizedConfig::new(clip_range);
    let config = if let Some(k) = top_k {
        config.with_top_k(k)
    } else {
        config
    };
    let fused = standardized_multi(&slices, config);
    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Standardized fusion (ERANK-style) for two ranked lists.
///
/// Uses z-score normalization with configurable clipping to handle different score distributions.
///
/// # Arguments
/// * `results_a`: List of (id, score) tuples from first retriever
/// * `results_b`: List of (id, score) tuples from second retriever
/// * `clip_range`: Tuple of (min, max) for clipping z-scores (default: (-3.0, 3.0))
/// * `top_k`: Maximum number of results to return (default: None = all)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[allow(deprecated)]
#[pyfunction(name = "standardized")]
#[pyo3(signature = (results_a, results_b, clip_range = (-3.0, 3.0), top_k = None))]
fn standardized_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    clip_range: (f32, f32),
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;

    let config = StandardizedConfig::new(clip_range);
    let config = if let Some(k) = top_k {
        config.with_top_k(k)
    } else {
        config
    };
    let fused = standardized_with_config(&a, &b, config);

    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Additive multi-task fusion (ResFlow-style) for two ranked lists.
///
/// Combines scores from multiple tasks with configurable weights and normalization.
/// Optimized for e-commerce ranking (e.g., CTR + CTCVR).
///
/// # Arguments
/// * `results_a`: List of (id, score) tuples from first task
/// * `results_b`: List of (id, score) tuples from second task
/// * `weights`: Tuple of (weight_a, weight_b) (default: (1.0, 1.0))
/// * `normalization`: Normalization method as string: "zscore", "minmax", "sum", "rank", "none" (default: "minmax")
/// * `top_k`: Maximum number of results to return (default: None = all)
///
/// # Returns
/// List of (id, score) tuples sorted by fused score (descending)
#[allow(deprecated)]
#[pyfunction(name = "additive_multi_task")]
#[pyo3(signature = (results_a, results_b, weights = (1.0, 1.0), normalization = "minmax", top_k = None))]
fn additive_multi_task_py(
    py: Python<'_>,
    results_a: &Bound<'_, PyList>,
    results_b: &Bound<'_, PyList>,
    weights: (f32, f32),
    normalization: &str,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let a: Vec<(String, f32)> = py_list_to_ranked(results_a)?;
    let b: Vec<(String, f32)> = py_list_to_ranked(results_b)?;

    let norm = match normalization.to_lowercase().as_str() {
        "zscore" => Normalization::ZScore,
        "minmax" => Normalization::MinMax,
        "sum" => Normalization::Sum,
        "rank" => Normalization::Rank,
        "none" => Normalization::None,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "normalization must be one of: zscore, minmax, sum, rank, none",
            ))
        }
    };

    let mut config = AdditiveMultiTaskConfig::new(weights).with_normalization(norm);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let fused = additive_multi_task_with_config(&a, &b, config);

    let result = PyList::empty(py);
    for (id, score) in fused {
        let tuple = make_result_tuple(py, id, score)?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// RRF with explainability.
#[allow(deprecated)]
#[pyfunction(name = "rrf_explain")]
#[pyo3(signature = (lists, retriever_ids, k = 60, top_k = None))]
fn rrf_explain_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    retriever_ids: &Bound<'_, PyList>,
    k: u32,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    if k == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "k must be >= 1 to avoid division by zero",
        ));
    }
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let mut ids = Vec::new();
    for id in retriever_ids.iter() {
        let id_str = id.extract::<String>()?;
        ids.push(RetrieverId::new(id_str));
    }
    if rust_lists.len() != ids.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lists and retriever_ids must have the same length",
        ));
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = RrfConfig::new(k);
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let explained = rrf_explain(&slices, &ids, config);
    let result = PyList::empty(py);
    for fused_result in explained {
        let py_result = FusedResultPy::from(fused_result);
        result.append(Bound::new(py, py_result)?)?;
    }
    Ok(result.into())
}

/// CombSUM with explainability.
#[allow(deprecated)]
#[pyfunction(name = "combsum_explain")]
#[pyo3(signature = (lists, retriever_ids, top_k = None))]
fn combsum_explain_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    retriever_ids: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let mut ids = Vec::new();
    for id in retriever_ids.iter() {
        let id_str = id.extract::<String>()?;
        ids.push(RetrieverId::new(id_str));
    }
    if rust_lists.len() != ids.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lists and retriever_ids must have the same length",
        ));
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let explained = combsum_explain(&slices, &ids, config);
    let result = PyList::empty(py);
    for fused_result in explained {
        let py_result = FusedResultPy::from(fused_result);
        result.append(Bound::new(py, py_result)?)?;
    }
    Ok(result.into())
}

/// CombMNZ with explainability.
#[allow(deprecated)]
#[pyfunction(name = "combmnz_explain")]
#[pyo3(signature = (lists, retriever_ids, top_k = None))]
fn combmnz_explain_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    retriever_ids: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let mut ids = Vec::new();
    for id in retriever_ids.iter() {
        let id_str = id.extract::<String>()?;
        ids.push(RetrieverId::new(id_str));
    }
    if rust_lists.len() != ids.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lists and retriever_ids must have the same length",
        ));
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let explained = combmnz_explain(&slices, &ids, config);
    let result = PyList::empty(py);
    for fused_result in explained {
        let py_result = FusedResultPy::from(fused_result);
        result.append(Bound::new(py, py_result)?)?;
    }
    Ok(result.into())
}

/// DBSF with explainability.
#[allow(deprecated)]
#[pyfunction(name = "dbsf_explain")]
#[pyo3(signature = (lists, retriever_ids, top_k = None))]
fn dbsf_explain_py(
    py: Python<'_>,
    lists: &Bound<'_, PyList>,
    retriever_ids: &Bound<'_, PyList>,
    top_k: Option<usize>,
) -> PyResult<Py<PyList>> {
    let mut rust_lists = Vec::new();
    for list in lists.iter() {
        let py_list = list.downcast::<PyList>()?;
        rust_lists.push(py_list_to_ranked(py_list)?);
    }
    let mut ids = Vec::new();
    for id in retriever_ids.iter() {
        let id_str = id.extract::<String>()?;
        ids.push(RetrieverId::new(id_str));
    }
    if rust_lists.len() != ids.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lists and retriever_ids must have the same length",
        ));
    }
    let slices: Vec<&[(String, f32)]> = rust_lists.iter().map(|v| v.as_slice()).collect();
    let mut config = FusionConfig::default();
    if let Some(k) = top_k {
        config = config.with_top_k(k);
    }
    let explained = dbsf_explain(&slices, &ids, config);
    let result = PyList::empty(py);
    for fused_result in explained {
        let py_result = FusedResultPy::from(fused_result);
        result.append(Bound::new(py, py_result)?)?;
    }
    Ok(result.into())
}

/// Python wrapper for RrfConfig.
#[pyclass(name = "RrfConfig")]
struct RrfConfigPy {
    inner: RrfConfig,
}

#[pymethods]
impl RrfConfigPy {
    #[new]
    fn new(k: u32) -> Self {
        Self {
            inner: RrfConfig::new(k),
        }
    }

    #[getter]
    fn k(&self) -> u32 {
        self.inner.k
    }

    fn with_k(&self, k: u32) -> Self {
        Self {
            inner: self.inner.with_k(k),
        }
    }
}

/// Python wrapper for FusionConfig.
#[pyclass(name = "FusionConfig")]
struct FusionConfigPy {
    inner: FusionConfig,
}

#[pymethods]
impl FusionConfigPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: FusionConfig::default(),
        }
    }

    fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            inner: self.inner.with_top_k(top_k),
        }
    }
}

/// Python wrapper for WeightedConfig.
#[pyclass(name = "WeightedConfig")]
struct WeightedConfigPy {
    inner: WeightedConfig,
}

#[pymethods]
impl WeightedConfigPy {
    #[new]
    fn new(weight_a: f32, weight_b: f32) -> Self {
        Self {
            inner: WeightedConfig::new(weight_a, weight_b),
        }
    }

    #[getter]
    fn weight_a(&self) -> f32 {
        self.inner.weight_a
    }

    #[getter]
    fn weight_b(&self) -> f32 {
        self.inner.weight_b
    }

    fn with_normalize(&self, normalize: bool) -> Self {
        Self {
            inner: self.inner.with_normalize(normalize),
        }
    }

    fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            inner: self.inner.with_top_k(top_k),
        }
    }
}

/// Python wrapper for StandardizedConfig.
#[pyclass(name = "StandardizedConfig")]
struct StandardizedConfigPy {
    inner: StandardizedConfig,
}

#[pymethods]
impl StandardizedConfigPy {
    #[new]
    #[pyo3(signature = (clip_range = (-3.0, 3.0)))]
    fn new(clip_range: (f32, f32)) -> Self {
        Self {
            inner: StandardizedConfig::new(clip_range),
        }
    }

    #[getter]
    fn clip_range(&self) -> (f32, f32) {
        self.inner.clip_range
    }

    fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            inner: self.inner.with_top_k(top_k),
        }
    }
}

/// Python wrapper for AdditiveMultiTaskConfig.
#[pyclass(name = "AdditiveMultiTaskConfig")]
struct AdditiveMultiTaskConfigPy {
    inner: AdditiveMultiTaskConfig,
}

#[pymethods]
impl AdditiveMultiTaskConfigPy {
    #[new]
    #[pyo3(signature = (weights = (1.0, 1.0), normalization = "minmax"))]
    fn new(weights: (f32, f32), normalization: &str) -> PyResult<Self> {
        let norm = match normalization.to_lowercase().as_str() {
            "zscore" => Normalization::ZScore,
            "minmax" => Normalization::MinMax,
            "sum" => Normalization::Sum,
            "rank" => Normalization::Rank,
            "none" => Normalization::None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "normalization must be one of: zscore, minmax, sum, rank, none",
                ))
            }
        };
        Ok(Self {
            inner: AdditiveMultiTaskConfig::new(weights).with_normalization(norm),
        })
    }

    #[getter]
    fn weights(&self) -> (f32, f32) {
        self.inner.weights
    }

    fn with_normalization(&self, normalization: &str) -> PyResult<Self> {
        let norm = match normalization.to_lowercase().as_str() {
            "zscore" => Normalization::ZScore,
            "minmax" => Normalization::MinMax,
            "sum" => Normalization::Sum,
            "rank" => Normalization::Rank,
            "none" => Normalization::None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "normalization must be one of: zscore, minmax, sum, rank, none",
                ))
            }
        };
        Ok(Self {
            inner: self.inner.with_normalization(norm),
        })
    }

    fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            inner: self.inner.with_top_k(top_k),
        }
    }
}

/// Python wrapper for FusedResult.
#[pyclass(name = "FusedResult")]
#[derive(Clone)]
struct FusedResultPy {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    explanation: ExplanationPy,
}

impl From<FusedResult<String>> for FusedResultPy {
    fn from(result: FusedResult<String>) -> Self {
        Self {
            id: result.id,
            score: result.score,
            rank: result.rank,
            explanation: ExplanationPy::from(result.explanation),
        }
    }
}

/// Python wrapper for Explanation.
#[pyclass(name = "Explanation")]
#[derive(Clone)]
struct ExplanationPy {
    #[pyo3(get)]
    sources: Vec<SourceContributionPy>,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    consensus_score: f32,
}

impl From<Explanation> for ExplanationPy {
    fn from(expl: Explanation) -> Self {
        Self {
            sources: expl
                .sources
                .into_iter()
                .map(SourceContributionPy::from)
                .collect(),
            method: expl.method.to_string(),
            consensus_score: expl.consensus_score,
        }
    }
}

/// Python wrapper for SourceContribution.
#[pyclass(name = "SourceContribution")]
#[derive(Clone)]
struct SourceContributionPy {
    #[pyo3(get)]
    retriever_id: String,
    #[pyo3(get)]
    original_rank: Option<usize>,
    #[pyo3(get)]
    original_score: Option<f32>,
    #[pyo3(get)]
    normalized_score: Option<f32>,
    #[pyo3(get)]
    contribution: f32,
}

impl From<SourceContribution> for SourceContributionPy {
    fn from(contrib: SourceContribution) -> Self {
        Self {
            retriever_id: contrib.retriever_id,
            original_rank: contrib.original_rank,
            original_score: contrib.original_score,
            normalized_score: contrib.normalized_score,
            contribution: contrib.contribution,
        }
    }
}

/// Python wrapper for RetrieverId.
#[pyclass(name = "RetrieverId")]
struct RetrieverIdPy {
    inner: RetrieverId,
}

#[pymethods]
impl RetrieverIdPy {
    #[new]
    fn new(id: String) -> Self {
        Self {
            inner: RetrieverId::new(id),
        }
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.as_str().to_string()
    }
}

/// Python wrapper for ConsensusReport.
#[pyclass(name = "ConsensusReport")]
#[derive(Clone)]
struct ConsensusReportPy {
    #[pyo3(get)]
    high_consensus: Vec<String>,
    #[pyo3(get)]
    single_source: Vec<String>,
    #[pyo3(get)]
    rank_disagreement: Vec<(String, Vec<(String, usize)>)>,
}

impl From<ConsensusReport<String>> for ConsensusReportPy {
    fn from(report: ConsensusReport<String>) -> Self {
        Self {
            high_consensus: report.high_consensus,
            single_source: report.single_source,
            rank_disagreement: report.rank_disagreement,
        }
    }
}

/// Python wrapper for RetrieverStats.
#[pyclass(name = "RetrieverStats")]
#[derive(Clone)]
struct RetrieverStatsPy {
    #[pyo3(get)]
    top_k_count: usize,
    #[pyo3(get)]
    avg_contribution: f32,
    #[pyo3(get)]
    unique_docs: usize,
}

impl From<RetrieverStats> for RetrieverStatsPy {
    fn from(stats: RetrieverStats) -> Self {
        Self {
            top_k_count: stats.top_k_count,
            avg_contribution: stats.avg_contribution,
            unique_docs: stats.unique_docs,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that fusion results are sorted by score (descending).
#[pyfunction(name = "validate_sorted")]
fn validate_sorted_py(results: &Bound<'_, PyList>) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate_sorted(&ranked);
    Ok(ValidationResultPy::from(result))
}

/// Validate that fusion results contain no duplicate document IDs.
#[pyfunction(name = "validate_no_duplicates")]
fn validate_no_duplicates_py(results: &Bound<'_, PyList>) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate_no_duplicates(&ranked);
    Ok(ValidationResultPy::from(result))
}

/// Validate that all scores are finite (not NaN or Infinity).
#[pyfunction(name = "validate_finite_scores")]
fn validate_finite_scores_py(results: &Bound<'_, PyList>) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate_finite_scores(&ranked);
    Ok(ValidationResultPy::from(result))
}

/// Validate that all scores are non-negative (warning only).
#[pyfunction(name = "validate_non_negative_scores")]
fn validate_non_negative_scores_py(results: &Bound<'_, PyList>) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate_non_negative_scores(&ranked);
    Ok(ValidationResultPy::from(result))
}

/// Validate that results are within expected bounds.
#[pyfunction(name = "validate_bounds")]
#[pyo3(signature = (results, max_results = None))]
fn validate_bounds_py(
    results: &Bound<'_, PyList>,
    max_results: Option<usize>,
) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate_bounds(&ranked, max_results);
    Ok(ValidationResultPy::from(result))
}

/// Comprehensive validation of fusion results.
#[pyfunction(name = "validate")]
#[pyo3(signature = (results, check_non_negative = false, max_results = None))]
fn validate_py(
    results: &Bound<'_, PyList>,
    check_non_negative: bool,
    max_results: Option<usize>,
) -> PyResult<ValidationResultPy> {
    let ranked: Vec<(String, f32)> = py_list_to_ranked(results)?;
    let result = validate(&ranked, check_non_negative, max_results);
    Ok(ValidationResultPy::from(result))
}

/// Python wrapper for ValidationResult.
#[pyclass(name = "ValidationResult")]
#[derive(Clone)]
struct ValidationResultPy {
    #[pyo3(get)]
    is_valid: bool,
    #[pyo3(get)]
    errors: Vec<String>,
    #[pyo3(get)]
    warnings: Vec<String>,
}

impl From<ValidationResult> for ValidationResultPy {
    fn from(result: ValidationResult) -> Self {
        Self {
            is_valid: result.is_valid,
            errors: result.errors,
            warnings: result.warnings,
        }
    }
}
