//! WebAssembly bindings for `rankops`.
//!
//! This crate exists to keep `rankops` itself small and dependency-light.

use wasm_bindgen::prelude::*;

use rankops::{
    additive_multi_task_with_config, standardized_with_config, AdditiveMultiTaskConfig,
    FusionConfig, Normalization, RrfConfig, StandardizedConfig, WeightedConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Helper to convert JS array of [id, score] pairs to Vec<(String, f32)>.
fn js_to_results(js: &JsValue) -> Result<Vec<(String, f32)>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    let len = array.length() as usize;
    let mut results = Vec::with_capacity(len);

    for (idx, item) in array.iter().enumerate() {
        let pair = item.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            JsValue::from_str(&format!("Expected [id, score] pair at index {}", idx))
        })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, score] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let score_val = pair.get(1).as_f64().ok_or_else(|| {
            JsValue::from_str(&format!("score must be a number at index {}", idx))
        })?;

        if !score_val.is_finite() {
            return Err(JsValue::from_str(&format!(
                "score must be a finite number at index {}, got {}",
                idx, score_val
            )));
        }

        results.push((id, score_val as f32));
    }
    Ok(results)
}

/// Helper to convert JS array of lists to Vec<Vec<(String, f32)>>.
fn js_to_multi(js: &JsValue) -> Result<Vec<Vec<(String, f32)>>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    let mut lists = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let list = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str(&format!("Expected list at index {}", idx)))?;
        lists.push(js_to_results(&list.into())?);
    }
    Ok(lists)
}

/// Helper to convert JS object {id: relevance} to qrels HashMap.
fn js_to_qrels(js: &JsValue) -> Result<std::collections::HashMap<String, u32>, JsValue> {
    use wasm_bindgen::JsCast;

    let obj = js
        .dyn_ref::<js_sys::Object>()
        .ok_or_else(|| JsValue::from_str("Expected object {id: relevance}"))?;

    let mut qrels = std::collections::HashMap::new();
    let entries = js_sys::Object::entries(obj);
    for entry in entries.iter() {
        let pair = entry
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| JsValue::from_str("Expected [key, value] entry"))?;
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str("qrels key must be string"))?;
        let rel =
            pair.get(1)
                .as_f64()
                .ok_or_else(|| JsValue::from_str("qrels value must be number"))? as u32;
        qrels.insert(id, rel);
    }
    Ok(qrels)
}

fn results_to_js(results: &[(String, f32)]) -> JsValue {
    let array = js_sys::Array::new();
    for (id, score) in results {
        let pair = js_sys::Array::new();
        pair.push(&JsValue::from_str(id));
        pair.push(&JsValue::from_f64(*score as f64));
        array.push(&pair);
    }
    array.into()
}

fn parse_normalization(s: &str) -> Result<Normalization, JsValue> {
    match s.to_lowercase().as_str() {
        "none" => Ok(Normalization::None),
        "minmax" => Ok(Normalization::MinMax),
        "zscore" => Ok(Normalization::ZScore),
        "quantile" => Ok(Normalization::Quantile),
        "sigmoid" => Ok(Normalization::Sigmoid),
        "sum" => Ok(Normalization::Sum),
        "rank" => Ok(Normalization::Rank),
        other => Err(JsValue::from_str(&format!(
            "unknown normalization: {other} (expected: none|minmax|zscore|quantile|sigmoid|sum|rank)"
        ))),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-list fusion
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn rrf(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str("k must be >= 1"));
    }

    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&rankops::rrf_with_config(&a, &b, config)))
}

#[wasm_bindgen]
pub fn isr(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(1);
    if k_val == 0 {
        return Err(JsValue::from_str("k must be >= 1"));
    }

    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&rankops::isr_with_config(&a, &b, config)))
}

/// Macro to generate simple two-list fusion bindings.
macro_rules! two_list_fusion {
    ($name:ident, $fn:path) => {
        #[wasm_bindgen]
        pub fn $name(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
            let a = js_to_results(results_a)?;
            let b = js_to_results(results_b)?;
            Ok(results_to_js(&$fn(&a, &b)))
        }
    };
}

two_list_fusion!(combsum, rankops::combsum);
two_list_fusion!(combmnz, rankops::combmnz);
two_list_fusion!(combmax, rankops::combmax);
two_list_fusion!(combmin, rankops::combmin);
two_list_fusion!(combmed, rankops::combmed);
two_list_fusion!(combanz, rankops::combanz);
two_list_fusion!(borda, rankops::borda);
two_list_fusion!(condorcet, rankops::condorcet);
two_list_fusion!(copeland, rankops::copeland);
two_list_fusion!(median_rank, rankops::median_rank);
two_list_fusion!(dbsf, rankops::dbsf);
two_list_fusion!(rbc, rankops::rbc);

#[wasm_bindgen]
pub fn weighted(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: f32,
    weight_b: f32,
    normalize: Option<bool>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    if !weight_a.is_finite() || !weight_b.is_finite() {
        return Err(JsValue::from_str("weights must be finite"));
    }

    let normalize = normalize.unwrap_or(true);
    let config = WeightedConfig {
        weight_a,
        weight_b,
        normalize,
        top_k,
    };
    Ok(results_to_js(&rankops::weighted(&a, &b, config)))
}

#[wasm_bindgen]
pub fn standardized(
    results_a: &JsValue,
    results_b: &JsValue,
    clip_min: Option<f32>,
    clip_max: Option<f32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let config = StandardizedConfig {
        clip_range: (clip_min.unwrap_or(-3.0), clip_max.unwrap_or(3.0)),
        top_k,
    };
    Ok(results_to_js(&standardized_with_config(&a, &b, config)))
}

#[wasm_bindgen]
pub fn additive_multi_task(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: Option<f32>,
    weight_b: Option<f32>,
    normalization: Option<String>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let norm = parse_normalization(&normalization.unwrap_or_else(|| "zscore".to_string()))?;

    let config = AdditiveMultiTaskConfig {
        weights: (weight_a.unwrap_or(1.0), weight_b.unwrap_or(1.0)),
        normalization: norm,
        top_k,
    };

    Ok(results_to_js(&additive_multi_task_with_config(
        &a, &b, config,
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-list fusion
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn rrf_multi(
    lists: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let rust_lists = js_to_multi(lists)?;
    if rust_lists.is_empty() {
        return Ok(js_sys::Array::new().into());
    }

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str("k must be >= 1"));
    }
    let config = RrfConfig { k: k_val, top_k };
    Ok(results_to_js(&rankops::rrf_multi(&rust_lists, config)))
}

/// Macro to generate multi-list fusion bindings.
macro_rules! multi_list_fusion {
    ($name:ident, $fn:path) => {
        #[wasm_bindgen]
        pub fn $name(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
            let rust_lists = js_to_multi(lists)?;
            if rust_lists.is_empty() {
                return Ok(js_sys::Array::new().into());
            }
            let config = FusionConfig { top_k };
            Ok(results_to_js(&$fn(&rust_lists, config)))
        }
    };
}

multi_list_fusion!(combsum_multi, rankops::combsum_multi);
multi_list_fusion!(combmnz_multi, rankops::combmnz_multi);
multi_list_fusion!(combmax_multi, rankops::combmax_multi);
multi_list_fusion!(combmin_multi, rankops::combmin_multi);
multi_list_fusion!(combmed_multi, rankops::combmed_multi);
multi_list_fusion!(combanz_multi, rankops::combanz_multi);
multi_list_fusion!(borda_multi, rankops::borda_multi);
multi_list_fusion!(condorcet_multi, rankops::condorcet_multi);
multi_list_fusion!(copeland_multi, rankops::copeland_multi);
multi_list_fusion!(median_rank_multi, rankops::median_rank_multi);
multi_list_fusion!(dbsf_multi, rankops::dbsf_multi);

// ─────────────────────────────────────────────────────────────────────────────
// Normalization
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn normalize_scores(results: &JsValue, method: &str) -> Result<JsValue, JsValue> {
    let r = js_to_results(results)?;
    let norm = parse_normalization(method)?;
    Ok(results_to_js(&rankops::normalize_scores(&r, norm)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn ndcg_at_k(results: &JsValue, qrels: &JsValue, k: usize) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::ndcg_at_k(&r, &q, k) as f64)
}

#[wasm_bindgen]
pub fn map(results: &JsValue, qrels: &JsValue) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::map(&r, &q) as f64)
}

#[wasm_bindgen]
pub fn map_at_k(results: &JsValue, qrels: &JsValue, k: usize) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::map_at_k(&r, &q, k) as f64)
}

#[wasm_bindgen]
pub fn mrr(results: &JsValue, qrels: &JsValue) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::mrr(&r, &q) as f64)
}

#[wasm_bindgen]
pub fn precision_at_k(results: &JsValue, qrels: &JsValue, k: usize) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::precision_at_k(&r, &q, k) as f64)
}

#[wasm_bindgen]
pub fn recall_at_k(results: &JsValue, qrels: &JsValue, k: usize) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::recall_at_k(&r, &q, k) as f64)
}

#[wasm_bindgen]
pub fn hit_rate(results: &JsValue, qrels: &JsValue, k: usize) -> Result<f64, JsValue> {
    let r = js_to_results(results)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::hit_rate(&r, &q, k) as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn overlap_ratio(results_a: &JsValue, results_b: &JsValue) -> Result<f64, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(rankops::diagnostics::overlap_ratio(&a, &b) as f64)
}

#[wasm_bindgen]
pub fn rank_correlation(results_a: &JsValue, results_b: &JsValue) -> Result<f64, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    Ok(rankops::diagnostics::rank_correlation(&a, &b) as f64)
}

#[wasm_bindgen]
pub fn complementarity(
    results_a: &JsValue,
    results_b: &JsValue,
    qrels: &JsValue,
) -> Result<f64, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    let q = js_to_qrels(qrels)?;
    Ok(rankops::diagnostics::complementarity(&a, &b, &q) as f64)
}
