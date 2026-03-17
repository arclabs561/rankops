//! Composable fusion pipeline: normalize, fuse, rerank, evaluate.
//!
//! Builds a typed pipeline for post-retrieval operations. Retrieval happens
//! upstream (tantivy, Qdrant, etc.) and feeds ranked lists into the pipeline.
//!
//! # Example
//!
//! ```rust
//! use rankops::pipeline::Pipeline;
//! use rankops::{FusionMethod, Normalization};
//!
//! let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
//! let dense = vec![("d2", 0.9), ("d1", 0.7)];
//!
//! let result = Pipeline::new()
//!     .add_run("bm25", &bm25)
//!     .add_run("dense", &dense)
//!     .normalize(Normalization::MinMax)
//!     .fuse(FusionMethod::rrf())
//!     .top_k(10)
//!     .execute();
//!
//! assert!(!result.is_empty());
//! ```

use crate::{FusionMethod, Normalization, Qrels};
use std::hash::Hash;

/// A composable fusion pipeline.
///
/// Stages: add runs -> normalize (optional) -> fuse -> top_k (optional) -> execute.
/// Each stage returns a new pipeline, so the API is chainable.
#[derive(Debug, Clone)]
pub struct Pipeline<I: Clone + Eq + Hash> {
    runs: Vec<PipelineRun<I>>,
    normalization: Option<Normalization>,
    method: Option<FusionMethod>,
    top_k: Option<usize>,
}

/// A named retrieval run in the pipeline.
#[derive(Debug, Clone)]
struct PipelineRun<I> {
    #[allow(dead_code)]
    name: String,
    results: Vec<(I, f32)>,
}

impl<I: Clone + Eq + Hash> Default for Pipeline<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Clone + Eq + Hash> Pipeline<I> {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            normalization: None,
            method: None,
            top_k: None,
        }
    }

    /// Add a named retrieval run.
    #[must_use]
    pub fn add_run(mut self, name: &str, results: &[(I, f32)]) -> Self {
        self.runs.push(PipelineRun {
            name: name.to_string(),
            results: results.to_vec(),
        });
        self
    }

    /// Set normalization applied to each run before fusion.
    #[must_use]
    pub fn normalize(mut self, method: Normalization) -> Self {
        self.normalization = Some(method);
        self
    }

    /// Set the fusion method.
    #[must_use]
    pub fn fuse(mut self, method: FusionMethod) -> Self {
        self.method = Some(method);
        self
    }

    /// Limit output to top-k results.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Execute the pipeline and return fused results.
    ///
    /// Returns an empty list if no runs or no fusion method is set.
    #[must_use]
    pub fn execute(&self) -> Vec<(I, f32)> {
        if self.runs.is_empty() {
            return Vec::new();
        }

        // Step 1: Normalize each run if requested
        let normalized: Vec<Vec<(I, f32)>> = if let Some(norm) = self.normalization {
            self.runs
                .iter()
                .map(|r| crate::normalize_scores(&r.results, norm))
                .collect()
        } else {
            self.runs.iter().map(|r| r.results.clone()).collect()
        };

        // Step 2: Fuse
        let method = self.method.unwrap_or_default();
        let refs: Vec<&[(I, f32)]> = normalized.iter().map(|v| v.as_slice()).collect();

        let mut fused = if refs.len() == 1 {
            refs[0].to_vec()
        } else if refs.len() == 2 {
            method.fuse(refs[0], refs[1])
        } else {
            method.fuse_multi(&refs)
        };

        // Step 3: Top-k truncation
        if let Some(k) = self.top_k {
            fused.truncate(k);
        }

        fused
    }

    /// Execute and evaluate against relevance judgments.
    ///
    /// Returns the fused results and a metrics report.
    pub fn execute_and_evaluate(&self, qrels: &Qrels<I>) -> PipelineResult<I> {
        let fused = self.execute();

        let metrics = PipelineMetrics {
            ndcg_5: crate::ndcg_at_k(&fused, qrels, 5),
            ndcg_10: crate::ndcg_at_k(&fused, qrels, 10),
            map: crate::map(&fused, qrels),
            map_10: crate::map_at_k(&fused, qrels, 10),
            mrr: crate::mrr(&fused, qrels),
            precision_5: crate::precision_at_k(&fused, qrels, 5),
            recall_10: crate::recall_at_k(&fused, qrels, 10),
            hit_rate_1: crate::hit_rate(&fused, qrels, 1),
        };

        PipelineResult { fused, metrics }
    }
}

/// Result of a pipeline execution with evaluation.
#[derive(Debug, Clone)]
pub struct PipelineResult<I> {
    /// Fused ranked list.
    pub fused: Vec<(I, f32)>,
    /// Evaluation metrics.
    pub metrics: PipelineMetrics,
}

/// Standard evaluation metrics from a pipeline run.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineMetrics {
    /// NDCG@5.
    pub ndcg_5: f32,
    /// NDCG@10.
    pub ndcg_10: f32,
    /// Mean Average Precision (full).
    pub map: f32,
    /// MAP@10.
    pub map_10: f32,
    /// Mean Reciprocal Rank.
    pub mrr: f32,
    /// Precision@5.
    pub precision_5: f32,
    /// Recall@10.
    pub recall_10: f32,
    /// Hit Rate@1.
    pub hit_rate_1: f32,
}

impl std::fmt::Display for PipelineMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NDCG@5={:.4} NDCG@10={:.4} MAP={:.4} MAP@10={:.4} MRR={:.4} P@5={:.4} R@10={:.4} Hit@1={:.4}",
            self.ndcg_5, self.ndcg_10, self.map, self.map_10, self.mrr,
            self.precision_5, self.recall_10, self.hit_rate_1
        )
    }
}

/// Compare multiple fusion configurations on the same data.
///
/// Returns results sorted by the specified metric (descending).
///
/// # Example
///
/// ```rust
/// use rankops::pipeline::compare;
/// use rankops::{FusionMethod, OptimizeMetric};
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d1", 0.7)];
/// let qrels = std::collections::HashMap::from([("d1", 2), ("d2", 1)]);
///
/// let configs = vec![
///     ("RRF", FusionMethod::rrf()),
///     ("CombSUM", FusionMethod::CombSum),
///     ("Copeland", FusionMethod::Copeland),
/// ];
///
/// let results = compare(&[&bm25, &dense], &qrels, &configs, OptimizeMetric::Ndcg { k: 10 });
/// // Results sorted by NDCG@10 descending
/// println!("Best method: {}", &results[0].0);
/// ```
pub fn compare<I: Clone + Eq + Hash>(
    runs: &[&[(I, f32)]],
    qrels: &Qrels<I>,
    configs: &[(&str, FusionMethod)],
    sort_by: crate::OptimizeMetric,
) -> Vec<(String, PipelineMetrics)> {
    let mut results: Vec<(String, PipelineMetrics)> = configs
        .iter()
        .map(|(name, method)| {
            let fused = if runs.len() == 2 {
                method.fuse(runs[0], runs[1])
            } else {
                method.fuse_multi(runs)
            };

            let metrics = PipelineMetrics {
                ndcg_5: crate::ndcg_at_k(&fused, qrels, 5),
                ndcg_10: crate::ndcg_at_k(&fused, qrels, 10),
                map: crate::map(&fused, qrels),
                map_10: crate::map_at_k(&fused, qrels, 10),
                mrr: crate::mrr(&fused, qrels),
                precision_5: crate::precision_at_k(&fused, qrels, 5),
                recall_10: crate::recall_at_k(&fused, qrels, 10),
                hit_rate_1: crate::hit_rate(&fused, qrels, 1),
            };

            (name.to_string(), metrics)
        })
        .collect();

    // Sort by specified metric (descending)
    results.sort_by(|a, b| {
        let score_a = metric_value(&a.1, sort_by);
        let score_b = metric_value(&b.1, sort_by);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

fn metric_value(m: &PipelineMetrics, metric: crate::OptimizeMetric) -> f32 {
    match metric {
        crate::OptimizeMetric::Ndcg { k } => {
            if k <= 5 {
                m.ndcg_5
            } else {
                m.ndcg_10
            }
        }
        crate::OptimizeMetric::Mrr => m.mrr,
        crate::OptimizeMetric::Recall { .. } => m.recall_10,
        crate::OptimizeMetric::Precision { .. } => m.precision_5,
        crate::OptimizeMetric::Map => m.map,
        crate::OptimizeMetric::MapAtK { .. } => m.map_10,
        crate::OptimizeMetric::HitRate { .. } => m.hit_rate_1,
    }
}

/// Fuse results from multiple query variations across multiple retrievers.
///
/// Common RAG pattern: LLM generates N query variations, each runs against M
/// retrievers, producing N*M ranked lists. This function fuses all of them
/// into a single ranked list.
///
/// # Arguments
///
/// * `query_results` - Outer vec: per query variation. Inner vec: per retriever.
/// * `method` - Fusion method to combine all lists.
///
/// # Example
///
/// ```rust
/// use rankops::pipeline::fuse_multi_query;
/// use rankops::FusionMethod;
///
/// // 2 query variations, each with 2 retriever results
/// let q1_bm25 = vec![("d1", 0.9), ("d2", 0.8)];
/// let q1_dense = vec![("d2", 0.95), ("d3", 0.85)];
/// let q2_bm25 = vec![("d1", 0.85), ("d4", 0.7)];
/// let q2_dense = vec![("d3", 0.9), ("d1", 0.8)];
///
/// let all_results = vec![
///     vec![q1_bm25, q1_dense],
///     vec![q2_bm25, q2_dense],
/// ];
///
/// let fused = fuse_multi_query(&all_results, FusionMethod::rrf());
/// // d1 appears in multiple query/retriever combinations
/// ```
pub fn fuse_multi_query<I: Clone + Eq + Hash>(
    query_results: &[Vec<Vec<(I, f32)>>],
    method: FusionMethod,
) -> Vec<(I, f32)> {
    // Flatten all query x retriever results into a single list of runs
    let all_runs: Vec<&[(I, f32)]> = query_results
        .iter()
        .flat_map(|query| query.iter().map(|run| run.as_slice()))
        .collect();

    if all_runs.is_empty() {
        return Vec::new();
    }
    if all_runs.len() == 1 {
        return all_runs[0].to_vec();
    }
    if all_runs.len() == 2 {
        return method.fuse(all_runs[0], all_runs[1]);
    }

    method.fuse_multi(&all_runs)
}

/// Fuse results from multiple query variations with per-query normalization.
///
/// Like [`fuse_multi_query`] but normalizes each retriever's results before
/// flattening, which helps when different query variations produce different
/// score ranges.
pub fn fuse_multi_query_normalized<I: Clone + Eq + Hash>(
    query_results: &[Vec<Vec<(I, f32)>>],
    normalization: Normalization,
    method: FusionMethod,
) -> Vec<(I, f32)> {
    let normalized: Vec<Vec<Vec<(I, f32)>>> = query_results
        .iter()
        .map(|query| {
            query
                .iter()
                .map(|run| crate::normalize_scores(run, normalization))
                .collect()
        })
        .collect();

    fuse_multi_query(&normalized, method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FusionMethod;
    use std::collections::HashMap;

    fn bm25() -> Vec<(&'static str, f32)> {
        vec![("d1", 12.5), ("d2", 11.0), ("d3", 9.0)]
    }

    fn dense() -> Vec<(&'static str, f32)> {
        vec![("d2", 0.95), ("d3", 0.88), ("d4", 0.70)]
    }

    fn qrels() -> Qrels<&'static str> {
        HashMap::from([("d1", 2), ("d2", 1), ("d3", 1)])
    }

    #[test]
    fn pipeline_basic() {
        let result = Pipeline::new()
            .add_run("bm25", &bm25())
            .add_run("dense", &dense())
            .fuse(FusionMethod::rrf())
            .execute();

        assert!(!result.is_empty());
        // d2 appears in both lists
        assert_eq!(result[0].0, "d2");
    }

    #[test]
    fn pipeline_with_normalization() {
        let result = Pipeline::new()
            .add_run("bm25", &bm25())
            .add_run("dense", &dense())
            .normalize(Normalization::MinMax)
            .fuse(FusionMethod::CombSum)
            .execute();

        assert!(!result.is_empty());
    }

    #[test]
    fn pipeline_with_top_k() {
        let result = Pipeline::new()
            .add_run("bm25", &bm25())
            .add_run("dense", &dense())
            .fuse(FusionMethod::rrf())
            .top_k(2)
            .execute();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn pipeline_evaluate() {
        let pr = Pipeline::new()
            .add_run("bm25", &bm25())
            .add_run("dense", &dense())
            .fuse(FusionMethod::rrf())
            .execute_and_evaluate(&qrels());

        assert!(pr.metrics.ndcg_10 > 0.0);
        assert!(pr.metrics.mrr > 0.0);
        assert!(pr.metrics.map > 0.0);
    }

    #[test]
    fn pipeline_empty() {
        let result = Pipeline::<&str>::new().execute();
        assert!(result.is_empty());
    }

    #[test]
    fn pipeline_single_run() {
        let result = Pipeline::new()
            .add_run("only", &bm25())
            .fuse(FusionMethod::rrf())
            .execute();

        // Single run: returned as-is
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn pipeline_three_way() {
        let sparse = vec![("d3", 8.0), ("d1", 7.0), ("d5", 6.0)];
        let result = Pipeline::new()
            .add_run("bm25", &bm25())
            .add_run("dense", &dense())
            .add_run("sparse", &sparse)
            .fuse(FusionMethod::Copeland)
            .execute();

        assert!(!result.is_empty());
    }

    #[test]
    fn compare_methods() {
        let b = bm25();
        let d = dense();
        let runs: Vec<&[(&str, f32)]> = vec![&b, &d];
        let q = qrels();

        let configs = vec![
            ("RRF", FusionMethod::rrf()),
            ("CombSUM", FusionMethod::CombSum),
            ("Copeland", FusionMethod::Copeland),
        ];

        let results = compare(&runs, &q, &configs, crate::OptimizeMetric::Ndcg { k: 10 });

        assert_eq!(results.len(), 3);
        // Sorted descending by NDCG@10
        assert!(results[0].1.ndcg_10 >= results[1].1.ndcg_10);
        assert!(results[1].1.ndcg_10 >= results[2].1.ndcg_10);
    }

    #[test]
    fn multi_query_fusion_basic() {
        let q1_bm25 = vec![("d1", 0.9), ("d2", 0.8)];
        let q1_dense = vec![("d2", 0.95), ("d3", 0.85)];
        let q2_bm25 = vec![("d1", 0.85), ("d4", 0.7)];
        let q2_dense = vec![("d3", 0.9), ("d1", 0.8)];

        let all = vec![vec![q1_bm25, q1_dense], vec![q2_bm25, q2_dense]];

        let fused = fuse_multi_query(&all, FusionMethod::rrf());

        assert!(!fused.is_empty());
        // d1 appears in 3 of 4 lists, should rank high
        let d1_pos = fused.iter().position(|(id, _)| *id == "d1").unwrap();
        assert!(d1_pos < 2, "d1 should be in top 2");
    }

    #[test]
    fn multi_query_fusion_normalized() {
        let q1 = vec![vec![("d1", 100.0), ("d2", 50.0)]];
        let q2 = vec![vec![("d1", 0.9), ("d3", 0.5)]];
        let all = vec![q1[0].clone(), q2[0].clone()];
        let all_wrapped = vec![all];

        let fused =
            fuse_multi_query_normalized(&all_wrapped, Normalization::MinMax, FusionMethod::rrf());
        assert!(!fused.is_empty());
    }

    #[test]
    fn multi_query_empty() {
        let empty: Vec<Vec<Vec<(&str, f32)>>> = vec![];
        let fused = fuse_multi_query(&empty, FusionMethod::rrf());
        assert!(fused.is_empty());
    }

    #[test]
    fn pipeline_metrics_display() {
        let m = PipelineMetrics {
            ndcg_5: 0.9,
            ndcg_10: 0.85,
            map: 0.8,
            map_10: 0.75,
            mrr: 1.0,
            precision_5: 0.8,
            recall_10: 0.9,
            hit_rate_1: 1.0,
        };
        let s = m.to_string();
        assert!(s.contains("NDCG@5=0.9000"));
        assert!(s.contains("MRR=1.0000"));
    }
}
