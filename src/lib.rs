#![warn(missing_docs)]
//! Operations on ranked lists: fuse multiple retrievers, then rerank.
//!
//! Pairs with **rankfns** (scoring kernels). Combine results from multiple retrievers
//! (BM25, dense, sparse) and rerank with MaxSim (ColBERT), diversity (MMR/DPP), or Matryoshka.
//!
//! ```rust
//! use rankops::rrf;
//!
//! let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
//! let dense = vec![("d2", 0.9), ("d3", 0.8)];
//! let fused = rrf(&bm25, &dense);
//! // d2 ranks highest (appears in both lists)
//! ```
//!
//! # Fusion Algorithms
//!
//! | Function | Uses Scores | Best For |
//! |----------|-------------|----------|
//! | [`rrf`] | No | Incompatible score scales |
//! | [`isr`] | No | When lower ranks matter more |
//! | [`combsum`] | Yes | Similar scales, trust scores |
//! | [`combmnz`] | Yes | Reward overlap between lists |
//! | [`borda`] | No | Simple voting |
//! | [`weighted`] | Yes | Custom retriever weights |
//! | [`dbsf`] | Yes | Different score distributions |
//! | [`condorcet`] | No | Pairwise voting, outlier-robust |
//! | [`copeland`] | No | Net pairwise wins, more discriminative than Condorcet |
//! | [`median_rank`] | No | Median rank across lists, outlier-robust |
//! | [`combmax`] | Yes | At least one retriever likes it |
//! | [`combmin`] | Yes | All retrievers must agree (conservative) |
//! | [`combmed`] | Yes | Median score, robust to outliers |
//!
//! All have `*_multi` variants for 3+ lists.
//!
//! # Diversity Reranking
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`mmr`] | Maximal Marginal Relevance (Carbonell & Goldstein, 1998) |
//! | [`mmr_with_matrix`] | MMR with precomputed similarity matrix |
//! | [`mmr_embeddings`] | MMR with embedding vectors (computes cosine similarity) |
//!
//! MMR balances relevance and diversity via tunable λ parameter.
//!
//! # Performance Notes
//!
//! `OpenSearch` benchmarks (BEIR) show RRF is ~3-4% lower NDCG than score-based
//! fusion (`CombSUM`), but ~1-2% faster. RRF excels when score scales are
//! incompatible or unknown. See [OpenSearch RRF blog](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/).

use std::collections::HashMap;
use std::hash::Hash;

/// Fusion diagnostics: complementarity, overlap, score distributions.
pub mod diagnostics;
/// Composable fusion pipeline and multi-query fusion.
pub mod pipeline;
/// Validation utilities for fusion results.
pub mod rerank;
pub mod validate;

#[cfg(test)]
mod proptests;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during fusion.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// Weights sum to zero or near-zero.
    ZeroWeights,
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroWeights => write!(f, "weights sum to zero"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Result type for fusion operations.
pub type Result<T> = std::result::Result<T, FusionError>;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration with Builder Pattern
// ─────────────────────────────────────────────────────────────────────────────

/// Threshold for treating weight sum as effectively zero.
///
/// Used in weighted fusion to detect invalid configurations where all weights
/// are zero or near-zero, which would cause division by zero.
const WEIGHT_EPSILON: f32 = 1e-9;

/// Threshold for treating score range as effectively zero (all scores equal).
///
/// Used in min-max normalization to detect degenerate cases where all scores
/// are identical, avoiding division by zero.
const SCORE_RANGE_EPSILON: f32 = 1e-9;

/// RRF configuration.
///
/// # Example
///
/// ```rust
/// use rankops::RrfConfig;
///
/// let config = RrfConfig::default()
///     .with_k(60)
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrfConfig {
    /// Smoothing constant (default: 60).
    ///
    /// **Must be >= 1** to avoid division by zero in the RRF formula.
    /// Values < 1 will cause panics during fusion.
    pub k: u32,
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60, top_k: None }
    }
}

impl RrfConfig {
    /// Create config with custom k.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` (would cause division by zero in RRF formula).
    ///
    /// # Example
    ///
    /// ```rust
    /// use rankops::RrfConfig;
    ///
    /// let config = RrfConfig::new(60);
    /// ```
    #[must_use]
    pub fn new(k: u32) -> Self {
        assert!(
            k >= 1,
            "k must be >= 1 to avoid division by zero in RRF formula"
        );
        Self { k, top_k: None }
    }

    /// Set the k parameter (smoothing constant).
    ///
    /// - `k=60` — Standard RRF, works well for most cases
    /// - `k=1` — Top positions dominate heavily
    /// - `k=100+` — More uniform contribution across ranks
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` (would cause division by zero in RRF formula).
    #[must_use]
    pub fn with_k(mut self, k: u32) -> Self {
        assert!(
            k >= 1,
            "k must be >= 1 to avoid division by zero in RRF formula"
        );
        self.k = k;
        self
    }

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Weighted fusion configuration.
///
/// # Example
///
/// ```rust
/// use rankops::WeightedConfig;
///
/// let config = WeightedConfig::default()
///     .with_weights(0.7, 0.3)
///     .with_normalize(true)
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedConfig {
    /// Weight for first list (default: 0.5).
    pub weight_a: f32,
    /// Weight for second list (default: 0.5).
    pub weight_b: f32,
    /// Normalize scores to `[0,1]` before combining (default: true).
    pub normalize: bool,
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for WeightedConfig {
    fn default() -> Self {
        Self {
            weight_a: 0.5,
            weight_b: 0.5,
            normalize: true,
            top_k: None,
        }
    }
}

impl WeightedConfig {
    /// Create config with custom weights.
    #[must_use]
    pub const fn new(weight_a: f32, weight_b: f32) -> Self {
        Self {
            weight_a,
            weight_b,
            normalize: true,
            top_k: None,
        }
    }

    /// Set weights for the two lists.
    #[must_use]
    pub const fn with_weights(mut self, weight_a: f32, weight_b: f32) -> Self {
        self.weight_a = weight_a;
        self.weight_b = weight_b;
        self
    }

    /// Enable/disable score normalization.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Configuration for rank-based fusion (Borda, `CombSUM`, `CombMNZ`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusionConfig {
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl FusionConfig {
    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prelude
// ─────────────────────────────────────────────────────────────────────────────

/// Prelude for common imports.
///
/// ```rust
/// use rankops::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        additive_multi_task, additive_multi_task_multi, additive_multi_task_with_config, borda,
        combanz, combmax, combmed, combmnz, combsum, condorcet, copeland, dbsf, isr,
        isr_with_config, median_rank, rrf, rrf_with_config, standardized, standardized_multi,
        standardized_with_config, weighted,
    };
    pub use crate::{
        evaluate_metric, hit_rate, map, map_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k,
    };
    pub use crate::{
        AdditiveMultiTaskConfig, FusionConfig, FusionError, FusionMethod, Normalization, Result,
        RrfConfig, StandardizedConfig, WeightedConfig,
    };
}

/// Explainability module for debugging and analysis.
///
/// Provides variants of fusion functions that return full provenance information,
/// showing which retrievers contributed each document and how scores were computed.
pub mod explain {
    pub use crate::{
        analyze_consensus, attribute_top_k, combmnz_explain, combsum_explain, dbsf_explain,
        rrf_explain, ConsensusReport, Explanation, FusedResult, RetrieverId, RetrieverStats,
        SourceContribution,
    };
}

// WASM bindings live in the separate `rankops-wasm` crate.

/// Strategy module for runtime fusion method selection.
///
/// Enables dynamic selection of fusion methods without trait objects.
pub mod strategy {
    pub use crate::FusionStrategy;
}

/// Validation module for fusion results.
///
/// Provides utilities to validate fusion results, ensuring they meet expected
/// properties (sorted, no duplicates, finite scores, etc.).
///
/// Re-exports all validation functions from the internal validate module.
pub use validate::{
    validate, validate_bounds, validate_finite_scores, validate_no_duplicates,
    validate_non_negative_scores, validate_sorted, ValidationResult,
};

// ─────────────────────────────────────────────────────────────────────────────
// Unified Fusion Method
// ─────────────────────────────────────────────────────────────────────────────

/// Unified fusion method for dispatching to different algorithms.
///
/// Provides a single entry point for all fusion algorithms with a consistent API.
///
/// # Example
///
/// ```rust
/// use rankops::FusionMethod;
///
/// let sparse = vec![("d1", 10.0), ("d2", 8.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.7)];
///
/// // Use RRF (rank-based, score-agnostic)
/// let fused = FusionMethod::Rrf { k: 60 }.fuse(&sparse, &dense);
///
/// // Use CombSUM (score-based)
/// let fused = FusionMethod::CombSum.fuse(&sparse, &dense);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion (ignores scores, uses rank position).
    Rrf {
        /// Smoothing constant (default: 60).
        k: u32,
    },
    /// Inverse Square Root rank fusion (gentler decay than RRF).
    Isr {
        /// Smoothing constant (default: 1).
        k: u32,
    },
    /// `CombSUM` — sum of normalized scores.
    CombSum,
    /// `CombMNZ` — sum × overlap count.
    CombMnz,
    /// Borda count — N - rank points.
    Borda,
    /// Condorcet — pairwise majority wins.
    Condorcet,
    /// Copeland — pairwise net wins (wins - losses). More discriminative than Condorcet.
    Copeland,
    /// Median Rank Aggregation — median rank across lists. Outlier-robust.
    MedianRank,
    /// `CombMAX` — maximum score across lists.
    CombMax,
    /// `CombMIN` — minimum score across lists (conservative).
    CombMin,
    /// `CombMED` — median score across lists.
    CombMed,
    /// `CombANZ` — average of non-zero scores.
    CombAnz,
    /// Rank-Biased Centroids with configurable persistence.
    Rbc {
        /// Persistence parameter (default: 0.8). Higher = more weight to lower ranks.
        persistence: f32,
    },
    /// Weighted combination with custom weights.
    Weighted {
        /// Weight for first list.
        weight_a: f32,
        /// Weight for second list.
        weight_b: f32,
        /// Whether to normalize scores before combining.
        normalize: bool,
    },
    /// Distribution-Based Score Fusion (z-score normalization).
    Dbsf,
    /// Standardization-based fusion (ERANK-style).
    ///
    /// Uses z-score normalization (standardization) instead of min-max normalization,
    /// then applies additive fusion. More robust to outliers and different score distributions.
    /// Based on ERANK (arXiv:2509.00520) which shows 2-5% NDCG improvement over CombSUM
    /// when score distributions differ significantly.
    Standardized {
        /// Clip z-scores to this range (default: [-3.0, 3.0]).
        clip_range: (f32, f32),
    },
    /// Additive multi-task fusion (ResFlow-style).
    ///
    /// Additive fusion of multi-task scores: `α·score_a + β·score_b`.
    /// ResFlow (arXiv:2411.09705) shows additive outperforms multiplicative for e-commerce.
    AdditiveMultiTask {
        /// Weight for first task.
        weight_a: f32,
        /// Weight for second task.
        weight_b: f32,
        /// Normalization method (default: ZScore for robustness).
        normalization: Normalization,
    },
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

impl FusionMethod {
    /// Create RRF method with default k=60.
    #[must_use]
    pub const fn rrf() -> Self {
        Self::Rrf { k: 60 }
    }

    /// Create RRF method with custom k.
    #[must_use]
    pub const fn rrf_with_k(k: u32) -> Self {
        Self::Rrf { k }
    }

    /// Create ISR method with default k=1.
    #[must_use]
    pub const fn isr() -> Self {
        Self::Isr { k: 1 }
    }

    /// Create ISR method with custom k.
    #[must_use]
    pub const fn isr_with_k(k: u32) -> Self {
        Self::Isr { k }
    }

    /// Create RBC method with default persistence 0.8.
    #[must_use]
    pub const fn rbc() -> Self {
        Self::Rbc { persistence: 0.8 }
    }

    /// Create RBC method with custom persistence.
    #[must_use]
    pub const fn rbc_with_persistence(persistence: f32) -> Self {
        Self::Rbc { persistence }
    }

    /// Create weighted method with custom weights.
    #[must_use]
    pub const fn weighted(weight_a: f32, weight_b: f32) -> Self {
        Self::Weighted {
            weight_a,
            weight_b,
            normalize: true,
        }
    }

    /// Create standardized fusion method (ERANK-style).
    ///
    /// Uses z-score normalization (standardization) with clipping to prevent outliers.
    /// More robust than min-max when score distributions differ significantly.
    #[must_use]
    pub const fn standardized(clip_range: (f32, f32)) -> Self {
        Self::Standardized { clip_range }
    }

    /// Create standardized fusion method with default clipping [-3.0, 3.0].
    #[must_use]
    pub const fn standardized_default() -> Self {
        Self::Standardized {
            clip_range: (-3.0, 3.0),
        }
    }

    /// Create additive multi-task fusion method (ResFlow-style).
    ///
    /// Additive fusion outperforms multiplicative for e-commerce ranking.
    /// ResFlow's optimal formula: `CTR + CTCVR × 20`.
    #[must_use]
    pub const fn additive_multi_task(weight_a: f32, weight_b: f32) -> Self {
        Self::AdditiveMultiTask {
            weight_a,
            weight_b,
            normalization: Normalization::ZScore,
        }
    }

    /// Create additive multi-task fusion with custom normalization.
    #[must_use]
    pub fn additive_multi_task_with_norm(
        weight_a: f32,
        weight_b: f32,
        normalization: Normalization,
    ) -> Self {
        Self::AdditiveMultiTask {
            weight_a,
            weight_b,
            normalization,
        }
    }

    /// Fuse two ranked lists using this method.
    ///
    /// # Arguments
    /// * `a` - First ranked list (ID, score pairs)
    /// * `b` - Second ranked list (ID, score pairs)
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    #[must_use]
    pub fn fuse<I: Clone + Eq + Hash>(&self, a: &[(I, f32)], b: &[(I, f32)]) -> Vec<(I, f32)> {
        match self {
            Self::Rrf { k } => {
                // Validate k at use time to avoid panics from invalid FusionMethod construction
                if *k == 0 {
                    return Vec::new();
                }
                crate::rrf_multi(&[a, b], RrfConfig::new(*k))
            }
            Self::Isr { k } => {
                if *k == 0 {
                    return Vec::new();
                }
                crate::isr_multi(&[a, b], RrfConfig::new(*k))
            }
            Self::CombSum => crate::combsum(a, b),
            Self::CombMnz => crate::combmnz(a, b),
            Self::Borda => crate::borda(a, b),
            Self::Condorcet => crate::condorcet(a, b),
            Self::Copeland => crate::copeland(a, b),
            Self::MedianRank => crate::median_rank(a, b),
            Self::CombMax => crate::combmax(a, b),
            Self::CombMin => crate::combmin(a, b),
            Self::CombMed => crate::combmed(a, b),
            Self::CombAnz => crate::combanz(a, b),
            Self::Rbc { persistence } => crate::rbc_multi(&[a, b], *persistence),
            Self::Weighted {
                weight_a,
                weight_b,
                normalize,
            } => crate::weighted(
                a,
                b,
                WeightedConfig::new(*weight_a, *weight_b).with_normalize(*normalize),
            ),
            Self::Dbsf => crate::dbsf(a, b),
            Self::Standardized { clip_range } => {
                crate::standardized_with_config(a, b, StandardizedConfig::new(*clip_range))
            }
            Self::AdditiveMultiTask {
                weight_a,
                weight_b,
                normalization,
            } => crate::additive_multi_task_with_config(
                a,
                b,
                AdditiveMultiTaskConfig::new((*weight_a, *weight_b))
                    .with_normalization(*normalization),
            ),
        }
    }

    /// Fuse multiple ranked lists using this method.
    ///
    /// # Arguments
    /// * `lists` - Slice of ranked lists
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    #[must_use]
    pub fn fuse_multi<I, L>(&self, lists: &[L]) -> Vec<(I, f32)>
    where
        I: Clone + Eq + Hash,
        L: AsRef<[(I, f32)]>,
    {
        match self {
            Self::Rrf { k } => crate::rrf_multi(lists, RrfConfig::new(*k)),
            Self::Isr { k } => crate::isr_multi(lists, RrfConfig::new(*k)),
            Self::CombSum => crate::combsum_multi(lists, FusionConfig::default()),
            Self::CombMnz => crate::combmnz_multi(lists, FusionConfig::default()),
            Self::Borda => crate::borda_multi(lists, FusionConfig::default()),
            Self::Condorcet => crate::condorcet_multi(lists, FusionConfig::default()),
            Self::Copeland => crate::copeland_multi(lists, FusionConfig::default()),
            Self::MedianRank => crate::median_rank_multi(lists, FusionConfig::default()),
            Self::CombMax => crate::combmax_multi(lists, FusionConfig::default()),
            Self::CombMin => crate::combmin_multi(lists, FusionConfig::default()),
            Self::CombMed => crate::combmed_multi(lists, FusionConfig::default()),
            Self::CombAnz => crate::combanz_multi(lists, FusionConfig::default()),
            Self::Rbc { persistence } => crate::rbc_multi(lists, *persistence),
            Self::Weighted { normalize, .. } => {
                if lists.len() == 2 {
                    self.fuse(lists[0].as_ref(), lists[1].as_ref())
                } else {
                    // Equal weights for 3+ lists
                    let weighted_lists: Vec<_> = lists
                        .iter()
                        .map(|l| (l.as_ref(), 1.0 / lists.len() as f32))
                        .collect();
                    crate::weighted_multi(&weighted_lists, *normalize, None).unwrap_or_default()
                }
            }
            Self::Dbsf => crate::dbsf_multi(lists, FusionConfig::default()),
            Self::Standardized { clip_range: _ } => {
                crate::standardized_multi(lists, StandardizedConfig::default())
            }
            Self::AdditiveMultiTask {
                weight_a,
                weight_b,
                normalization,
            } => {
                // For multi-list, use equal weights (users should use additive_multi_task_multi directly)
                if lists.len() == 2 {
                    self.fuse(lists[0].as_ref(), lists[1].as_ref())
                } else {
                    // For 3+ lists, convert to weighted lists format
                    let weighted_lists: Vec<_> = lists
                        .iter()
                        .map(|l| (l.as_ref(), 1.0 / lists.len() as f32))
                        .collect();
                    crate::additive_multi_task_multi(
                        &weighted_lists,
                        AdditiveMultiTaskConfig::new((*weight_a, *weight_b))
                            .with_normalization(*normalization),
                    )
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RRF (Reciprocal Rank Fusion)
// ─────────────────────────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion of two result lists with default config (k=60).
///
/// Formula: `score(d) = Σ 1/(k + rank)` where rank is 0-indexed.
///
/// **Why RRF?** Different retrievers use incompatible score scales (BM25: 0-100,
/// dense: 0-1). RRF solves this by ignoring scores entirely and using only rank
/// positions. The reciprocal formula ensures:
/// - Top positions dominate (rank 0 gets 1/60 = 0.017, rank 5 gets 1/65 = 0.015)
/// - Multiple list agreement is rewarded (documents appearing in both lists score higher)
/// - No normalization needed (works with any score distribution)
///
/// **When to use**: Hybrid search with incompatible score scales, zero-configuration needs.
/// **When NOT to use**: When score scales are compatible, CombSUM achieves ~3-4% better NDCG.
///
/// Use [`rrf_with_config`] to customize the k parameter (lower k = more top-heavy).
///
/// # Duplicate Document IDs
///
/// If a document ID appears multiple times in the same list, **all occurrences contribute**
/// to the RRF score based on their respective ranks. For example, if "d1" appears at
/// rank 0 and rank 5 in list A, its contribution from list A is `1/(k+0) + 1/(k+5)`.
/// This differs from some implementations that take only the first occurrence.
///
/// # Complexity
///
/// O(n log n) where n = |a| + |b| (dominated by final sort).
///
/// # Input Validation
///
/// This function does not validate inputs. For validation, use `validate()`
/// after fusion. Edge cases handled:
/// - Empty lists: Returns items from non-empty list(s)
/// - k=0: Returns empty Vec (use `validate()` to catch this)
/// - Non-finite scores: Ignored (RRF is rank-based)
///
/// # Example
///
/// ```rust
/// use rankops::rrf;
///
/// let sparse = vec![("d1", 0.9), ("d2", 0.5)];
/// let dense = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = rrf(&sparse, &dense);
/// assert_eq!(fused[0].0, "d2"); // appears in both lists (consensus)
/// ```
#[must_use]
/// Reciprocal Rank Fusion (RRF) with default configuration (k=60).
///
/// RRF is the recommended fusion method when combining rankings with incompatible
/// score scales. It uses only rank positions, ignoring score magnitudes entirely.
///
/// # Formula
///
/// `RRF(d) = Σ 1/(k + rank_r(d))` where:
/// - `k` = smoothing constant (default: 60)
/// - `rank_r(d)` = position of document d in ranking r (0-indexed)
///
/// # Why RRF?
///
/// - **Score Scale Independent**: Works with any scoring system (BM25: 0-100, embeddings: 0-1)
/// - **Robust**: Handles missing documents gracefully (documents not in a list contribute 0)
/// - **Effective**: Proven to outperform individual rankers in hybrid search
/// - **Fast**: O(n log n) complexity where n = total unique documents
///
/// # Arguments
///
/// * `results_a` - First ranked list: `Vec<(document_id, score)>`
/// * `results_b` - Second ranked list: `Vec<(document_id, score)>`
///
/// Note: Scores are ignored; only rank positions matter.
///
/// # Returns
///
/// Fused ranking sorted by RRF score (descending). Documents appearing in both
/// lists rank higher than those appearing in only one list.
///
/// # Example
///
/// ```rust
/// use rankops::rrf;
///
/// // BM25 results (high scores = better)
/// let bm25 = vec![
///     ("doc1", 12.5),
///     ("doc2", 11.0),
///     ("doc3", 10.0),
/// ];
///
/// // Dense embedding results (different scale: 0-1)
/// let dense = vec![
///     ("doc2", 0.9),
///     ("doc3", 0.8),
///     ("doc1", 0.7),
/// ];
///
/// // RRF ignores scores, uses only rank positions
/// let fused = rrf(&bm25, &dense);
/// // doc2 ranks highest (appears in both lists at high positions)
/// // doc1 and doc3 follow (appear in both lists but at different positions)
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = |results_a| + |results_b| (dominated by final sort).
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Combining BM25 + dense embeddings (different scales)
/// - ✅ Combining sparse + dense retrieval
/// - ✅ Unknown or incompatible score scales
/// - ✅ Need robust fusion without normalization tuning
///
/// # When NOT to Use
///
/// - ❌ Scores are already normalized and comparable (use `combsum` or `weighted`)
/// - ❌ You trust score magnitudes (use score-based fusion)
/// - ❌ Need fine-grained control over retriever importance (use `rrf_weighted`)
pub fn rrf<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    rrf_with_config(results_a, results_b, RrfConfig::default())
}

/// RRF with custom configuration.
///
/// Use this when you need to tune the k parameter:
/// - **k=20-40**: Top positions dominate more. Use when top retrievers are highly reliable.
/// - **k=60**: Default (empirically chosen by Cormack et al., 2009). Balanced for most scenarios.
/// - **k=100+**: More uniform contribution. Use when lower-ranked items are still valuable.
///
/// **Sensitivity**: k=10 gives 1.5x ratio (rank 0 vs rank 5), k=60 gives 1.1x, k=100 gives 1.05x.
///
/// # Example
///
/// ```rust
/// use rankops::{rrf_with_config, RrfConfig};
///
/// let a = vec![("d1", 0.9), ("d2", 0.5)];
/// let b = vec![("d2", 0.8), ("d3", 0.3)];
///
/// // k=20: emphasize top positions (strong consensus required)
/// let fused = rrf_with_config(&a, &b, RrfConfig::new(20));
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
) -> Vec<(I, f32)> {
    // Validate k >= 1 to avoid division by zero (k=0 would cause 1/0 for rank 0)
    if config.k == 0 {
        return Vec::new();
    }
    let k = config.k as f32;
    // Pre-allocate capacity to avoid reallocations during insertion
    let estimated_size = results_a.len() + results_b.len();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }

    finalize(scores, config.top_k)
}

/// RRF with preallocated output buffer.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_into<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
    output: &mut Vec<(I, f32)>,
) {
    output.clear();
    let k = config.k as f32;
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(results_a.len() + results_b.len());

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }

    output.extend(scores);
    sort_scored_desc(output);
    if let Some(top_k) = config.top_k {
        output.truncate(top_k);
    }
}

/// RRF for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. If some lists are empty,
/// they contribute zero scores (documents not appearing in those lists
/// receive no contribution from them).
///
/// # Complexity
///
/// O(L×N + U×log U) where L = number of lists, N = average list size,
/// U = number of unique document IDs across all lists.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_multi<I, L>(lists: &[L], config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    // Validate k >= 1 to avoid division by zero
    if config.k == 0 {
        return Vec::new();
    }
    let k = config.k as f32;
    // Estimate capacity: sum of all list sizes (may overestimate due to duplicates)
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for list in lists {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32);
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

/// Weighted RRF: per-retriever weights applied to rank-based scores.
///
/// Unlike standard RRF which treats all lists equally, weighted RRF allows
/// assigning different importance to different retrievers based on domain
/// knowledge or tuning.
///
/// Formula: `score(d) = Σ w_i / (k + rank_i(d))`
///
/// # Example
///
/// ```rust
/// use rankops::{rrf_weighted, RrfConfig};
///
/// let bm25 = vec![("d1", 0.0), ("d2", 0.0)];   // scores ignored
/// let dense = vec![("d2", 0.0), ("d3", 0.0)];
///
/// // Trust dense retriever 2x more than BM25
/// let weights = [0.33, 0.67];
/// let fused = rrf_weighted(&[&bm25[..], &dense[..]], &weights, RrfConfig::default());
/// ```
///
/// # Errors
///
/// - Returns [`FusionError::ZeroWeights`] if weights sum to zero.
/// - Returns [`FusionError::InvalidConfig`] if `lists.len() != weights.len()`.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_weighted<I, L>(lists: &[L], weights: &[f32], config: RrfConfig) -> Result<Vec<(I, f32)>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.len() != weights.len() {
        return Err(FusionError::InvalidConfig(format!(
            "lists.len() ({}) != weights.len() ({}). Each list must have a corresponding weight.",
            lists.len(),
            weights.len()
        )));
    }
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum.abs() < WEIGHT_EPSILON {
        return Err(FusionError::ZeroWeights);
    }

    let k = config.k as f32;
    // Pre-allocate capacity
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, &weight) in lists.iter().zip(weights.iter()) {
        let normalized_weight = weight / weight_sum;
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = normalized_weight / (k + rank as f32);
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    Ok(finalize(scores, config.top_k))
}

// ─────────────────────────────────────────────────────────────────────────────
// ISR (Inverse Square Root Rank)
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse Square Root rank fusion with default config (k=1).
///
/// ISR uses a gentler decay than RRF, giving lower-ranked documents more
/// relative contribution compared to top positions. This makes it useful when
/// you believe relevant documents may appear deeper in the ranking lists.
///
/// # Formula
///
/// `score(d) = Σ 1/sqrt(k + rank)` where rank is 0-indexed.
///
/// Compared to RRF's `1/(k + rank)`, ISR's `1/sqrt(k + rank)` decays more slowly,
/// meaning rank 10 vs rank 20 has a smaller relative difference than in RRF.
///
/// # Arguments
///
/// * `results_a` - First ranked list (scores ignored, only positions matter)
/// * `results_b` - Second ranked list (scores ignored, only positions matter)
///
/// # Returns
///
/// Fused ranking sorted by ISR score (descending). Documents appearing in both
/// lists rank higher, with less emphasis on exact position than RRF.
///
/// # Example
///
/// ```rust
/// use rankops::isr;
///
/// let sparse = vec![("d1", 0.9), ("d2", 0.5), ("d3", 0.3)];
/// let dense = vec![("d2", 0.8), ("d3", 0.7), ("d4", 0.2)];
///
/// let fused = isr(&sparse, &dense);
/// // d2 and d3 appear in both lists, so they rank highest
/// // ISR gives more weight to lower positions than RRF
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = |results_a| + |results_b| (dominated by final sort).
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Relevant documents may appear deeper in lists (rank 20-50 still valuable)
/// - ✅ Want gentler position-based decay than RRF
/// - ✅ Combining retrievers where position uncertainty is high
///
/// # When NOT to Use
///
/// - ❌ Top positions are highly reliable (use RRF with k=20-40)
/// - ❌ Need score-based fusion (use `combsum` or `weighted`)
/// - ❌ Unknown score scales but want standard decay (use RRF)
///
/// # Trade-offs vs RRF
///
/// - **Decay**: Gentler (lower ranks contribute more)
/// - **Top emphasis**: Less emphasis on exact top positions
/// - **Use case**: Better for noisy retrievers or when depth matters
#[must_use]
pub fn isr<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    isr_with_config(results_a, results_b, RrfConfig::new(1))
}

/// ISR with custom configuration.
///
/// The k parameter controls decay steepness:
/// - Lower k (e.g., 1): Top positions dominate more
/// - Higher k (e.g., 10): More uniform contribution across positions
///
/// # Example
///
/// ```rust
/// use rankops::{isr_with_config, RrfConfig};
///
/// let a = vec![("d1", 0.9), ("d2", 0.5)];
/// let b = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = isr_with_config(&a, &b, RrfConfig::new(1));
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn isr_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
) -> Vec<(I, f32)> {
    // Validate k >= 1 to avoid division by zero (k=0 would cause 1/0 for rank 0)
    if config.k == 0 {
        return Vec::new();
    }
    let k = config.k as f32;
    let estimated_size = results_a.len() + results_b.len();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32).sqrt();
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32).sqrt();
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }

    finalize(scores, config.top_k)
}

/// ISR for 3+ result lists.
///
/// # Invalid Configuration
///
/// If `config.k == 0`, returns an empty result to avoid division by zero.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn isr_multi<I, L>(lists: &[L], config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    // Validate k >= 1 to avoid division by zero
    if config.k == 0 {
        return Vec::new();
    }
    let k = config.k as f32;
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32).sqrt();
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Score-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted score fusion with configurable retriever trust.
///
/// Formula: `score(d) = w_a × norm(s_a) + w_b × norm(s_b)`
///
/// Use when you know one retriever is more reliable for your domain.
/// Weights are normalized to sum to 1.
///
/// # Complexity
///
/// O(n log n) where n = total items across all lists.
#[must_use]
pub fn weighted<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: WeightedConfig,
) -> Vec<(I, f32)> {
    weighted_impl(
        &[(results_a, config.weight_a), (results_b, config.weight_b)],
        config.normalize,
        config.top_k,
    )
}

/// Weighted fusion for 3+ result lists.
///
/// Each list is paired with its weight. Weights are normalized to sum to 1.
///
/// # Errors
///
/// Returns `Err(FusionError::ZeroWeights)` if weights sum to zero.
pub fn weighted_multi<I, L>(
    lists: &[(L, f32)],
    normalize: bool,
    top_k: Option<usize>,
) -> Result<Vec<(I, f32)>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < WEIGHT_EPSILON {
        return Err(FusionError::ZeroWeights);
    }

    let estimated_size: usize = lists.iter().map(|(l, _)| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            let contribution = w * (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    Ok(finalize(scores, top_k))
}

/// Internal weighted implementation (infallible for two-list case).
fn weighted_impl<I, L>(lists: &[(L, f32)], normalize: bool, top_k: Option<usize>) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < WEIGHT_EPSILON {
        return Vec::new();
    }

    let estimated_size: usize = lists.iter().map(|(l, _)| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            let contribution = w * (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, top_k)
}

/// Sum of min-max normalized scores (CombSUM).
///
/// CombSUM normalizes each list to [0, 1] using min-max normalization, then sums
/// the normalized scores. This preserves score magnitudes while handling different scales.
///
/// # Formula
///
/// For each list: `normalized = (score - min) / (max - min)`
/// Final score: `score(d) = Σ normalized_scores(d)`
///
/// # Arguments
///
/// * `results_a` - First ranked list: `Vec<(document_id, score)>`
/// * `results_b` - Second ranked list: `Vec<(document_id, score)>`
///
/// # Returns
///
/// Fused ranking sorted by combined score (descending). Documents with higher
/// normalized scores across lists rank higher.
///
/// # Example
///
/// ```rust
/// use rankops::combsum;
///
/// // Both lists use cosine similarity (0-1 scale)
/// let sparse = vec![
///     ("doc1", 0.9),
///     ("doc2", 0.8),
///     ("doc3", 0.7),
/// ];
///
/// let dense = vec![
///     ("doc2", 0.95),
///     ("doc1", 0.85),
///     ("doc3", 0.75),
/// ];
///
/// let fused = combsum(&sparse, &dense);
/// // doc2 ranks highest (0.8 + 0.95 = 1.75 after normalization)
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = total items across all lists.
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Scores are on similar scales (e.g., all cosine similarities 0-1)
/// - ✅ You trust score magnitudes (scores represent true relevance)
/// - ✅ Need better accuracy than RRF (CombSUM typically 3-4% higher NDCG)
///
/// # When NOT to Use
///
/// - ❌ Incompatible score scales (BM25: 0-100 vs embeddings: 0-1) - use RRF
/// - ❌ Score distributions differ significantly - use `standardized` or `dbsf`
/// - ❌ Unknown score scales - use RRF
///
/// # Trade-offs
///
/// - **Accuracy**: Typically 3-4% higher NDCG than RRF (OpenSearch benchmarks)
/// - **Robustness**: Less robust to outliers than RRF (min-max is sensitive)
/// - **Speed**: Similar to RRF (~1-2% faster due to simpler computation)
#[must_use]
pub fn combsum<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combsum_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombSUM` with configuration.
#[must_use]
pub fn combsum_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combsum_multi(&[results_a, results_b], config)
}

/// `CombSUM` for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no contribution from them).
#[must_use]
pub fn combsum_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            let contribution = (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

/// Normalized sum × overlap count (CombMNZ).
///
/// CombMNZ multiplies the CombSUM score by the number of lists containing the document,
/// rewarding documents that appear in multiple retrievers (consensus signal).
///
/// # Formula
///
/// `score(d) = CombSUM(d) × |{lists containing d}|`
///
/// Where CombSUM(d) is the sum of min-max normalized scores, and the multiplier
/// is the number of lists that contain document d.
///
/// # Arguments
///
/// * `results_a` - First ranked list: `Vec<(document_id, score)>`
/// * `results_b` - Second ranked list: `Vec<(document_id, score)>`
///
/// # Returns
///
/// Fused ranking sorted by CombMNZ score (descending). Documents appearing in
/// both lists get a 2x multiplier, significantly boosting their scores.
///
/// # Example
///
/// ```rust
/// use rankops::combmnz;
///
/// let sparse = vec![("doc1", 0.9), ("doc2", 0.8)];
/// let dense = vec![("doc2", 0.95), ("doc3", 0.7)];
///
/// let fused = combmnz(&sparse, &dense);
/// // doc2 ranks highest: (0.8 + 0.95) × 2 = 3.5 (appears in both lists)
/// // doc1: 0.9 × 1 = 0.9 (only in sparse)
/// // doc3: 0.7 × 1 = 0.7 (only in dense)
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = total items across all lists.
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Overlap between retrievers signals higher relevance
/// - ✅ Want to strongly favor documents found by multiple retrievers
/// - ✅ Combining complementary retrievers (e.g., keyword + semantic)
///
/// # When NOT to Use
///
/// - ❌ Retrievers are highly correlated (overlap doesn't add information)
/// - ❌ Single-source documents are still valuable (CombMNZ penalizes them)
/// - ❌ Need fine-grained control (use `weighted` with custom weights)
///
/// # Trade-offs
///
/// - **Consensus**: Strongly favors documents in multiple lists
/// - **Diversity**: May reduce diversity (single-source documents rank lower)
/// - **Accuracy**: Typically similar to CombSUM, better when overlap is informative
#[must_use]
pub fn combmnz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmnz_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombMNZ` with configuration.
#[must_use]
pub fn combmnz_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combmnz_multi(&[results_a, results_b], config)
}

/// `CombMNZ` for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores and don't affect overlap counts.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combmnz_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, (f32, u32)> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
            let contribution = (s - off) * norm;
            if let Some(entry) = scores.get_mut(id) {
                entry.0 += contribution;
                entry.1 += 1;
            } else {
                scores.insert(id.clone(), (contribution, 1));
            }
        }
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (sum, n))| (id, sum * n as f32))
        .collect();
    sort_scored_desc(&mut results);
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Borda count voting — position-based scoring.
///
/// Borda count assigns points based on position: the first item gets N points,
/// the second gets N-1 points, etc., where N is the list length. Simple and
/// robust when you don't trust score magnitudes.
///
/// # Formula
///
/// `score(d) = Σ (N - rank)` where:
/// - N = list length
/// - rank = 0-indexed position in the list
///
/// Each list contributes independently, and scores are summed across lists.
///
/// # Arguments
///
/// * `results_a` - First ranked list (scores ignored, only positions matter)
/// * `results_b` - Second ranked list (scores ignored, only positions matter)
///
/// # Returns
///
/// Fused ranking sorted by Borda score (descending). Documents appearing in
/// both lists at high positions rank highest.
///
/// # Example
///
/// ```rust
/// use rankops::borda;
///
/// // List 1: 3 items (positions 0, 1, 2 → scores 3, 2, 1)
/// let list1 = vec![("d1", 0.9), ("d2", 0.5), ("d3", 0.3)];
///
/// // List 2: 2 items (positions 0, 1 → scores 2, 1)
/// let list2 = vec![("d2", 0.8), ("d4", 0.7)];
///
/// let fused = borda(&list1, &list2);
/// // d2: 2 (from list1) + 2 (from list2) = 4 points
/// // d1: 3 (from list1) = 3 points
/// // d3: 1 (from list1) = 1 point
/// // d4: 1 (from list2) = 1 point
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = total items across all lists.
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Simple voting-based fusion needed
/// - ✅ Don't trust score magnitudes (only positions matter)
/// - ✅ Need interpretable scoring (easy to explain: "sum of position points")
///
/// # When NOT to Use
///
/// - ❌ Lists have very different lengths (longer lists dominate)
/// - ❌ Need position decay (use RRF or ISR for exponential/square-root decay)
/// - ❌ Score magnitudes are reliable (use `combsum` or `weighted`)
///
/// # Trade-offs
///
/// - **Simplicity**: Very simple to understand and implement
/// - **Fairness**: Treats all positions linearly (no decay)
/// - **Length bias**: Longer lists contribute more total points
#[must_use]
pub fn borda<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    borda_with_config(results_a, results_b, FusionConfig::default())
}

/// Borda count with configuration.
#[must_use]
pub fn borda_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    borda_multi(&[results_a, results_b], config)
}

/// Borda count for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no Borda points from them).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn borda_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let n = items.len() as f32;
        for (rank, (id, _)) in items.iter().enumerate() {
            let contribution = n - rank as f32;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Distribution-Based Score Fusion (DBSF)
// ─────────────────────────────────────────────────────────────────────────────

/// Distribution-Based Score Fusion (DBSF).
///
/// DBSF uses z-score normalization (standardization) with mean ± 3σ clipping,
/// then sums the normalized scores. More robust than min-max normalization
/// when score distributions differ significantly or contain outliers.
///
/// # Algorithm
///
/// For each list:
/// 1. Compute mean (μ) and standard deviation (σ)
/// 2. Normalize: `z = (score - μ) / σ`, clipped to [-3, 3]
/// 3. Sum normalized z-scores across lists
///
/// The ±3σ clipping prevents extreme outliers from dominating the fusion.
///
/// # Arguments
///
/// * `results_a` - First ranked list with scores
/// * `results_b` - Second ranked list with scores
///
/// # Returns
///
/// Fused ranking sorted by combined z-score (descending). Documents with
/// consistently high z-scores across lists rank highest.
///
/// # Example
///
/// ```rust
/// use rankops::dbsf;
///
/// // BM25 scores (high variance, different scale)
/// let bm25 = vec![("d1", 15.0), ("d2", 12.0), ("d3", 8.0)];
///
/// // Dense embedding scores (low variance, different scale)
/// let dense = vec![("d2", 0.9), ("d3", 0.7), ("d4", 0.5)];
///
/// let fused = dbsf(&bm25, &dense);
/// // Z-scores normalize both lists to comparable scales
/// // d2 and d3 appear in both lists, so they rank highest
/// ```
///
/// # Performance
///
/// Time complexity: O(n log n) where n = total items across all lists.
/// Requires computing mean and std for each list (O(n) per list).
/// For typical workloads (100-1000 items per list), fusion completes in <1ms.
///
/// # When to Use
///
/// - ✅ Score distributions differ significantly (BM25: 0-100, embeddings: 0-1)
/// - ✅ Outliers are present (z-score clipping handles them)
/// - ✅ Need robust normalization (more robust than min-max)
///
/// # When NOT to Use
///
/// - ❌ Score scales are similar (use `combsum` for simplicity)
/// - ❌ Need configurable clipping (use `standardized` with custom range)
/// - ❌ Unknown score scales (use RRF to avoid normalization)
///
/// # Trade-offs vs CombSUM
///
/// - **Robustness**: More robust to outliers (z-score vs min-max)
/// - **Complexity**: Slightly more complex (requires mean/std computation)
/// - **Clipping**: Fixed [-3, 3] range (use `standardized` for custom range)
///
/// # Differences from Standardized
///
/// - DBSF uses fixed [-3, 3] clipping
/// - Standardized allows configurable clipping range
/// - Both use the same z-score approach
#[must_use]
pub fn dbsf<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    dbsf_with_config(results_a, results_b, FusionConfig::default())
}

/// DBSF with configuration.
#[must_use]
pub fn dbsf_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    dbsf_multi(&[results_a, results_b], config)
}

/// DBSF for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no z-score contribution from them).
///
/// # Degenerate Cases
///
/// If all scores in a list are equal (zero variance), that list contributes
/// z-score=0.0 for all documents, which is mathematically correct but
/// effectively ignores that list's contribution.
#[must_use]
pub fn dbsf_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (id, s) in items {
            // Z-score normalize and clip to [-3, 3]
            let z = if std > SCORE_RANGE_EPSILON {
                ((s - mean) / std).clamp(-3.0, 3.0)
            } else {
                0.0 // All scores equal
            };
            if let Some(score) = scores.get_mut(id) {
                *score += z;
            } else {
                scores.insert(id.clone(), z);
            }
        }
    }

    finalize(scores, config.top_k)
}

/// Compute mean and standard deviation for z-score normalization.
#[inline(always)]
fn zscore_params<I>(results: &[(I, f32)]) -> (f32, f32) {
    if results.is_empty() {
        return (0.0, 1.0);
    }

    let n = results.len() as f32;
    let mean = results.iter().map(|(_, s)| s).sum::<f32>() / n;
    let variance = results.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    (mean, std)
}

// ─────────────────────────────────────────────────────────────────────────────
// Standardization-Based Fusion (ERANK-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for standardization-based fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardizedConfig {
    /// Clip z-scores to this range (default: [-3.0, 3.0]).
    pub clip_range: (f32, f32),
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for StandardizedConfig {
    fn default() -> Self {
        Self {
            clip_range: (-3.0, 3.0),
            top_k: None,
        }
    }
}

impl StandardizedConfig {
    /// Create new config with custom clipping range.
    #[must_use]
    pub const fn new(clip_range: (f32, f32)) -> Self {
        Self {
            clip_range,
            top_k: None,
        }
    }

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Standardization-based fusion (ERANK-style).
///
/// Uses z-score normalization (standardization) instead of min-max normalization,
/// then applies additive fusion. More robust to outliers and different score distributions.
///
/// Based on ERANK (arXiv:2509.00520) which shows 2-5% NDCG improvement over CombSUM
/// when score distributions differ significantly.
///
/// # Algorithm
///
/// For each list:
/// 1. Compute mean (μ) and standard deviation (σ)
/// 2. Normalize: `z = (score - μ) / σ`, clipped to `clip_range`
/// 3. Sum normalized scores across lists
///
/// # Differences from DBSF
///
/// - DBSF uses fixed [-3, 3] clipping
/// - Standardized allows configurable clipping range
/// - Both use the same z-score approach, but standardized is more flexible
///
/// # Example
///
/// ```rust
/// use rankops::standardized;
///
/// let bm25 = vec![("d1", 15.0), ("d2", 12.0), ("d3", 8.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.7), ("d4", 0.5)];
/// let fused = standardized(&bm25, &dense);
/// ```
#[must_use]
pub fn standardized<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    standardized_with_config(results_a, results_b, StandardizedConfig::default())
}

/// Standardized fusion with configuration.
#[must_use]
pub fn standardized_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: StandardizedConfig,
) -> Vec<(I, f32)> {
    standardized_multi(&[results_a, results_b], config)
}

/// Standardized fusion for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no z-score contribution from them).
///
/// # Degenerate Cases
///
/// If all scores in a list are equal (zero variance), that list contributes
/// z-score=0.0 for all documents, which is mathematically correct but
/// effectively ignores that list's contribution.
#[must_use]
pub fn standardized_multi<I, L>(lists: &[L], config: StandardizedConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);
    let (clip_min, clip_max) = config.clip_range;

    for list in lists {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (id, s) in items {
            // Z-score normalize and clip to configurable range
            let z = if std > SCORE_RANGE_EPSILON {
                ((s - mean) / std).clamp(clip_min, clip_max)
            } else {
                0.0 // All scores equal
            };
            if let Some(score) = scores.get_mut(id) {
                *score += z;
            } else {
                scores.insert(id.clone(), z);
            }
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Additive Multi-Task Fusion (ResFlow-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for additive multi-task fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdditiveMultiTaskConfig {
    /// Weights for each task: (weight_a, weight_b).
    pub weights: (f32, f32),
    /// Normalization method (default: ZScore for robustness).
    pub normalization: Normalization,
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for AdditiveMultiTaskConfig {
    fn default() -> Self {
        Self {
            weights: (1.0, 1.0),
            normalization: Normalization::ZScore,
            top_k: None,
        }
    }
}

impl AdditiveMultiTaskConfig {
    /// Create new config with custom weights.
    ///
    /// ResFlow's optimal formula for e-commerce: `CTR + CTCVR × 20`.
    /// This would be `AdditiveMultiTaskConfig::new((1.0, 20.0))`.
    #[must_use]
    pub const fn new(weights: (f32, f32)) -> Self {
        Self {
            weights,
            normalization: Normalization::ZScore,
            top_k: None,
        }
    }

    /// Set normalization method.
    #[must_use]
    pub const fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Additive multi-task fusion (ResFlow-style).
///
/// Additive fusion of multi-task scores: `α·score_a + β·score_b`.
///
/// ResFlow (arXiv:2411.09705) shows additive outperforms multiplicative for e-commerce
/// ranking tasks. The optimal formula is typically `CTR + CTCVR × 20`, where CTR and
/// CTCVR are normalized scores from different tasks.
///
/// # Algorithm
///
/// 1. Normalize each list using the specified normalization method
/// 2. Compute weighted sum: `α·norm_a + β·norm_b`
/// 3. Sort by combined score (descending)
///
/// # Example
///
/// ```rust
/// use rankops::{additive_multi_task, AdditiveMultiTaskConfig};
///
/// let ctr_scores = vec![("item1", 0.05), ("item2", 0.03), ("item3", 0.01)];
/// let ctcvr_scores = vec![("item1", 0.02), ("item2", 0.01), ("item3", 0.005)];
///
/// // ResFlow optimal: CTR + CTCVR × 20
/// let config = AdditiveMultiTaskConfig::new((1.0, 20.0));
/// let fused = additive_multi_task(&ctr_scores, &ctcvr_scores, config);
/// ```
#[must_use]
pub fn additive_multi_task<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: AdditiveMultiTaskConfig,
) -> Vec<(I, f32)> {
    additive_multi_task_with_config(results_a, results_b, config)
}

/// Additive multi-task fusion with configuration.
#[must_use]
pub fn additive_multi_task_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: AdditiveMultiTaskConfig,
) -> Vec<(I, f32)> {
    let weighted_lists = vec![(results_a, config.weights.0), (results_b, config.weights.1)];
    additive_multi_task_multi(&weighted_lists, config)
}

/// Additive multi-task fusion for 3+ weighted lists.
///
/// # Arguments
///
/// * `weighted_lists` - Slice of (list, weight) pairs. Each list is normalized independently,
///   then combined using weighted sum.
///
/// # Example
///
/// ```rust
/// use rankops::{additive_multi_task_multi, AdditiveMultiTaskConfig};
///
/// let task1 = vec![("d1", 0.9), ("d2", 0.7)];
/// let task2 = vec![("d1", 0.8), ("d2", 0.6)];
/// let task3 = vec![("d1", 0.5), ("d2", 0.4)];
///
/// let weighted = vec![
///     (&task1[..], 1.0),
///     (&task2[..], 2.0),
///     (&task3[..], 0.5),
/// ];
///
/// let config = AdditiveMultiTaskConfig::default();
/// let fused = additive_multi_task_multi(&weighted, config);
/// ```
#[must_use]
pub fn additive_multi_task_multi<I, L>(
    weighted_lists: &[(L, f32)],
    config: AdditiveMultiTaskConfig,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if weighted_lists.is_empty() {
        return Vec::new();
    }

    // Normalize each list independently
    let normalized: Vec<_> = weighted_lists
        .iter()
        .map(|(list, _)| normalize_scores(list.as_ref(), config.normalization))
        .collect();

    // Compute weighted sum
    let estimated_size: usize = normalized.iter().map(|n| n.len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (normalized_list, (_, weight)) in normalized.iter().zip(weighted_lists.iter()) {
        for (id, norm_score) in normalized_list {
            if let Some(score) = scores.get_mut(id) {
                *score += weight * norm_score;
            } else {
                scores.insert(id.clone(), weight * norm_score);
            }
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sort scores descending and optionally truncate.
///
/// Uses `total_cmp` for deterministic NaN handling (NaN sorts after valid values).
#[inline]
fn finalize<I>(scores: HashMap<I, f32>, top_k: Option<usize>) -> Vec<(I, f32)> {
    let capacity = top_k.map(|k| k.min(scores.len())).unwrap_or(scores.len());
    let mut results = Vec::with_capacity(capacity);
    results.extend(scores);
    sort_scored_desc(&mut results);
    if let Some(k) = top_k {
        results.truncate(k);
    }
    results
}

/// Sort scored results in descending order.
///
/// Uses `f32::total_cmp` for deterministic ordering of NaN values.
#[inline]
fn sort_scored_desc<I>(results: &mut [(I, f32)]) {
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
}

/// Score normalization methods.
///
/// Different retrievers produce scores on different scales. Normalization
/// puts them on a common scale before combining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// Min-max normalization: `(score - min) / (max - min)` → [0, 1]
    ///
    /// Best when score distributions are similar. Sensitive to outliers.
    #[default]
    MinMax,
    /// Z-score normalization: `(score - mean) / std`, clipped to [-3, 3]
    ///
    /// More robust to outliers. Better when distributions differ.
    ZScore,
    /// Sum normalization: `score / sum(scores)`
    ///
    /// Preserves relative magnitudes. Useful when scores represent probabilities.
    Sum,
    /// Rank-based: convert scores to ranks, then normalize
    ///
    /// Sorts input by score (descending), assigns ranks 0..n-1, then normalizes
    /// to [0, 1] range where rank 0 (best) → 1.0, rank n-1 (worst) → 1/n.
    /// Ignores score magnitudes entirely. Most robust but loses information.
    Rank,
    /// Quantile normalization: maps scores to their percentile rank in [0, 1].
    ///
    /// More robust than min-max for non-Gaussian score distributions.
    /// Each score becomes `rank_among_scores / (n - 1)`.
    /// Referenced as an alternative to 3-sigma DBSF normalization
    /// when cosine similarity scores are not normally distributed.
    Quantile,
    /// Sigmoid normalization: `1 / (1 + exp(-score))` → (0, 1).
    ///
    /// Squashes unbounded scores to (0, 1) while preserving relative ordering.
    /// Useful for cross-encoder logits or other unbounded score ranges.
    Sigmoid,
    /// No normalization: use raw scores
    ///
    /// Only use when all retrievers use the same scale.
    None,
}

/// Normalize a list of scores using the specified method.
///
/// Returns a vector of (id, normalized_score) pairs.
pub fn normalize_scores<I: Clone>(results: &[(I, f32)], method: Normalization) -> Vec<(I, f32)> {
    if results.is_empty() {
        return Vec::new();
    }

    match method {
        Normalization::MinMax => {
            let (norm, off) = min_max_params(results);
            results
                .iter()
                .map(|(id, s)| (id.clone(), (s - off) * norm))
                .collect()
        }
        Normalization::ZScore => {
            let (mean, std) = zscore_params(results);
            results
                .iter()
                .map(|(id, s)| {
                    let z = if std > SCORE_RANGE_EPSILON {
                        ((s - mean) / std).clamp(-3.0, 3.0)
                    } else {
                        0.0
                    };
                    (id.clone(), z)
                })
                .collect()
        }
        Normalization::Sum => {
            let sum: f32 = results.iter().map(|(_, s)| s).sum();
            if sum.abs() < SCORE_RANGE_EPSILON {
                return results.to_vec();
            }
            results
                .iter()
                .map(|(id, s)| (id.clone(), s / sum))
                .collect()
        }
        Normalization::Rank => {
            // Sort by score (descending) first, then assign ranks
            // This ensures higher scores get better (lower) ranks
            // NaN/Inf values are treated as worst (sorted to end) so valid scores rank first
            let mut sorted: Vec<_> = results.to_vec();
            sorted.sort_by(|a, b| {
                // Custom comparison: finite values sort normally (descending),
                // non-finite values (NaN, Inf) sort to the end
                match (a.1.is_finite(), b.1.is_finite()) {
                    (true, true) => b.1.total_cmp(&a.1), // Both finite: descending
                    (true, false) => std::cmp::Ordering::Less, // a is finite, b is not: a first
                    (false, true) => std::cmp::Ordering::Greater, // b is finite, a is not: b first
                    (false, false) => std::cmp::Ordering::Equal, // Both non-finite: equal
                }
            });

            let n = sorted.len() as f32;
            sorted
                .iter()
                .enumerate()
                .map(|(rank, (id, _))| (id.clone(), 1.0 - (rank as f32 / n)))
                .collect()
        }
        Normalization::Quantile => {
            // Sort scores to assign percentile ranks
            let mut indexed: Vec<(usize, f32)> = results
                .iter()
                .enumerate()
                .map(|(i, (_, s))| (i, *s))
                .collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let n = indexed.len();
            let mut quantiles = vec![0.0f32; n];
            if n == 1 {
                quantiles[0] = 0.5; // Single item gets middle quantile
            } else {
                for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
                    quantiles[orig_idx] = rank as f32 / (n - 1) as f32;
                }
            }

            results
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (id.clone(), quantiles[i]))
                .collect()
        }
        Normalization::Sigmoid => results
            .iter()
            .map(|(id, s)| (id.clone(), 1.0 / (1.0 + (-s).exp())))
            .collect(),
        Normalization::None => results.to_vec(),
    }
}

/// Returns `(norm_factor, offset)` for min-max normalization.
///
/// Normalized score = `(score - offset) * norm_factor`
///
/// For single-element lists or lists where all scores are equal,
/// returns `(0.0, 0.0)` so each element contributes its raw score.
#[inline(always)]
fn min_max_params<I>(results: &[(I, f32)]) -> (f32, f32) {
    if results.is_empty() {
        return (1.0, 0.0);
    }
    let (min, max) = results
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, s)| {
            (lo.min(*s), hi.max(*s))
        });
    let range = max - min;
    if range < SCORE_RANGE_EPSILON {
        // All scores equal: just pass through the score (norm=1, offset=0)
        (1.0, 0.0)
    } else {
        (1.0 / range, min)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explainability
// ─────────────────────────────────────────────────────────────────────────────

/// A fused result with full provenance information for debugging and analysis.
///
/// Unlike the simple `Vec<(DocId, f32)>` returned by standard fusion functions,
/// `FusedResult` preserves which retrievers contributed each document, their
/// original ranks and scores, and how much each source contributed to the final score.
///
/// # Example
///
/// ```rust
/// use rankops::explain::{rrf_explain, RetrieverId};
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let retrievers = vec![
///     RetrieverId::new("bm25"),
///     RetrieverId::new("dense"),
/// ];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &retrievers,
///     rankops::RrfConfig::default(),
/// );
///
/// // d2 appears in both lists, so it has 2 source contributions
/// let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
/// assert_eq!(d2.explanation.sources.len(), 2);
/// assert_eq!(d2.explanation.consensus_score, 1.0); // 2/2 lists
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FusedResult<K> {
    /// Document identifier.
    pub id: K,
    /// Final fused score.
    pub score: f32,
    /// Final rank position (0-indexed, highest score = rank 0).
    pub rank: usize,
    /// Explanation of how this score was computed.
    pub explanation: Explanation,
}

/// Explanation of how a fused score was computed.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Explanation {
    /// Contributions from each retriever that contained this document.
    pub sources: Vec<SourceContribution>,
    /// Fusion method used (e.g., "rrf", "combsum").
    pub method: &'static str,
    /// Consensus score: fraction of retrievers that contained this document (0.0-1.0).
    ///
    /// - 1.0 = document appeared in all retrievers (strong consensus)
    /// - 0.5 = document appeared in half of retrievers
    /// - < 0.3 = document appeared in few retrievers (outlier)
    pub consensus_score: f32,
}

/// Contribution from a single retriever to a document's final score.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceContribution {
    /// Identifier for this retriever (e.g., "bm25", "dense_vector").
    pub retriever_id: String,
    /// Original rank in this retriever's list (0-indexed, None if not present).
    pub original_rank: Option<usize>,
    /// Original score from this retriever (None for rank-based methods or if not present).
    pub original_score: Option<f32>,
    /// Normalized score (for score-based methods, None for rank-based).
    pub normalized_score: Option<f32>,
    /// How much this source contributed to the final fused score.
    ///
    /// For RRF: `1/(k + rank)` or `weight / (k + rank)` for weighted.
    /// For CombSUM: normalized score.
    /// For CombMNZ: normalized score × overlap count.
    pub contribution: f32,
}

/// Retriever identifier for explainability.
///
/// Used to label which retriever each list comes from when calling explain variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RetrieverId {
    id: String,
}

impl RetrieverId {
    /// Create a new retriever identifier.
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self { id: id.into() }
    }

    /// Get the identifier string.
    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl From<&str> for RetrieverId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for RetrieverId {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

/// RRF with explainability: returns full provenance for each result.
///
/// This variant preserves which retrievers contributed each document, their
/// original ranks, and how much each source contributed to the final RRF score.
///
/// # Arguments
///
/// * `lists` - Ranked lists from each retriever
/// * `retriever_ids` - Identifiers for each retriever (must match `lists.len()`)
/// * `config` - RRF configuration
///
/// # Returns
///
/// Results sorted by fused score (descending), with full explanation metadata.
///
/// # Example
///
/// ```rust
/// use rankops::explain::{rrf_explain, RetrieverId};
/// use rankops::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let retrievers = vec![
///     RetrieverId::new("bm25"),
///     RetrieverId::new("dense"),
/// ];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &retrievers,
///     RrfConfig::default(),
/// );
///
/// // d2 appears in both lists at rank 1 and 0 respectively
/// let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
/// assert_eq!(d2.explanation.sources.len(), 2);
/// assert_eq!(d2.explanation.consensus_score, 1.0); // in both lists
///
/// // Check contributions
/// let bm25_contrib = d2.explanation.sources.iter()
///     .find(|s| s.retriever_id == "bm25")
///     .unwrap();
/// assert_eq!(bm25_contrib.original_rank, Some(1));
/// assert!(bm25_contrib.contribution > 0.0);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: RrfConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let k = config.k as f32;
    let num_retrievers = lists.len() as f32;

    // Track scores and provenance
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        for (rank, (id, original_score)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32);

            // Update score
            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            // Track provenance
            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: None, // RRF doesn't normalize
                    contribution,
                });
        }
    }

    // Build results with explanations
    let mut results: Vec<FusedResult<I>> = scores
        .into_iter()
        .map(|(id, score)| {
            let sources = provenance.remove(&id).unwrap_or_default();
            let consensus_score = sources.len() as f32 / num_retrievers;

            FusedResult {
                id,
                score,
                rank: 0, // Will be set after sorting
                explanation: Explanation {
                    sources,
                    method: "rrf",
                    consensus_score,
                },
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Set ranks
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    // Apply top_k
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }

    results
}

/// Analyze consensus patterns across retrievers.
///
/// Returns statistics about how retrievers agree or disagree on document relevance.
///
/// # Example
///
/// ```rust
/// use rankops::explain::{rrf_explain, analyze_consensus, RetrieverId};
/// use rankops::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
///     RrfConfig::default(),
/// );
///
/// let consensus = analyze_consensus(&explained);
/// // consensus.high_consensus contains documents in all retrievers
/// // consensus.single_source contains documents only in one retriever
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConsensusReport<K> {
    /// Documents that appeared in all retrievers (consensus_score == 1.0).
    pub high_consensus: Vec<K>,
    /// Documents that appeared in only one retriever (consensus_score < 0.5).
    pub single_source: Vec<K>,
    /// Documents with large rank disagreements across retrievers.
    ///
    /// A document might appear at rank 0 in one retriever but rank 50 in another,
    /// indicating retriever disagreement.
    pub rank_disagreement: Vec<(K, Vec<(String, usize)>)>,
}

/// Analyze consensus across fused results, identifying high-agreement and single-source items.
pub fn analyze_consensus<K: Clone + Eq + Hash>(results: &[FusedResult<K>]) -> ConsensusReport<K> {
    let mut high_consensus = Vec::new();
    let mut single_source = Vec::new();
    let mut rank_disagreement = Vec::new();

    for result in results {
        // High consensus: in all retrievers
        if result.explanation.consensus_score >= 1.0 - 1e-6 {
            high_consensus.push(result.id.clone());
        }

        // Single source: in only one retriever
        if result.explanation.sources.len() == 1 {
            single_source.push(result.id.clone());
        }

        // Rank disagreement: large spread in ranks
        if result.explanation.sources.len() > 1 {
            let ranks: Vec<usize> = result
                .explanation
                .sources
                .iter()
                .filter_map(|s| s.original_rank)
                .collect();
            if let (Some(&min_rank), Some(&max_rank)) = (ranks.iter().min(), ranks.iter().max()) {
                if max_rank - min_rank > 10 {
                    // Large disagreement threshold
                    let rank_info: Vec<(String, usize)> = result
                        .explanation
                        .sources
                        .iter()
                        .filter_map(|s| s.original_rank.map(|r| (s.retriever_id.clone(), r)))
                        .collect();
                    rank_disagreement.push((result.id.clone(), rank_info));
                }
            }
        }
    }

    ConsensusReport {
        high_consensus,
        single_source,
        rank_disagreement,
    }
}

/// Attribution statistics for each retriever.
///
/// Shows how much each retriever contributed to the top-k results.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RetrieverStats {
    /// Number of top-k documents this retriever contributed.
    pub top_k_count: usize,
    /// Average contribution strength for documents in top-k.
    pub avg_contribution: f32,
    /// Documents that only this retriever found (unique to this retriever).
    pub unique_docs: usize,
}

/// Attribute top-k results to retrievers.
///
/// Returns statistics showing which retrievers contributed most to the top-k results.
///
/// # Example
///
/// ```rust
/// use rankops::explain::{rrf_explain, attribute_top_k, RetrieverId};
/// use rankops::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
///     RrfConfig::default(),
/// );
///
/// let attribution = attribute_top_k(&explained, 5);
/// // attribution["bm25"].top_k_count shows how many top-5 docs came from BM25
/// ```
pub fn attribute_top_k<K: Clone + Eq + Hash>(
    results: &[FusedResult<K>],
    k: usize,
) -> std::collections::HashMap<String, RetrieverStats> {
    let top_k = results.iter().take(k);
    let mut stats: std::collections::HashMap<String, RetrieverStats> =
        std::collections::HashMap::new();

    // Track which documents each retriever found
    let mut retriever_docs: std::collections::HashMap<String, std::collections::HashSet<K>> =
        std::collections::HashMap::new();

    for result in top_k {
        for source in &result.explanation.sources {
            let entry =
                stats
                    .entry(source.retriever_id.clone())
                    .or_insert_with(|| RetrieverStats {
                        top_k_count: 0,
                        avg_contribution: 0.0,
                        unique_docs: 0,
                    });

            entry.top_k_count += 1;
            entry.avg_contribution += source.contribution;

            retriever_docs
                .entry(source.retriever_id.clone())
                .or_default()
                .insert(result.id.clone());
        }
    }

    // Calculate averages and unique counts
    for (retriever_id, stat) in &mut stats {
        if stat.top_k_count > 0 {
            stat.avg_contribution /= stat.top_k_count as f32;
        }

        // Count unique documents (only in this retriever)
        let this_retriever_docs = retriever_docs
            .get(retriever_id)
            .cloned()
            .unwrap_or_default();
        let other_retriever_docs: std::collections::HashSet<K> = retriever_docs
            .iter()
            .filter(|(id, _)| *id != retriever_id)
            .flat_map(|(_, docs)| docs.iter().cloned())
            .collect();

        stat.unique_docs = this_retriever_docs
            .difference(&other_retriever_docs)
            .count();
    }

    stats
}

/// CombSUM with explainability.
#[must_use]
pub fn combsum_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (rank, (id, original_score)) in items.iter().enumerate() {
            let normalized_score = (original_score - off) * norm;
            let contribution = normalized_score;

            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(normalized_score),
                    contribution,
                });
        }
    }

    build_explained_results(scores, provenance, num_retrievers, "combsum", config.top_k)
}

/// CombMNZ with explainability.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combmnz_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, (f32, u32)> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (rank, (id, original_score)) in items.iter().enumerate() {
            let normalized_score = (original_score - off) * norm;
            let contribution = normalized_score;

            let entry = scores.entry(id.clone()).or_insert((0.0, 0));
            entry.0 += contribution;
            entry.1 += 1;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(normalized_score),
                    contribution,
                });
        }
    }

    // Apply CombMNZ multiplier (overlap count)
    let mut final_scores: HashMap<I, f32> = HashMap::new();
    let mut final_provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (id, (sum, overlap_count)) in scores {
        let final_score = sum * overlap_count as f32;
        final_scores.insert(id.clone(), final_score);

        // Update contributions to reflect multiplier
        if let Some(mut sources) = provenance.remove(&id) {
            for source in &mut sources {
                source.contribution *= overlap_count as f32;
            }
            final_provenance.insert(id, sources);
        }
    }

    build_explained_results(
        final_scores,
        final_provenance,
        num_retrievers,
        "combmnz",
        config.top_k,
    )
}

/// DBSF with explainability.
#[must_use]
pub fn dbsf_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (rank, (id, original_score)) in items.iter().enumerate() {
            let z = if std > SCORE_RANGE_EPSILON {
                ((original_score - mean) / std).clamp(-3.0, 3.0)
            } else {
                0.0
            };
            let contribution = z;

            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(z),
                    contribution,
                });
        }
    }

    build_explained_results(scores, provenance, num_retrievers, "dbsf", config.top_k)
}

/// Helper to build explained results from scores and provenance.
fn build_explained_results<I: Clone + Eq + Hash>(
    scores: HashMap<I, f32>,
    mut provenance: HashMap<I, Vec<SourceContribution>>,
    num_retrievers: f32,
    method: &'static str,
    top_k: Option<usize>,
) -> Vec<FusedResult<I>> {
    let mut results: Vec<FusedResult<I>> = scores
        .into_iter()
        .map(|(id, score)| {
            let sources = provenance.remove(&id).unwrap_or_default();
            let consensus_score = sources.len() as f32 / num_retrievers;

            FusedResult {
                id,
                score,
                rank: 0, // Will be set after sorting
                explanation: Explanation {
                    sources,
                    method,
                    consensus_score,
                },
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));

    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    if let Some(k) = top_k {
        results.truncate(k);
    }

    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait-Based Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Fusion strategy enum for runtime dispatch.
///
/// This enables dynamic selection of fusion methods without trait objects.
///
/// # Example
///
/// ```rust
/// use rankops::FusionStrategy;
///
/// let list1 = vec![("d1", 1.0), ("d2", 0.5)];
/// let list2 = vec![("d2", 0.9), ("d3", 0.8)];
/// let strategy = FusionStrategy::rrf(60);
/// let result = strategy.fuse(&[&list1[..], &list2[..]]);
/// ```
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// RRF with custom k.
    Rrf {
        /// Smoothing constant (typically 60).
        k: u32,
    },
    /// CombSUM.
    CombSum,
    /// CombMNZ.
    CombMnz,
    /// Weighted fusion with custom weights.
    Weighted {
        /// Per-source fusion weights.
        weights: Vec<f32>,
        /// Whether to normalize scores before weighting.
        normalize: bool,
    },
}

impl FusionStrategy {
    /// Fuse multiple ranked lists.
    ///
    /// # Arguments
    /// * `runs` - Slice of ranked lists, each as (ID, score) pairs
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    pub fn fuse<I: Clone + Eq + Hash>(&self, runs: &[&[(I, f32)]]) -> Vec<(I, f32)> {
        match self {
            Self::Rrf { k } => rrf_multi(runs, RrfConfig::new(*k)),
            Self::CombSum => combsum_multi(runs, FusionConfig::default()),
            Self::CombMnz => combmnz_multi(runs, FusionConfig::default()),
            Self::Weighted { weights, normalize } => {
                if runs.len() != weights.len() {
                    // Mismatched lengths: return empty rather than panic
                    // Callers should validate inputs before calling fuse()
                    return Vec::new();
                }
                let lists: Vec<_> = runs
                    .iter()
                    .zip(weights.iter())
                    .map(|(run, &w)| (*run, w))
                    .collect();
                // Unwrap is safe here because we validate weights in weighted_multi
                // If weights sum to zero, returns empty Vec (graceful degradation)
                weighted_multi(&lists, *normalize, None).unwrap_or_default()
            }
        }
    }

    /// Human-readable name of this fusion method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rrf { .. } => "rrf",
            Self::CombSum => "combsum",
            Self::CombMnz => "combmnz",
            Self::Weighted { .. } => "weighted",
        }
    }

    /// Whether this method uses score values (true) or only ranks (false).
    pub fn uses_scores(&self) -> bool {
        match self {
            Self::Rrf { .. } => false,
            Self::CombSum | Self::CombMnz | Self::Weighted { .. } => true,
        }
    }
}

// Convenience constructors for FusionStrategy
impl FusionStrategy {
    /// Create RRF strategy with custom k.
    #[must_use]
    pub fn rrf(k: u32) -> Self {
        assert!(k >= 1, "k must be >= 1");
        Self::Rrf { k }
    }

    /// Create RRF strategy with default k=60.
    #[must_use]
    pub fn rrf_default() -> Self {
        Self::Rrf { k: 60 }
    }

    /// Create CombSUM strategy.
    #[must_use]
    pub fn combsum() -> Self {
        Self::CombSum
    }

    /// Create CombMNZ strategy.
    #[must_use]
    pub fn combmnz() -> Self {
        Self::CombMnz
    }

    /// Create weighted strategy with custom weights.
    #[must_use]
    pub fn weighted(weights: Vec<f32>, normalize: bool) -> Self {
        Self::Weighted { weights, normalize }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional Algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// CombMAX: maximum score across all lists.
///
/// Formula: `score(d) = max(s_r(d))` for all retrievers r containing d.
///
/// Use as a baseline or when you want to favor documents that score highly
/// in at least one retriever.
#[must_use]
pub fn combmax<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmax_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombMAX for 3+ result lists.
#[must_use]
pub fn combmax_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            scores
                .entry(id.clone())
                .and_modify(|max_score| *max_score = max_score.max(*s))
                .or_insert(*s);
        }
    }

    finalize(scores, config.top_k)
}

/// CombMIN: minimum score across all lists.
///
/// Formula: `score(d) = min(s_r(d))` for all retrievers r containing d.
///
/// # Historical Context
///
/// CombMIN emerged from the information retrieval meta-search literature of
/// the late 1990s alongside CombSUM, CombMAX, and CombMNZ. The "Comb" family
/// was systematically studied by Fox & Shaw (1994) and later by Lee (1997).
///
/// | Method | Formula | Intuition |
/// |--------|---------|-----------|
/// | CombSUM | Σ s_r(d) | Agreement across all retrievers |
/// | CombMAX | max s_r(d) | At least one retriever likes it |
/// | CombMIN | min s_r(d) | All retrievers agree (conservative) |
/// | CombMNZ | Σ s_r(d) × count | Reward overlap explicitly |
///
/// # When to Use CombMIN
///
/// - **High-precision requirements**: When false positives are costly
/// - **Consensus retrieval**: Only surface documents all systems agree on
/// - **Spam filtering**: A document must pass multiple filters
///
/// CombMIN is inherently **conservative**: a document with scores [0.9, 0.1]
/// gets score 0.1, while CombMAX would give 0.9.
///
/// # Caution
///
/// Documents appearing in only one list will have that single score as their
/// CombMIN. To require presence in multiple lists, combine with a threshold
/// on occurrence count.
///
/// # Reference
///
/// Fox & Shaw, "Combination of Multiple Searches", NIST TREC 1994.
/// Lee, "Analyses of Multiple Evidence Combination", SIGIR 1997.
#[must_use]
pub fn combmin<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmin_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombMIN for 3+ result lists.
#[must_use]
pub fn combmin_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            scores
                .entry(id.clone())
                .and_modify(|min_score| *min_score = min_score.min(*s))
                .or_insert(*s);
        }
    }

    finalize(scores, config.top_k)
}

/// CombMED: median score across all lists.
///
/// Formula: `score(d) = median(s_r(d))` for all retrievers r containing d.
///
/// More robust to outliers than CombMAX or CombSUM.
#[must_use]
pub fn combmed<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmed_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombMED for 3+ result lists.
#[must_use]
pub fn combmed_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut score_lists: HashMap<I, Vec<f32>> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            score_lists.entry(id.clone()).or_default().push(*s);
        }
    }

    let mut scores: HashMap<I, f32> = HashMap::new();
    for (id, mut score_vec) in score_lists {
        score_vec.sort_by(|a, b| a.total_cmp(b));
        let median = if score_vec.len() % 2 == 0 {
            let mid = score_vec.len() / 2;
            (score_vec[mid - 1] + score_vec[mid]) / 2.0
        } else {
            score_vec[score_vec.len() / 2]
        };
        scores.insert(id, median);
    }

    finalize(scores, config.top_k)
}

/// CombANZ: average of non-zero scores.
///
/// Formula: `score(d) = mean(s_r(d))` for all retrievers r containing d.
///
/// Similar to CombSUM but divides by count (average instead of sum).
#[must_use]
pub fn combanz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combanz_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombANZ for 3+ result lists.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combanz_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<I, (f32, usize)> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            let entry = scores.entry(id.clone()).or_insert((0.0, 0));
            entry.0 += s;
            entry.1 += 1;
        }
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (sum, count))| {
            // count is always >= 1 because we only add entries when we see items
            debug_assert!(count > 0, "Count should always be > 0 for CombANZ");
            (id, sum / count as f32)
        })
        .collect();
    sort_scored_desc(&mut results);
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }
    results
}

/// Rank-Biased Centroids (RBC) fusion.
///
/// Handles variable-length lists gracefully by using a geometric discount
/// that depends on list length. More robust than RRF when lists have very
/// different lengths.
///
/// Formula: `score(d) = Σ (1 - p)^rank / (1 - p^N)` where:
/// - `p` is the persistence parameter (default 0.8, higher = more top-heavy)
/// - `N` is the list length
/// - `rank` is 0-indexed
///
/// From Bailey et al. (2017). Better than RRF when lists have different lengths.
#[must_use]
pub fn rbc<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    rbc_multi(&[results_a, results_b], 0.8)
}

/// RBC for 3+ result lists with custom persistence.
///
/// # Arguments
/// * `lists` - Ranked lists to fuse
/// * `persistence` - Persistence parameter (0.0-1.0), default 0.8. Higher = more top-heavy.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rbc_multi<I, L>(lists: &[L], persistence: f32) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    let p = persistence.clamp(0.0, 1.0);
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let n = items.len() as f32;
        let denominator = 1.0 - p.powi(n as i32);

        for (rank, (id, _)) in items.iter().enumerate() {
            let numerator = (1.0 - p).powi(rank as i32);
            let contribution = if denominator > 1e-9 {
                numerator / denominator
            } else {
                0.0
            };

            *scores.entry(id.clone()).or_insert(0.0) += contribution;
        }
    }

    finalize(scores, None)
}

/// Condorcet fusion (pairwise comparison voting).
///
/// For each pair of documents, counts how many retrievers prefer one over the other.
/// Documents that beat all others in pairwise comparisons win.
///
/// This is a simplified Condorcet method. Full Condorcet (Kemeny optimal) is NP-hard.
///
/// # Algorithm
///
/// 1. For each document pair (d1, d2), count retrievers where d1 ranks higher than d2
/// 2. Document d1 "beats" d2 if majority of retrievers prefer d1
/// 3. Score = number of documents that this document beats
///
/// More robust to outliers than score-based methods.
#[must_use]
pub fn condorcet<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    condorcet_multi(&[results_a, results_b], FusionConfig::default())
}

/// Condorcet for 3+ result lists.
#[must_use]
pub fn condorcet_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    // Build rank maps: doc_id -> rank in each list
    let mut doc_ranks: HashMap<I, Vec<Option<usize>>> = HashMap::new();
    let mut all_docs: std::collections::HashSet<I> = std::collections::HashSet::new();

    for list in lists {
        let items = list.as_ref();
        for doc_id in items.iter().map(|(id, _)| id) {
            all_docs.insert(doc_id.clone());
        }
    }

    // Initialize all docs with None ranks
    for doc_id in &all_docs {
        doc_ranks.insert(doc_id.clone(), vec![None; lists.len()]);
    }

    // Fill in actual ranks
    for (list_idx, list) in lists.iter().enumerate() {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            if let Some(ranks) = doc_ranks.get_mut(id) {
                ranks[list_idx] = Some(rank);
            }
        }
    }

    // For each document, count how many others it beats
    let mut scores: HashMap<I, f32> = HashMap::new();
    let doc_vec: Vec<I> = all_docs.into_iter().collect();

    for (i, d1) in doc_vec.iter().enumerate() {
        let mut wins = 0;

        for (j, d2) in doc_vec.iter().enumerate() {
            if i == j {
                continue;
            }

            // Count lists where d1 ranks better than d2
            let d1_ranks = &doc_ranks[d1];
            let d2_ranks = &doc_ranks[d2];

            let mut d1_wins = 0;
            for (r1, r2) in d1_ranks.iter().zip(d2_ranks.iter()) {
                match (r1, r2) {
                    (Some(rank1), Some(rank2)) if rank1 < rank2 => d1_wins += 1,
                    (Some(_), None) => d1_wins += 1, // d1 present, d2 not
                    _ => {}
                }
            }

            // Majority wins
            if d1_wins > lists.len() / 2 {
                wins += 1;
            }
        }

        scores.insert(d1.clone(), wins as f32);
    }

    finalize(scores, config.top_k)
}

/// Copeland fusion -- pairwise net wins across ranked lists.
///
/// For each document pair (d1, d2), counts how many input lists rank d1 above d2.
/// Score = (pairwise wins) - (pairwise losses). This provides a complete ranking
/// where Condorcet only counts wins.
///
/// Copeland is more discriminative than Condorcet and Borda, and satisfies the
/// Condorcet winner criterion (a document beating all others pairwise always ranks first).
///
/// # Reference
///
/// Tyomkin & Kurland, "Analyzing Fusion Methods Using the Condorcet Rule," SIGIR 2024.
/// Shows Copeland beats both CondorcetFuse and Borda on TREC tracks.
///
/// # Complexity
///
/// O(n^2 * m) where n = total documents, m = number of input lists.
#[must_use]
pub fn copeland<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    copeland_multi(&[results_a, results_b], FusionConfig::default())
}

/// Copeland fusion for 3+ result lists.
#[must_use]
pub fn copeland_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    // Build rank maps: doc_id -> rank in each list (None = absent)
    let mut doc_ranks: HashMap<I, Vec<Option<usize>>> = HashMap::new();
    let mut all_docs: std::collections::HashSet<I> = std::collections::HashSet::new();

    for list in lists {
        for (id, _) in list.as_ref() {
            all_docs.insert(id.clone());
        }
    }

    for doc_id in &all_docs {
        doc_ranks.insert(doc_id.clone(), vec![None; lists.len()]);
    }

    for (list_idx, list) in lists.iter().enumerate() {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            if let Some(ranks) = doc_ranks.get_mut(id) {
                ranks[list_idx] = Some(rank);
            }
        }
    }

    // Copeland: score = wins - losses (net pairwise preference)
    let mut scores: HashMap<I, f32> = HashMap::new();
    let doc_vec: Vec<I> = all_docs.into_iter().collect();

    for (i, d1) in doc_vec.iter().enumerate() {
        let mut net = 0i32;

        for (j, d2) in doc_vec.iter().enumerate() {
            if i == j {
                continue;
            }

            let d1_ranks = &doc_ranks[d1];
            let d2_ranks = &doc_ranks[d2];

            let mut d1_preferred = 0;
            let mut d2_preferred = 0;

            for (r1, r2) in d1_ranks.iter().zip(d2_ranks.iter()) {
                match (r1, r2) {
                    (Some(rank1), Some(rank2)) => {
                        if rank1 < rank2 {
                            d1_preferred += 1;
                        } else if rank2 < rank1 {
                            d2_preferred += 1;
                        }
                    }
                    (Some(_), None) => d1_preferred += 1, // present beats absent
                    (None, Some(_)) => d2_preferred += 1,
                    (None, None) => {}
                }
            }

            // Majority rule
            if d1_preferred > d2_preferred {
                net += 1; // win
            } else if d2_preferred > d1_preferred {
                net -= 1; // loss
            }
            // tie: net unchanged
        }

        scores.insert(d1.clone(), net as f32);
    }

    finalize(scores, config.top_k)
}

/// Median Rank Aggregation.
///
/// Scores each document by the median of its ranks across all input lists.
/// Documents not in a list receive a penalty rank of `max_rank + 1`.
///
/// Lower median rank = higher fusion score. The output is normalized to
/// descending scores (higher = better) for consistency with other fusion methods.
///
/// Outlier-robust: a single bad retriever rank has minimal effect on the median.
#[must_use]
pub fn median_rank<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    median_rank_multi(&[results_a, results_b], FusionConfig::default())
}

/// Median Rank Aggregation for 3+ result lists.
#[must_use]
pub fn median_rank_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    // Find penalty rank: max list length + 1
    let max_len = lists.iter().map(|l| l.as_ref().len()).max().unwrap_or(0);
    let penalty_rank = max_len + 1;

    // Collect all docs and their ranks
    let mut doc_ranks: HashMap<I, Vec<usize>> = HashMap::new();

    for list in lists {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            doc_ranks.entry(id.clone()).or_default().push(rank);
        }
    }

    // Compute median rank for each doc, converting to a descending score
    let mut scores: HashMap<I, f32> = HashMap::new();

    for (id, mut ranks) in doc_ranks {
        // Pad with penalty_rank for lists where doc is absent
        while ranks.len() < lists.len() {
            ranks.push(penalty_rank);
        }
        ranks.sort_unstable();

        let median = if ranks.len() % 2 == 1 {
            ranks[ranks.len() / 2] as f32
        } else {
            (ranks[ranks.len() / 2 - 1] + ranks[ranks.len() / 2]) as f32 / 2.0
        };

        // Invert: lower median rank -> higher score
        // Use 1/(1+median) so scores are in (0, 1] and descending
        scores.insert(id, 1.0 / (1.0 + median));
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Diversity-Aware Reranking
// ─────────────────────────────────────────────────────────────────────────────

/// Maximal Marginal Relevance (MMR) configuration.
///
/// # Background
///
/// MMR was introduced by Carbonell & Goldstein (1998) to address a fundamental
/// tension in information retrieval: **relevance vs. diversity**.
///
/// Traditional ranking optimizes relevance only, leading to redundant results.
/// If the top-5 results are all about the same aspect of a topic, the user
/// gains little from results 2-5.
///
/// MMR balances:
/// - **Relevance**: How well does this document match the query?
/// - **Diversity**: How different is this document from already-selected ones?
///
/// # Historical Context
///
/// | Year | Development |
/// |------|-------------|
/// | 1998 | MMR introduced (Carbonell & Goldstein) |
/// | 2008 | xQuAD extends MMR with explicit subtopics |
/// | 2012 | PM-2 proportional model |
/// | 2020s | MMR widely used in RAG to reduce redundancy |
///
/// MMR remains the go-to algorithm for diversity because:
/// 1. Simple to implement and explain
/// 2. Single tunable parameter (λ)
/// 3. Works with any similarity function
/// 4. Greedy selection is fast
///
/// # Mathematical Formulation
///
/// At each step, select the document that maximizes:
///
/// ```text
/// MMR(d) = λ · Sim(d, q) - (1-λ) · max_{s∈S} Sim(d, s)
/// ```
///
/// Where:
/// - `d` is a candidate document
/// - `q` is the query
/// - `S` is the set of already-selected documents
/// - `Sim(d, q)` is relevance (query-document similarity)
/// - `max_{s∈S} Sim(d, s)` is redundancy (max similarity to any selected doc)
/// - `λ` in `[0,1]` balances relevance and diversity
///
/// # The λ Parameter
///
/// | λ Value | Effect |
/// |---------|--------|
/// | λ = 1.0 | Pure relevance (standard ranking) |
/// | λ = 0.7 | Mild diversity (typical for search) |
/// | λ = 0.5 | Balanced relevance/diversity |
/// | λ = 0.3 | Strong diversity preference |
/// | λ = 0.0 | Pure diversity (maximally spread results) |
///
/// # Computational Complexity
///
/// - Naïve: O(k·n·|S|) where k=results wanted, n=candidates, |S|=selected set
/// - In practice: O(k·n²) worst case, often much better with pruning
///
/// # Reference
///
/// Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for
/// Reordering Documents and Producing Summaries", SIGIR 1998.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmrConfig {
    /// Balance parameter λ ∈ `[0,1]`.
    /// - λ = 1.0: pure relevance (no diversity)
    /// - λ = 0.5: balanced
    /// - λ = 0.0: pure diversity
    pub lambda: f32,
    /// Maximum results to return.
    pub top_k: usize,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            lambda: 0.7,
            top_k: 10,
        }
    }
}

impl MmrConfig {
    /// Create MMR config with specified lambda.
    ///
    /// # Arguments
    ///
    /// - `lambda`: Balance between relevance and diversity. Valid: `[0.0, 1.0]`.
    ///
    /// # Panics
    ///
    /// Panics if lambda is outside [0.0, 1.0].
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "lambda must be in [0.0, 1.0], got {lambda}"
        );
        Self { lambda, top_k: 10 }
    }

    /// Set the number of results to return.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

/// Maximal Marginal Relevance reranking.
///
/// Reranks candidates to balance relevance and diversity using greedy selection.
/// At each step, selects the candidate that maximizes:
///
/// ```text
/// MMR(d) = λ · relevance(d) - (1-λ) · max_redundancy(d, selected)
/// ```
///
/// # Arguments
///
/// - `candidates`: List of (id, relevance_score) tuples, typically from initial retrieval
/// - `similarities`: Function returning similarity between two IDs. Should return
///   values in `[0,1]` where 1 = identical, 0 = completely different.
/// - `config`: MMR configuration (lambda, top_k)
///
/// # Returns
///
/// Reranked list of (id, mmr_score) tuples, length = min(top_k, candidates.len())
///
/// # Example
///
/// ```rust
/// use rankops::{mmr, MmrConfig};
/// use std::collections::HashMap;
///
/// // Candidates with relevance scores
/// let candidates = vec![
///     ("doc1".to_string(), 0.95),
///     ("doc2".to_string(), 0.90), // Similar to doc1
///     ("doc3".to_string(), 0.85), // Different topic
///     ("doc4".to_string(), 0.80),
/// ];
///
/// // Similarity matrix (doc1 and doc2 are similar)
/// let mut sims: HashMap<(String, String), f32> = HashMap::new();
/// sims.insert(("doc1".to_string(), "doc2".to_string()), 0.9);
/// sims.insert(("doc2".to_string(), "doc1".to_string()), 0.9);
/// // All other pairs: 0.1 (different)
///
/// let similarity = |a: &String, b: &String| -> f32 {
///     if a == b { return 1.0; }
///     *sims.get(&(a.clone(), b.clone())).unwrap_or(&0.1)
/// };
///
/// let config = MmrConfig::new(0.7).with_top_k(3);
/// let results = mmr(&candidates, similarity, config);
///
/// // doc1 selected first (highest relevance)
/// // doc3 likely selected second (doc2 penalized for similarity to doc1)
/// ```
///
/// # Implementation Notes
///
/// 1. Relevance scores are normalized to `[0,1]` before MMR computation
/// 2. First document is always the highest-relevance candidate
/// 3. Ties broken by original relevance score
///
/// # Performance
///
/// For n candidates and k results: O(k·n) similarity evaluations.
/// With caching in the similarity function, this is typically fast.
#[must_use]
pub fn mmr<I, F>(candidates: &[(I, f32)], similarity: F, config: MmrConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    F: Fn(&I, &I) -> f32,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    let k = config.top_k.min(candidates.len());
    let lambda = config.lambda;

    // Normalize relevance scores to [0,1]
    let max_rel = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_rel = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let rel_range = max_rel - min_rel;

    let normalized: Vec<(I, f32)> = if rel_range > SCORE_RANGE_EPSILON {
        candidates
            .iter()
            .map(|(id, s)| (id.clone(), (s - min_rel) / rel_range))
            .collect()
    } else {
        // All scores equal—treat as uniform relevance
        candidates.iter().map(|(id, _)| (id.clone(), 1.0)).collect()
    };

    // Track selected documents and remaining candidates
    let mut selected: Vec<(I, f32)> = Vec::with_capacity(k);
    let mut remaining: Vec<(I, f32)> = normalized;

    // Greedy selection
    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (idx, (cand_id, cand_rel)) in remaining.iter().enumerate() {
            // Relevance term: λ · Sim(d, q)
            let relevance_term = lambda * cand_rel;

            // Redundancy term: (1-λ) · max_{s∈S} Sim(d, s)
            let redundancy_term = if selected.is_empty() {
                0.0
            } else {
                let max_sim = selected
                    .iter()
                    .map(|(sel_id, _)| similarity(cand_id, sel_id))
                    .fold(0.0_f32, f32::max);
                (1.0 - lambda) * max_sim
            };

            // MMR score
            let mmr_score = relevance_term - redundancy_term;

            if mmr_score > best_mmr {
                best_mmr = mmr_score;
                best_idx = idx;
            }
        }

        // Move best candidate to selected set
        let (id, _) = remaining.remove(best_idx);
        selected.push((id, best_mmr));
    }

    selected
}

/// MMR with precomputed similarity matrix.
///
/// More efficient when similarities are already computed (e.g., from embeddings).
///
/// # Arguments
///
/// - `candidates`: List of (id, relevance_score) tuples
/// - `sim_matrix`: Maps (id_a, id_b) -> similarity. Missing pairs treated as 0.
/// - `config`: MMR configuration
///
/// # Example
///
/// ```rust
/// use rankops::{mmr_with_matrix, MmrConfig};
/// use std::collections::HashMap;
///
/// let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];
///
/// let mut matrix: HashMap<(&str, &str), f32> = HashMap::new();
/// matrix.insert(("a", "b"), 0.8); // a and b are similar
/// matrix.insert(("b", "a"), 0.8);
/// matrix.insert(("a", "c"), 0.2); // a and c are different
/// matrix.insert(("c", "a"), 0.2);
/// matrix.insert(("b", "c"), 0.3);
/// matrix.insert(("c", "b"), 0.3);
///
/// let config = MmrConfig::new(0.5).with_top_k(2);
/// let results = mmr_with_matrix(&candidates, &matrix, config);
/// ```
#[must_use]
pub fn mmr_with_matrix<I: Clone + Eq + Hash>(
    candidates: &[(I, f32)],
    sim_matrix: &HashMap<(I, I), f32>,
    config: MmrConfig,
) -> Vec<(I, f32)> {
    let similarity =
        |a: &I, b: &I| -> f32 { *sim_matrix.get(&(a.clone(), b.clone())).unwrap_or(&0.0) };
    mmr(candidates, similarity, config)
}

/// MMR for embedding-based retrieval.
///
/// Computes cosine similarity between embedding vectors on-the-fly.
/// Use this when you have dense embeddings and want diversity without
/// precomputing the full similarity matrix.
///
/// # Arguments
///
/// - `candidates`: List of (id, relevance_score, embedding) tuples
/// - `config`: MMR configuration
///
/// # Returns
///
/// Reranked list of (id, mmr_score) tuples.
///
/// # Performance Note
///
/// Cosine similarity is computed on-demand. For very large candidate sets
/// (>1000), consider precomputing top-k similarities per candidate.
#[must_use]
pub fn mmr_embeddings<I: Clone + Eq + Hash>(
    candidates: &[(I, f32, Vec<f32>)],
    config: MmrConfig,
) -> Vec<(I, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Build embedding lookup
    let embeddings: HashMap<I, &[f32]> = candidates
        .iter()
        .map(|(id, _, emb)| (id.clone(), emb.as_slice()))
        .collect();

    // Convert to (id, score) for mmr()
    let id_scores: Vec<(I, f32)> = candidates
        .iter()
        .map(|(id, score, _)| (id.clone(), *score))
        .collect();

    let similarity = |a: &I, b: &I| -> f32 {
        match (embeddings.get(a), embeddings.get(b)) {
            (Some(emb_a), Some(emb_b)) => cosine_similarity(emb_a, emb_b),
            _ => 0.0,
        }
    };

    mmr(&id_scores, similarity, config)
}

/// Cosine similarity between two vectors.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimization and Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Relevance judgments (qrels) for a query.
///
/// Maps document IDs to relevance scores (typically 0=not relevant, 1=relevant, 2=highly relevant).
pub type Qrels<K> = std::collections::HashMap<K, u32>;

/// Normalized Discounted Cumulative Gain at k.
///
/// Measures ranking quality by rewarding relevant documents that appear early.
/// NDCG@k ranges from 0.0 (worst) to 1.0 (perfect).
///
/// # Formula
///
/// NDCG@k = DCG@k / IDCG@k
///
/// where:
/// - DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i in [0, k)
/// - IDCG@k = DCG@k of the ideal ranking (sorted by relevance descending)
pub fn ndcg_at_k<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    if qrels.is_empty() || results.is_empty() {
        return 0.0;
    }

    let k = k.min(results.len());
    let mut dcg = 0.0;

    for (i, (id, _)) in results.iter().take(k).enumerate() {
        if let Some(&rel) = qrels.get(id) {
            let gain = (2.0_f32.powi(rel as i32) - 1.0) / ((i + 2) as f32).log2();
            dcg += gain;
        }
    }

    // Compute IDCG (ideal DCG)
    let mut ideal_relevances: Vec<u32> = qrels.values().copied().collect();
    ideal_relevances.sort_by(|a, b| b.cmp(a)); // Descending

    let mut idcg = 0.0;
    for (i, &rel) in ideal_relevances.iter().take(k).enumerate() {
        let gain = (2.0_f32.powi(rel as i32) - 1.0) / ((i + 2) as f32).log2();
        idcg += gain;
    }

    if idcg > 1e-9 {
        dcg / idcg
    } else {
        0.0
    }
}

/// Mean Reciprocal Rank.
///
/// Measures the rank of the first relevant document. MRR ranges from 0.0 to 1.0.
///
/// Formula: MRR = 1 / rank_of_first_relevant
pub fn mrr<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>) -> f32 {
    for (rank, (id, _)) in results.iter().enumerate() {
        if qrels.contains_key(id) && qrels[id] > 0 {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Recall at k.
///
/// Fraction of relevant documents that appear in the top-k results.
///
/// Formula: Recall@k = |relevant_docs_in_top_k| / |total_relevant_docs|
pub fn recall_at_k<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    let total_relevant = qrels.values().filter(|&&rel| rel > 0).count();
    if total_relevant == 0 {
        return 0.0;
    }

    let k = k.min(results.len());
    let relevant_in_top_k = results
        .iter()
        .take(k)
        .filter(|(id, _)| qrels.get(id).is_some_and(|&rel| rel > 0))
        .count();

    relevant_in_top_k as f32 / total_relevant as f32
}

/// Precision at k.
///
/// Fraction of top-k results that are relevant.
///
/// Formula: Precision@k = |relevant_docs_in_top_k| / k
pub fn precision_at_k<K: Clone + Eq + Hash>(
    results: &[(K, f32)],
    qrels: &Qrels<K>,
    k: usize,
) -> f32 {
    if k == 0 || results.is_empty() {
        return 0.0;
    }

    let k = k.min(results.len());
    let relevant_in_top_k = results
        .iter()
        .take(k)
        .filter(|(id, _)| qrels.get(id).is_some_and(|&rel| rel > 0))
        .count();

    relevant_in_top_k as f32 / k as f32
}

/// Mean Average Precision (MAP).
///
/// Average of precision values at each rank where a relevant document appears.
/// MAP is the default metric for MTEB Reranking and TREC evaluations.
///
/// Formula: MAP = (1/|R|) * Σ Precision@k * rel(k)
///
/// where R is the set of relevant documents and rel(k) is 1 if the document
/// at rank k is relevant.
pub fn map<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>) -> f32 {
    let total_relevant = qrels.values().filter(|&&rel| rel > 0).count();
    if total_relevant == 0 || results.is_empty() {
        return 0.0;
    }

    let mut sum_precision = 0.0;
    let mut relevant_seen = 0;

    for (i, (id, _)) in results.iter().enumerate() {
        if qrels.get(id).is_some_and(|&rel| rel > 0) {
            relevant_seen += 1;
            // Precision at this rank position
            sum_precision += relevant_seen as f32 / (i + 1) as f32;
        }
    }

    sum_precision / total_relevant as f32
}

/// Mean Average Precision at k (MAP@k).
///
/// Like [`map`] but only considers the top-k results.
/// Used by MTEB Reranking (MAP@10) and TREC evaluations.
pub fn map_at_k<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    let total_relevant = qrels.values().filter(|&&rel| rel > 0).count();
    if total_relevant == 0 || results.is_empty() || k == 0 {
        return 0.0;
    }

    let k = k.min(results.len());
    let mut sum_precision = 0.0;
    let mut relevant_seen = 0;

    for (i, (id, _)) in results.iter().take(k).enumerate() {
        if qrels.get(id).is_some_and(|&rel| rel > 0) {
            relevant_seen += 1;
            sum_precision += relevant_seen as f32 / (i + 1) as f32;
        }
    }

    // Divide by min(total_relevant, k) for MAP@k — standard IR convention
    // when k < total_relevant, we can only observe k documents
    sum_precision / total_relevant.min(k) as f32
}

/// Hit Rate (Success@k).
///
/// Binary: 1.0 if any relevant document appears in top-k, 0.0 otherwise.
/// Commonly reported in RAG evaluation pipelines.
pub fn hit_rate<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    if k == 0 || results.is_empty() {
        return 0.0;
    }

    let k = k.min(results.len());
    let hit = results
        .iter()
        .take(k)
        .any(|(id, _)| qrels.get(id).is_some_and(|&rel| rel > 0));

    if hit {
        1.0
    } else {
        0.0
    }
}

/// Optimization configuration for hyperparameter search.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Fusion method to optimize.
    pub method: FusionMethod,
    /// Metric to optimize (NDCG, MRR, or Recall).
    pub metric: OptimizeMetric,
    /// Parameter grid to search.
    pub param_grid: ParamGrid,
}

/// Metric to optimize during hyperparameter search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeMetric {
    /// NDCG@k (default k=10).
    Ndcg {
        /// Cutoff depth for NDCG evaluation.
        k: usize,
    },
    /// Mean Reciprocal Rank.
    Mrr,
    /// Recall@k (default k=10).
    Recall {
        /// Cutoff depth for recall evaluation.
        k: usize,
    },
    /// Precision@k.
    Precision {
        /// Cutoff depth for precision evaluation.
        k: usize,
    },
    /// Mean Average Precision (full ranking).
    Map,
    /// MAP@k (truncated at k).
    MapAtK {
        /// Cutoff depth for MAP evaluation.
        k: usize,
    },
    /// Hit Rate / Success@k.
    HitRate {
        /// Cutoff depth.
        k: usize,
    },
}

impl Default for OptimizeMetric {
    fn default() -> Self {
        Self::Ndcg { k: 10 }
    }
}

/// Parameter grid for optimization.
#[derive(Debug, Clone)]
pub enum ParamGrid {
    /// Grid search over RRF k values.
    RrfK {
        /// k values to search over.
        values: Vec<u32>,
    },
    /// Grid search over weighted fusion weights.
    Weighted {
        /// Weight vectors to evaluate.
        weight_combinations: Vec<Vec<f32>>,
    },
}

/// Optimized parameters from hyperparameter search.
#[derive(Debug, Clone)]
pub struct OptimizedParams {
    /// Best metric value found.
    pub best_score: f32,
    /// Parameters that achieved best score.
    pub best_params: String,
}

/// Evaluate a ranked list using the specified metric.
///
/// Convenience function that dispatches to the appropriate metric function.
pub fn evaluate_metric<K: Clone + Eq + Hash>(
    results: &[(K, f32)],
    qrels: &Qrels<K>,
    metric: OptimizeMetric,
) -> f32 {
    match metric {
        OptimizeMetric::Ndcg { k } => ndcg_at_k(results, qrels, k),
        OptimizeMetric::Mrr => mrr(results, qrels),
        OptimizeMetric::Recall { k } => recall_at_k(results, qrels, k),
        OptimizeMetric::Precision { k } => precision_at_k(results, qrels, k),
        OptimizeMetric::Map => map(results, qrels),
        OptimizeMetric::MapAtK { k } => map_at_k(results, qrels, k),
        OptimizeMetric::HitRate { k } => hit_rate(results, qrels, k),
    }
}

/// Optimize fusion hyperparameters using grid search.
///
/// Given relevance judgments (qrels) and multiple retrieval runs, searches
/// over parameter space to find the best configuration.
///
/// # Example
///
/// ```rust
/// use rankops::optimize::{optimize_fusion, OptimizeConfig, OptimizeMetric, ParamGrid};
/// use rankops::FusionMethod;
///
/// let qrels = std::collections::HashMap::from([
///     ("doc1", 2), // highly relevant
///     ("doc2", 1), // relevant
/// ]);
///
/// let runs = vec![
///     vec![("doc1", 0.9), ("doc2", 0.8)],
///     vec![("doc2", 0.9), ("doc1", 0.7)],
/// ];
///
/// let config = OptimizeConfig {
///     method: FusionMethod::Rrf { k: 60 }, // will be overridden
///     metric: OptimizeMetric::Ndcg { k: 10 },
///     param_grid: ParamGrid::RrfK {
///         values: vec![20, 40, 60, 100],
///     },
/// };
///
/// let optimized = optimize_fusion(&qrels, &runs, config);
/// println!("Best k: {}, score: {:.4}", optimized.best_params, optimized.best_score);
/// ```
pub fn optimize_fusion<K: Clone + Eq + Hash>(
    qrels: &Qrels<K>,
    runs: &[Vec<(K, f32)>],
    config: OptimizeConfig,
) -> OptimizedParams {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_params = String::new();

    match config.param_grid {
        ParamGrid::RrfK { values } => {
            for k in values {
                let method = FusionMethod::Rrf { k };
                let fused = method.fuse_multi(runs);

                let score = evaluate_metric(&fused, qrels, config.metric);

                if score > best_score {
                    best_score = score;
                    best_params = format!("k={}", k);
                }
            }
        }
        ParamGrid::Weighted {
            ref weight_combinations,
        } => {
            for weights in weight_combinations {
                if weights.len() != runs.len() {
                    continue;
                }
                let lists: Vec<(&[(K, f32)], f32)> = runs
                    .iter()
                    .zip(weights.iter())
                    .map(|(run, &w)| (run.as_slice(), w))
                    .collect();

                if let Ok(fused) = weighted_multi(&lists, true, None) {
                    let score = evaluate_metric(&fused, qrels, config.metric);

                    if score > best_score {
                        best_score = score;
                        best_params = format!("weights={:?}", weights);
                    }
                }
            }
        }
    }

    OptimizedParams {
        best_score,
        best_params,
    }
}

/// Optimization module exports.
pub mod optimize {
    pub use crate::{
        evaluate_metric, hit_rate, map, map_at_k, mrr, ndcg_at_k, optimize_fusion, precision_at_k,
        recall_at_k, OptimizeConfig, OptimizeMetric, OptimizedParams, ParamGrid, Qrels,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked<'a>(ids: &[&'a str]) -> Vec<(&'a str, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, 1.0 - i as f32 * 0.1))
            .collect()
    }

    #[test]
    fn rrf_basic() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = rrf(&a, &b);

        assert!(f.iter().position(|(id, _)| *id == "d2").unwrap() < 2);
    }

    #[test]
    fn rrf_with_top_k() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(2));

        assert_eq!(f.len(), 2);
    }

    #[test]
    fn rrf_into_works() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let mut out = Vec::new();

        rrf_into(&a, &b, RrfConfig::default(), &mut out);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, "d2");
    }

    #[test]
    fn rrf_score_formula() {
        let a = vec![("d1", 1.0)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

        let expected = 1.0 / 60.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    /// Verify RRF score formula: score(d) = Σ 1/(k + rank) for all lists containing d
    #[test]
    fn rrf_exact_score_computation() {
        // d1 at rank 0 in list A, rank 2 in list B
        // With k=60: score = 1/(60+0) + 1/(60+2) = 1/60 + 1/62
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7)];

        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

        // Find d1's score
        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 1.0 / 60.0 + 1.0 / 62.0; // rank 0 in A + rank 2 in B

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    /// Verify ISR score formula: score(d) = Σ 1/sqrt(k + rank)
    #[test]
    fn isr_exact_score_computation() {
        // d1 at rank 0 in list A, rank 2 in list B
        // With k=1: score = 1/sqrt(1+0) + 1/sqrt(1+2) = 1 + 1/sqrt(3)
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7)];

        let f = isr_with_config(&a, &b, RrfConfig::new(1));

        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 1.0 / 1.0_f32.sqrt() + 1.0 / 3.0_f32.sqrt();

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    /// Verify Borda score formula: score(d) = Σ (N - rank) where N = list length
    #[test]
    fn borda_exact_score_computation() {
        // List A: 3 items, d1 at rank 0 -> score = 3-0 = 3
        // List B: 4 items, d1 at rank 2 -> score = 4-2 = 2
        // Total d1 score = 3 + 2 = 5
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7), ("d6", 0.6)];

        let f = borda(&a, &b);

        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 3.0 + 2.0; // (3-0) + (4-2)

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    #[test]
    fn rrf_weighted_applies_weights() {
        // d1 appears in list_a (rank 0), d2 appears in list_b (rank 0)
        let list_a = [("d1", 0.0)];
        let list_b = [("d2", 0.0)];

        // Weight list_b 3x more than list_a
        let weights = [0.25, 0.75];
        let f = rrf_weighted(&[&list_a[..], &list_b[..]], &weights, RrfConfig::new(60)).unwrap();

        // d2 should rank higher because its list has 3x the weight
        assert_eq!(f[0].0, "d2", "weighted RRF should favor higher-weight list");

        // Verify score formula: w / (k + rank)
        // d1: 0.25 / 60 = 0.00417
        // d2: 0.75 / 60 = 0.0125
        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let d2_score = f.iter().find(|(id, _)| *id == "d2").unwrap().1;
        assert!(
            d2_score > d1_score * 2.0,
            "d2 should score ~3x higher than d1"
        );
    }

    #[test]
    fn rrf_weighted_zero_weights_error() {
        let list_a = [("d1", 0.0)];
        let list_b = [("d2", 0.0)];
        let weights = [0.0, 0.0];

        let result = rrf_weighted(&[&list_a[..], &list_b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::ZeroWeights)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ISR Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn isr_basic() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = isr(&a, &b);

        // d2 appears in both lists, should rank high
        assert!(f.iter().position(|(id, _)| *id == "d2").unwrap() < 2);
    }

    #[test]
    fn isr_score_formula() {
        // Single item in one list: score = 1/sqrt(k + 0) = 1/sqrt(k)
        let a = vec![("d1", 1.0)];
        let b: Vec<(&str, f32)> = vec![];
        let f = isr_with_config(&a, &b, RrfConfig::new(1));

        let expected = 1.0 / 1.0_f32.sqrt(); // 1/sqrt(1) = 1.0
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn isr_gentler_decay_than_rrf() {
        // ISR should have a gentler decay than RRF
        // At rank 0 and rank 3 (with k=1):
        // RRF: 1/1 vs 1/4 = ratio of 4
        // ISR: 1/sqrt(1) vs 1/sqrt(4) = 1 vs 0.5 = ratio of 2
        let a = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8), ("d4", 0.7)];
        let b: Vec<(&str, f32)> = vec![];

        let rrf_result = rrf_with_config(&a, &b, RrfConfig::new(1));
        let isr_result = isr_with_config(&a, &b, RrfConfig::new(1));

        // Calculate ratio of first to last score
        let rrf_ratio = rrf_result[0].1 / rrf_result[3].1;
        let isr_ratio = isr_result[0].1 / isr_result[3].1;

        // ISR should have smaller ratio (gentler decay)
        assert!(
            isr_ratio < rrf_ratio,
            "ISR should have gentler decay: ISR ratio={}, RRF ratio={}",
            isr_ratio,
            rrf_ratio
        );
    }

    #[test]
    fn isr_multi_works() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let c = ranked(&["d3", "d4"]);
        let f = isr_multi(&[&a, &b, &c], RrfConfig::new(1));

        // All items should be present
        assert_eq!(f.len(), 4);
        // d2 and d3 appear in 2 lists each, d1 and d4 in 1
        // d2 at rank 1,0 => 1/sqrt(2) + 1/sqrt(1)
        // d3 at rank 1,0 => 1/sqrt(2) + 1/sqrt(1)
        // They should be top
        let top_2: Vec<_> = f.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_2.contains(&"d2") && top_2.contains(&"d3"));
    }

    #[test]
    fn isr_with_top_k() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = isr_with_config(&a, &b, RrfConfig::new(1).with_top_k(2));

        assert_eq!(f.len(), 2);
    }

    #[test]
    fn isr_empty_lists() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = ranked(&["d1"]);

        assert_eq!(isr(&empty, &non_empty).len(), 1);
        assert_eq!(isr(&non_empty, &empty).len(), 1);
        assert_eq!(isr(&empty, &empty).len(), 0);
    }

    #[test]
    fn fusion_method_isr() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::isr().fuse(&a, &b);
        assert_eq!(f[0].0, "d2");

        // With custom k
        let f = FusionMethod::isr_with_k(10).fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_isr_multi() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let c = ranked(&["d3", "d4"]);
        let lists = [&a[..], &b[..], &c[..]];

        let f = FusionMethod::isr().fuse_multi(&lists);
        assert!(!f.is_empty());
    }

    #[test]
    fn combmnz_rewards_overlap() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let f = combmnz(&a, &b);

        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn combsum_basic() {
        let a = vec![("d1", 0.5), ("d2", 1.0)];
        let b = vec![("d2", 1.0), ("d3", 0.5)];
        let f = combsum(&a, &b);

        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn weighted_skewed() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d2", 1.0)];

        let f = weighted(
            &a,
            &b,
            WeightedConfig::default()
                .with_weights(0.9, 0.1)
                .with_normalize(false),
        );
        assert_eq!(f[0].0, "d1");

        let f = weighted(
            &a,
            &b,
            WeightedConfig::default()
                .with_weights(0.1, 0.9)
                .with_normalize(false),
        );
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn borda_symmetric() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);
        let f = borda(&a, &b);

        let scores: Vec<f32> = f.iter().map(|(_, s)| *s).collect();
        assert!((scores[0] - scores[1]).abs() < 0.01);
        assert!((scores[1] - scores[2]).abs() < 0.01);
    }

    #[test]
    fn rrf_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];
        let f = rrf_multi(&lists, RrfConfig::default());

        assert_eq!(f.len(), 3);
    }

    #[test]
    fn borda_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];
        let f = borda_multi(&lists, FusionConfig::default());
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn combsum_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            vec![("d1", 1.0), ("d2", 0.5)],
            vec![("d2", 1.0), ("d3", 0.5)],
            vec![("d1", 1.0), ("d3", 0.5)],
        ];
        let f = combsum_multi(&lists, FusionConfig::default());
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn combmnz_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            vec![("d1", 1.0)],
            vec![("d1", 1.0), ("d2", 0.5)],
            vec![("d1", 1.0), ("d2", 0.5)],
        ];
        let f = combmnz_multi(&lists, FusionConfig::default());
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn weighted_multi_works() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d2", 1.0)];
        let c = vec![("d3", 1.0)];

        let f = weighted_multi(&[(&a, 1.0), (&b, 1.0), (&c, 1.0)], false, None).unwrap();
        assert_eq!(f.len(), 3);

        let f = weighted_multi(&[(&a, 10.0), (&b, 1.0), (&c, 1.0)], false, None).unwrap();
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn weighted_multi_zero_weights() {
        let a = vec![("d1", 1.0)];
        let result = weighted_multi(&[(&a, 0.0)], false, None);
        assert!(matches!(result, Err(FusionError::ZeroWeights)));
    }

    #[test]
    fn empty_inputs() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = ranked(&["d1"]);

        assert_eq!(rrf(&empty, &non_empty).len(), 1);
        assert_eq!(rrf(&non_empty, &empty).len(), 1);
    }

    #[test]
    fn both_empty() {
        let empty: Vec<(&str, f32)> = vec![];
        assert_eq!(rrf(&empty, &empty).len(), 0);
        assert_eq!(combsum(&empty, &empty).len(), 0);
        assert_eq!(borda(&empty, &empty).len(), 0);
    }

    #[test]
    fn duplicate_ids_in_same_list() {
        let a = vec![("d1", 1.0), ("d1", 0.5)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

        assert_eq!(f.len(), 1);
        let expected = 1.0 / 60.0 + 1.0 / 61.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn builder_pattern() {
        let config = RrfConfig::default().with_k(30).with_top_k(5);
        assert_eq!(config.k, 30);
        assert_eq!(config.top_k, Some(5));

        let config = WeightedConfig::default()
            .with_weights(0.8, 0.2)
            .with_normalize(false)
            .with_top_k(10);
        assert_eq!(config.weight_a, 0.8);
        assert!(!config.normalize);
        assert_eq!(config.top_k, Some(10));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Edge Case Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn nan_scores_handled() {
        let a = vec![("d1", f32::NAN), ("d2", 0.5)];
        let b = vec![("d2", 0.9), ("d3", 0.1)];

        // Should not panic
        let _ = rrf(&a, &b);
        let _ = combsum(&a, &b);
        let _ = combmnz(&a, &b);
        let _ = borda(&a, &b);
    }

    #[test]
    fn inf_scores_handled() {
        let a = vec![("d1", f32::INFINITY), ("d2", 0.5)];
        let b = vec![("d2", f32::NEG_INFINITY), ("d3", 0.1)];

        // Should not panic
        let _ = rrf(&a, &b);
        let _ = combsum(&a, &b);
    }

    #[test]
    fn zero_scores() {
        let a = vec![("d1", 0.0), ("d2", 0.0)];
        let b = vec![("d2", 0.0), ("d3", 0.0)];

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn negative_scores() {
        let a = vec![("d1", -1.0), ("d2", -0.5)];
        let b = vec![("d2", -0.9), ("d3", -0.1)];

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 3);
        // Should normalize properly
    }

    #[test]
    fn large_k_value() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        // k = u32::MAX should not overflow
        let f = rrf_with_config(&a, &b, RrfConfig::new(u32::MAX));
        assert!(!f.is_empty());
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn k_zero_panics() {
        let _ = RrfConfig::new(0);
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn k_zero_with_k_panics() {
        let _ = RrfConfig::default().with_k(0);
    }

    #[test]
    fn all_nan_scores() {
        let a = vec![("d1", f32::NAN), ("d2", f32::NAN)];
        let b = vec![("d3", f32::NAN), ("d4", f32::NAN)];

        // Should not panic, but results may contain NaN
        let f = rrf(&a, &b);
        assert_eq!(f.len(), 4);
        // NaN values are valid RRF scores (1/(k+rank) is always finite)
        // But if all scores are NaN, the RRF calculation still works
        // Actually, RRF ignores scores, so NaN scores don't matter
        // All documents get RRF scores based on ranks, which are finite
        for (_, score) in &f {
            assert!(
                score.is_finite(),
                "RRF scores should be finite (based on ranks, not input scores)"
            );
        }
    }

    #[test]
    fn empty_lists_multi() {
        let empty: Vec<Vec<(&str, f32)>> = vec![];
        assert_eq!(rrf_multi(&empty, RrfConfig::default()).len(), 0);
        assert_eq!(combsum_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(combmnz_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(borda_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(dbsf_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(isr_multi(&empty, RrfConfig::default()).len(), 0);
    }

    #[test]
    fn rrf_weighted_list_weight_mismatch() {
        let a = [("d1", 1.0)];
        let b = [("d2", 1.0)];
        let weights = [0.5, 0.5, 0.0]; // 3 weights for 2 lists

        let result = rrf_weighted(&[&a[..], &b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn rrf_weighted_list_weight_mismatch_short() {
        let a = [("d1", 1.0)];
        let b = [("d2", 1.0)];
        let weights = [0.5]; // 1 weight for 2 lists

        let result = rrf_weighted(&[&a[..], &b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn duplicate_ids_commutative() {
        // Test that duplicate handling is commutative
        let a = vec![("d1", 1.0), ("d1", 0.5), ("d2", 0.3)];
        let b = vec![("d2", 0.9), ("d3", 0.7)];

        let ab = rrf(&a, &b);
        let ba = rrf(&b, &a);

        // Should have same document IDs (order may differ due to ties)
        let ab_ids: Vec<&str> = ab.iter().map(|(id, _)| *id).collect();
        let ba_ids: Vec<&str> = ba.iter().map(|(id, _)| *id).collect();
        assert_eq!(ab_ids.len(), ba_ids.len());
        // All IDs should appear in both
        for id in &ab_ids {
            assert!(ba_ids.contains(id));
        }
    }

    #[test]
    fn dbsf_zero_variance() {
        // All scores equal in one list
        let a = vec![("d1", 1.0), ("d2", 1.0), ("d3", 1.0)];
        let b = vec![("d1", 0.9), ("d2", 0.5), ("d3", 0.1)];

        // Should not panic, list a contributes z-score=0.0 for all
        let f = dbsf(&a, &b);
        assert_eq!(f.len(), 3);
        // d1 should win (0.0 + positive z-score from b)
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn single_item_lists() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d1", 1.0)];

        let f = rrf(&a, &b);
        assert_eq!(f.len(), 1);

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 1);

        let f = borda(&a, &b);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn disjoint_lists() {
        let a = vec![("d1", 1.0), ("d2", 0.9)];
        let b = vec![("d3", 1.0), ("d4", 0.9)];

        let f = rrf(&a, &b);
        assert_eq!(f.len(), 4);

        let f = combmnz(&a, &b);
        assert_eq!(f.len(), 4);
        // No overlap bonus
    }

    #[test]
    fn identical_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d1", "d2", "d3"]);

        let f = rrf(&a, &b);
        // Order should be preserved
        assert_eq!(f[0].0, "d1");
        assert_eq!(f[1].0, "d2");
        assert_eq!(f[2].0, "d3");
    }

    #[test]
    fn reversed_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);

        let f = rrf(&a, &b);
        // All items appear in both lists, so all have same total RRF score
        // d2 at rank 1 in both gets: 2 * 1/(60+1) = 2/61
        // d1 at rank 0,2 gets: 1/60 + 1/62
        // d3 at rank 2,0 gets: 1/62 + 1/60
        // d1 and d3 tie, d2 is slightly lower (rank 1+1 vs 0+2)
        // Just check we get all 3
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn top_k_larger_than_result() {
        let a = ranked(&["d1"]);
        let b = ranked(&["d2"]);

        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(100));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn top_k_zero() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(0));
        assert_eq!(f.len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FusionMethod Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn fusion_method_rrf() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::rrf().fuse(&a, &b);
        assert_eq!(f[0].0, "d2"); // Appears in both
    }

    #[test]
    fn fusion_method_combsum() {
        // Use scores where d2 clearly wins after normalization
        // a: d1=1.0 (norm: 1.0), d2=0.5 (norm: 0.0)
        // b: d2=1.0 (norm: 1.0), d3=0.5 (norm: 0.0)
        // Final: d1=1.0, d2=1.0, d3=0.0 - still a tie!
        // Use 3 elements to break the tie:
        let a = vec![("d1", 1.0_f32), ("d2", 0.6), ("d4", 0.2)];
        let b = vec![("d2", 1.0_f32), ("d3", 0.5)];
        // a norms: d1=(1.0-0.2)/0.8=1.0, d2=(0.6-0.2)/0.8=0.5, d4=0.0
        // b norms: d2=(1.0-0.5)/0.5=1.0, d3=0.0
        // Final: d1=1.0, d2=0.5+1.0=1.5, d3=0.0, d4=0.0

        let f = FusionMethod::CombSum.fuse(&a, &b);
        // d2 appears in both lists with high scores, should win
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_combmnz() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::CombMnz.fuse(&a, &b);
        assert_eq!(f[0].0, "d2"); // Overlap bonus
    }

    #[test]
    fn fusion_method_borda() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::Borda.fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_weighted() {
        let a = vec![("d1", 1.0f32)];
        let b = vec![("d2", 1.0f32)];

        // Heavy weight on first list
        let f = FusionMethod::weighted(0.9, 0.1).fuse(&a, &b);
        assert_eq!(f[0].0, "d1");

        // Heavy weight on second list
        let f = FusionMethod::weighted(0.1, 0.9).fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_multi() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];

        let f = FusionMethod::rrf().fuse_multi(&lists);
        assert_eq!(f.len(), 3);
        // d1 and d2 both appear in 2 lists, should be top 2
    }

    #[test]
    fn fusion_method_default_is_rrf() {
        let method = FusionMethod::default();
        assert!(matches!(method, FusionMethod::Rrf { k: 60 }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MMR Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn mmr_basic() {
        // Candidates with relevance scores
        let candidates = vec![("d1", 0.95), ("d2", 0.90), ("d3", 0.85)];

        // Similarity: d1 and d2 are very similar, d3 is different
        let similarity = |a: &&str, b: &&str| -> f32 {
            if a == b {
                1.0
            } else if (*a == "d1" && *b == "d2") || (*a == "d2" && *b == "d1") {
                0.95 // d1 and d2 are near-duplicates
            } else {
                0.1 // other pairs are different
            }
        };

        let config = MmrConfig::new(0.5).with_top_k(3);
        let results = mmr(&candidates, similarity, config);

        assert_eq!(results.len(), 3);
        // d1 should be first (highest relevance)
        assert_eq!(results[0].0, "d1");
        // d3 should be second (d2 penalized for similarity to d1)
        assert_eq!(results[1].0, "d3");
        // d2 should be last
        assert_eq!(results[2].0, "d2");
    }

    #[test]
    fn mmr_pure_relevance() {
        // With lambda=1.0, MMR should be equivalent to standard ranking
        let candidates = vec![
            ("d1", 0.9),
            ("d2", 0.95), // Highest relevance
            ("d3", 0.8),
        ];

        let similarity = |_a: &&str, _b: &&str| -> f32 { 0.5 };

        let config = MmrConfig::new(1.0).with_top_k(3);
        let results = mmr(&candidates, similarity, config);

        // Should be sorted by relevance only
        assert_eq!(results[0].0, "d2"); // 0.95
        assert_eq!(results[1].0, "d1"); // 0.9
        assert_eq!(results[2].0, "d3"); // 0.8
    }

    #[test]
    fn mmr_pure_diversity() {
        // With lambda=0.0, MMR should maximize diversity (spread)
        let candidates = vec![
            ("d1", 0.9),
            ("d2", 0.9), // Same relevance as d1, but similar to d1
            ("d3", 0.9), // Same relevance, but different
        ];

        // d1-d2 are similar, d3 is different from both
        let similarity = |a: &&str, b: &&str| -> f32 {
            if a == b {
                1.0
            } else if (*a == "d1" && *b == "d2") || (*a == "d2" && *b == "d1") {
                0.9
            } else {
                0.1
            }
        };

        let config = MmrConfig::new(0.0).with_top_k(2);
        let results = mmr(&candidates, similarity, config);

        // First selection: any (all equal relevance, no penalty yet)
        // Second selection: should pick the most different from first
        // If d1 was first, d3 should be second (not d2 which is similar to d1)
        if results[0].0 == "d1" || results[0].0 == "d2" {
            assert_eq!(results[1].0, "d3", "should pick diverse document second");
        }
    }

    #[test]
    fn mmr_config_lambda_bounds() {
        // Valid lambda values
        let _ = MmrConfig::new(0.0);
        let _ = MmrConfig::new(0.5);
        let _ = MmrConfig::new(1.0);
    }

    #[test]
    #[should_panic(expected = "lambda must be in [0.0, 1.0]")]
    fn mmr_config_lambda_negative() {
        let _ = MmrConfig::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "lambda must be in [0.0, 1.0]")]
    fn mmr_config_lambda_too_large() {
        let _ = MmrConfig::new(1.1);
    }

    #[test]
    fn mmr_empty_candidates() {
        let candidates: Vec<(&str, f32)> = vec![];
        let similarity = |_a: &&str, _b: &&str| -> f32 { 0.0 };
        let results = mmr(&candidates, similarity, MmrConfig::default());
        assert!(results.is_empty());
    }

    #[test]
    fn mmr_single_candidate() {
        let candidates = vec![("d1", 0.9)];
        let similarity = |_a: &&str, _b: &&str| -> f32 { 1.0 };
        let results = mmr(&candidates, similarity, MmrConfig::default());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "d1");
    }

    #[test]
    fn mmr_matrix_based() {
        let candidates = vec![("a", 0.9), ("b", 0.85), ("c", 0.8)];

        let mut matrix: HashMap<(&str, &str), f32> = HashMap::new();
        matrix.insert(("a", "b"), 0.9); // a and b are similar
        matrix.insert(("b", "a"), 0.9);
        matrix.insert(("a", "c"), 0.1);
        matrix.insert(("c", "a"), 0.1);
        matrix.insert(("b", "c"), 0.2);
        matrix.insert(("c", "b"), 0.2);

        let config = MmrConfig::new(0.5).with_top_k(2);
        let results = mmr_with_matrix(&candidates, &matrix, config);

        assert_eq!(results.len(), 2);
        // a first (highest relevance), c second (diverse from a)
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "c");
    }

    #[test]
    fn mmr_embedding_based() {
        // Test embedding-based MMR
        let candidates = vec![
            ("d1", 0.9, vec![1.0, 0.0, 0.0]),  // Points along x-axis
            ("d2", 0.85, vec![0.9, 0.1, 0.0]), // Similar to d1
            ("d3", 0.8, vec![0.0, 1.0, 0.0]),  // Points along y-axis (orthogonal)
        ];

        let config = MmrConfig::new(0.5).with_top_k(2);
        let results = mmr_embeddings(&candidates, config);

        assert_eq!(results.len(), 2);
        // d1 first (highest relevance)
        assert_eq!(results[0].0, "d1");
        // d3 should be second (orthogonal to d1, maximal diversity)
        assert_eq!(results[1].0, "d3");
    }

    #[test]
    fn cosine_sim_basic() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-6);

        // Similar vectors
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[0.9, 0.1, 0.0]);
        assert!(sim > 0.9); // Should be close to 1

        // Empty vectors
        assert_eq!(cosine_similarity(&[], &[]), 0.0);

        // Different lengths
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    // ── Normalization Tests ────────────────────────────────────────────────

    #[test]
    fn quantile_normalization() {
        let results = vec![
            ("a", 10.0),
            ("b", 20.0),
            ("c", 30.0),
            ("d", 40.0),
            ("e", 50.0),
        ];
        let normed = normalize_scores(&results, Normalization::Quantile);

        // Sorted ascending: a(10)=0.0, b(20)=0.25, c(30)=0.5, d(40)=0.75, e(50)=1.0
        assert!((normed[0].1 - 0.0).abs() < 1e-6, "a should be 0.0");
        assert!((normed[1].1 - 0.25).abs() < 1e-6, "b should be 0.25");
        assert!((normed[2].1 - 0.5).abs() < 1e-6, "c should be 0.5");
        assert!((normed[4].1 - 1.0).abs() < 1e-6, "e should be 1.0");
    }

    #[test]
    fn quantile_normalization_single() {
        let results = vec![("a", 42.0)];
        let normed = normalize_scores(&results, Normalization::Quantile);
        assert!((normed[0].1 - 0.5).abs() < 1e-6, "single item gets 0.5");
    }

    #[test]
    fn sigmoid_normalization() {
        let results = vec![("a", -10.0), ("b", 0.0), ("c", 10.0)];
        let normed = normalize_scores(&results, Normalization::Sigmoid);

        // sigmoid(-10) ~ 0.0000454, sigmoid(0) = 0.5, sigmoid(10) ~ 0.99995
        assert!(normed[0].1 < 0.01, "sigmoid(-10) should be near 0");
        assert!((normed[1].1 - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");
        assert!(normed[2].1 > 0.99, "sigmoid(10) should be near 1");
    }

    #[test]
    fn sigmoid_preserves_order() {
        let results = vec![("a", 1.0), ("b", 3.0), ("c", 2.0)];
        let normed = normalize_scores(&results, Normalization::Sigmoid);

        // b(3.0) > c(2.0) > a(1.0) should hold after sigmoid
        assert!(normed[1].1 > normed[2].1);
        assert!(normed[2].1 > normed[0].1);
    }

    #[test]
    fn quantile_handles_non_gaussian() {
        // Scores with extreme outlier -- quantile should be robust
        let results = vec![
            ("a", 0.1),
            ("b", 0.2),
            ("c", 0.3),
            ("d", 100.0), // extreme outlier
        ];
        let normed = normalize_scores(&results, Normalization::Quantile);

        // Quantile normalization: ranks are 0/3, 1/3, 2/3, 3/3
        assert!((normed[0].1 - 0.0).abs() < 1e-6);
        assert!((normed[1].1 - 1.0 / 3.0).abs() < 1e-6);
        assert!((normed[2].1 - 2.0 / 3.0).abs() < 1e-6);
        assert!((normed[3].1 - 1.0).abs() < 1e-6);
    }

    // ── Copeland & Median Rank Tests ───────────────────────────────────────

    #[test]
    fn copeland_basic() {
        // Three lists, d2 is preferred by majority in all pairwise comparisons
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d1", "d3"]);
        let c = ranked(&["d2", "d3", "d1"]);

        let f = copeland_multi(&[&a, &b, &c], FusionConfig::default());
        // d2 ranks first in 2/3 lists vs d1, and first in all vs d3
        assert_eq!(f[0].0, "d2", "d2 should be Copeland winner");
    }

    #[test]
    fn copeland_net_wins() {
        // d1 at rank 0 in both lists. d2 at rank 1 in both. d3 at rank 2 in both.
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];

        let f = copeland(&a, &b);
        // d1 beats both d2 and d3 in both lists: net = +2
        // d2 loses to d1, beats d3: net = 0
        // d3 loses to both: net = -2
        assert_eq!(f[0].0, "d1");
        assert!((f[0].1 - 2.0).abs() < 1e-6, "d1 net wins should be 2");
        assert_eq!(f[2].0, "d3");
        assert!((f[2].1 - (-2.0)).abs() < 1e-6, "d3 net wins should be -2");
    }

    #[test]
    fn copeland_vs_condorcet_more_discriminative() {
        // Copeland distinguishes between "close loser" and "total loser"
        // Condorcet only counts wins, so both losers get the same score
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d1", 0.9), ("d3", 0.8), ("d2", 0.7)];

        let cope = copeland(&a, &b);
        let cond = condorcet(&a, &b);

        // Condorcet: d1=2 wins, d2=0 wins (ties with d3), d3=0 wins
        // Copeland: d1=+2, d2=0 (1 win, 1 loss), d3=0 (1 win, 1 loss)
        // Both give d1 first place
        assert_eq!(cope[0].0, "d1");
        assert_eq!(cond[0].0, "d1");
    }

    #[test]
    fn copeland_commutative() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d1", "d2"]);

        let f1 = copeland(&a, &b);
        let f2 = copeland(&b, &a);

        // Same results regardless of input order
        assert_eq!(f1.len(), f2.len());
        for (r1, r2) in f1.iter().zip(f2.iter()) {
            assert_eq!(r1.0, r2.0);
            assert!((r1.1 - r2.1).abs() < 1e-6);
        }
    }

    #[test]
    fn median_rank_basic() {
        // d1: rank 0 in both lists -> median 0 -> score 1/(1+0) = 1.0
        // d2: rank 1 in list a, rank 0 in list b -> median 0.5 -> score 1/1.5
        // d3: rank 2 in list a, absent in list b -> ranks [2, 3] -> median 2.5
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d1", 0.9), ("d2", 0.8)];

        let f = median_rank(&a, &b);
        assert_eq!(f[0].0, "d1", "d1 should rank first (median rank 0)");
        assert!((f[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn median_rank_outlier_robust() {
        // d1: ranks [0, 0, 100] -> median 0 (outlier ignored)
        // d2: ranks [1, 1, 1] -> median 1
        // With 3 lists, d1 should still rank above d2 despite one terrible rank
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d1", 0.9), ("d2", 0.8)];
        // In list c, d1 is at rank 5 (far down), d2 is at rank 0
        let c: Vec<(&str, f32)> = vec![
            ("x1", 0.9),
            ("x2", 0.8),
            ("x3", 0.7),
            ("x4", 0.6),
            ("x5", 0.5),
            ("d1", 0.4),
            ("d2", 0.3),
        ];

        let f = median_rank_multi(&[&a, &b, &c], FusionConfig::default());

        let d1_pos = f.iter().position(|(id, _)| *id == "d1").unwrap();
        let d2_pos = f.iter().position(|(id, _)| *id == "d2").unwrap();
        // d1 median rank = 0 (ranks: [0, 0, 5] -> sorted [0, 0, 5] -> median 0)
        // d2 median rank = 1 (ranks: [1, 1, 6] -> sorted [1, 1, 6] -> median 1)
        assert!(
            d1_pos < d2_pos,
            "d1 should rank above d2 (outlier-robust median)"
        );
    }

    #[test]
    fn median_rank_commutative() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d1", "d2"]);

        let f1 = median_rank(&a, &b);
        let f2 = median_rank(&b, &a);

        assert_eq!(f1.len(), f2.len());
        for (r1, r2) in f1.iter().zip(f2.iter()) {
            assert_eq!(r1.0, r2.0);
            assert!((r1.1 - r2.1).abs() < 1e-6);
        }
    }

    #[test]
    fn fusion_method_copeland_dispatch() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d1", "d3"]);

        let direct = copeland(&a, &b);
        let via_enum = FusionMethod::Copeland.fuse(&a, &b);

        // Compare as score maps (tie-breaking order may differ)
        let direct_map: HashMap<_, _> = direct.into_iter().collect();
        let enum_map: HashMap<_, _> = via_enum.into_iter().collect();
        assert_eq!(direct_map.len(), enum_map.len());
        for (id, score) in &direct_map {
            let other = enum_map.get(id).expect("same keys");
            assert!((score - other).abs() < 1e-6);
        }
    }

    #[test]
    fn fusion_method_median_rank_dispatch() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d1", "d2"]);

        let direct = median_rank(&a, &b);
        let via_enum = FusionMethod::MedianRank.fuse(&a, &b);

        let direct_map: HashMap<_, _> = direct.into_iter().collect();
        let enum_map: HashMap<_, _> = via_enum.into_iter().collect();
        assert_eq!(direct_map.len(), enum_map.len());
        for (id, score) in &direct_map {
            let other = enum_map.get(id).expect("same keys");
            assert!((score - other).abs() < 1e-6);
        }
    }

    // ── Evaluation Metric Tests ──────────────────────────────────────────────

    fn make_qrels() -> Qrels<&'static str> {
        // d1=highly relevant, d2=relevant, d3=relevant, d4/d5=not relevant
        HashMap::from([("d1", 2), ("d2", 1), ("d3", 1)])
    }

    #[test]
    fn precision_at_k_basic() {
        let qrels = make_qrels();
        // Results: d1(rel=2), d4(not rel), d2(rel=1), d5(not rel), d3(rel=1)
        let results = vec![
            ("d1", 0.9),
            ("d4", 0.8),
            ("d2", 0.7),
            ("d5", 0.6),
            ("d3", 0.5),
        ];

        // P@1 = 1/1 = 1.0 (d1 is relevant)
        assert!((precision_at_k(&results, &qrels, 1) - 1.0).abs() < 1e-6);
        // P@2 = 1/2 = 0.5 (d1 relevant, d4 not)
        assert!((precision_at_k(&results, &qrels, 2) - 0.5).abs() < 1e-6);
        // P@3 = 2/3 (d1, d2 relevant out of 3)
        assert!((precision_at_k(&results, &qrels, 3) - 2.0 / 3.0).abs() < 1e-6);
        // P@5 = 3/5 = 0.6
        assert!((precision_at_k(&results, &qrels, 5) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn precision_at_k_edge_cases() {
        let qrels = make_qrels();
        let results = vec![("d1", 0.9)];

        assert_eq!(precision_at_k(&results, &qrels, 0), 0.0);
        assert_eq!(precision_at_k(&[], &qrels, 5), 0.0);
        // k > results.len() clamps
        assert!((precision_at_k(&results, &qrels, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn map_basic() {
        let qrels = make_qrels(); // d1=2, d2=1, d3=1 (3 relevant)
                                  // Perfect ranking: all relevant docs first
        let perfect = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6)];
        // MAP = (1/3) * (1/1 + 2/2 + 3/3) = (1/3) * 3 = 1.0
        assert!((map(&perfect, &qrels) - 1.0).abs() < 1e-6);

        // Interleaved: d1, d4, d2, d5, d3
        let interleaved = vec![
            ("d1", 0.9),
            ("d4", 0.8),
            ("d2", 0.7),
            ("d5", 0.6),
            ("d3", 0.5),
        ];
        // P@1=1/1=1.0 (d1 relevant), P@3=2/3 (d2 relevant), P@5=3/5 (d3 relevant)
        // MAP = (1/3) * (1.0 + 2/3 + 3/5) = (1/3) * (1.0 + 0.6667 + 0.6) = 0.7556
        let expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0;
        assert!((map(&interleaved, &qrels) - expected).abs() < 1e-4);
    }

    #[test]
    fn map_at_k_truncation() {
        let qrels = make_qrels(); // 3 relevant
        let results = vec![
            ("d4", 0.9), // not relevant
            ("d1", 0.8), // relevant
            ("d5", 0.7), // not relevant
            ("d2", 0.6), // relevant (beyond k=3)
        ];

        // MAP@3: only consider first 3 results
        // Relevant at position 2: P@2 = 1/2
        // MAP@3 = (1/2) / min(3, 3) = 0.5/3 = 0.1667
        let expected = (1.0 / 2.0) / 3.0;
        assert!(
            (map_at_k(&results, &qrels, 3) - expected).abs() < 1e-4,
            "MAP@3 = {}, expected {}",
            map_at_k(&results, &qrels, 3),
            expected
        );
    }

    #[test]
    fn map_empty() {
        let qrels = make_qrels();
        assert_eq!(map(&[], &qrels), 0.0);
        assert_eq!(map_at_k(&[], &qrels, 10), 0.0);

        let empty_qrels: Qrels<&str> = HashMap::new();
        let results = vec![("d1", 0.9)];
        assert_eq!(map(&results, &empty_qrels), 0.0);
    }

    #[test]
    fn hit_rate_basic() {
        let qrels = make_qrels();
        // First result is relevant
        let results = vec![("d1", 0.9), ("d4", 0.8)];
        assert_eq!(hit_rate(&results, &qrels, 1), 1.0);
        assert_eq!(hit_rate(&results, &qrels, 2), 1.0);

        // First result is NOT relevant
        let results2 = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7)];
        assert_eq!(hit_rate(&results2, &qrels, 1), 0.0);
        assert_eq!(hit_rate(&results2, &qrels, 2), 0.0);
        assert_eq!(hit_rate(&results2, &qrels, 3), 1.0);
    }

    #[test]
    fn hit_rate_edge_cases() {
        let qrels = make_qrels();
        assert_eq!(hit_rate(&[], &qrels, 5), 0.0);
        assert_eq!(hit_rate(&[("d4", 0.9)], &qrels, 1), 0.0); // d4 not relevant
    }

    #[test]
    fn evaluate_metric_dispatch() {
        let qrels = make_qrels();
        let results = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];

        // Verify dispatch matches direct calls
        let ndcg = evaluate_metric(&results, &qrels, OptimizeMetric::Ndcg { k: 3 });
        assert!((ndcg - ndcg_at_k(&results, &qrels, 3)).abs() < 1e-6);

        let m = evaluate_metric(&results, &qrels, OptimizeMetric::Map);
        assert!((m - map(&results, &qrels)).abs() < 1e-6);

        let p = evaluate_metric(&results, &qrels, OptimizeMetric::Precision { k: 2 });
        assert!((p - precision_at_k(&results, &qrels, 2)).abs() < 1e-6);

        let h = evaluate_metric(&results, &qrels, OptimizeMetric::HitRate { k: 1 });
        assert!((h - hit_rate(&results, &qrels, 1)).abs() < 1e-6);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────
// Property tests are in a separate module (proptests.rs) to avoid macro expansion issues
