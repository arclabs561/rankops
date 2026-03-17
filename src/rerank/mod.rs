//! Reranking: MaxSim (ColBERT), cosine similarity, diversity (MMR/DPP), matryoshka.
//!
//! # Pipeline Position
//!
//! Reranking sits after first-stage retrieval and fusion, refining a candidate set:
//! ```text
//! retrieve → fuse (rankops) → rerank (rankops::rerank) → final results
//! ```
//!
//! # Key Types
//!
//! - [`RerankConfig`](crate::rerank::RerankConfig) — blending weight and top-k truncation
//! - [`colbert`](crate::rerank::colbert) — MaxSim / late interaction scoring
//! - [`scoring`](crate::rerank::scoring) — Dense, MaxSim, and CrossEncoder traits
//! - [`matryoshka`](crate::rerank::matryoshka) — two-stage retrieval with nested embeddings
//! - [`diversity`](crate::rerank::diversity) — DPP diversity reranking

pub mod colbert;
pub mod diversity;
pub mod embedding;
pub mod explain;
pub mod matryoshka;
pub mod quantization;
pub mod scoring;
pub mod simd;

pub use colbert::{rank as maxsim_rank, refine as maxsim_refine};
pub use diversity::{dpp, mmr as diversity_mmr, DppConfig, MmrConfig as DiversityMmrConfig};
pub use matryoshka::refine as matryoshka_refine;
pub use quantization::{dequantize_int8, quantize_int8, QuantizationError};
pub use scoring::{Scorer, TokenScorer};

// ─────────────────────────────────────────────────────────────────────────────
// Core Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from reranking operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RerankError {
    /// `head_dims` must be less than `query.len()` for tail refinement.
    InvalidHeadDims {
        /// Requested head dimensions.
        head_dims: usize,
        /// Actual query vector length.
        query_len: usize,
    },
    /// Vector dimensions must match.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension received.
        got: usize,
    },
    /// Pool factor must be >= 1.
    InvalidPoolFactor {
        /// Invalid pool factor value.
        pool_factor: usize,
    },
    /// Window size must be >= 1.
    InvalidWindowSize {
        /// Invalid window size value.
        window_size: usize,
    },
}

impl std::fmt::Display for RerankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeadDims {
                head_dims,
                query_len,
            } => write!(
                f,
                "invalid head_dims: {head_dims} >= query length {query_len}"
            ),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} dimensions, got {got}")
            }
            Self::InvalidPoolFactor { pool_factor } => {
                write!(f, "pool_factor must be >= 1, got {pool_factor}")
            }
            Self::InvalidWindowSize { window_size } => {
                write!(f, "window_size must be >= 1, got {window_size}")
            }
        }
    }
}

impl std::error::Error for RerankError {}

/// Result type for reranking operations.
pub type Result<T> = std::result::Result<T, RerankError>;

/// Configuration for score blending and top-k truncation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankConfig {
    /// Blending weight: 0.0 = all refinement, 1.0 = all original. Default: 0.5.
    pub alpha: f32,
    /// Truncate to top k results. Default: None (return all).
    pub top_k: Option<usize>,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            top_k: None,
        }
    }
}

impl RerankConfig {
    /// Set blending weight (clamped to [0, 1]).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }
    /// Limit output to top k.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    /// Only use refinement scores (alpha = 0).
    #[must_use]
    pub const fn refinement_only() -> Self {
        Self {
            alpha: 0.0,
            top_k: None,
        }
    }
    /// Only use original scores (alpha = 1).
    #[must_use]
    pub const fn original_only() -> Self {
        Self {
            alpha: 1.0,
            top_k: None,
        }
    }
}

/// Sort scored results in descending order (highest score first).
#[inline]
pub(crate) fn sort_scored_desc<T>(results: &mut [(T, f32)]) {
    results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
}
