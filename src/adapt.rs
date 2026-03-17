//! Adapters for converting retriever outputs into rankops input format.
//!
//! Retrieval systems produce results in different formats:
//! - Vector search (vicinity, faiss): `(doc_id: u32, distance: f32)` -- lower is better
//! - Lexical search (lexir, tantivy): `(doc_id, score: f32)` -- higher is better
//! - Cross-encoders: `(doc_id, logit: f32)` -- unbounded
//!
//! rankops expects `(I, f32)` where higher scores are better.
//! This module provides zero-cost conversion functions.

use std::hash::Hash;

/// Convert distance-based results (lower = closer) to score-based (higher = better).
///
/// Uses `1 / (1 + distance)` which maps distances in [0, inf) to scores in (0, 1].
/// Suitable for L2/Euclidean distance results from vector search.
///
/// # Example
///
/// ```rust
/// use rankops::adapt::from_distances;
///
/// // Vicinity-style results: (doc_id, distance)
/// let ann_results = vec![(42u32, 0.1), (17, 0.5), (99, 1.2)];
/// let ranked = from_distances(&ann_results);
/// // doc 42 (closest) now has highest score
/// assert_eq!(ranked[0].0, 42);
/// assert!(ranked[0].1 > ranked[1].1);
/// ```
pub fn from_distances<I: Clone + Eq + Hash>(results: &[(I, f32)]) -> Vec<(I, f32)> {
    let mut scored: Vec<(I, f32)> = results
        .iter()
        .map(|(id, dist)| (id.clone(), 1.0 / (1.0 + dist)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Convert distance-based results with ID remapping.
///
/// Maps integer doc IDs to arbitrary types using a lookup function.
///
/// # Example
///
/// ```rust
/// use rankops::adapt::from_distances_mapped;
///
/// let doc_names = ["intro.md", "chapter1.md", "chapter2.md", "appendix.md"];
/// let ann_results = vec![(1u32, 0.2), (3, 0.5), (0, 0.8)];
///
/// let ranked = from_distances_mapped(&ann_results, |id| doc_names[*id as usize]);
/// assert_eq!(ranked[0].0, "chapter1.md"); // closest
/// ```
pub fn from_distances_mapped<I, O, F>(results: &[(I, f32)], map_id: F) -> Vec<(O, f32)>
where
    I: Clone,
    O: Clone + Eq + Hash,
    F: Fn(&I) -> O,
{
    let mut scored: Vec<(O, f32)> = results
        .iter()
        .map(|(id, dist)| (map_id(id), 1.0 / (1.0 + dist)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Convert cosine similarity results (already higher = better, range [-1, 1]).
///
/// Passes through scores as-is, just ensures descending sort order.
/// Use for results from cosine similarity search where scores are already
/// in the right direction.
pub fn from_similarities<I: Clone + Eq + Hash>(results: &[(I, f32)]) -> Vec<(I, f32)> {
    let mut scored: Vec<(I, f32)> = results.to_vec();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Convert cosine similarity results with ID remapping.
pub fn from_similarities_mapped<I, O, F>(results: &[(I, f32)], map_id: F) -> Vec<(O, f32)>
where
    I: Clone,
    O: Clone + Eq + Hash,
    F: Fn(&I) -> O,
{
    let mut scored: Vec<(O, f32)> = results.iter().map(|(id, s)| (map_id(id), *s)).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Convert inner product / dot product results (already higher = better, unbounded).
///
/// Alias for [`from_similarities`] since both are higher-is-better.
pub fn from_inner_product<I: Clone + Eq + Hash>(results: &[(I, f32)]) -> Vec<(I, f32)> {
    from_similarities(results)
}

/// Convert cross-encoder logits (unbounded) to normalized scores via sigmoid.
///
/// Maps logits in (-inf, inf) to scores in (0, 1) using the logistic function.
/// Useful for MonoT5, mMiniLMv2, or other cross-encoder reranker outputs.
///
/// # Example
///
/// ```rust
/// use rankops::adapt::from_logits;
///
/// let reranker_output = vec![("d1", 2.5), ("d2", -1.0), ("d3", 0.0)];
/// let ranked = from_logits(&reranker_output);
/// // d1 (highest logit) ranks first; scores are in (0, 1)
/// assert!(ranked[0].1 > 0.9);
/// assert!((ranked[2].1 - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
/// ```
pub fn from_logits<I: Clone + Eq + Hash>(results: &[(I, f32)]) -> Vec<(I, f32)> {
    let mut scored: Vec<(I, f32)> = results
        .iter()
        .map(|(id, logit)| (id.clone(), 1.0 / (1.0 + (-logit).exp())))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

/// Convert cross-encoder logits with ID remapping.
pub fn from_logits_mapped<I, O, F>(results: &[(I, f32)], map_id: F) -> Vec<(O, f32)>
where
    I: Clone,
    O: Clone + Eq + Hash,
    F: Fn(&I) -> O,
{
    let mut scored: Vec<(O, f32)> = results
        .iter()
        .map(|(id, logit)| (map_id(id), 1.0 / (1.0 + (-logit).exp())))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_to_scores() {
        let results = vec![(1u32, 0.0), (2, 1.0), (3, 9.0)];
        let ranked = from_distances(&results);

        // Closest (dist=0) should have highest score (1.0)
        assert_eq!(ranked[0].0, 1);
        assert!((ranked[0].1 - 1.0).abs() < 1e-6);

        // dist=1 -> score = 0.5
        assert_eq!(ranked[1].0, 2);
        assert!((ranked[1].1 - 0.5).abs() < 1e-6);

        // dist=9 -> score = 0.1
        assert_eq!(ranked[2].0, 3);
        assert!((ranked[2].1 - 0.1).abs() < 1e-6);
    }

    #[test]
    fn distances_mapped() {
        let names = ["a", "b", "c", "d"];
        let results = vec![(2u32, 0.5), (0, 0.1)];
        let ranked = from_distances_mapped(&results, |id| names[*id as usize]);

        assert_eq!(ranked[0].0, "a"); // dist 0.1 -> highest score
        assert_eq!(ranked[1].0, "c"); // dist 0.5 -> lower score
    }

    #[test]
    fn similarities_passthrough() {
        let results = vec![("d1", 0.3), ("d2", 0.9), ("d3", 0.6)];
        let ranked = from_similarities(&results);

        // Should be sorted descending
        assert_eq!(ranked[0].0, "d2");
        assert_eq!(ranked[1].0, "d3");
        assert_eq!(ranked[2].0, "d1");
    }

    #[test]
    fn logits_conversion() {
        let results = vec![("d1", 0.0), ("d2", 5.0), ("d3", -5.0)];
        let ranked = from_logits(&results);

        // Highest logit first
        assert_eq!(ranked[0].0, "d2");
        assert!(ranked[0].1 > 0.99);

        // sigmoid(0) = 0.5
        assert_eq!(ranked[1].0, "d1");
        assert!((ranked[1].1 - 0.5).abs() < 1e-6);

        // Lowest logit last
        assert_eq!(ranked[2].0, "d3");
        assert!(ranked[2].1 < 0.01);
    }

    #[test]
    fn empty_inputs() {
        let empty: Vec<(u32, f32)> = vec![];
        assert!(from_distances(&empty).is_empty());
        assert!(from_similarities(&empty).is_empty());
        assert!(from_logits(&empty).is_empty());
    }

    #[test]
    fn adapter_then_fuse() {
        // Simulate: BM25 + ANN -> fuse with RRF
        let bm25 = vec![("d1", 12.0), ("d2", 10.0), ("d3", 8.0)];
        let ann_distances = vec![("d2", 0.1), ("d4", 0.3), ("d1", 0.9)];

        let ann_scores = from_distances(&ann_distances);

        // Feed both into RRF
        let fused = crate::rrf(&bm25, &ann_scores);

        // d1 and d2 appear in both lists
        assert!(!fused.is_empty());
        let d2_pos = fused.iter().position(|(id, _)| *id == "d2").unwrap();
        assert!(d2_pos < 2, "d2 should rank high (in both lists)");
    }
}
