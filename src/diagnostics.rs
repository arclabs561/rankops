//! Fusion diagnostics: decide whether to fuse, and which method to use.
//!
//! Based on Louis et al., "Know When to Fuse" (COLING 2025, arXiv 2409.01357).
//! Key finding: fusion helps in zero-shot settings (~82% of combinations improve)
//! but often hurts in-domain trained models (~70% degrade) unless weights are tuned.
//!
//! These diagnostics help answer:
//! - Are the retrievers complementary (do they find different relevant docs)?
//! - Do score distributions align (can scores be meaningfully combined)?
//! - What fraction of documents overlap between lists?

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Score distribution statistics for a single retriever's output.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreStats {
    /// Number of results.
    pub count: usize,
    /// Minimum score.
    pub min: f32,
    /// Maximum score.
    pub max: f32,
    /// Arithmetic mean.
    pub mean: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Median score.
    pub median: f32,
    /// Score at the 25th percentile.
    pub p25: f32,
    /// Score at the 75th percentile.
    pub p75: f32,
}

/// Compute score distribution statistics for a ranked list.
///
/// Returns `None` if the list is empty.
pub fn score_stats<I>(results: &[(I, f32)]) -> Option<ScoreStats> {
    if results.is_empty() {
        return None;
    }

    let mut scores: Vec<f32> = results.iter().map(|(_, s)| *s).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = scores.len();
    let min = scores[0];
    let max = scores[count - 1];
    let sum: f32 = scores.iter().sum();
    let mean = sum / count as f32;

    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / count as f32;
    let std_dev = variance.sqrt();

    let median = percentile(&scores, 50.0);
    let p25 = percentile(&scores, 25.0);
    let p75 = percentile(&scores, 75.0);

    Some(ScoreStats {
        count,
        min,
        max,
        mean,
        std_dev,
        median,
        p25,
        p75,
    })
}

/// Overlap ratio between two ranked lists.
///
/// Returns the fraction of documents that appear in both lists (Jaccard index).
/// Range: 0.0 (disjoint) to 1.0 (identical document sets).
pub fn overlap_ratio<I: Eq + Hash>(a: &[(I, f32)], b: &[(I, f32)]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<_> = a.iter().map(|(id, _)| id).collect();
    let set_b: HashSet<_> = b.iter().map(|(id, _)| id).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Overlap ratio at k: considers only top-k of each list.
pub fn overlap_at_k<I: Eq + Hash>(a: &[(I, f32)], b: &[(I, f32)], k: usize) -> f32 {
    let a_k: Vec<_> = a.iter().take(k).map(|(id, s)| (id, *s)).collect();
    let b_k: Vec<_> = b.iter().take(k).map(|(id, s)| (id, *s)).collect();

    if a_k.is_empty() && b_k.is_empty() {
        return 0.0;
    }

    let set_a: HashSet<_> = a_k.iter().map(|(id, _)| *id).collect();
    let set_b: HashSet<_> = b_k.iter().map(|(id, _)| *id).collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Complementarity score between two ranked lists given relevance judgments.
///
/// Measures how often one retriever finds relevant documents the other misses.
/// Range: 0.0 (fully redundant) to 1.0 (fully complementary).
///
/// Formula: |relevant_only_in_A ∪ relevant_only_in_B| / |all_relevant_found|
///
/// High complementarity (>0.5) suggests fusion will help.
/// Low complementarity (<0.2) suggests the retrievers are redundant.
pub fn complementarity<I: Clone + Eq + Hash>(
    a: &[(I, f32)],
    b: &[(I, f32)],
    qrels: &HashMap<I, u32>,
) -> f32 {
    let relevant_in_a: HashSet<_> = a
        .iter()
        .filter(|(id, _)| qrels.get(id).is_some_and(|&r| r > 0))
        .map(|(id, _)| id.clone())
        .collect();

    let relevant_in_b: HashSet<_> = b
        .iter()
        .filter(|(id, _)| qrels.get(id).is_some_and(|&r| r > 0))
        .map(|(id, _)| id.clone())
        .collect();

    let all_relevant_found: HashSet<_> = relevant_in_a.union(&relevant_in_b).collect();
    if all_relevant_found.is_empty() {
        return 0.0;
    }

    let only_in_a: HashSet<_> = relevant_in_a.difference(&relevant_in_b).collect();
    let only_in_b: HashSet<_> = relevant_in_b.difference(&relevant_in_a).collect();

    let unique_relevant = only_in_a.len() + only_in_b.len();

    unique_relevant as f32 / all_relevant_found.len() as f32
}

/// Rank correlation (Kendall's tau-b) between two ranked lists on shared documents.
///
/// Measures agreement between two rankings. Range: -1.0 (reversed) to 1.0 (identical).
/// Values near 0 indicate independent rankings -- good candidates for fusion.
///
/// Only considers documents present in both lists.
pub fn rank_correlation<I: Clone + Eq + Hash>(a: &[(I, f32)], b: &[(I, f32)]) -> f32 {
    // Build rank maps for shared docs
    let rank_a: HashMap<_, _> = a
        .iter()
        .enumerate()
        .map(|(r, (id, _))| (id.clone(), r))
        .collect();
    let rank_b: HashMap<_, _> = b
        .iter()
        .enumerate()
        .map(|(r, (id, _))| (id.clone(), r))
        .collect();

    let shared: Vec<I> = rank_a
        .keys()
        .filter(|id| rank_b.contains_key(id))
        .cloned()
        .collect();

    let n = shared.len();
    if n < 2 {
        return 0.0;
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_a: i64 = 0;
    let mut ties_b: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let ra_i = rank_a[&shared[i]];
            let ra_j = rank_a[&shared[j]];
            let rb_i = rank_b[&shared[i]];
            let rb_j = rank_b[&shared[j]];

            let sign_a = (ra_i as i64 - ra_j as i64).signum();
            let sign_b = (rb_i as i64 - rb_j as i64).signum();

            if sign_a == 0 && sign_b == 0 {
                // Both tied -- not counted
            } else if sign_a == 0 {
                ties_a += 1;
            } else if sign_b == 0 {
                ties_b += 1;
            } else if sign_a == sign_b {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let n0 = concordant + discordant + ties_a + ties_b;
    if n0 == 0 {
        return 0.0;
    }

    let denom_a = concordant + discordant + ties_a;
    let denom_b = concordant + discordant + ties_b;

    if denom_a == 0 || denom_b == 0 {
        return 0.0;
    }

    (concordant - discordant) as f32 / ((denom_a as f64 * denom_b as f64).sqrt() as f32)
}

/// Full diagnostic report for a pair of ranked lists.
#[derive(Debug, Clone)]
pub struct FusionDiagnostics<I> {
    /// Score distribution of list A.
    pub stats_a: Option<ScoreStats>,
    /// Score distribution of list B.
    pub stats_b: Option<ScoreStats>,
    /// Jaccard overlap ratio (full lists).
    pub overlap: f32,
    /// Jaccard overlap at k.
    pub overlap_at_k: f32,
    /// Complementarity given relevance judgments (None if no qrels provided).
    pub complementarity: Option<f32>,
    /// Kendall's tau-b rank correlation on shared documents.
    pub rank_correlation: f32,
    /// Suggested fusion approach based on diagnostics.
    pub suggestion: FusionSuggestion,
    /// Documents unique to list A.
    pub unique_to_a: Vec<I>,
    /// Documents unique to list B.
    pub unique_to_b: Vec<I>,
}

/// Fusion suggestion based on diagnostic analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionSuggestion {
    /// Retrievers are complementary and fusion is likely to help.
    FuseRecommended {
        /// Suggested reason.
        reason: &'static str,
    },
    /// Fusion may help with careful weight tuning.
    FuseWithCaution {
        /// Suggested reason.
        reason: &'static str,
    },
    /// Fusion is unlikely to help; use the best single retriever.
    SkipFusion {
        /// Suggested reason.
        reason: &'static str,
    },
}

/// Run full fusion diagnostics on two ranked lists.
///
/// If relevance judgments are available, complementarity is computed.
/// The `k` parameter controls the depth for overlap_at_k analysis.
pub fn diagnose<I: Clone + Eq + Hash>(
    a: &[(I, f32)],
    b: &[(I, f32)],
    qrels: Option<&HashMap<I, u32>>,
    k: usize,
) -> FusionDiagnostics<I> {
    let stats_a = score_stats(a);
    let stats_b = score_stats(b);
    let overlap = overlap_ratio(a, b);
    let overlap_k = overlap_at_k(a, b, k);
    let comp = qrels.map(|q| complementarity(a, b, q));
    let tau = rank_correlation(a, b);

    // Unique documents
    let set_a: HashSet<_> = a.iter().map(|(id, _)| id).collect();
    let set_b: HashSet<_> = b.iter().map(|(id, _)| id).collect();
    let unique_a: Vec<I> = a
        .iter()
        .filter(|(id, _)| !set_b.contains(id))
        .map(|(id, _)| id.clone())
        .collect();
    let unique_b: Vec<I> = b
        .iter()
        .filter(|(id, _)| !set_a.contains(id))
        .map(|(id, _)| id.clone())
        .collect();

    // Suggestion logic
    let suggestion = if let Some(c) = comp {
        if c > 0.5 {
            FusionSuggestion::FuseRecommended {
                reason: "high complementarity (>0.5): retrievers find different relevant docs",
            }
        } else if c > 0.2 {
            FusionSuggestion::FuseWithCaution {
                reason: "moderate complementarity: tune weights for best results",
            }
        } else {
            FusionSuggestion::SkipFusion {
                reason: "low complementarity (<0.2): retrievers are redundant",
            }
        }
    } else if overlap < 0.1 {
        FusionSuggestion::FuseRecommended {
            reason: "very low overlap: retrievers see different document sets",
        }
    } else if tau.abs() < 0.3 {
        FusionSuggestion::FuseRecommended {
            reason: "low rank correlation: retrievers disagree on ordering",
        }
    } else if tau > 0.8 {
        FusionSuggestion::SkipFusion {
            reason: "very high rank correlation (>0.8): retrievers agree, fusion adds little",
        }
    } else {
        FusionSuggestion::FuseWithCaution {
            reason: "moderate agreement: fusion may help with tuned weights",
        }
    };

    FusionDiagnostics {
        stats_a,
        stats_b,
        overlap,
        overlap_at_k: overlap_k,
        complementarity: comp,
        rank_correlation: tau,
        suggestion,
        unique_to_a: unique_a,
        unique_to_b: unique_b,
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = (p / 100.0) * (sorted.len() - 1) as f32;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f32;

    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_stats_basic() {
        let results = vec![("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0), ("e", 5.0)];
        let stats = score_stats(&results).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.median - 3.0).abs() < 1e-6);
    }

    #[test]
    fn score_stats_empty() {
        let results: Vec<(&str, f32)> = vec![];
        assert!(score_stats(&results).is_none());
    }

    #[test]
    fn overlap_ratio_disjoint() {
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d3", 0.9), ("d4", 0.8)];
        assert!((overlap_ratio(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn overlap_ratio_identical() {
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d1", 0.7), ("d2", 0.6)];
        assert!((overlap_ratio(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_ratio_partial() {
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d2", 0.7), ("d3", 0.6)];
        // Intersection: {d2}, Union: {d1, d2, d3}
        assert!((overlap_ratio(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn complementarity_high() {
        let qrels: HashMap<&str, u32> = HashMap::from([("d1", 1), ("d2", 1), ("d3", 1), ("d4", 1)]);
        // A finds d1, d2 (relevant). B finds d3, d4 (relevant). No overlap.
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d3", 0.9), ("d4", 0.8)];

        let c = complementarity(&a, &b, &qrels);
        // All 4 relevant docs are unique to one list: 4/4 = 1.0
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn complementarity_zero() {
        let qrels: HashMap<&str, u32> = HashMap::from([("d1", 1), ("d2", 1)]);
        // Both retrievers find the same relevant docs
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d1", 0.7), ("d2", 0.6)];

        let c = complementarity(&a, &b, &qrels);
        // 0 unique relevant / 2 total relevant = 0.0
        assert!(c.abs() < 1e-6);
    }

    #[test]
    fn rank_correlation_identical() {
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let tau = rank_correlation(&a, &b);
        assert!(
            (tau - 1.0).abs() < 1e-6,
            "identical rankings should have tau=1.0"
        );
    }

    #[test]
    fn rank_correlation_reversed() {
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d3", 0.9), ("d2", 0.8), ("d1", 0.7)];
        let tau = rank_correlation(&a, &b);
        assert!(
            (tau - (-1.0)).abs() < 1e-6,
            "reversed rankings should have tau=-1.0, got {}",
            tau
        );
    }

    #[test]
    fn diagnose_complementary() {
        let qrels: HashMap<&str, u32> = HashMap::from([("d1", 1), ("d2", 1), ("d3", 1), ("d4", 1)]);
        let a = vec![("d1", 0.9), ("d2", 0.8), ("x1", 0.5)];
        let b = vec![("d3", 0.9), ("d4", 0.8), ("x2", 0.5)];

        let diag = diagnose(&a, &b, Some(&qrels), 3);

        assert!(diag.complementarity.unwrap() > 0.9);
        assert!(matches!(
            diag.suggestion,
            FusionSuggestion::FuseRecommended { .. }
        ));
        assert_eq!(diag.unique_to_a.len(), 3); // all of a is unique
        assert_eq!(diag.unique_to_b.len(), 3);
    }

    #[test]
    fn diagnose_redundant() {
        let qrels: HashMap<&str, u32> = HashMap::from([("d1", 1), ("d2", 1)]);
        let a = vec![("d1", 0.9), ("d2", 0.8)];
        let b = vec![("d1", 0.7), ("d2", 0.6)];

        let diag = diagnose(&a, &b, Some(&qrels), 2);

        assert!(diag.complementarity.unwrap() < 0.01);
        assert!(matches!(
            diag.suggestion,
            FusionSuggestion::SkipFusion { .. }
        ));
    }
}
