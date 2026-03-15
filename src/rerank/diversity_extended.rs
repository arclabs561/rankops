use super::simd;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Similarity metric used by diversification algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Cosine similarity (typically for normalized embeddings).
    Cosine,
    /// Raw dot-product similarity (will be rescaled internally for stability).
    Dot,
}

/// Configuration for Maximal Marginal Relevance (MMR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmrConfig {
    /// Trade-off between relevance and diversity.
    /// - `λ=1.0`: pure relevance (no diversity)
    /// - `λ=0.0`: pure diversity (maximize distance from selected)
    /// - `λ=0.5`: balanced (common default)
    pub lambda: f32,
    /// Number of results to select.
    pub k: usize,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self { lambda: 0.5, k: 10 }
    }
}

impl MmrConfig {
    /// Create config with custom lambda and k. Lambda clamped to \[0, 1\].
    #[must_use]
    pub fn new(lambda: f32, k: usize) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
            k,
        }
    }

    /// Set lambda (relevance-diversity tradeoff). Clamped to \[0, 1\].
    ///
    /// - `0.0` = maximum diversity (ignore relevance)
    /// - `0.5` = balanced (default)
    /// - `1.0` = pure relevance (ignore diversity)
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Set k (number of results to select).
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
}

/// Configuration for Max Sum Dispersion (MSD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsdConfig {
    /// Number of results to select.
    pub k: usize,
}

impl Default for MsdConfig {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl MsdConfig {
    /// Create a new MSD config with the given result count.
    #[must_use]
    pub const fn new(k: usize) -> Self {
        Self { k }
    }
}

/// Configuration for Sliding Spectrum Decomposition (SSD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsdConfig {
    /// Number of results to select.
    pub k: usize,
    /// Trade-off between relevance and novelty.
    /// Similar to MMR lambda but applied to projection magnitude.
    pub lambda: f32,
    /// Sliding window size for "recent" items (default 10).
    pub window_size: usize,
}

impl Default for SsdConfig {
    fn default() -> Self {
        Self {
            k: 10,
            lambda: 0.5,
            window_size: 10,
        }
    }
}

impl SsdConfig {
    /// Create a new SSD config with result count, diversity weight, and sliding window size.
    #[must_use]
    pub fn new(k: usize, lambda: f32, window_size: usize) -> Self {
        Self {
            k,
            lambda: lambda.clamp(0.0, 1.0),
            window_size,
        }
    }
}

/// Configuration for concave coverage (COVER / facility-location style).
///
/// This is adapted from the `pyversity` `cover()` strategy:
/// - Coverage is **summed similarity** per item: `c_j = Σ_{i∈S} sim(j, i)`
/// - Objective uses a concave transform: `Σ_j (c_j)^γ`, with `γ ∈ (0, 1]`
/// - Greedy selection uses marginal gain in that objective, blended with relevance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverConfig {
    /// Number of items to select.
    pub k: usize,
    /// Trade-off between relevance and diversity/coverage in \[0, 1\].
    ///
    /// - `0.0` = pure relevance
    /// - `1.0` = pure coverage
    pub diversity: f32,
    /// Concavity parameter for coverage in `(0, 1]`.
    ///
    /// Lower values emphasize diversity (stronger diminishing returns).
    pub gamma: f32,
    /// Similarity metric used for coverage.
    pub metric: SimilarityMetric,
    /// Whether to normalize embeddings to unit length before cosine similarity.
    pub normalize_embeddings: bool,
}

impl Default for CoverConfig {
    fn default() -> Self {
        Self {
            k: 10,
            diversity: 0.5,
            gamma: 0.5,
            metric: SimilarityMetric::Cosine,
            normalize_embeddings: true,
        }
    }
}

impl CoverConfig {
    /// Create a new config, clamping parameters to safe ranges.
    #[must_use]
    pub fn new(
        k: usize,
        diversity: f32,
        gamma: f32,
        metric: SimilarityMetric,
        normalize_embeddings: bool,
    ) -> Self {
        // gamma must be > 0 to avoid powf(0, 0) and to preserve concavity semantics.
        let gamma = gamma.clamp(1e-6, 1.0);
        Self {
            k,
            diversity: diversity.clamp(0.0, 1.0),
            gamma,
            metric,
            normalize_embeddings,
        }
    }

    /// Set k (number of results to select).
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set diversity trade-off. Clamped to \[0, 1\].
    #[must_use]
    pub fn with_diversity(mut self, diversity: f32) -> Self {
        self.diversity = diversity.clamp(0.0, 1.0);
        self
    }

    /// Set concavity parameter. Clamped to `(0, 1]`.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma.clamp(1e-6, 1.0);
        self
    }

    /// Set similarity metric.
    #[must_use]
    pub const fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set whether to normalize embeddings for cosine similarity.
    #[must_use]
    pub const fn with_normalize_embeddings(mut self, normalize: bool) -> Self {
        self.normalize_embeddings = normalize;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithms
// ─────────────────────────────────────────────────────────────────────────────

// ... (existing MMR and DPP implementations) ...

/// Max Sum Dispersion (MSD).
///
/// Selects items that maximize the sum of pairwise distances in the selected set.
/// Often used when you want a "representative set" covering the convex hull.
///
/// Greedy approximation:
/// 1. Select pair with max distance.
/// 2. Iteratively add item maximizing sum of distances to already selected items.
#[must_use]
pub fn msd<I: Clone, V: AsRef<[f32]>>(
    candidates: &[(I, f32)],
    embeddings: &[V],
    config: MsdConfig,
) -> Vec<(I, f32)> {
    let n = candidates.len();
    assert_eq!(embeddings.len(), n);

    if n == 0 || config.k == 0 {
        return Vec::new();
    }

    if config.k >= n {
        return candidates.to_vec();
    }

    let mut selected_indices = Vec::with_capacity(config.k);
    let mut remaining: Vec<usize> = (0..n).collect();

    // Step 1: Find pair with max distance (initial 2 items)
    // For efficiency, we can just pick the most relevant item first,
    // or run O(n^2) search. Borodin et al suggests greedy start is fine.
    // Let's pick max relevance item first to anchor the set (hybrid approach).
    // Pure MSD doesn't care about relevance, but for search/RAG we usually want
    // at least one highly relevant item.

    // Variant: Greedy Max Sum
    // Pick item maximizing avg distance to selected.

    // Initialize with highest relevance item
    let mut best_first_idx = 0;
    let mut best_first_score = f32::NEG_INFINITY;
    for (i, (_, score)) in candidates.iter().enumerate() {
        if *score > best_first_score {
            best_first_score = *score;
            best_first_idx = i;
        }
    }

    let first = remaining.swap_remove(remaining.iter().position(|&x| x == best_first_idx).unwrap());
    selected_indices.push(first);

    while selected_indices.len() < config.k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_dispersion = f32::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            let cand_emb = embeddings[cand_idx].as_ref();

            // Sum of distances to selected items
            let mut sum_dist = 0.0;
            for &sel_idx in &selected_indices {
                // Use cosine distance: 1 - cosine_similarity
                // or L2 distance. Cosine is standard for embeddings.
                let sel_emb = embeddings[sel_idx].as_ref();
                let dist = 1.0 - simd::cosine(cand_emb, sel_emb);
                sum_dist += dist;
            }

            if sum_dist > best_dispersion {
                best_dispersion = sum_dist;
                best_idx = pos;
            }
        }

        let chosen = remaining.swap_remove(best_idx);
        selected_indices.push(chosen);
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
}

/// Sliding Spectrum Decomposition (SSD).
///
/// Sequence-aware diversification. Ideal for chat history or feeds.
/// It penalizes items that are similar to the "recent subspace" spanned
/// by the last `window_size` items.
///
/// The penalty is based on the magnitude of projection onto the subspace.
#[must_use]
pub fn ssd<I: Clone, V: AsRef<[f32]>>(
    candidates: &[(I, f32)],
    embeddings: &[V],
    config: SsdConfig,
) -> Vec<(I, f32)> {
    // This requires subspace projection (Gram-Schmidt or similar).
    // For small window_size (e.g. 5-10), we can do modified Gram-Schmidt.

    let n = candidates.len();
    assert_eq!(embeddings.len(), n);

    if n == 0 || config.k == 0 {
        return Vec::new();
    }

    let mut selected_indices = Vec::with_capacity(config.k);
    let mut remaining: Vec<usize> = (0..n).collect();

    // Normalize relevance
    // ... (reuse normalization logic from MMR) ...
    let (rel_min, rel_max) = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
    let rel_range = rel_max - rel_min;
    let rel_norm: Vec<f32> = if rel_range > 1e-9 {
        candidates
            .iter()
            .map(|(_, s)| (s - rel_min) / rel_range)
            .collect()
    } else {
        vec![1.0; n]
    };

    // Basis vectors for the sliding window subspace (orthonormalized)
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(config.window_size);

    for _ in 0..config.k.min(n) {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (pos, &cand_idx) in remaining.iter().enumerate() {
            let relevance = rel_norm[cand_idx];
            let cand_vec = embeddings[cand_idx].as_ref();

            // Compute projection magnitude onto basis
            let mut projection_mag_sq = 0.0;
            for basis_vec in &basis {
                let dot = simd::dot(cand_vec, basis_vec);
                projection_mag_sq += dot * dot;
            }
            let projection_mag = projection_mag_sq.sqrt();

            // Novelty = 1 - projection_mag (roughly, assuming normalized vectors)
            // Score = λ * Rel + (1-λ) * Novelty
            //       = λ * Rel + (1-λ) * (1 - projection_mag)
            let score = config.lambda * relevance + (1.0 - config.lambda) * (1.0 - projection_mag);

            if score > best_score {
                best_score = score;
                best_idx = pos;
            }
        }

        let chosen_idx = remaining.swap_remove(best_idx);
        selected_indices.push(chosen_idx);

        // Update basis with chosen vector (Gram-Schmidt)
        let mut v = embeddings[chosen_idx].as_ref().to_vec();

        // Orthogonalize against current basis
        for basis_vec in &basis {
            let dot = simd::dot(&v, basis_vec);
            for (i, val) in v.iter_mut().enumerate() {
                *val -= dot * basis_vec[i];
            }
        }

        // Normalize
        let norm = simd::norm(&v);
        if norm > 1e-9 {
            for val in &mut v {
                *val /= norm;
            }

            if basis.len() >= config.window_size {
                basis.remove(0); // Slide window
            }
            basis.push(v);
        }
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
}

/// Concave coverage selection (COVER / facility-location style).
///
/// This greedily selects `k` items maximizing a blend of:
///
/// - **Relevance**: normalized to \[0, 1\]
/// - **Coverage gain**: marginal increase in `Σ_j (c_j)^γ`, where
///   `c_j = Σ_{i∈S} sim(j, i)` accumulates non-negative similarities
///
/// The blend weight follows `pyversity`: `theta = 1 - diversity`.
/// - `diversity = 0.0` ⇒ `theta = 1.0` ⇒ pure relevance
/// - `diversity = 1.0` ⇒ `theta = 0.0` ⇒ pure coverage
///
/// # Notes on scaling
///
/// The raw coverage gain is a sum over `n` items. To keep the blend stable
/// across candidate set sizes, we normalize coverage gains by `n`.
#[must_use]
pub fn cover<I: Clone, V: AsRef<[f32]>>(
    candidates: &[(I, f32)],
    embeddings: &[V],
    config: CoverConfig,
) -> Vec<(I, f32)> {
    let n = candidates.len();
    assert_eq!(embeddings.len(), n, "embeddings must match candidates");

    if n == 0 || config.k == 0 {
        return Vec::new();
    }

    let k = config.k.min(n);

    // Normalize relevance scores to [0, 1] (robust vs arbitrary retrieval score scales).
    let (rel_min, rel_max) = candidates
        .iter()
        .map(|(_, s)| *s)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
    let rel_range = rel_max - rel_min;
    let rel_norm: Vec<f32> = if rel_range > 1e-9 {
        candidates
            .iter()
            .map(|(_, s)| (s - rel_min) / rel_range)
            .collect()
    } else {
        vec![1.0; n]
    };

    // pyversity: theta = 1 - diversity
    let theta = 1.0 - config.diversity;
    if theta >= 1.0 - 1e-9 {
        // Pure relevance: select top-k by normalized relevance, return original scores.
        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.sort_unstable_by(|&a, &b| rel_norm[b].total_cmp(&rel_norm[a]).then_with(|| a.cmp(&b)));
        return idxs
            .into_iter()
            .take(k)
            .map(|idx| candidates[idx].clone())
            .collect();
    }

    // Optionally normalize embeddings for cosine similarity.
    let owned_normalized: Vec<Vec<f32>>;
    let emb: Vec<&[f32]> = match (config.metric, config.normalize_embeddings) {
        (SimilarityMetric::Cosine, true) => {
            owned_normalized = embeddings
                .iter()
                .map(|e| {
                    let v = e.as_ref();
                    let norm = simd::norm(v);
                    if norm > 1e-9 {
                        v.iter().map(|&x| x / norm).collect()
                    } else {
                        vec![0.0; v.len()]
                    }
                })
                .collect();
            owned_normalized.iter().map(|v| v.as_slice()).collect()
        }
        _ => embeddings.iter().map(|e| e.as_ref()).collect(),
    };

    // Precompute non-negative similarity matrix, row-major sim[i*n + j] = sim(i, j).
    let mut sim: Vec<f32> = vec![0.0; n * n];
    let mut max_sim = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let s = match config.metric {
                SimilarityMetric::Cosine => simd::cosine(emb[i], emb[j]).clamp(0.0, 1.0),
                SimilarityMetric::Dot => simd::dot(emb[i], emb[j]).max(0.0),
            };
            sim[i * n + j] = s;
            if s.is_finite() {
                max_sim = max_sim.max(s);
            }
        }
    }

    // For dot-product, rescale similarities into [0, 1] so coverage terms remain comparable.
    if config.metric == SimilarityMetric::Dot && max_sim > 0.0 {
        let inv = 1.0 / max_sim;
        for s in &mut sim {
            *s *= inv;
        }
    }

    let gamma = config.gamma;
    let inv_n = 1.0 / (n as f32);

    let mut accumulated = vec![0.0f32; n];
    let mut selected = vec![false; n];
    let mut selected_indices = Vec::with_capacity(k);

    for _step in 0..k {
        // concave_before[j] = accumulated[j]^gamma
        let concave_before: Vec<f32> = accumulated
            .iter()
            .map(|&c| c.max(0.0).powf(gamma))
            .collect();

        let mut best_i: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..n {
            if selected[i] {
                continue;
            }

            // coverage_gain(i) = Σ_j ((acc[j] + sim[j,i])^gamma - acc[j]^gamma)
            let mut gain = 0.0f32;
            for j in 0..n {
                let add = sim[j * n + i];
                let after = (accumulated[j] + add).max(0.0).powf(gamma);
                gain += after - concave_before[j];
            }
            gain *= inv_n; // normalize by n for stable blending

            let score = theta * rel_norm[i] + (1.0 - theta) * gain;

            // Deterministic tie-break: higher score, then lower index.
            if score > best_score
                || (score == best_score && best_i.map(|bi| i < bi).unwrap_or(true))
            {
                best_score = score;
                best_i = Some(i);
            }
        }

        let best = match best_i {
            Some(i) => i,
            None => break,
        };

        selected[best] = true;
        selected_indices.push(best);

        // accumulated[j] += sim[j, best]
        for j in 0..n {
            accumulated[j] += sim[j * n + best];
        }
    }

    selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cover_tests {
    use super::*;

    #[test]
    fn cover_empty_is_empty() {
        let candidates: Vec<(&str, f32)> = vec![];
        let embeddings: Vec<Vec<f32>> = vec![];
        let out = cover(&candidates, &embeddings, CoverConfig::default());
        assert!(out.is_empty());
    }

    #[test]
    fn cover_k_zero_is_empty() {
        let candidates = vec![("a", 0.9), ("b", 0.8)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let out = cover(&candidates, &embeddings, CoverConfig::default().with_k(0));
        assert!(out.is_empty());
    }

    #[test]
    fn cover_pure_relevance_matches_relevance_order() {
        let candidates = vec![("a", 10.0), ("b", 5.0), ("c", 7.0)];
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let out = cover(
            &candidates,
            &embeddings,
            CoverConfig::default().with_k(3).with_diversity(0.0),
        );
        // relevance order: a (10), c (7), b (5)
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "c");
        assert_eq!(out[2].0, "b");
    }

    #[test]
    fn cover_prefers_coverage_when_diversity_is_one() {
        // Construct 3 items:
        // - a and b are near-identical
        // - c is orthogonal
        // Relevance favors a strongly, but pure coverage should select a and c (not a and b).
        let candidates = vec![("a", 1.0), ("b", 0.9), ("c", 0.0)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.99, 0.01], vec![0.0, 1.0]];

        let out = cover(
            &candidates,
            &embeddings,
            CoverConfig::default()
                .with_k(2)
                .with_diversity(1.0)
                .with_gamma(0.5)
                .with_metric(SimilarityMetric::Cosine),
        );

        assert_eq!(out.len(), 2);
        // We expect c to be included because it increases coverage for the orthogonal dimension.
        let ids: std::collections::HashSet<_> = out.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn cover_is_deterministic_under_ties() {
        // All equal relevance and identical embeddings -> score ties everywhere.
        // We expect deterministic selection (by index).
        let candidates = vec![("a", 1.0), ("b", 1.0), ("c", 1.0)];
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let out = cover(
            &candidates,
            &embeddings,
            CoverConfig::default().with_k(2).with_diversity(1.0),
        );
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cover_proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cover_output_len_bounded(
            n in 0usize..20,
            k in 0usize..30,
            diversity in 0.0f32..1.0,
            gamma in 1e-3f32..1.0,
        ) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, (i as f32 * 0.37).sin()))
                .collect();

            // Fixed-dimensional embeddings; if n=0, keep empty.
            let dim = 4usize;
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let mut v = vec![0.0; dim];
                    v[i % dim] = 1.0;
                    v
                })
                .collect();

            let out = cover(
                &candidates,
                &embeddings,
                CoverConfig::new(
                    k,
                    diversity,
                    gamma,
                    SimilarityMetric::Cosine,
                    true,
                ),
            );

            prop_assert!(out.len() <= k.min(n));
        }

        #[test]
        fn cover_returns_unique_ids(n in 0usize..20, k in 0usize..20) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, 1.0 - (i as f32 * 0.01)))
                .collect();
            let dim = 3usize;
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    let mut v = vec![0.0; dim];
                    v[i % dim] = 1.0;
                    v
                })
                .collect();

            let out = cover(
                &candidates,
                &embeddings,
                CoverConfig::default().with_k(k),
            );

            let mut seen = std::collections::HashSet::new();
            for (id, _) in out {
                prop_assert!(seen.insert(id), "duplicate id selected");
            }
        }

        #[test]
        fn cover_handles_dot_metric_without_nan(n in 0usize..12, k in 0usize..12) {
            let candidates: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, (i as f32 * 0.13).cos()))
                .collect();
            let embeddings: Vec<Vec<f32>> = (0..n)
                .map(|i| vec![i as f32, 1.0, -0.5])
                .collect();

            let out = cover(
                &candidates,
                &embeddings,
                CoverConfig::default()
                    .with_k(k)
                    .with_metric(SimilarityMetric::Dot)
                    .with_diversity(0.7)
            );

            // Scores returned are the original candidate scores (not internal objective),
            // so just ensure they are finite.
            for (_id, score) in out {
                prop_assert!(score.is_finite());
            }
        }
    }
}
