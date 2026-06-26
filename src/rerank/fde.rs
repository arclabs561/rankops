//! Fixed-dimensional encodings for late-interaction retrieval.
//!
//! Late-interaction scoring keeps one vector per query token and one vector per
//! document token, then computes `MaxSim`. That gives useful token-level
//! matching, but exact scoring is not a single-vector nearest-neighbor problem.
//!
//! This module provides a small fixed-dimensional proxy:
//!
//! - assign tokens to deterministic SimHash buckets,
//! - encode a query by summing token vectors per bucket,
//! - encode a document by storing a centroid per bucket,
//! - score the two fixed vectors with an inner product.
//!
//! The result is a first-stage approximation for candidate generation. Use
//! [`crate::rerank::colbert`] to rerank the shortlisted candidates with exact
//! `MaxSim`.

use super::{simd, RerankError, Result};

const MAX_SIMHASH_BITS: u8 = 12;
const DEFAULT_SEED: u64 = 0x7a5d_12c3_8e91_b6f0;

/// Configuration for fixed-dimensional late-interaction encodings.
///
/// The default uses 4 repetitions of 4 buckets each. Increasing buckets or
/// repetitions usually improves the proxy but also increases vector length:
/// `repetitions * 2^simhash_bits * token_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdeConfig {
    simhash_bits: u8,
    repetitions: usize,
    seed: u64,
    fill_empty_doc_clusters: bool,
}

impl Default for FdeConfig {
    fn default() -> Self {
        Self {
            simhash_bits: 2,
            repetitions: 4,
            seed: DEFAULT_SEED,
            fill_empty_doc_clusters: true,
        }
    }
}

impl FdeConfig {
    /// Construct the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of SimHash bits used per repetition.
    #[must_use]
    pub const fn simhash_bits(&self) -> u8 {
        self.simhash_bits
    }

    /// Number of independent SimHash repetitions.
    #[must_use]
    pub const fn repetitions(&self) -> usize {
        self.repetitions
    }

    /// Seed for deterministic SimHash projections.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether empty document buckets are filled from the nearest non-empty bucket.
    #[must_use]
    pub const fn fills_empty_doc_clusters(&self) -> bool {
        self.fill_empty_doc_clusters
    }

    /// Set SimHash bits per repetition.
    ///
    /// `0` is allowed and produces one bucket. Values above 12 are rejected to
    /// avoid accidentally constructing very large vectors.
    pub fn with_simhash_bits(mut self, simhash_bits: u8) -> Result<Self> {
        if simhash_bits > MAX_SIMHASH_BITS {
            return Err(RerankError::InvalidFdeConfig {
                reason: "simhash_bits must be <= 12",
            });
        }
        self.simhash_bits = simhash_bits;
        Ok(self)
    }

    /// Set the number of independent repetitions.
    pub fn with_repetitions(mut self, repetitions: usize) -> Result<Self> {
        if repetitions == 0 {
            return Err(RerankError::InvalidFdeConfig {
                reason: "repetitions must be >= 1",
            });
        }
        self.repetitions = repetitions;
        Ok(self)
    }

    /// Set the deterministic projection seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable empty document bucket filling.
    #[must_use]
    pub const fn with_empty_doc_cluster_fill(mut self, fill: bool) -> Self {
        self.fill_empty_doc_clusters = fill;
        self
    }

    /// Number of buckets per repetition.
    #[must_use]
    pub const fn buckets(&self) -> usize {
        1usize << self.simhash_bits
    }

    /// Encoded vector length for token vectors with `dim` dimensions.
    #[must_use]
    pub fn encoded_len(&self, dim: usize) -> usize {
        self.repetitions * self.buckets() * dim
    }

    /// Encode query tokens as bucket-wise vector sums.
    pub fn encode_query(&self, tokens: &[Vec<f32>]) -> Result<FixedDimEncoding> {
        let dim = validate_tokens(tokens)?;
        let buckets = self.buckets();
        let mut values = vec![0.0; self.encoded_len(dim)];

        for repetition in 0..self.repetitions {
            for token in tokens {
                let bucket = simhash_bucket(token, repetition, self.simhash_bits, self.seed);
                let offset = block_offset(repetition, bucket, buckets, dim);
                add_into(&mut values[offset..offset + dim], token);
            }
        }

        Ok(FixedDimEncoding {
            values,
            repetitions: self.repetitions,
            buckets,
            dim,
        })
    }

    /// Encode document tokens as bucket centroids.
    ///
    /// Empty buckets are filled by default from the nearest non-empty bucket in
    /// Hamming distance. This keeps every query bucket able to score against a
    /// document representative, which is closer to `MaxSim` than all-zero empty
    /// buckets on small examples.
    pub fn encode_document(&self, tokens: &[Vec<f32>]) -> Result<FixedDimEncoding> {
        let dim = validate_tokens(tokens)?;
        let buckets = self.buckets();
        let mut values = vec![0.0; self.encoded_len(dim)];
        let mut counts = vec![0usize; self.repetitions * buckets];

        for repetition in 0..self.repetitions {
            for token in tokens {
                let bucket = simhash_bucket(token, repetition, self.simhash_bits, self.seed);
                counts[repetition * buckets + bucket] += 1;
                let offset = block_offset(repetition, bucket, buckets, dim);
                add_into(&mut values[offset..offset + dim], token);
            }
        }

        for repetition in 0..self.repetitions {
            for bucket in 0..buckets {
                let count = counts[repetition * buckets + bucket];
                if count > 0 {
                    let offset = block_offset(repetition, bucket, buckets, dim);
                    scale_in_place(&mut values[offset..offset + dim], 1.0 / count as f32);
                }
            }
        }

        if self.fill_empty_doc_clusters {
            fill_empty_document_buckets(&mut values, &counts, self.repetitions, buckets, dim);
        }

        Ok(FixedDimEncoding {
            values,
            repetitions: self.repetitions,
            buckets,
            dim,
        })
    }

    /// Approximate late-interaction score from raw query and document tokens.
    pub fn score(&self, query: &[Vec<f32>], document: &[Vec<f32>]) -> Result<f32> {
        let query = self.encode_query(query)?;
        let document = self.encode_document(document)?;
        query.score(&document)
    }

    /// Rank documents by the fixed-dimensional proxy score.
    pub fn rank<I: Clone>(
        &self,
        query: &[Vec<f32>],
        docs: &[(I, Vec<Vec<f32>>)],
    ) -> Result<Vec<(I, f32)>> {
        let query = self.encode_query(query)?;
        let mut results = Vec::with_capacity(docs.len());

        for (id, tokens) in docs {
            let document = self.encode_document(tokens)?;
            results.push((id.clone(), query.score(&document)?));
        }

        super::sort_scored_desc(&mut results);
        Ok(results)
    }
}

/// A fixed-dimensional representation of a multi-vector item.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedDimEncoding {
    values: Vec<f32>,
    repetitions: usize,
    buckets: usize,
    dim: usize,
}

impl FixedDimEncoding {
    /// Encoded values suitable for inner-product ANN.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Consume the encoding and return its vector values.
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }

    /// Number of SimHash repetitions used to build this encoding.
    #[must_use]
    pub const fn repetitions(&self) -> usize {
        self.repetitions
    }

    /// Number of buckets per repetition.
    #[must_use]
    pub const fn buckets(&self) -> usize {
        self.buckets
    }

    /// Token vector dimension used by this encoding.
    #[must_use]
    pub const fn token_dim(&self) -> usize {
        self.dim
    }

    /// Raw encoded vector length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the encoded vector has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Approximate score against another fixed-dimensional encoding.
    ///
    /// This returns the average inner product across repetitions. Multiplying by
    /// the repetition count gives the raw concatenated-vector inner product,
    /// which has the same ranking order for a fixed configuration.
    pub fn score(&self, other: &Self) -> Result<f32> {
        self.check_compatible(other)?;
        Ok(simd::dot(&self.values, &other.values) / self.repetitions as f32)
    }

    fn check_compatible(&self, other: &Self) -> Result<()> {
        if self.values.len() != other.values.len() {
            return Err(RerankError::DimensionMismatch {
                expected: self.values.len(),
                got: other.values.len(),
            });
        }
        if self.repetitions != other.repetitions {
            return Err(RerankError::DimensionMismatch {
                expected: self.repetitions,
                got: other.repetitions,
            });
        }
        if self.buckets != other.buckets {
            return Err(RerankError::DimensionMismatch {
                expected: self.buckets,
                got: other.buckets,
            });
        }
        if self.dim != other.dim {
            return Err(RerankError::DimensionMismatch {
                expected: self.dim,
                got: other.dim,
            });
        }
        Ok(())
    }
}

fn validate_tokens(tokens: &[Vec<f32>]) -> Result<usize> {
    let Some(first) = tokens.first() else {
        return Err(RerankError::InvalidFdeConfig {
            reason: "token list must be non-empty",
        });
    };
    let dim = first.len();
    if dim == 0 {
        return Err(RerankError::InvalidFdeConfig {
            reason: "token dimension must be >= 1",
        });
    }
    for token in tokens {
        if token.len() != dim {
            return Err(RerankError::DimensionMismatch {
                expected: dim,
                got: token.len(),
            });
        }
    }
    Ok(dim)
}

fn simhash_bucket(token: &[f32], repetition: usize, bits: u8, seed: u64) -> usize {
    let mut bucket = 0usize;
    for bit in 0..bits {
        let mut sum = 0.0;
        for (dim, value) in token.iter().enumerate() {
            let sign = projection_sign(seed, repetition, bit, dim);
            sum += sign * value;
        }
        if sum >= 0.0 {
            bucket |= 1usize << bit;
        }
    }
    bucket
}

fn projection_sign(seed: u64, repetition: usize, bit: u8, dim: usize) -> f32 {
    let mut x = seed;
    x ^= (repetition as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= (bit as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= (dim as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    if splitmix64(x) & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn block_offset(repetition: usize, bucket: usize, buckets: usize, dim: usize) -> usize {
    (repetition * buckets + bucket) * dim
}

fn add_into(accumulator: &mut [f32], token: &[f32]) {
    for (acc, value) in accumulator.iter_mut().zip(token) {
        *acc += value;
    }
}

fn scale_in_place(values: &mut [f32], factor: f32) {
    for value in values {
        *value *= factor;
    }
}

fn fill_empty_document_buckets(
    values: &mut [f32],
    counts: &[usize],
    repetitions: usize,
    buckets: usize,
    dim: usize,
) {
    for repetition in 0..repetitions {
        for bucket in 0..buckets {
            if counts[repetition * buckets + bucket] > 0 {
                continue;
            }
            let Some(nearest) = nearest_non_empty_bucket(counts, repetition, bucket, buckets)
            else {
                continue;
            };
            let dst = block_offset(repetition, bucket, buckets, dim);
            let src = block_offset(repetition, nearest, buckets, dim);
            for offset in 0..dim {
                values[dst + offset] = values[src + offset];
            }
        }
    }
}

fn nearest_non_empty_bucket(
    counts: &[usize],
    repetition: usize,
    bucket: usize,
    buckets: usize,
) -> Option<usize> {
    (0..buckets)
        .filter(|candidate| counts[repetition * buckets + candidate] > 0)
        .min_by_key(|candidate| {
            (
                (bucket ^ candidate).count_ones(),
                bucket.abs_diff(*candidate),
                *candidate,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_len_matches_configuration() {
        let config = FdeConfig::new()
            .with_simhash_bits(3)
            .unwrap()
            .with_repetitions(5)
            .unwrap();

        assert_eq!(config.buckets(), 8);
        assert_eq!(config.encoded_len(7), 5 * 8 * 7);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(matches!(
            FdeConfig::new().with_repetitions(0),
            Err(RerankError::InvalidFdeConfig { .. })
        ));
        assert!(matches!(
            FdeConfig::new().with_simhash_bits(13),
            Err(RerankError::InvalidFdeConfig { .. })
        ));
    }

    #[test]
    fn rejects_mixed_token_dimensions() {
        let tokens = vec![vec![1.0, 0.0], vec![1.0]];
        assert!(matches!(
            FdeConfig::new().encode_query(&tokens),
            Err(RerankError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    #[test]
    fn single_bucket_single_token_matches_dot_product() {
        let config = FdeConfig::new()
            .with_simhash_bits(0)
            .unwrap()
            .with_repetitions(1)
            .unwrap();
        let query = vec![vec![1.0, 2.0, 3.0]];
        let document = vec![vec![4.0, 5.0, 6.0]];

        let score = config.score(&query, &document).unwrap();

        assert_eq!(score, 32.0);
    }

    #[test]
    fn fixed_vectors_score_if_compatible() {
        let config = FdeConfig::new()
            .with_simhash_bits(0)
            .unwrap()
            .with_repetitions(2)
            .unwrap();
        let query = config.encode_query(&[vec![1.0, 0.0]]).unwrap();
        let document = config.encode_document(&[vec![0.5, 0.5]]).unwrap();

        assert_eq!(query.repetitions(), 2);
        assert_eq!(query.buckets(), 1);
        assert_eq!(query.token_dim(), 2);
        assert_eq!(query.score(&document).unwrap(), 0.5);
    }

    #[test]
    fn rank_sorts_by_proxy_score() {
        let config = FdeConfig::new()
            .with_simhash_bits(0)
            .unwrap()
            .with_repetitions(1)
            .unwrap();
        let query = vec![vec![1.0, 0.0]];
        let docs = vec![
            ("weak", vec![vec![0.2, 0.0]]),
            ("strong", vec![vec![0.9, 0.0]]),
        ];

        let ranked = config.rank(&query, &docs).unwrap();

        assert_eq!(ranked[0].0, "strong");
        assert!(ranked[0].1 > ranked[1].1);
    }
}
