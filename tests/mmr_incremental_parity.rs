//! The three MMR implementations keep a running max similarity per candidate
//! instead of re-folding over the selected set every round. These references
//! are the previous from-scratch versions; outputs must match bit for bit
//! (same ids, same order, same score bits) on random inputs with ties,
//! duplicate vectors and zero vectors.
#![cfg(feature = "rerank")]

use rankops::rerank::diversity::{self, MmrConfig as DivConfig};
use rankops::rerank::simd;
use std::collections::HashMap;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Quarter-step values in [-1, 1], so ties are common.
    fn coarse(&mut self) -> f32 {
        (self.below(9) as f32 - 4.0) / 4.0
    }
}

fn normalize(scores: &[f32], eps: f32) -> Vec<f32> {
    let lo = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = hi - lo;
    if range > eps {
        scores.iter().map(|s| (s - lo) / range).collect()
    } else {
        vec![1.0; scores.len()]
    }
}

/// Previous `rankops::mmr`: returns (index, mmr score).
fn reference_generic(
    scores: &[f32],
    sim: impl Fn(usize, usize) -> f32,
    lambda: f32,
    k: usize,
) -> Vec<(usize, f32)> {
    let rel = normalize(scores, 1e-9);
    let mut remaining: Vec<(usize, f32)> = rel.into_iter().enumerate().collect();
    let mut selected: Vec<(usize, f32)> = Vec::new();
    while selected.len() < k.min(scores.len()) && !remaining.is_empty() {
        let (mut best_idx, mut best) = (0, f32::NEG_INFINITY);
        for (idx, &(cand, r)) in remaining.iter().enumerate() {
            let red = if selected.is_empty() {
                0.0
            } else {
                (1.0 - lambda)
                    * selected
                        .iter()
                        .map(|&(s, _)| sim(cand, s))
                        .fold(f32::NEG_INFINITY, f32::max)
            };
            let score = lambda * r - red;
            if score > best {
                best = score;
                best_idx = idx;
            }
        }
        let (id, _) = remaining.remove(best_idx);
        selected.push((id, best));
    }
    selected
}

/// Previous `diversity::mmr` / `mmr_cosine` selection order.
fn reference_indices(
    scores: &[f32],
    sim: impl Fn(usize, usize) -> f32,
    lambda: f32,
    k: usize,
) -> Vec<usize> {
    let rel = normalize(scores, 1e-9);
    let n = scores.len();
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..n).collect();
    for _ in 0..k.min(n) {
        let (mut best_idx, mut best) = (0, f32::NEG_INFINITY);
        for (pos, &c) in remaining.iter().enumerate() {
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|&s| sim(c, s))
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            let score = lambda * rel[c] - (1.0 - lambda) * max_sim;
            if score > best {
                best = score;
                best_idx = pos;
            }
        }
        selected.push(remaining.remove(best_idx));
    }
    selected
}

struct Case {
    scores: Vec<f32>,
    vectors: Vec<Vec<f32>>,
    lambda: f32,
    k: usize,
}

fn case(rng: &mut XorShift) -> Case {
    let n = 1 + rng.below(40) as usize;
    let dim = 1 + rng.below(6) as usize;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let v = match rng.below(8) {
            0 => vec![0.0; dim],
            1 if !vectors.is_empty() => vectors[rng.below(vectors.len() as u64) as usize].clone(),
            _ => (0..dim).map(|_| rng.coarse()).collect(),
        };
        vectors.push(v);
    }
    Case {
        scores: (0..n).map(|_| rng.coarse()).collect(),
        vectors,
        lambda: rng.below(5) as f32 / 4.0,
        k: rng.below(n as u64 + 3) as usize,
    }
}

#[test]
fn all_three_mmr_paths_match_the_from_scratch_versions() {
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    for round in 0..2000 {
        let c = case(&mut rng);
        let n = c.scores.len();
        let cos = |a: usize, b: usize| simd::cosine(&c.vectors[a], &c.vectors[b]);
        let ids: Vec<(usize, f32)> = c.scores.iter().copied().enumerate().collect();

        // rankops::mmr with a similarity closure.
        let got = rankops::mmr(
            &ids,
            |a: &usize, b: &usize| cos(*a, *b),
            rankops::MmrConfig::new(c.lambda).with_top_k(c.k),
        );
        let want = reference_generic(&c.scores, cos, c.lambda, c.k);
        let bits =
            |v: &[(usize, f32)]| v.iter().map(|(i, s)| (*i, s.to_bits())).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want), "rankops::mmr, round {round}");

        // diversity::mmr over a precomputed matrix.
        let matrix: Vec<f32> = (0..n * n).map(|i| cos(i / n, i % n)).collect();
        let got = diversity::mmr(&ids, &matrix, DivConfig::new(c.lambda, c.k));
        let want: Vec<(usize, f32)> =
            reference_indices(&c.scores, |a, b| matrix[a * n + b], c.lambda, c.k)
                .into_iter()
                .map(|i| ids[i])
                .collect();
        assert_eq!(bits(&got), bits(&want), "diversity::mmr, round {round}");

        // diversity::mmr_cosine over raw embeddings.
        let got = diversity::mmr_cosine(&ids, &c.vectors, DivConfig::new(c.lambda, c.k));
        let want: Vec<(usize, f32)> = reference_indices(&c.scores, cos, c.lambda, c.k)
            .into_iter()
            .map(|i| ids[i])
            .collect();
        assert_eq!(bits(&got), bits(&want), "mmr_cosine, round {round}");
    }
}

#[test]
fn mmr_embeddings_still_matches_closure_mmr() {
    let mut rng = XorShift(7);
    for _ in 0..300 {
        let c = case(&mut rng);
        let cands: Vec<(usize, f32, Vec<f32>)> = c
            .scores
            .iter()
            .zip(&c.vectors)
            .enumerate()
            .map(|(i, (s, v))| (i, *s, v.clone()))
            .collect();
        let by_id: HashMap<usize, &Vec<f32>> = c.vectors.iter().enumerate().collect();
        let config = rankops::MmrConfig::new(c.lambda).with_top_k(c.k);
        let got = rankops::mmr_embeddings(&cands, config);
        let want = reference_generic(
            &c.scores,
            |a, b| simd::cosine(by_id[&a], by_id[&b]).clamp(-1.0, 1.0),
            c.lambda,
            c.k,
        );
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!((g.0, g.1.to_bits()), (w.0, w.1.to_bits()));
        }
    }
}
