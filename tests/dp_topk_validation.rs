//! Differentiable top-k against its exact (hard) limit.
//!
//! `dp_topk(scores, k, τ)` is a smooth k-selection: the k largest scores get
//! selection ~1, the rest ~0, and the selection sums to k. These tests check
//! that against the exact hard top-k indicator across 80 shuffled inputs.
//!
//! Numerical floor: the f32 marginal `exp((log_pick - opt)/τ)` loses accuracy
//! below roughly τ=1e-2 (cancellation amplified by 1/τ), where the sum drifts
//! below k and top items collapse toward 0. The tests run in the usable regime
//! (τ=2e-2) and do not assert anything below the floor; an f64 internal pass
//! would extend the range.

use rankops::dp_topk::dp_topk;

/// Usable regime: small enough to be sharp, large enough to be f32-stable.
const TAU: f32 = 2e-2;

/// One case: scores, k, and the exact top-k indicator.
struct Case {
    scores: Vec<f32>,
    k: usize,
    in_topk: Vec<bool>,
}

/// 80 cases with n in 3..=16 and k in 1..=n.
///
/// Scores are shuffled multiples of 5, so the k/k+1 boundary is unambiguous;
/// close-boundary scores legitimately split probability and are not a hard
/// top-k at finite temperature.
fn cases() -> Vec<Case> {
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };

    (0..80)
        .map(|trial| {
            let n = 3 + (trial % 14);
            let k = 1 + (trial % n);
            let mut scores: Vec<f32> = (0..n).map(|i| i as f32 * 5.0).collect();
            for i in (1..n).rev() {
                let j = ((next() * (i as f32 + 1.0)) as usize).min(i);
                scores.swap(i, j);
            }

            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            let mut in_topk = vec![false; n];
            for &i in idx.iter().take(k) {
                in_topk[i] = true;
            }
            Case { scores, k, in_topk }
        })
        .collect()
}

#[test]
fn selection_sums_to_k() {
    let violations: Vec<String> = cases()
        .iter()
        .filter_map(|c| {
            let sum: f32 = dp_topk(&c.scores, c.k, TAU).iter().sum();
            ((sum - c.k as f32).abs() >= 0.1).then(|| {
                format!(
                    "n={} k={}: selection sum {sum:.4} != k",
                    c.scores.len(),
                    c.k
                )
            })
        })
        .collect();
    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn selection_matches_hard_topk_indicator() {
    let mut violations = Vec::new();
    for c in cases() {
        let n = c.scores.len();
        let sel = dp_topk(&c.scores, c.k, TAU);
        for (i, (&p, &top)) in sel.iter().zip(&c.in_topk).enumerate() {
            if top && p <= 0.85 {
                violations.push(format!(
                    "n={n} k={}: top-k item {i} sel {p:.4} <= 0.85",
                    c.k
                ));
            } else if !top && p >= 0.15 {
                violations.push(format!(
                    "n={n} k={}: non-top item {i} sel {p:.4} >= 0.15",
                    c.k
                ));
            }
        }
    }
    assert!(violations.is_empty(), "{violations:#?}");
}
