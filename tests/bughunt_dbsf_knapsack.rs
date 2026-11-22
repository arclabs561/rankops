//! Regressions for `dbsf` ordering and `dp_knapsack` marginals.

use rankops::dbsf;
use rankops::dp_topk::dp_knapsack;

fn pos(v: &[(&str, f32)], id: &str) -> usize {
    v.iter().position(|(d, _)| *d == id).unwrap()
}

/// The `dbsf` rustdoc example states "d2 and d3 appear in both lists, so they
/// rank highest". DBSF (Mazzeschi 2023; Qdrant) maps each list with
/// (s - (mu - 3 sigma)) / (6 sigma) and sums, so an absent document gets 0 and a
/// present one ~[0, 1]: expected order d2, d3, d1, d4. Summing raw clipped
/// z-scores makes "absent" equal to "at the list mean", so d3 (middle of B,
/// bottom of A) falls below d4 (only in B, at its bottom).
#[test]
fn dbsf_doc_example_ranks_shared_docs_first() {
    let bm25 = vec![("d1", 15.0), ("d2", 12.0), ("d3", 8.0)];
    let dense = vec![("d2", 0.9), ("d3", 0.7), ("d4", 0.5)];
    let fused = dbsf(&bm25, &dense);
    let order: Vec<&str> = fused.iter().map(|(d, _)| *d).collect();
    assert!(
        pos(&fused, "d3") < pos(&fused, "d4") && pos(&fused, "d3") < pos(&fused, "d1"),
        "dbsf order {order:?} (scores {fused:?}); DBSF reference order [d2, d3, d1, d4]"
    );
}

/// Exact Gibbs marginals of the entropy-smoothed knapsack:
/// p(S) ∝ exp(Σ_{i∈S} s_i / t) over subsets with Σ w_i <= capacity.
fn brute_knapsack(scores: &[f64], weights: &[usize], cap: usize, t: f64) -> Vec<f64> {
    let n = scores.len();
    let mut z = 0.0;
    let mut m = vec![0.0; n];
    for mask in 0u32..(1 << n) {
        let w: usize = (0..n)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| weights[i])
            .sum();
        if w > cap {
            continue;
        }
        let s: f64 = (0..n)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| scores[i])
            .sum();
        let p = (s / t).exp();
        z += p;
        for (i, mi) in m.iter_mut().enumerate() {
            if mask >> i & 1 == 1 {
                *mi += p;
            }
        }
    }
    m.iter().map(|x| x / z).collect()
}

#[test]
fn dp_knapsack_marginals_match_brute_force() {
    // Two unit-weight items, capacity 1: feasible sets {}, {0}, {1} -> 1/3 each.
    let sel = dp_knapsack(&[0.0, 0.0], &[1, 1], 1, 1.0);
    let exact = brute_knapsack(&[0.0, 0.0], &[1, 1], 1, 1.0);
    for i in 0..2 {
        assert!(
            (sel[i] as f64 - exact[i]).abs() < 1e-4,
            "item {i}: dp_knapsack {} vs exact {}  (all: {sel:?} vs {exact:?})",
            sel[i],
            exact[i]
        );
    }
}

#[test]
fn dp_knapsack_respects_capacity_in_hard_limit() {
    // Capacity 1, unit weights: only one item fits; item 1 (score 10) should win.
    let sel = dp_knapsack(&[1.0, 10.0], &[1, 1], 1, 0.1);
    let total: f32 = sel.iter().sum();
    assert!(
        total <= 1.0 + 1e-3,
        "expected selection mass <= 1 (capacity 1, unit weights), got {sel:?}"
    );
    assert!(
        sel[0] < 0.01,
        "low-score item should not be selected: {sel:?}"
    );
}

#[test]
fn dp_knapsack_random_cases_match_brute_force() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut worst = 0.0f64;
    for _ in 0..50 {
        let n = 2 + (next() % 4) as usize;
        let scores: Vec<f64> = (0..n).map(|_| (next() % 7) as f64 * 0.5).collect();
        let weights: Vec<usize> = (0..n).map(|_| 1 + (next() % 3) as usize).collect();
        let cap = 1 + (next() % 5) as usize;
        let t = 1.0;
        let s32: Vec<f32> = scores.iter().map(|&x| x as f32).collect();
        let sel = dp_knapsack(&s32, &weights, cap, t as f32);
        let exact = brute_knapsack(&scores, &weights, cap, t);
        for i in 0..n {
            worst = worst.max((sel[i] as f64 - exact[i]).abs());
        }
    }
    assert!(worst < 1e-3, "max |dp_knapsack - exact marginal| = {worst}");
}
