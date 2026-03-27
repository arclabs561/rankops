//! Differentiable top-k selection via dynamic programming on a smooth semiring.
//!
//! Casts top-k as a DP where the hard max/add operations are replaced with
//! log-sum-exp smooth approximations. As temperature approaches 0, the output
//! converges to hard top-k selection.
//!
//! Reference: Vivier-Ardisson, Sander, Parmentier, Blondel 2026
//! "Differentiable Knapsack and Top-k Operators"
//!
//! # Example
//!
//! ```rust
//! use rankops::dp_topk::dp_topk;
//!
//! let scores = vec![3.0, 1.0, 4.0, 1.5, 2.0];
//! let selection = dp_topk(&scores, 2, 0.1);
//! // Items at index 0 (score=3.0) and 2 (score=4.0) selected
//! assert!(selection[2] > 0.9);
//! assert!(selection[0] > 0.9);
//! assert!(selection[1] < 0.1);
//! ```

/// Smooth maximum: `t * log(exp(a/t) + exp(b/t))`.
///
/// Numerically stable via the log-sum-exp trick.
#[inline]
fn smooth_max(a: f32, b: f32, temperature: f32) -> f32 {
    let inv_t = temperature.recip();
    let m = a.max(b);
    if m == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    m + temperature * (((a - m) * inv_t).exp() + ((b - m) * inv_t).exp()).ln()
}

/// Log-sum-exp over a slice, returning `t * log(sum_i exp(x_i / t))`.
#[inline]
fn log_sum_exp(xs: &[f32], temperature: f32) -> f32 {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let inv_t = temperature.recip();
    let sum: f32 = xs.iter().map(|&x| ((x - m) * inv_t).exp()).sum();
    m + temperature * sum.ln()
}

/// Forward DP pass for top-k selection.
///
/// Returns the DP table `v` of shape `(n+1) x (k+1)`.
/// `v[i][j]` = smooth-max total score of selecting exactly `j` items from the first `i`.
fn forward_dp(scores: &[f32], k: usize, temperature: f32) -> Vec<Vec<f32>> {
    let n = scores.len();
    let mut v = vec![vec![f32::NEG_INFINITY; k + 1]; n + 1];
    v[0][0] = 0.0;

    for i in 1..=n {
        let s = scores[i - 1];
        for j in 0..=k.min(i) {
            let skip = v[i - 1][j];
            let pick = if j > 0 {
                v[i - 1][j - 1] + s
            } else {
                f32::NEG_INFINITY
            };
            v[i][j] = smooth_max(skip, pick, temperature);
        }
    }
    v
}

/// Backward pass: compute soft selection probabilities from the DP table.
///
/// Uses the smooth semiring structure to extract marginal selection probabilities
/// via the "backward" values `w[i][j]` = smooth-max total of selecting `k - j`
/// items from items `i+1..n`.
fn backward_selection(scores: &[f32], k: usize, temperature: f32, v: &[Vec<f32>]) -> Vec<f32> {
    let n = scores.len();
    // Backward table: w[i][j] = best way to pick (k - j) items from items i..n-1
    let mut w = vec![vec![f32::NEG_INFINITY; k + 1]; n + 1];
    w[n][k] = 0.0;

    for i in (1..=n).rev() {
        let s = scores[i - 1];
        for j in 0..=k.min(i) {
            // Propagate backward: if we're at state (i, j), we came from (i-1, j) [skip]
            // or (i-1, j-1) [pick].
            let cur_w = w[i][j];
            if cur_w == f32::NEG_INFINITY {
                continue;
            }
            // Skip contribution to w[i-1][j]
            w[i - 1][j] = smooth_max(w[i - 1][j], cur_w, temperature);
            // Pick contribution to w[i-1][j-1]
            if j > 0 {
                w[i - 1][j - 1] = smooth_max(w[i - 1][j - 1], cur_w + s, temperature);
            }
        }
    }

    // The optimal value is v[n][k].
    let opt = v[n][k];
    if opt == f32::NEG_INFINITY {
        return vec![0.0; n];
    }

    // Selection probability for item i:
    // p_i = sum_j exp((v[i-1][j-1] + s_i + w[i][j]) / t) / exp(opt / t)
    //      = exp(lse_j(v[i-1][j-1] + s_i + w[i][j]) / t - opt / t)
    let inv_t = temperature.recip();
    let mut selection = vec![0.0_f32; n];

    for i in 1..=n {
        let s = scores[i - 1];
        let mut pick_vals = Vec::with_capacity(k.min(i));

        for j in 1..=k.min(i) {
            let fwd = v[i - 1][j - 1];
            let bwd = w[i][j];
            if fwd > f32::NEG_INFINITY && bwd > f32::NEG_INFINITY {
                pick_vals.push(fwd + s + bwd);
            }
        }

        if pick_vals.is_empty() {
            continue;
        }

        let log_pick = log_sum_exp(&pick_vals, temperature);
        let diff = (log_pick - opt) * inv_t;
        selection[i - 1] = diff.exp().clamp(0.0, 1.0);
    }

    selection
}

/// Differentiable top-k selection via dynamic programming.
///
/// Returns a soft selection vector of length `n` where values are in `[0, 1]`.
/// Selected items have values near 1, unselected near 0.
/// The sum of the vector is approximately `k`.
///
/// As `temperature` approaches 0, the output converges to hard top-k selection.
///
/// # Arguments
///
/// * `scores` - Score for each item (higher = more likely selected).
/// * `k` - Number of items to select. Must be `<= scores.len()`.
/// * `temperature` - Smoothing temperature. Lower = sharper. Must be `> 0`.
///
/// # Panics
///
/// Panics if `k > scores.len()` or `temperature <= 0`.
pub fn dp_topk(scores: &[f32], k: usize, temperature: f32) -> Vec<f32> {
    assert!(k <= scores.len(), "k ({k}) > n ({})", scores.len());
    assert!(
        temperature > 0.0,
        "temperature must be positive, got {temperature}"
    );

    let n = scores.len();
    if n == 0 || k == 0 {
        return vec![0.0; n];
    }
    if k == n {
        return vec![1.0; n];
    }

    let v = forward_dp(scores, k, temperature);
    backward_selection(scores, k, temperature, &v)
}

/// Differentiable top-k with score gradients.
///
/// Returns `(selection, gradients)` where:
/// - `selection` is the soft selection vector (same as [`dp_topk`]).
/// - `gradients` is `d(optimal_value) / d(score_i)` for each item, equal to
///   the selection probability. This is the key property of the smooth semiring
///   formulation: the gradient of the optimal value with respect to each score
///   is exactly the marginal selection probability.
///
/// # Arguments
///
/// Same as [`dp_topk`].
pub fn dp_topk_with_grad(scores: &[f32], k: usize, temperature: f32) -> (Vec<f32>, Vec<f32>) {
    let sel = dp_topk(scores, k, temperature);
    // In the smooth semiring formulation, d(V*)/d(s_i) = p_i (selection probability).
    let grad = sel.clone();
    (sel, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_max_approaches_hard_max() {
        let a = 3.0_f32;
        let b = 5.0;
        // At low temperature, smooth_max -> max(a, b)
        let result = smooth_max(a, b, 0.01);
        assert!((result - 5.0).abs() < 0.05, "got {result}");
        // At high temperature, smooth_max -> soft blend
        let soft = smooth_max(a, b, 10.0);
        assert!(soft > 5.0, "soft max should exceed hard max, got {soft}");
    }

    #[test]
    fn returns_correct_length() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sel = dp_topk(&scores, 3, 0.5);
        assert_eq!(sel.len(), 5);
    }

    #[test]
    fn values_in_unit_interval() {
        let scores = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        let sel = dp_topk(&scores, 3, 0.5);
        for (i, &v) in sel.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "sel[{i}] = {v} out of [0, 1]");
        }
    }

    #[test]
    fn sum_approximately_k() {
        let scores = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        for k in 1..=5 {
            let sel = dp_topk(&scores, k, 0.5);
            let sum: f32 = sel.iter().sum();
            assert!(
                (sum - k as f32).abs() < 1.0,
                "k={k}, sum={sum}, expected ~{k}"
            );
        }
    }

    #[test]
    fn low_temperature_matches_hard_topk() {
        let scores = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6];
        let k = 2;
        let sel = dp_topk(&scores, k, 0.01);
        // Top-2 are index 4 (9.0) and index 2 (4.0)
        assert!(sel[4] > 0.9, "top item sel={}", sel[4]);
        assert!(sel[2] > 0.9, "second item sel={}", sel[2]);
        // Others should be near 0
        assert!(sel[1] < 0.1, "non-top item sel={}", sel[1]);
        assert!(sel[3] < 0.1, "non-top item sel={}", sel[3]);
    }

    #[test]
    fn monotonicity_higher_scores_higher_selection() {
        // Distinct scores so ordering is unambiguous
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sel = dp_topk(&scores, 3, 0.5);
        // Item with score 5 should have higher selection than item with score 1
        assert!(
            sel[4] > sel[0],
            "sel[4]={} should > sel[0]={}",
            sel[4],
            sel[0]
        );
        // Items sorted by score should have roughly monotone selection
        for i in 0..4 {
            assert!(
                sel[i] <= sel[i + 1] + 0.05,
                "sel[{i}]={} > sel[{}]={} (non-monotone)",
                sel[i],
                i + 1,
                sel[i + 1]
            );
        }
    }

    #[test]
    fn gradient_nonzero_for_all_scores() {
        let scores = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        let (_, grad) = dp_topk_with_grad(&scores, 2, 1.0);
        // At moderate temperature, all gradients should be nonzero
        for (i, &g) in grad.iter().enumerate() {
            assert!(g > 0.0, "grad[{i}] = {g}, expected > 0");
        }
    }

    #[test]
    fn edge_case_k_zero() {
        let scores = vec![1.0, 2.0, 3.0];
        let sel = dp_topk(&scores, 0, 1.0);
        assert_eq!(sel, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn edge_case_k_equals_n() {
        let scores = vec![1.0, 2.0, 3.0];
        let sel = dp_topk(&scores, 3, 1.0);
        assert_eq!(sel, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn edge_case_empty() {
        let scores: Vec<f32> = vec![];
        let sel = dp_topk(&scores, 0, 1.0);
        assert!(sel.is_empty());
    }

    #[test]
    fn edge_case_single_item() {
        let scores = vec![42.0];
        let sel = dp_topk(&scores, 1, 0.5);
        assert_eq!(sel, vec![1.0]);
    }

    #[test]
    fn temperature_effect() {
        let scores = vec![3.0, 1.0, 4.0, 1.5, 2.0];
        let sharp = dp_topk(&scores, 2, 0.01);
        let smooth = dp_topk(&scores, 2, 5.0);

        // Sharp selection should be more extreme (closer to 0/1)
        // Measure by variance: sharp has higher variance (values near 0 or 1)
        let mean_sharp = sharp.iter().sum::<f32>() / sharp.len() as f32;
        let var_sharp: f32 = sharp.iter().map(|&p| (p - mean_sharp).powi(2)).sum();
        let mean_smooth = smooth.iter().sum::<f32>() / smooth.len() as f32;
        let var_smooth: f32 = smooth.iter().map(|&p| (p - mean_smooth).powi(2)).sum();
        assert!(
            var_sharp > var_smooth,
            "sharp variance {var_sharp} should > smooth variance {var_smooth}"
        );
    }

    #[test]
    #[should_panic(expected = "k (4) > n (3)")]
    fn panics_k_exceeds_n() {
        dp_topk(&[1.0, 2.0, 3.0], 4, 1.0);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn panics_negative_temperature() {
        dp_topk(&[1.0, 2.0], 1, -0.5);
    }

    #[test]
    fn equal_scores_uniform_selection() {
        let scores = vec![1.0; 5];
        let sel = dp_topk(&scores, 2, 1.0);
        // With equal scores, selection should be roughly uniform
        let mean = sel.iter().sum::<f32>() / sel.len() as f32;
        for (i, &v) in sel.iter().enumerate() {
            assert!(
                (v - mean).abs() < 0.15,
                "sel[{i}]={v} deviates from mean={mean}"
            );
        }
    }
}
