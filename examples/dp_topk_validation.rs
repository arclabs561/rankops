//! Validate differentiable top-k against its exact (hard) limit.
//!
//! `dp_topk(scores, k, τ)` is a smooth k-selection: the k largest scores get
//! selection ~1, the rest ~0, and the selection sums to k. This checks that
//! against the exact hard top-k indicator across random distinct inputs.
//!
//! Numerical floor: the f32 marginal `exp((log_pick - opt)/τ)` loses accuracy
//! below roughly τ=1e-2 (cancellation amplified by 1/τ), where the sum drifts
//! below k and top items collapse toward 0. This eval runs in the usable regime
//! (τ=2e-2) and ALSO probes smaller τ to surface that degradation rather than
//! assert against it (an f64 internal pass would extend the range).
//!
//! ```sh
//! cargo run --release --example dp_topk_validation
//! ```

use std::process::ExitCode;

use rankops::dp_topk::dp_topk;

fn main() -> ExitCode {
    let mut failures = 0u64;
    let mut checks = 0u64;
    let mut check = |cond: bool, what: String| {
        checks += 1;
        if !cond {
            failures += 1;
            if failures <= 10 {
                eprintln!("  VIOLATION: {what}");
            }
        }
    };

    let mut s = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };

    for trial in 0..80 {
        let n = 3 + (trial % 14);
        let k = 1 + (trial % n);
        // Well-separated scores (shuffled multiples) so the k/k+1 boundary is
        // unambiguous; close-boundary scores legitimately split probability and
        // are not a hard top-k at finite temperature.
        let mut scores: Vec<f32> = (0..n).map(|i| i as f32 * 5.0).collect();
        for i in (1..n).rev() {
            let j = ((next() * (i as f32 + 1.0)) as usize).min(i);
            scores.swap(i, j);
        }

        // Exact top-k index set (distinct scores).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let in_topk: Vec<bool> = {
            let mut v = vec![false; n];
            for &i in idx.iter().take(k) {
                v[i] = true;
            }
            v
        };

        // Usable regime: τ small enough to be sharp, large enough to be f32-stable.
        let sel = dp_topk(&scores, k, 2e-2);
        let sum: f32 = sel.iter().sum();
        check(
            (sum - k as f32).abs() < 0.1,
            format!("n={n} k={k}: selection sum {sum:.4} != k"),
        );
        for i in 0..n {
            if in_topk[i] {
                check(
                    sel[i] > 0.85,
                    format!("n={n} k={k}: top-k item {i} sel {:.4} <= 0.85", sel[i]),
                );
            } else {
                check(
                    sel[i] < 0.15,
                    format!("n={n} k={k}: non-top item {i} sel {:.4} >= 0.15", sel[i]),
                );
            }
        }
    }

    // Probe the numerical floor (informational, not asserted): the sum should
    // hold near k in the usable regime and drift below k as f32 cancellation bites.
    {
        let probe: Vec<f32> = (0..16)
            .map(|i| (i as f32 * 0.61).sin() * 5.0 + i as f32)
            .collect();
        println!("numerical-floor probe (k=10, sum should stay near 10 then degrade):");
        for t in [1e-1f32, 1e-2, 1e-3, 1e-4, 1e-5] {
            let s: f32 = dp_topk(&probe, 10, t).iter().sum();
            println!("  τ={t:<7} sum={s:.4}");
        }
    }

    println!("{checks} checks, {failures} violations");
    if failures == 0 {
        println!("PASS: dp_topk collapses to the hard top-k indicator as temperature->0");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL: dp_topk diverged from the hard top-k limit");
        ExitCode::FAILURE
    }
}
