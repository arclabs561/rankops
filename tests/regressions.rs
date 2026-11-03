//! Regression tests for normalization, fusion, metrics, diversity, and fp16
//! quantization behavior. Each test states the reference it is checked against.

use rankops::{combsum, normalize_scores, Normalization};

/// `Normalization::MinMax` is documented as mapping to [0, 1]. A constant
/// list (here a single BM25 hit) must not pass its raw score through.
#[test]
fn minmax_constant_list_stays_in_unit_interval() {
    let single = vec![("x", 12.5)];
    let n = normalize_scores(&single, Normalization::MinMax);
    assert!(
        (0.0..=1.0).contains(&n[0].1),
        "min-max of a constant list = {} (documented range [0, 1])",
        n[0].1
    );
}

/// CombSUM min-max normalizes each list (ranx `fuse(method="sum",
/// norm="min-max")`), so a single-hit list cannot outrank the top of a
/// two-hit list with its raw score.
#[test]
fn combsum_single_hit_list_does_not_dominate() {
    let bm25 = vec![("x", 12.5)];
    let dense = vec![("y", 0.9), ("z", 0.1)];
    let fused = combsum(&bm25, &dense);
    let x = fused.iter().find(|(d, _)| *d == "x").unwrap().1;
    assert!(x <= 1.0, "combsum x = {x}; fused = {fused:?}");
    assert_eq!(fused[0].0, "y", "fused = {fused:?}");
}

/// `Normalization::Quantile` maps each score to its percentile rank, which is
/// a function of the score: identical scores get identical values.
#[test]
fn quantile_normalization_gives_tied_scores_equal_values() {
    let results = vec![("a", 0.5f32), ("b", 0.5), ("c", 1.0)];
    let out = normalize_scores(&results, Normalization::Quantile);
    let get = |id: &str| out.iter().find(|(d, _)| *d == id).unwrap().1;
    assert_eq!(
        get("a"),
        get("b"),
        "identical scores must get the same percentile rank: {out:?}"
    );
}

/// `rerank::diversity::mmr` is documented to apply the same greedy rule as
/// `crate::mmr`, which breaks ties toward the candidate that comes first. With
/// lambda = 1.0 (pure relevance) and tied relevance, input order must hold.
#[test]
fn diversity_mmr_pure_relevance_keeps_input_order_on_ties() {
    use rankops::rerank::diversity::{mmr, MmrConfig};
    let cands = vec![
        ("a", 1.0f32),
        ("b", 0.5),
        ("c", 0.5),
        ("d", 0.5),
        ("e", 0.5),
    ];
    let n = cands.len();
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        sim[i * n + i] = 1.0;
    }
    let out = mmr(&cands, &sim, MmrConfig::new(1.0, 5));
    let ids: Vec<&str> = out.iter().map(|(d, _)| *d).collect();
    let top = rankops::mmr(
        &cands,
        |a: &&str, b: &&str| if a == b { 1.0 } else { 0.0 },
        rankops::MmrConfig::new(1.0).with_top_k(5),
    );
    let top_ids: Vec<&str> = top.iter().map(|(d, _)| *d).collect();
    assert_eq!(top_ids, vec!["a", "b", "c", "d", "e"]);
    assert_eq!(
        ids, top_ids,
        "rerank::diversity::mmr tie order differs from crate::mmr"
    );
}

/// `mmr_cosine` computes the similarity on the fly but applies the same rule
/// as `mmr`, so tied candidates also keep input order.
#[test]
fn diversity_mmr_cosine_pure_relevance_keeps_input_order_on_ties() {
    use rankops::rerank::diversity::{mmr_cosine, MmrConfig};
    let cands = vec![
        ("a", 1.0f32),
        ("b", 0.5),
        ("c", 0.5),
        ("d", 0.5),
        ("e", 0.5),
    ];
    let embeddings: Vec<Vec<f32>> = (0..cands.len())
        .map(|i| {
            let mut v = vec![0.0f32; cands.len()];
            v[i] = 1.0;
            v
        })
        .collect();
    let out = mmr_cosine(&cands, &embeddings, MmrConfig::new(1.0, 5));
    let ids: Vec<&str> = out.iter().map(|(d, _)| *d).collect();
    assert_eq!(ids, vec!["a", "b", "c", "d", "e"]);
}

/// A NaN score for the true entity is not comparable to any candidate and
/// must not count as a perfect (rank 1) prediction.
#[test]
fn compute_rank_nan_target_is_not_rank_one() {
    use rankops::metrics::compute_rank;
    let r = compute_rank(f64::NAN, &[0.1, 0.5, 0.9, f64::NAN], true);
    assert!(r > 1, "NaN target got rank {r}");
    assert_eq!(r, 4, "NaN target ranks last among 4 candidates");
}

/// `quantize_fp16` returns IEEE 754 binary16 bit patterns (round to nearest,
/// ties to even; subnormals; overflow to infinity). Reference bits from
/// `np.float32(x).astype(np.float16).view(np.uint16)`.
#[test]
fn fp16_bits_match_ieee_binary16() {
    use rankops::rerank::quantization::quantize_fp16;
    let xs = [
        1e-5f32,
        3.051_757_8e-5,
        4.5e-5,
        70000.0,
        0.333_333_34,
        1.001_465_8,
        0.1,
        65504.0,
    ];
    let want: [u16; 8] = [
        0x00a8, 0x0200, 0x02f3, 0x7c00, 0x3555, 0x3c02, 0x2e66, 0x7bff,
    ];
    let got = quantize_fp16(&xs);
    let bad: Vec<String> = xs
        .iter()
        .zip(got.iter().zip(want.iter()))
        .filter(|(_, (g, w))| g != w)
        .map(|(x, (g, w))| format!("{x:e}: got {g:#06x} want {w:#06x}"))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Round trip: a finite value above the f16 maximum comes back as infinity,
/// not NaN, and f16 subnormals (2^-15, 1e-5) survive.
#[test]
fn fp16_round_trip_keeps_values() {
    use rankops::rerank::quantization::{dequantize_fp16, quantize_fp16};
    let xs = [70000.0f32, 3.051_757_8e-5, 1e-5];
    let back = dequantize_fp16(&quantize_fp16(&xs));
    assert!(!back[0].is_nan(), "70000 -> {}", back[0]);
    assert!(back[1] > 0.0, "2^-15 -> {}", back[1]);
    assert!((back[2] - 1e-5).abs() < 1e-7, "1e-5 -> {}", back[2]);
}
