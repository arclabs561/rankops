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

/// Candidates without the embeddings a method needs are ranked after every
/// scored candidate, ordered by their original score; a BM25 12.5 must not
/// outrank an exact cosine match.
#[test]
fn rerank_batch_ranks_unscored_candidates_after_scored_ones() {
    use rankops::rerank::explain::{rerank_batch, Candidate, RerankMethod, RerankerInput};
    let q = vec![1.0f32, 0.0];
    let exact = vec![1.0f32, 0.0];
    let off = vec![0.0f32, 1.0];
    let candidates = vec![
        Candidate {
            id: "bm25_a",
            original_score: 12.5,
            dense_embedding: None,
            token_embeddings: None,
            text: None,
        },
        Candidate {
            id: "off",
            original_score: 1.0,
            dense_embedding: Some(&off),
            token_embeddings: None,
            text: None,
        },
        Candidate {
            id: "bm25_b",
            original_score: 20.0,
            dense_embedding: None,
            token_embeddings: None,
            text: None,
        },
        Candidate {
            id: "exact",
            original_score: 3.0,
            dense_embedding: Some(&exact),
            token_embeddings: None,
            text: None,
        },
    ];
    let input = RerankerInput {
        query_dense: Some(&q),
        query_tokens: None,
        candidates,
    };
    let ids: Vec<_> = rerank_batch(input, RerankMethod::DenseCosine, 10)
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(ids, ["exact", "off", "bm25_b", "bm25_a"]);
}

/// With a range that fits MaxSim (one similarity per query token, so [0, 2]
/// for a two-token query), fine-grained scores separate the candidates; the
/// default [-1, 1] range is for cosine-based methods.
#[test]
fn fine_grained_scores_distinguish_maxsim_candidates_with_a_fitting_range() {
    use rankops::rerank::explain::{
        rerank_fine_grained, Candidate, FineGrainedConfig, RerankMethod, RerankerInput,
    };
    let q = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let d1 = vec![vec![0.9f32, 0.1], vec![0.1, 0.9]];
    let d2 = vec![vec![0.5f32, 0.5]];
    let input = RerankerInput {
        query_dense: None,
        query_tokens: Some(&q),
        candidates: vec![
            Candidate {
                id: "doc1",
                original_score: 0.8,
                dense_embedding: None,
                token_embeddings: Some(&d1),
                text: None,
            },
            Candidate {
                id: "doc2",
                original_score: 0.7,
                dense_embedding: None,
                token_embeddings: Some(&d2),
                text: None,
            },
        ],
    };
    let mut config = FineGrainedConfig::default().without_weighting();
    config.min_score = 0.0;
    config.max_score = 2.0;
    let r = rerank_fine_grained(input, RerankMethod::MaxSim, config, 10);
    assert_eq!(r[0].id, "doc1");
    assert!(r[0].fine_score > r[1].fine_score, "{r:?}");
}

/// A single run goes through the fusion method, so it comes back sorted and
/// top_k keeps the best document, as with fuse_multi.
#[test]
fn pipeline_single_run_is_fused_like_multi_run() {
    use rankops::pipeline::Pipeline;
    use rankops::FusionMethod;
    let a = vec![("low", 0.1f32), ("high", 0.9)];
    let via_fuse_multi = FusionMethod::CombSum.fuse_multi(&[a.as_slice()]);
    let out = Pipeline::new()
        .add_run("a", &a)
        .fuse(FusionMethod::CombSum)
        .top_k(1)
        .execute();
    assert_eq!(out[0].0, "high");
    assert_eq!(out[0].0, via_fuse_multi[0].0);
}

/// Expected rank under random tie-breaking: a model that scores all 100
/// candidates the same gets rank 50.5, not the optimistic 1.
#[test]
fn expected_rank_splits_ties() {
    use rankops::metrics::{compute_expected_rank, compute_rank};
    assert_eq!(compute_rank(0.5, &[0.5; 100], true), 1);
    assert_eq!(compute_expected_rank(0.5, &[0.5; 100], true), 50.5);
    assert_eq!(compute_expected_rank(0.5, &[0.9, 0.5, 0.1], true), 2.0);
    assert_eq!(compute_expected_rank(0.1, &[0.9, 0.5, 0.1], false), 1.0);
    assert_eq!(compute_expected_rank(f64::NAN, &[0.9, 0.5], true), 2.0);
}
