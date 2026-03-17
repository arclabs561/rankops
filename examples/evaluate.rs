//! Evaluation example: compare fusion methods on realistic retrieval data.
//!
//! Simulates a hybrid search scenario with BM25, dense, and sparse retrievers
//! on 20 documents with known relevance judgments. Demonstrates:
//!
//! - All fusion methods with score comparison
//! - All evaluation metrics (NDCG, MAP, MRR, Precision, Recall, Hit Rate)
//! - Fusion diagnostics (complementarity, overlap, score distributions)
//! - Normalization variants (MinMax, ZScore, Quantile, Sigmoid)
//! - Hyperparameter optimization (RRF k, weighted alpha)
//!
//! Run: `cargo run --example evaluate`

use rankops::diagnostics::{diagnose, diagnose_multi, score_stats};
use rankops::{
    borda, combmnz, combsum, condorcet, copeland, dbsf, hit_rate, map, map_at_k, median_rank, mrr,
    ndcg_at_k, normalize_scores, optimize_fusion, precision_at_k, recall_at_k, rrf, weighted,
    FusionMethod, Normalization, OptimizeConfig, OptimizeMetric, ParamGrid, Qrels, WeightedConfig,
};
use std::collections::HashMap;

fn main() {
    // ── Simulated retrieval data ──────────────────────────────────────────
    // 20 documents, 3 retrievers with different characteristics:
    // - BM25: good at exact keyword matches, scores 0-25
    // - Dense: good at semantic similarity, scores 0.0-1.0
    // - Sparse (SPLADE): good at expanded terms, scores 0-15

    // Relevance judgments (ground truth): 0=not relevant, 1=relevant, 2=highly relevant
    let qrels: Qrels<&str> = HashMap::from([
        ("d01", 2), // highly relevant
        ("d02", 2), // highly relevant
        ("d03", 1), // relevant
        ("d04", 1), // relevant
        ("d05", 1), // relevant
        ("d06", 1), // relevant
        ("d07", 0), // not relevant
        ("d08", 0), // not relevant
        ("d09", 0), // not relevant
        ("d10", 0), // not relevant
    ]);

    // BM25: finds d01, d03 well; misses d02 (synonym match); ranks some irrelevant high
    let bm25: Vec<(&str, f32)> = vec![
        ("d01", 22.3), // highly relevant, exact keyword match
        ("d07", 18.7), // not relevant, keyword overlap
        ("d03", 17.1), // relevant
        ("d08", 15.2), // not relevant
        ("d04", 13.8), // relevant
        ("d11", 12.1), // unjudged
        ("d05", 10.5), // relevant
        ("d09", 8.3),  // not relevant
        ("d12", 6.7),  // unjudged
        ("d13", 4.2),  // unjudged
    ];

    // Dense: finds d02 via semantic similarity; different ordering
    let dense: Vec<(&str, f32)> = vec![
        ("d02", 0.94), // highly relevant, semantic match
        ("d01", 0.91), // highly relevant
        ("d06", 0.87), // relevant (BM25 missed this)
        ("d04", 0.82), // relevant
        ("d10", 0.78), // not relevant
        ("d03", 0.75), // relevant
        ("d14", 0.71), // unjudged
        ("d05", 0.68), // relevant
        ("d15", 0.62), // unjudged
        ("d16", 0.55), // unjudged
    ];

    // Sparse (SPLADE): expanded terms, partially overlapping results
    let sparse: Vec<(&str, f32)> = vec![
        ("d01", 14.2), // highly relevant
        ("d02", 12.8), // highly relevant
        ("d05", 11.3), // relevant
        ("d03", 10.7), // relevant
        ("d17", 9.1),  // unjudged
        ("d06", 8.4),  // relevant
        ("d07", 7.2),  // not relevant
        ("d04", 6.5),  // relevant
        ("d18", 5.1),  // unjudged
        ("d08", 3.8),  // not relevant
    ];

    // ── Diagnostics ───────────────────────────────────────────────────────
    println!("=== Fusion Diagnostics ===\n");

    println!("Score distributions:");
    for (name, list) in [("BM25", &bm25), ("Dense", &dense), ("Sparse", &sparse)] {
        if let Some(stats) = score_stats(list) {
            println!(
                "  {name:8} range=[{:.2}, {:.2}]  mean={:.2}  std={:.2}  median={:.2}",
                stats.min, stats.max, stats.mean, stats.std_dev, stats.median
            );
        }
    }

    println!("\nPairwise diagnostics:");
    for (name, a, b) in [
        ("BM25 vs Dense", bm25.as_slice(), dense.as_slice()),
        ("BM25 vs Sparse", bm25.as_slice(), sparse.as_slice()),
        ("Dense vs Sparse", dense.as_slice(), sparse.as_slice()),
    ] {
        let diag = diagnose(a, b, Some(&qrels), 5);
        println!("  {name}:");
        println!(
            "    overlap={:.2}  overlap@5={:.2}  tau={:.2}  complementarity={:.2}",
            diag.overlap,
            diag.overlap_at_k,
            diag.rank_correlation,
            diag.complementarity.unwrap_or(0.0),
        );
        println!("    suggestion: {:?}", diag.suggestion);
    }

    // Multi-retriever diagnostics (all 3 at once)
    let multi_diag = diagnose_multi(
        &[
            ("BM25", bm25.as_slice()),
            ("Dense", dense.as_slice()),
            ("Sparse", sparse.as_slice()),
        ],
        Some(&qrels),
    );
    println!("\nMulti-retriever diagnostics:");
    println!("  full overlap (all 3): {:.2}", multi_diag.full_overlap);
    println!("  suggestion: {:?}", multi_diag.suggestion);

    // ── Fusion method comparison ──────────────────────────────────────────
    println!("\n=== Fusion Method Comparison (BM25 + Dense) ===\n");

    let methods: Vec<(&str, Vec<(&str, f32)>)> = vec![
        ("RRF (k=60)", rrf(&bm25, &dense)),
        (
            "RRF (k=20)",
            FusionMethod::Rrf { k: 20 }.fuse(&bm25, &dense),
        ),
        ("CombSUM", combsum(&bm25, &dense)),
        ("CombMNZ", combmnz(&bm25, &dense)),
        ("Borda", borda(&bm25, &dense)),
        ("Condorcet", condorcet(&bm25, &dense)),
        ("Copeland", copeland(&bm25, &dense)),
        ("MedianRank", median_rank(&bm25, &dense)),
        ("DBSF", dbsf(&bm25, &dense)),
        (
            "Weighted(.3)",
            weighted(
                &bm25,
                &dense,
                WeightedConfig::new(0.3, 0.7).with_normalize(true),
            ),
        ),
    ];

    // Header
    println!(
        "{:14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Method", "NDCG@5", "NDCG@10", "MAP", "MAP@10", "MRR", "P@5", "Hit@1"
    );
    println!("{}", "-".repeat(86));

    for (name, fused) in &methods {
        println!(
            "{:14} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name,
            ndcg_at_k(fused, &qrels, 5),
            ndcg_at_k(fused, &qrels, 10),
            map(fused, &qrels),
            map_at_k(fused, &qrels, 10),
            mrr(fused, &qrels),
            precision_at_k(fused, &qrels, 5),
            hit_rate(fused, &qrels, 1),
        );
    }

    // ── Three-way fusion ──────────────────────────────────────────────────
    println!("\n=== Three-Way Fusion (BM25 + Dense + Sparse) ===\n");

    let lists_3: Vec<&[(&str, f32)]> = vec![&bm25, &dense, &sparse];

    let methods_3: Vec<(&str, Vec<(&str, f32)>)> = vec![
        ("RRF", FusionMethod::rrf().fuse_multi(&lists_3)),
        ("CombSUM", FusionMethod::CombSum.fuse_multi(&lists_3)),
        ("CombMNZ", FusionMethod::CombMnz.fuse_multi(&lists_3)),
        ("Borda", FusionMethod::Borda.fuse_multi(&lists_3)),
        ("Copeland", FusionMethod::Copeland.fuse_multi(&lists_3)),
        ("MedianRank", FusionMethod::MedianRank.fuse_multi(&lists_3)),
        ("DBSF", FusionMethod::Dbsf.fuse_multi(&lists_3)),
    ];

    println!(
        "{:14} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Method", "NDCG@5", "MAP@10", "MRR", "P@5", "R@10"
    );
    println!("{}", "-".repeat(62));

    for (name, fused) in &methods_3 {
        println!(
            "{:14} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name,
            ndcg_at_k(fused, &qrels, 5),
            map_at_k(fused, &qrels, 10),
            mrr(fused, &qrels),
            precision_at_k(fused, &qrels, 5),
            recall_at_k(fused, &qrels, 10),
        );
    }

    // ── Normalization comparison ──────────────────────────────────────────
    println!("\n=== Normalization Effect on CombSUM (BM25 + Dense) ===\n");

    let norms = [
        ("MinMax", Normalization::MinMax),
        ("ZScore", Normalization::ZScore),
        ("Quantile", Normalization::Quantile),
        ("Sigmoid", Normalization::Sigmoid),
        ("Rank", Normalization::Rank),
    ];

    println!("{:10} {:>8} {:>8}  top-3", "Norm", "NDCG@5", "MAP@10");
    println!("{}", "-".repeat(50));

    for (name, norm) in &norms {
        let norm_bm25 = normalize_scores(&bm25, *norm);
        let norm_dense = normalize_scores(&dense, *norm);

        // Manual combsum on pre-normalized scores (using Normalization::None)
        let fused = combsum(&norm_bm25, &norm_dense);

        let top3: Vec<_> = fused.iter().take(3).map(|(id, _)| *id).collect();
        println!(
            "{:10} {:>8.4} {:>8.4}  {:?}",
            name,
            ndcg_at_k(&fused, &qrels, 5),
            map_at_k(&fused, &qrels, 10),
            top3,
        );
    }

    // ── Hyperparameter optimization ───────────────────────────────────────
    println!("\n=== Hyperparameter Optimization ===\n");

    // Optimize RRF k parameter
    let runs = vec![bm25.clone(), dense.clone()];
    let rrf_config = OptimizeConfig {
        method: FusionMethod::rrf(),
        metric: OptimizeMetric::Ndcg { k: 10 },
        param_grid: ParamGrid::RrfK {
            values: vec![1, 5, 10, 20, 40, 60, 80, 100],
        },
    };
    let best_rrf = optimize_fusion(&qrels, &runs, rrf_config);
    println!(
        "Best RRF: {} (NDCG@10 = {:.4})",
        best_rrf.best_params, best_rrf.best_score
    );

    // Optimize weighted alpha (Bruch et al. recommend alpha ~ 0.2-0.3 for lexical weight)
    let alphas: Vec<Vec<f32>> = (0..=10)
        .map(|i| {
            let a = i as f32 * 0.1;
            vec![a, 1.0 - a]
        })
        .collect();

    let weighted_config = OptimizeConfig {
        method: FusionMethod::weighted(0.5, 0.5),
        metric: OptimizeMetric::Ndcg { k: 10 },
        param_grid: ParamGrid::Weighted {
            weight_combinations: alphas,
        },
    };
    let best_weighted = optimize_fusion(&qrels, &runs, weighted_config);
    println!(
        "Best Weighted: {} (NDCG@10 = {:.4})",
        best_weighted.best_params, best_weighted.best_score
    );

    // Optimize for MAP (MTEB Reranking metric)
    let map_config = OptimizeConfig {
        method: FusionMethod::rrf(),
        metric: OptimizeMetric::MapAtK { k: 10 },
        param_grid: ParamGrid::RrfK {
            values: vec![10, 20, 40, 60, 100],
        },
    };
    let best_map = optimize_fusion(&qrels, &runs, map_config);
    println!(
        "Best RRF for MAP@10: {} (MAP@10 = {:.4})",
        best_map.best_params, best_map.best_score
    );

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n=== Summary ===\n");
    println!("Individual retrievers (NDCG@10):");
    println!("  BM25:   {:.4}", ndcg_at_k(&bm25, &qrels, 10));
    println!("  Dense:  {:.4}", ndcg_at_k(&dense, &qrels, 10));
    println!("  Sparse: {:.4}", ndcg_at_k(&sparse, &qrels, 10));

    // Best 2-way fusion
    let best_2way = methods
        .iter()
        .max_by(|a, b| {
            ndcg_at_k(&a.1, &qrels, 10)
                .partial_cmp(&ndcg_at_k(&b.1, &qrels, 10))
                .unwrap()
        })
        .unwrap();
    println!(
        "\nBest 2-way fusion: {} (NDCG@10 = {:.4})",
        best_2way.0,
        ndcg_at_k(&best_2way.1, &qrels, 10)
    );

    // Best 3-way fusion
    let best_3way = methods_3
        .iter()
        .max_by(|a, b| {
            ndcg_at_k(&a.1, &qrels, 10)
                .partial_cmp(&ndcg_at_k(&b.1, &qrels, 10))
                .unwrap()
        })
        .unwrap();
    println!(
        "Best 3-way fusion: {} (NDCG@10 = {:.4})",
        best_3way.0,
        ndcg_at_k(&best_3way.1, &qrels, 10)
    );

    // ── Pipeline API ──────────────────────────────────────────────────────
    println!("\n=== Pipeline API ===\n");

    use rankops::pipeline::{compare, Pipeline};

    let pipeline_result = Pipeline::new()
        .add_run("bm25", &bm25)
        .add_run("dense", &dense)
        .add_run("sparse", &sparse)
        .normalize(Normalization::MinMax)
        .fuse(FusionMethod::Copeland)
        .top_k(10)
        .execute_and_evaluate(&qrels);

    println!("Pipeline (3-way, MinMax, Copeland, top-10):");
    println!("  {}", pipeline_result.metrics);

    // Compare all methods at once
    let b_ref = bm25.as_slice();
    let d_ref = dense.as_slice();
    let s_ref = sparse.as_slice();
    let all_runs: Vec<&[(&str, f32)]> = vec![b_ref, d_ref, s_ref];

    let configs = vec![
        ("RRF", FusionMethod::rrf()),
        ("CombSUM", FusionMethod::CombSum),
        ("CombMNZ", FusionMethod::CombMnz),
        ("CombMAX", FusionMethod::CombMax),
        ("Copeland", FusionMethod::Copeland),
        ("MedianRank", FusionMethod::MedianRank),
        ("DBSF", FusionMethod::Dbsf),
    ];

    let comparison = compare(&all_runs, &qrels, &configs, OptimizeMetric::Ndcg { k: 10 });
    println!("\nMethod comparison (sorted by NDCG@10):");
    for (name, m) in &comparison {
        println!(
            "  {:12} NDCG@10={:.4}  MAP={:.4}  MRR={:.4}",
            name, m.ndcg_10, m.map, m.mrr
        );
    }
}
