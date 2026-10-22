//! End-to-end hybrid search: BM25 + dense vector retrieval -> fusion -> rerank -> evaluate.
//!
//! Covers:
//! - **Scoring**: BM25 via `rankfns` kernels
//! - **Fusion**: Combine BM25 and dense results via `rankops`
//! - **Reranking**: Diversity selection via MMR
//! - **Diagnostics**: Check if fusion is worthwhile
//! - **Evaluation**: NDCG, MAP, MRR against ground truth
//!
//! This example uses inline data. External retrievers can provide the same
//! ranked-list input shapes.
//!
//! Run: `cargo run --example hybrid_search`

use rankops::adapt::from_distances;
use rankops::diagnostics::{diagnose, diagnose_multi};
use rankops::pipeline::Pipeline;
use rankops::{
    map, mrr, ndcg_at_k, precision_at_k, recall_at_k, FusionMethod, Normalization, Qrels,
};
use std::collections::HashMap;

fn main() {
    println!("=== Hybrid Search Pipeline ===\n");

    // ── Step 1: Simulated retrieval results ───────────────────────────────
    // In production these come from lexir (BM25) and vicinity (ANN).
    // Here we inline realistic data: 30 docs, 3 retrievers, 1 query.

    // Ground truth: which docs are relevant for query "neural network optimization"
    let qrels: Qrels<&str> = HashMap::from([
        ("sgd_paper", 2),           // highly relevant
        ("adam_paper", 2),          // highly relevant
        ("backprop_paper", 1),      // relevant
        ("dropout_paper", 1),       // relevant
        ("batch_norm_paper", 1),    // relevant
        ("learning_rate_paper", 1), // relevant
        ("relu_paper", 1),          // relevant
    ]);

    // BM25 retriever: good at keyword matches, scores in [0, 25]
    let bm25: Vec<(&str, f32)> = vec![
        ("sgd_paper", 22.1),           // exact match: "optimization"
        ("gradient_paper", 19.3),      // keyword overlap but not relevant
        ("adam_paper", 17.8),          // relevant
        ("momentum_paper", 16.2),      // not relevant
        ("backprop_paper", 14.5),      // relevant
        ("convex_paper", 12.1),        // not relevant
        ("dropout_paper", 10.3),       // relevant
        ("loss_landscape_paper", 8.7), // not relevant
        ("batch_norm_paper", 7.2),     // relevant
        ("regularization_paper", 5.1), // not relevant
    ];

    // Dense retriever (vicinity/HNSW): distances (lower = closer)
    let dense_distances: Vec<(&str, f32)> = vec![
        ("adam_paper", 0.12),          // semantically close
        ("learning_rate_paper", 0.18), // relevant, BM25 missed
        ("sgd_paper", 0.22),           // relevant
        ("relu_paper", 0.31),          // relevant, BM25 missed
        ("batch_norm_paper", 0.35),    // relevant
        ("transformer_paper", 0.41),   // not relevant
        ("dropout_paper", 0.48),       // relevant
        ("attention_paper", 0.55),     // not relevant
        ("resnet_paper", 0.62),        // not relevant
        ("backprop_paper", 0.71),      // relevant (far but found)
    ];

    // Convert distances to scores using the adapt module
    let dense = from_distances(&dense_distances);

    // Sparse retriever (SPLADE-style): scores in [0, 15]
    let sparse: Vec<(&str, f32)> = vec![
        ("sgd_paper", 13.7),
        ("adam_paper", 12.1),
        ("backprop_paper", 11.5),
        ("learning_rate_paper", 10.8),
        ("batch_norm_paper", 9.2),
        ("dropout_paper", 8.6),
        ("relu_paper", 7.1),
        ("weight_init_paper", 6.3),    // not relevant
        ("gradient_paper", 5.7),       // not relevant
        ("loss_landscape_paper", 4.2), // not relevant
    ];

    // ── Step 2: Diagnostics ───────────────────────────────────────────────
    println!("--- Diagnostics ---\n");

    // Should we fuse? Check complementarity.
    let diag = diagnose(&bm25, &dense, Some(&qrels), 5);
    println!(
        "BM25 vs Dense: overlap={:.2}, tau={:.2}, complementarity={:.2}",
        diag.overlap,
        diag.rank_correlation,
        diag.complementarity.unwrap_or(0.0),
    );
    println!("  -> {:?}\n", diag.suggestion);

    let multi = diagnose_multi(
        &[
            ("BM25", bm25.as_slice()),
            ("Dense", dense.as_slice()),
            ("Sparse", sparse.as_slice()),
        ],
        Some(&qrels),
    );
    println!(
        "3-way: full_overlap={:.2}, suggestion={:?}\n",
        multi.full_overlap, multi.suggestion
    );

    // ── Step 3: Individual retriever baselines ────────────────────────────
    println!("--- Individual Retrievers ---\n");

    for (name, results) in [("BM25", &bm25), ("Dense", &dense), ("Sparse", &sparse)] {
        println!(
            "  {:8} NDCG@5={:.4}  MAP={:.4}  MRR={:.4}  P@5={:.4}  R@10={:.4}",
            name,
            ndcg_at_k(results, &qrels, 5),
            map(results, &qrels),
            mrr(results, &qrels),
            precision_at_k(results, &qrels, 5),
            recall_at_k(results, &qrels, 10),
        );
    }

    // ── Step 4: Two-way fusion (BM25 + Dense) ────────────────────────────
    println!("\n--- Two-Way Fusion (BM25 + Dense) ---\n");

    let methods_2way = [
        ("RRF", FusionMethod::rrf()),
        ("CombSUM", FusionMethod::CombSum),
        ("Copeland", FusionMethod::Copeland),
        ("Weighted(.3/.7)", FusionMethod::weighted(0.3, 0.7)),
        ("DBSF", FusionMethod::Dbsf),
    ];

    println!(
        "{:18} {:>8} {:>8} {:>8} {:>8}",
        "Method", "NDCG@5", "MAP", "MRR", "R@10"
    );
    println!("{}", "-".repeat(58));

    for (name, method) in &methods_2way {
        let fused = method.fuse(&bm25, &dense);
        println!(
            "{:18} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name,
            ndcg_at_k(&fused, &qrels, 5),
            map(&fused, &qrels),
            mrr(&fused, &qrels),
            recall_at_k(&fused, &qrels, 10),
        );
    }

    // ── Step 5: Three-way fusion via Pipeline ─────────────────────────────
    println!("\n--- Three-Way Pipeline (BM25 + Dense + Sparse) ---\n");

    let methods_3way = [
        ("RRF", FusionMethod::rrf(), Normalization::None),
        (
            "CombSUM+MinMax",
            FusionMethod::CombSum,
            Normalization::MinMax,
        ),
        (
            "CombSUM+Quantile",
            FusionMethod::CombSum,
            Normalization::Quantile,
        ),
        ("Copeland", FusionMethod::Copeland, Normalization::None),
        ("DBSF", FusionMethod::Dbsf, Normalization::None),
    ];

    println!(
        "{:18} {:>8} {:>8} {:>8} {:>8}",
        "Method", "NDCG@5", "MAP", "MRR", "R@10"
    );
    println!("{}", "-".repeat(58));

    for (name, method, norm) in &methods_3way {
        let result = Pipeline::new()
            .add_run("bm25", &bm25)
            .add_run("dense", &dense)
            .add_run("sparse", &sparse)
            .normalize(*norm)
            .fuse(*method)
            .top_k(10)
            .execute_and_evaluate(&qrels);

        println!(
            "{:18} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            name,
            result.metrics.ndcg_5,
            result.metrics.map,
            result.metrics.mrr,
            result.metrics.recall_10,
        );
    }

    // ── Step 6: Best pipeline with diversity reranking ─────────────────────
    println!("\n--- Best Pipeline + Diversity ---\n");

    // Fuse first, then apply MMR for diversity
    let fused = Pipeline::new()
        .add_run("bm25", &bm25)
        .add_run("dense", &dense)
        .add_run("sparse", &sparse)
        .normalize(Normalization::MinMax)
        .fuse(FusionMethod::CombSum)
        .execute();

    println!("After fusion (CombSUM+MinMax): {} results", fused.len());
    println!(
        "  NDCG@5={:.4}  MAP={:.4}  R@10={:.4}",
        ndcg_at_k(&fused, &qrels, 5),
        map(&fused, &qrels),
        recall_at_k(&fused, &qrels, 10),
    );

    // Show top-5 fused results
    println!("\n  Top-5 fused:");
    for (i, (id, score)) in fused.iter().take(5).enumerate() {
        let rel = qrels.get(id).copied().unwrap_or(0);
        let marker = match rel {
            2 => " **",
            1 => " *",
            _ => "",
        };
        println!("    {}: {} ({:.4}){}", i + 1, id, score, marker);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!("\n--- Summary ---\n");
    println!("Key findings:");
    println!("  - Dense retriever found docs BM25 missed (learning_rate_paper, relu_paper)");
    println!("  - Sparse already had R@10=1.000; three-way fusion matched it on this fixture");
    println!("  - Two-way RRF beat CombSUM on NDCG@5 and MAP here; both had R@10=1.000");
    println!("  - Three-way CombSUM+Quantile had MAP=1.000 vs MinMax=0.9821; NDCG@5 and R@10 tied");
    println!("  - The adapt module converted vicinity distances to scores automatically");

    println!();
    per_query_oracle_demo();
}

struct QueryCase {
    query: &'static str,
    qrels: Qrels<&'static str>,
    bm25: Vec<(&'static str, f32)>,
    dense: Vec<(&'static str, f32)>,
}

/// Compare one fixed retriever with an ideal per-query selector on explicit
/// synthetic cases. The oracle uses qrels and is not an online routing method.
fn per_query_oracle_demo() {
    let cases = [
        QueryCase {
            query: "q1",
            qrels: HashMap::from([("a", 1)]),
            bm25: vec![("a", 10.0), ("x", 9.0), ("y", 8.0)],
            dense: vec![("x", 0.9), ("y", 0.8), ("a", 0.7)],
        },
        QueryCase {
            query: "q2",
            qrels: HashMap::from([("b", 1)]),
            bm25: vec![("x", 10.0), ("y", 9.0), ("b", 8.0)],
            dense: vec![("b", 0.9), ("x", 0.8), ("y", 0.7)],
        },
    ];

    let mut bm25_total = 0.0;
    let mut dense_total = 0.0;
    let mut oracle_total = 0.0;
    let mut rrf_total = 0.0;

    println!("=== Synthetic Per-Query Oracle Comparison (NDCG@1) ===");
    println!(
        "{:8} {:>8} {:>8} {:>8} {:>8}",
        "Query", "BM25", "Dense", "Oracle", "RRF"
    );

    for case in &cases {
        let bm25_score = ndcg_at_k(&case.bm25, &case.qrels, 1);
        let dense_score = ndcg_at_k(&case.dense, &case.qrels, 1);
        let oracle_score = bm25_score.max(dense_score);
        let fused = FusionMethod::rrf().fuse(&case.bm25, &case.dense);
        let rrf_score = ndcg_at_k(&fused, &case.qrels, 1);

        bm25_total += bm25_score;
        dense_total += dense_score;
        oracle_total += oracle_score;
        rrf_total += rrf_score;

        println!(
            "{query:8} {bm25_score:>8.3} {dense_score:>8.3} {oracle_score:>8.3} {rrf_score:>8.3}",
            query = case.query,
        );
    }

    let query_count = cases.len() as f32;
    let best_single = bm25_total.max(dense_total) / query_count;
    println!(
        "{:8} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
        "Mean",
        bm25_total / query_count,
        dense_total / query_count,
        oracle_total / query_count,
        rrf_total / query_count,
    );
    println!(
        "Best single mean (selected once): {best_single:.3}. Oracle mean (selects using qrels): {:.3}.",
        oracle_total / query_count
    );
    println!("The oracle is label-informed and only bounds these fixtures; it does not establish an achievable routing payoff.");
}
