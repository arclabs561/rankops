//! Rank fusion: combine results from multiple retrievers.
//!
//! Run: `cargo run --example fusion`

use rankops::{
    borda, combmnz, copeland, median_rank, rrf, rrf_with_config, FusionMethod, RrfConfig,
};

fn main() {
    // Two ranked lists from different retrievers.
    // Scores are on incompatible scales (BM25 vs cosine similarity).
    let bm25: Vec<(&str, f32)> = vec![
        ("doc_a", 12.5),
        ("doc_b", 11.0),
        ("doc_c", 9.2),
        ("doc_d", 7.1),
    ];
    let dense: Vec<(&str, f32)> = vec![
        ("doc_b", 0.95),
        ("doc_e", 0.88),
        ("doc_a", 0.82),
        ("doc_f", 0.70),
    ];

    // ── RRF (Reciprocal Rank Fusion) ────────────────────────────────────
    // Score-agnostic: uses only rank positions, so incompatible scales are fine.
    let fused_rrf = rrf(&bm25, &dense);
    println!("RRF (default k=60):");
    for (id, score) in &fused_rrf {
        println!("  {id:8} {score:.6}");
    }
    // doc_b and doc_a appear in both lists, so they rank highest.

    // RRF with custom k: lower k gives more weight to top positions.
    let config = RrfConfig::new(20).with_top_k(3);
    let top3 = rrf_with_config(&bm25, &dense, config);
    println!("\nRRF (k=20, top_k=3):");
    for (id, score) in &top3 {
        println!("  {id:8} {score:.6}");
    }

    // ── CombMNZ ─────────────────────────────────────────────────────────
    // Score-based: sums normalized scores, then multiplies by overlap count.
    // Rewards documents that appear in multiple lists.
    let fused_mnz = combmnz(&bm25, &dense);
    println!("\nCombMNZ:");
    for (id, score) in &fused_mnz {
        println!("  {id:8} {score:.6}");
    }

    // ── Borda count ─────────────────────────────────────────────────────
    // Score-agnostic: each document gets (N - rank) points per list.
    let fused_borda = borda(&bm25, &dense);
    println!("\nBorda:");
    for (id, score) in &fused_borda {
        println!("  {id:8} {score:.6}");
    }

    // ── Copeland (net pairwise wins) ────────────────────────────────────
    // More discriminative than Condorcet: score = wins - losses.
    let fused_copeland = copeland(&bm25, &dense);
    println!("\nCopeland:");
    for (id, score) in &fused_copeland {
        println!("  {id:8} {score:.1}");
    }

    // ── Median Rank (outlier-robust) ──────────────────────────────────
    // Takes the median rank across lists. Robust to a single bad retriever.
    let fused_median = median_rank(&bm25, &dense);
    println!("\nMedian Rank:");
    for (id, score) in &fused_median {
        println!("  {id:8} {score:.6}");
    }

    // ── FusionMethod enum (runtime dispatch) ────────────────────────────
    // Useful when the fusion algorithm is selected at runtime (e.g. from config).
    let methods = [
        ("RRF", FusionMethod::rrf()),
        ("CombMNZ", FusionMethod::CombMnz),
        ("Borda", FusionMethod::Borda),
        ("DBSF", FusionMethod::Dbsf),
        ("Copeland", FusionMethod::Copeland),
        ("MedRank", FusionMethod::MedianRank),
    ];
    println!("\nFusionMethod dispatch — top result per method:");
    for (name, method) in &methods {
        let result = method.fuse(&bm25, &dense);
        if let Some((id, score)) = result.first() {
            println!("  {name:8} -> {id} ({score:.6})");
        }
    }
}
