//! Pareto-optimal reranking: select results that are non-dominated across
//! multiple objectives (relevance, diversity, recency).
//!
//! Shows how to combine `rankops` (fusion) with `pare` (Pareto frontier)
//! for multi-objective result selection.
//!
//! Run: `cargo run --example pareto_rerank --features pareto`
//!
//! Requires the `pareto` feature: `rankops = { features = ["pareto"] }`

fn main() {
    #[cfg(feature = "pareto")]
    pareto_demo();

    #[cfg(not(feature = "pareto"))]
    eprintln!("This example requires the `pareto` feature: cargo run --example pareto_rerank --features pareto");
}

#[cfg(feature = "pareto")]
fn pareto_demo() {
    use pare::{Direction, ParetoFrontier};
    use rankops::{ndcg_at_k, rrf, Qrels};
    use std::collections::HashMap;

    println!("=== Pareto-Optimal Reranking ===\n");

    // Two retrievers with different strengths
    let bm25: Vec<(&str, f32)> = vec![
        ("paper_a", 22.0), // highly relevant, old
        ("paper_b", 18.0), // relevant
        ("paper_c", 15.0), // relevant, very recent
        ("paper_d", 12.0), // marginally relevant, recent
        ("paper_e", 8.0),  // not relevant
    ];

    let dense: Vec<(&str, f32)> = vec![
        ("paper_c", 0.95), // relevant, very recent
        ("paper_a", 0.90), // highly relevant, old
        ("paper_f", 0.85), // relevant, recent
        ("paper_b", 0.80), // relevant
        ("paper_g", 0.70), // not relevant
    ];

    // Fuse with RRF first
    let fused = rrf(&bm25, &dense);

    // Now add a second objective: recency (higher = more recent)
    let recency: HashMap<&str, f64> = HashMap::from([
        ("paper_a", 0.2),  // old (2019)
        ("paper_b", 0.5),  // medium (2022)
        ("paper_c", 0.95), // very recent (2025)
        ("paper_d", 0.8),  // recent (2024)
        ("paper_e", 0.3),  // old
        ("paper_f", 0.9),  // recent (2025)
        ("paper_g", 0.6),  // medium
    ]);

    // Build Pareto frontier: maximize both relevance and recency
    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize, Direction::Maximize]);

    for (id, relevance_score) in &fused {
        let rec = recency.get(id).copied().unwrap_or(0.0);
        frontier.push(vec![*relevance_score as f64, rec], *id);
    }

    println!("Fused results ({} docs):", fused.len());
    for (id, score) in &fused {
        let rec = recency.get(id).copied().unwrap_or(0.0);
        let on_frontier = frontier.points().iter().any(|p| p.data == *id);
        let marker = if on_frontier { " [Pareto]" } else { "" };
        println!(
            "  {:10} relevance={:.4}  recency={:.2}{}",
            id, score, rec, marker
        );
    }

    println!("\nPareto frontier ({} non-dominated):", frontier.len());
    for point in frontier.points() {
        println!(
            "  {:10} relevance={:.4}  recency={:.2}",
            point.data, point.values[0], point.values[1]
        );
    }

    // Use knee point to find the best tradeoff
    if let Some(knee_idx) = frontier.knee_index() {
        let knee = &frontier.points()[knee_idx];
        println!(
            "\nKnee point (best tradeoff): {} (relevance={:.4}, recency={:.2})",
            knee.data, knee.values[0], knee.values[1]
        );
    }

    // Compare with relevance-only ranking
    let qrels: Qrels<&str> = HashMap::from([
        ("paper_a", 2),
        ("paper_b", 1),
        ("paper_c", 1),
        ("paper_f", 1),
    ]);
    println!(
        "\nRelevance-only NDCG@5: {:.4}",
        ndcg_at_k(&fused, &qrels, 5)
    );

    // Pareto-selected results (frontier points, sorted by relevance)
    let mut pareto_results: Vec<(&str, f32)> = frontier
        .points()
        .iter()
        .map(|p| (p.data, p.values[0] as f32))
        .collect();
    pareto_results.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!(
        "Pareto-selected NDCG@5: {:.4}",
        ndcg_at_k(&pareto_results, &qrels, 5)
    );
}
