//! MaxSim (ColBERT late interaction) reranking with sample token embeddings.
//!
//! Run: `cargo run --example rerank_maxsim --features rerank`

use rankops::rerank::colbert;

fn main() {
    // Query: 3 tokens, each a 4-dimensional embedding.
    // In practice these come from a ColBERT/ColPali encoder.
    let query: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0], // token "what"
        vec![0.0, 1.0, 0.0, 0.0], // token "is"
        vec![0.0, 0.0, 1.0, 0.0], // token "rust"
    ];

    // Three candidate documents, each with per-token embeddings.
    let docs: Vec<(&str, Vec<Vec<f32>>)> = vec![
        (
            "doc_rust_intro",
            vec![
                vec![0.9, 0.0, 0.0, 0.0],  // "what"  — strong match
                vec![0.0, 0.8, 0.0, 0.0],  // "is"    — strong match
                vec![0.0, 0.0, 0.95, 0.0], // "rust"  — strong match
                vec![0.0, 0.0, 0.1, 0.9],  // "lang"  — no query match
            ],
        ),
        (
            "doc_python",
            vec![
                vec![0.7, 0.1, 0.0, 0.0], // partial "what"
                vec![0.0, 0.6, 0.0, 0.0], // partial "is"
                vec![0.0, 0.0, 0.1, 0.9], // "python" — no match on "rust"
            ],
        ),
        (
            "doc_rust_borrow",
            vec![
                vec![0.3, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.85, 0.0], // "rust"  — strong
                vec![0.0, 0.0, 0.7, 0.3],  // "borrow" — partial "rust"
            ],
        ),
    ];

    // ── MaxSim ranking ──────────────────────────────────────────────────
    // Score(Q, D) = sum over query tokens of max dot product with any doc token.
    let ranked = colbert::rank(&query, &docs);
    println!("MaxSim ranking:");
    for (id, score) in &ranked {
        println!("  {id:20} score={score:.4}");
    }
    // doc_rust_intro wins: all three query tokens have strong doc matches.

    // ── Token-level alignment ───────────────────────────────────────────
    // Shows which doc token best matches each query token.
    let best_doc = &docs[0]; // doc_rust_intro
    let alignments = colbert::alignments(&query, &best_doc.1);
    println!("\nAlignments (query_tok -> doc_tok, similarity):");
    for (qi, di, sim) in &alignments {
        println!("  query[{qi}] -> doc[{di}]  sim={sim:.4}");
    }

    // ── Highlight ───────────────────────────────────────────────────────
    // Which document tokens exceed a similarity threshold against any query token.
    let highlighted = colbert::highlight(&query, &best_doc.1, 0.7);
    println!("\nHighlighted doc tokens (threshold=0.7): {highlighted:?}");
}
