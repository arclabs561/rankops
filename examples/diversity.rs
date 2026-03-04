//! Diversity reranking with MMR and DPP.
//!
//! Run: `cargo run --example diversity --features rerank`

use rankops::rerank::diversity::{dpp, mmr, mmr_cosine, DppConfig, MmrConfig};

fn main() {
    // Five search results about "async programming".
    // Several are near-duplicates (Python async tutorials).
    let candidates: Vec<(&str, f32)> = vec![
        ("python_async_await", 0.95),
        ("python_asyncio_guide", 0.92),
        ("rust_async_await", 0.88),
        ("javascript_promises", 0.85),
        ("python_coroutines", 0.90),
    ];

    // Pairwise similarity matrix (flattened, row-major).
    // High values between the three Python docs; low across languages.
    #[rustfmt::skip]
    let similarity: Vec<f32> = vec![
        // py_async  py_asyncio  rust_async  js_promise  py_corou
        1.0,        0.92,       0.25,       0.20,       0.90,   // python_async_await
        0.92,       1.0,        0.22,       0.18,       0.88,   // python_asyncio_guide
        0.25,       0.22,       1.0,        0.30,       0.20,   // rust_async_await
        0.20,       0.18,       0.30,       1.0,        0.15,   // javascript_promises
        0.90,       0.88,       0.20,       0.15,       1.0,    // python_coroutines
    ];

    // ── MMR with precomputed similarity matrix ──────────────────────────
    // lambda=1.0: pure relevance (no diversity).
    let pure_relevance = mmr(&candidates, &similarity, MmrConfig::new(1.0, 3));
    println!("MMR lambda=1.0 (pure relevance):");
    for (id, score) in &pure_relevance {
        println!("  {id:25} {score:.4}");
    }
    // All three Python docs — redundant.

    // lambda=0.5: balanced relevance + diversity.
    let balanced = mmr(&candidates, &similarity, MmrConfig::new(0.5, 3));
    println!("\nMMR lambda=0.5 (balanced):");
    for (id, score) in &balanced {
        println!("  {id:25} {score:.4}");
    }
    // Picks one Python doc, then Rust and JS for diversity.

    // ── MMR with raw embeddings ─────────────────────────────────────────
    // When you have embeddings instead of a precomputed matrix.
    let embeddings: Vec<Vec<f32>> = vec![
        vec![0.9, 0.1, 0.0, 0.0],   // python_async_await
        vec![0.88, 0.12, 0.0, 0.0], // python_asyncio_guide (similar)
        vec![0.1, 0.0, 0.9, 0.1],   // rust_async_await
        vec![0.0, 0.1, 0.1, 0.9],   // javascript_promises
        vec![0.85, 0.15, 0.0, 0.0], // python_coroutines (similar)
    ];

    let config = MmrConfig::default().with_lambda(0.5).with_k(3);
    let diverse = mmr_cosine(&candidates, &embeddings, config);
    println!("\nMMR-cosine lambda=0.5 (from embeddings):");
    for (id, score) in &diverse {
        println!("  {id:25} {score:.4}");
    }

    // ── DPP (Determinantal Point Process) ───────────────────────────────
    // Models joint diversity via determinants — prefers orthogonal items.
    let dpp_config = DppConfig::default().with_k(3).with_alpha(1.0);
    let dpp_result = dpp(&candidates, &embeddings, dpp_config);
    println!("\nDPP alpha=1.0:");
    for (id, score) in &dpp_result {
        println!("  {id:25} {score:.4}");
    }
}
