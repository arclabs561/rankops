//! Hybrid retrieval with real indexes: lexir BM25 and a vicinity HNSW index
//! each rank a small corpus, rankops fuses the two lists with RRF, and
//! rankops' metrics score all three against graded judgments.
//!
//! `hybrid_search.rs` shows the same stages on hand-written ranked lists.
//! Here the lists come from the sibling crates, so the example also shows the
//! glue: document ids are `u32` in both indexes, lexir returns scores (higher
//! is better) and vicinity returns distances (lower is better), and
//! `rankops::adapt::from_distances` converts the second into the first.
//!
//! The dense side needs no model download. Each text is embedded as a hashed
//! bag of character trigrams, which is enough to match spelling variants that
//! exact-token BM25 misses ("optimiser" and "optimizer", "tokenise" and
//! "tokenization"). It is a stand-in for a learned encoder, not a substitute.
//! The corpus is 24 sentences and 4 queries, so the numbers illustrate the
//! plumbing and say nothing about retrieval quality in general.
//!
//! Run: `cargo run --features composed-example --example composed_retrieval`

use lexir::bm25::{Bm25Params, InvertedIndex};
use rankops::adapt::from_distances;
use rankops::{ndcg_at_k, recall_at_k, rrf, Qrels};
use std::collections::HashMap;
use vicinity::hnsw::HNSWIndex;

const DIM: usize = 256;
const K: usize = 5;

const DOCS: &[&str] = &[
    /* 0 */ "Stochastic gradient descent updates weights from a mini-batch gradient.",
    /* 1 */ "The Adam optimiser adapts a per-parameter learning rate from gradient moments.",
    /* 2 */ "Momentum accumulates past gradients to damp oscillation during optimisation.",
    /* 3 */ "Learning rate warm-up avoids divergence in the first training steps.",
    /* 4 */ "Weight decay regularises a model by shrinking parameters toward zero.",
    /* 5 */ "Dropout randomly zeroes activations to reduce co-adaptation.",
    /* 6 */ "Batch normalisation rescales activations using mini-batch statistics.",
    /* 7 */
    "BM25 scores a document by term frequency, inverse document frequency and length.",
    /* 8 */ "An inverted index maps each term to the list of documents containing it.",
    /* 9 */ "Tokenisation splits text into terms before indexing.",
    /* 10 */ "Stemming reduces inflected words to a common root for matching.",
    /* 11 */ "Query expansion adds related terms to improve recall.",
    /* 12 */
    "HNSW builds a layered proximity graph for approximate nearest-neighbour search.",
    /* 13 */ "Product quantisation compresses vectors into short codes.",
    /* 14 */ "Cosine similarity compares the direction of two embedding vectors.",
    /* 15 */ "A vector database stores embeddings and answers nearest-neighbour queries.",
    /* 16 */ "Reciprocal rank fusion combines ranked lists using only ranks.",
    /* 17 */
    "A cross-encoder reranks candidates by reading the query and document together.",
    /* 18 */
    "Normalised discounted cumulative gain rewards relevant documents near the top.",
    /* 19 */ "Mean reciprocal rank averages the inverse rank of the first relevant result.",
    /* 20 */ "A write-ahead log makes index updates durable across crashes.",
    /* 21 */ "Segment merging compacts an index and drops deleted documents.",
    /* 22 */ "Bloom filters answer set-membership queries with no false negatives.",
    /* 23 */ "Consistent hashing spreads keys across shards with little movement on resize.",
];

/// (query, [(doc, grade)]). Grade 2 answers the query; grade 1 is related.
const QUERIES: &[(&str, &[(u32, u32)])] = &[
    (
        "adam optimizer learning rate",
        &[(1, 2), (3, 1), (2, 1), (0, 1)],
    ),
    (
        "tokenization and stemming for an inverted index",
        &[(9, 2), (10, 2), (8, 2), (7, 1)],
    ),
    (
        "approximate nearest neighbor search over vectors",
        &[(12, 2), (15, 2), (13, 1), (14, 1)],
    ),
    (
        "rank fusion and ndcg evaluation",
        &[(16, 2), (18, 2), (19, 1), (17, 1)],
    ),
];

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Hashed bag of character trigrams, L2-normalised by the index on insert.
fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for token in tokens(text) {
        let padded: Vec<char> = format!("  {token} ").chars().collect();
        for gram in padded.windows(3) {
            // FNV-1a over the trigram.
            let mut h: u64 = 0xcbf29ce484222325;
            for c in gram {
                h = (h ^ *c as u64).wrapping_mul(0x100000001b3);
            }
            v[(h % DIM as u64) as usize] += 1.0;
        }
    }
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Index the corpus twice: once lexically, once as vectors.
    let mut bm25 = InvertedIndex::new();
    let mut ann = HNSWIndex::builder(DIM)
        .ef_search(64)
        .auto_normalize(true)
        .build()?;
    for (id, doc) in DOCS.iter().enumerate() {
        bm25.add_document(id as u32, &tokens(doc));
        ann.add_slice(id as u32, &embed(doc))?;
    }
    ann.build()?;

    println!(
        "{:<50} {:>8} {:>8} {:>8}",
        "nDCG@5 per query", "bm25", "dense", "rrf"
    );
    let mut sums = [0.0f32; 3];
    let mut recalls = [0.0f32; 3];
    for (query, judged) in QUERIES {
        let qrels: Qrels<u32> = judged.iter().copied().collect::<HashMap<_, _>>();

        // lexir: (doc, score), higher is better. A query with no indexed term
        // returns an empty list, which is a valid input to fusion.
        let lexical = bm25.retrieve(&tokens(query), K * 2, Bm25Params::default())?;
        // vicinity: (doc, distance), lower is better. Convert to scores.
        let dense = from_distances(&ann.search(&embed(query), K * 2, 64)?);
        // RRF uses ranks only, so the two score scales need no calibration.
        let fused = rrf(&lexical, &dense);

        let lists = [&lexical, &dense, &fused];
        let mut row = [0.0f32; 3];
        for (i, list) in lists.iter().enumerate() {
            row[i] = ndcg_at_k(list, &qrels, K);
            sums[i] += row[i];
            recalls[i] += recall_at_k(list, &qrels, K);
        }
        println!(
            "{:<50} {:>8.3} {:>8.3} {:>8.3}",
            query, row[0], row[1], row[2]
        );
        assert!(!fused.is_empty(), "fusion returned nothing for {query:?}");
    }
    let n = QUERIES.len() as f32;
    println!(
        "{:<50} {:>8.3} {:>8.3} {:>8.3}",
        "mean nDCG@5",
        sums[0] / n,
        sums[1] / n,
        sums[2] / n
    );
    println!(
        "{:<50} {:>8.3} {:>8.3} {:>8.3}",
        "mean recall@5",
        recalls[0] / n,
        recalls[1] / n,
        recalls[2] / n
    );
    Ok(())
}
