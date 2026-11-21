# rankops examples

These examples cover the main post-retrieval decisions: fuse, diagnose, rerank,
and evaluate.

## Start here

| I want to... | Run | Output |
| --- | --- | --- |
| Compare fusion methods on incompatible score scales | `cargo run --release --example fusion` | fused rankings and method scores |
| Explain one document in two saved TREC runs | `cargo run --example trec_rrf_document -- first.run second.run QUERY_ID DOC_ID` | per-run rank and score contribution, fused position or tie range |
| Run a hybrid-search loop | `cargo run --release --example hybrid_search` | source rankings, fusion diagnostics, and qrels metrics |
| Fuse lists from real indexes | `cargo run --features composed-example --example composed_retrieval` | lexir BM25 and vicinity HNSW over one corpus, fused with RRF and scored |
| Evaluate hybrid retrieval on judged collections | [SciFact](scifact/README.md) · [NFCorpus](scifact/NFCorpus.md) | TREC runs, paired changes per query, candidate coverage and a NIST cross-check |
| Check collection-level IR metrics | `cargo run --release --example trec_eval` | parsed qrels/run files and collection-level means |
| Compare metric and diagnostic APIs | `cargo run --release --example evaluate` | metric tables, diagnostics, normalization variants, and parameter search |
| Diversify near-duplicate results | `cargo run --release --example diversity` | MMR and DPP selections |
| Score late-interaction candidates | `cargo run --release --example rerank_maxsim` | MaxSim scores and token alignments |
| Approximate late-interaction candidates | `cargo run --release --example rerank_fde` | fixed-dimensional proxy scores and exact MaxSim rerank |
| Select a multi-objective frontier | `cargo run --release --example pareto_rerank --features pareto` | non-dominated relevance/recency candidates |

## Fusion and evaluation

- `fusion.rs` is the shortest entry point. It compares RRF, CombMNZ, Borda, Copeland, median-rank, and runtime `FusionMethod` dispatch on two result lists with incompatible score scales.
- `composed_retrieval.rs` builds a lexir BM25 index and a vicinity HNSW index over 24 sentences, fuses their results and scores all three. It needs the `composed-example` feature, which adds lexir and vicinity; nothing else in the crate depends on them.
- `hybrid_search.rs` is the main workflow example. It starts from BM25 scores, dense distances, and sparse scores, then converts distances, diagnoses complementarity, fuses runs, and evaluates the result.
- `evaluate.rs` expands the same idea into metric comparison, diagnostics, normalization variants, and fusion-parameter search.
- `trec_eval.rs` parses qrels and run files, then reports collection-level means.

### Why did the shelf-life paper leave the first page?

After generating the two [NFCorpus input runs](scifact/NFCorpus.md), ask about the abstract that BM25 ranked first:

```sh
cargo run --example trec_rrf_document -- target/nf-runs/bm25.run target/nf-runs/trigram.run PLAIN-2081 MED-4303
```

The example reports BM25 rank 1, contribution 0.0163934. The trigram run contributed zero. Fourteen documents received enough credit from both retrievers to pass it, leaving MED-4303 at fused rank 15. Replace the two paths and IDs to inspect another search. The files are the original retrievers’ rankings. The final fused run alone cannot recover their contributions.

## Reranking

- `diversity.rs` runs MMR and DPP on a result set with near-duplicate Python async documents.
- `rerank_maxsim.rs` runs ColBERT-style MaxSim scoring and prints token alignments on inline token embeddings.
- `rerank_fde.rs` runs fixed-dimensional proxy scoring for late-interaction candidate generation.
- `pareto_rerank.rs` needs the `pareto` feature and adds recency as a second objective after fusion.

## Checks

The small examples run on inline data. The optional BEIR examples download the
selected checksummed collection on first use. When changing an example, run at least:

```sh
cargo build --examples --all-features
cargo run --release --example hybrid_search
cargo run --release --example trec_eval
```

Checks against a known answer live in `tests/`, not here, so `cargo test` runs
them. For example, `tests/dp_topk_validation.rs` compares `dp_topk` with the
exact hard top-k indicator.
