# rankops examples

These examples cover the main post-retrieval decisions: fuse, diagnose, rerank,
and evaluate.

## Start here

| I want to... | Run | Output |
| --- | --- | --- |
| Compare fusion methods on incompatible score scales | `cargo run --release --example fusion` | fused rankings and method scores |
| Run a hybrid-search loop | `cargo run --release --example hybrid_search` | source rankings, fusion diagnostics, and qrels metrics |
| Check collection-level IR metrics | `cargo run --release --example trec_eval` | parsed qrels/run files and collection-level means |
| Compare metric and diagnostic APIs | `cargo run --release --example evaluate` | metric tables, diagnostics, normalization variants, and parameter search |
| Diversify near-duplicate results | `cargo run --release --example diversity` | MMR and DPP selections |
| Score late-interaction candidates | `cargo run --release --example rerank_maxsim` | MaxSim scores and token alignments |
| Approximate late-interaction candidates | `cargo run --release --example rerank_fde` | fixed-dimensional proxy scores and exact MaxSim rerank |
| Select a multi-objective frontier | `cargo run --release --example pareto_rerank --features pareto` | non-dominated relevance/recency candidates |

## Fusion and evaluation

- `fusion.rs` is the shortest entry point. It compares RRF, CombMNZ, Borda, Copeland, median-rank, and runtime `FusionMethod` dispatch on two result lists with incompatible score scales.
- `hybrid_search.rs` is the main workflow example. It starts from BM25 scores, dense distances, and sparse scores, then converts distances, diagnoses complementarity, fuses runs, and evaluates the result.
- `evaluate.rs` expands the same idea into metric comparison, diagnostics, normalization variants, and fusion-parameter search.
- `trec_eval.rs` parses qrels and run files, then reports collection-level means.

## Reranking

- `diversity.rs` runs MMR and DPP on a result set with near-duplicate Python async documents.
- `rerank_maxsim.rs` runs ColBERT-style MaxSim scoring and prints token alignments on inline token embeddings.
- `rerank_fde.rs` runs fixed-dimensional proxy scoring for late-interaction candidate generation.
- `pareto_rerank.rs` needs the `pareto` feature and adds recency as a second objective after fusion.

## Checks

The examples are self-contained and use inline data. They run without downloads
or external files. When changing an example, run at least:

```sh
cargo build --examples --all-features
cargo run --release --example hybrid_search
cargo run --release --example trec_eval
```
