# rankops examples

These examples cover the main post-retrieval decisions: fuse, diagnose, rerank,
and evaluate.

## Start here

| I want to... | Run | What to look for |
| --- | --- | --- |
| Compare fusion methods on incompatible score scales | `cargo run --release --example fusion` | RRF and rank-based methods agree on overlap-heavy results without trusting raw scores. |
| See a complete hybrid-search loop | `cargo run --release --example hybrid_search` | BM25, dense, and sparse results are fused, diagnosed, and scored against qrels. |
| Check collection-level IR metrics | `cargo run --release --example trec_eval` | qrels/run parsing plus collection-level means, matching the `trec_eval`, BEIR, ranx, and pytrec_eval workflow. |
| Compare metric and diagnostic APIs | `cargo run --release --example evaluate` | Fusion diagnostics, metric tables, normalization variants, and parameter search. |
| Diversify near-duplicate results | `cargo run --release --example diversity` | MMR and DPP trade some relevance for topic coverage. |
| Score late-interaction candidates | `cargo run --release --example rerank_maxsim` | MaxSim ranks token-level document matches and prints token alignments. |
| Approximate late-interaction candidates | `cargo run --release --example rerank_fde` | A fixed-dimensional proxy shortlists candidates before exact MaxSim. |
| Select a multi-objective frontier | `cargo run --release --example pareto_rerank --features pareto` | Pareto selection keeps non-dominated relevance/recency tradeoffs. |

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
