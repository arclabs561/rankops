# rankops-wasm

WASM bindings for **rankops** (fusion + reranking). Keeps the main rankops crate dependency-light.

Part of the **rankops** crate (this directory: `rankops/crates/rankops-wasm`).

## TREC evaluation

`evaluate_trec(qrels, run, k)` parses TREC-format strings and returns an object
with `num_queries`, `k`, `ndcg_at_k`, `map`, `mrr`, `recall_at_k`,
`precision_at_k`, and `judged_at_k`. `judged_at_k` is the share of top-k
results with any qrels judgment and does not change the metric denominators.
It throws for malformed input, duplicate documents, or non-finite run scores.

`evaluate_trec_detailed(qrels, run, k)` adds deterministic per-query metrics
and retrieved/relevant/judged-document counts.

## License

MIT OR Apache-2.0
