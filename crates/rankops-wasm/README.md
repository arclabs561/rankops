# rankops-wasm

WASM bindings for **rankops** (fusion + reranking). Keeps the main rankops crate dependency-light.

Part of the **rankops** crate (this directory: `rankops/crates/rankops-wasm`).

## TREC evaluation

`evaluate_trec(qrels, run, k)` parses TREC-format strings and returns an object
with `num_queries`, `k`, `ndcg_at_k`, `map`, `mrr`, `recall_at_k`, and
`precision_at_k`. It throws for malformed input, duplicate documents, or
non-finite run scores.

`evaluate_trec_detailed(qrels, run, k)` adds deterministic per-query metrics,
retrieved/relevant/judged-document counts, and `judged_at_k`: the mean share of
top-k results with any qrels judgment. It is diagnostic only and does not
change metric denominators.

## License

MIT OR Apache-2.0
