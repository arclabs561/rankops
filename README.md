# rankops

[![crates.io](https://img.shields.io/crates/v/rankops.svg)](https://crates.io/crates/rankops)
[![Documentation](https://docs.rs/rankops/badge.svg)](https://docs.rs/rankops)
[![CI](https://github.com/arclabs561/rankops/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rankops/actions/workflows/ci.yml)

Rank fusion and reranking.

Includes ranked-list fusion, reranking, and IR evaluation metrics.

## Quickstart

```toml
[dependencies]
rankops = "0.2"
```

### Fusion

Fuse two ranked lists with Reciprocal Rank Fusion (score-agnostic, works across incompatible scales):

```rust
use rankops::rrf;

let bm25  = vec![("doc_a", 12.5), ("doc_b", 11.0), ("doc_c", 9.2)];
let dense = vec![("doc_b", 0.95), ("doc_c", 0.88), ("doc_d", 0.70)];

let fused = rrf(&bm25, &dense);
// doc_b ranks highest: appears in both lists
assert_eq!(fused[0].0, "doc_b");
```

Score-based fusion when scales are comparable:

```rust
use rankops::combmnz;

let fused = combmnz(&bm25, &dense);
// CombMNZ: sum of normalized scores * overlap count
```

Select the algorithm at runtime via `FusionMethod`:

```rust
use rankops::FusionMethod;

let method = FusionMethod::Rrf { k: 60 };
let result = method.fuse(&bm25, &dense);
```

### Diversity reranking (requires `rerank` feature, on by default)

MMR selects the next document by balancing relevance against redundancy:

$$\text{MMR} = \arg\max_{d_i \in R \setminus S}\left[\lambda \cdot \text{rel}(d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)\right]$$

```rust
use rankops::rerank::diversity::{mmr, MmrConfig};

let candidates = vec![("d1", 0.95), ("d2", 0.90), ("d3", 0.85)];
let similarity = vec![
    1.0, 0.9, 0.2,
    0.9, 1.0, 0.3,
    0.2, 0.3, 1.0,
];

let config = MmrConfig::default().with_lambda(0.5).with_k(2);
let selected = mmr(&candidates, &similarity, config);
// Picks d1 (highest relevance), then d3 (diverse from d1)
```

## Fusion algorithms

15 fusion methods: rank-based (`rrf`, `isr`, `borda`, `condorcet`, `copeland`, `median_rank`) and score-based (`combsum`, `combmnz`, `combmax`, `combmin`, `combmed`, `combanz`, `weighted`, `dbsf`, `standardized`). All two-list functions have `*_multi` variants for 3+ lists. Explainability variants (`rrf_explain`, `combsum_explain`, etc.) return full provenance. Score normalization via `Normalization` enum (MinMax, ZScore, Quantile, Sigmoid, Sum, Rank).

## Evaluation

Single-list IR metrics: `ndcg_at_k`, `map`, `mrr`, `precision_at_k`, `recall_at_k`, and `hit_rate`. These low-level functions accept the ordering supplied by the caller; validate external lists with `validate` before evaluating. Use the `trec` module for parsed TREC runs, multi-query aggregation, and explicit coverage semantics. `optimize_fusion` is a grid-search helper; choose parameters on held-out queries before reporting final results.

The `trec` module validates TREC qrels and run files and reports collection-level mean metrics over every qrels query (`trec_eval -c` semantics). It supports linear-gain nDCG@k, MAP, reciprocal rank, recall@k, and P@k. Use `evaluate_detailed` or `evaluate_detailed_with_config` to retain deterministic per-query values and coverage counts.

```rust
use rankops::trec::{evaluate_with_config, parse_qrels, parse_run, TrecEvalConfig};
use std::fs::File;

let qrels = parse_qrels(File::open("qrels.txt")?)?;
let run = parse_run(File::open("run.txt")?)?;
let config = TrecEvalConfig::new(10).with_relevance_level(1);
let summary = evaluate_with_config(&run, &qrels, config);
```

`evaluate(&run, &qrels, k)` remains the short form for the default configuration. Validated `TrecRun` and `TrecQrels` values can also be built from in-memory records. The checked-in parity corpus is part of `cargo test`; with `trec_eval` installed, run `scripts/verify-trec-parity.sh` to compare both the default judged-only and `-c` query-coverage modes.

## Diagnostics

The `diagnostics` module (`diagnose`, `overlap_ratio`, `complementarity`, `rank_correlation`) helps decide whether fusion is beneficial, based on Louis et al., "Know When to Fuse" (2024).

## Adapters

The `adapt` module converts retriever outputs (distances, similarities, logits, inner products) to rankops format with optional ID mapping.

## Pipeline

The `pipeline` module provides composable post-retrieval operations:

```rust
use rankops::pipeline::Pipeline;
use rankops::{FusionMethod, Normalization};

let result = Pipeline::new()
    .add_run("bm25", &bm25)
    .add_run("dense", &dense)
    .normalize(Normalization::MinMax)
    .fuse(FusionMethod::rrf())
    .top_k(10)
    .execute();
```

Also: `compare()` for method comparison, `fuse_multi_query()` for the N-queries x M-retrievers RAG pattern.

## Reranking (feature: `rerank`)

ColBERT MaxSim scoring, MMR/DPP diversity selection, Matryoshka two-stage reranking, and int8 quantization for token embeddings. On by default; also available: `hierarchical` (ColBERT clustering) and `serde`.

## Examples

See [examples/README.md](examples/README.md) for the example list.

```sh
cargo run --example fusion            # All fusion methods side-by-side
cargo run --example hybrid_search     # End-to-end: BM25 + dense -> fuse -> rerank -> eval
cargo run --example evaluate          # Metrics, diagnostics, pipeline, method comparison
cargo run --example trec_eval         # Parse TREC qrels/run files, collection-level metrics
cargo run --example rerank_maxsim     # ColBERT MaxSim scoring
cargo run --example diversity         # MMR and DPP diversity reranking
```

## License

MIT OR Apache-2.0
