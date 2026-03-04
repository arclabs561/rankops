# rankops

[![crates.io](https://img.shields.io/crates/v/rankops.svg)](https://crates.io/crates/rankops)
[![Documentation](https://docs.rs/rankops/badge.svg)](https://docs.rs/rankops)
[![CI](https://github.com/arclabs561/rankops/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rankops/actions/workflows/ci.yml)

Operations on ranked lists: fuse multiple retrievers, then rerank. Pairs with [**rankfns**](https://crates.io/crates/rankfns) (scoring kernels).

`rankops` covers the post-retrieval pipeline:

- **Fusion** -- combine ranked lists from heterogeneous retrievers (BM25, dense, sparse)
- **Reranking** -- MaxSim/ColBERT late interaction, MMR/DPP diversity, Matryoshka two-stage
- **Evaluation** -- NDCG, MRR, recall@k, fusion parameter optimization

## Quickstart

```toml
[dependencies]
rankops = "0.1.0"
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

| Function | Uses scores | Description |
|----------|:-----------:|-------------|
| `rrf` | No | Reciprocal Rank Fusion -- rank-based, works across incompatible scales |
| `isr` | No | Inverse Square Root fusion -- gentler rank decay than RRF |
| `borda` | No | Borda count -- (N - rank) voting points |
| `condorcet` | No | Pairwise Condorcet voting -- outlier-robust |
| `combsum` | Yes | Sum of min-max normalized scores |
| `combmnz` | Yes | CombSUM * overlap count -- rewards multi-list presence |
| `combmax` | Yes | Max score across lists |
| `combmin` | Yes | Min score -- conservative, requires all retrievers to agree |
| `combmed` | Yes | Median score -- robust to outliers |
| `weighted` | Yes | Weighted combination with per-list weights |
| `dbsf` | Yes | Distribution-Based Score Fusion (z-score normalization) |
| `standardized` | Yes | ERANK-style z-score fusion with clipping |

All two-list functions have `*_multi` variants for 3+ lists. Explainability variants (`rrf_explain`, `combsum_explain`, etc.) return full provenance.

## Reranking (feature: `rerank`)

| Module | Description |
|--------|-------------|
| `rerank::colbert` | MaxSim late interaction scoring (ColBERT, ColPali, Jina-ColBERT) |
| `rerank::diversity` | MMR and DPP diversity selection |
| `rerank::matryoshka` | Two-stage reranking with nested (Matryoshka) embeddings |
| `rerank::embedding` | Normalized vectors, masked token MaxSim |
| `rerank::quantization` | int8 quantization/dequantization for token embeddings |

## Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `rerank` | Yes | MaxSim, diversity, Matryoshka reranking (depends on `innr` for SIMD) |
| `hierarchical` | No | Hierarchical ColBERT clustering (depends on `kodama`) |
| `serde` | No | Serialization for configs and types |

## Examples

```sh
cargo run --example fusion                        # RRF, CombMNZ, Borda
cargo run --example rerank_maxsim --features rerank  # ColBERT MaxSim scoring
cargo run --example diversity --features rerank      # MMR and DPP diversity
```

## See also

- [**rankfns**](https://crates.io/crates/rankfns) -- scoring kernels (BM25, TF-IDF, cosine) that pair with `rankops`
- [**innr**](https://crates.io/crates/innr) -- SIMD dot product and MaxSim primitives used by the `rerank` feature

## License

MIT OR Apache-2.0
