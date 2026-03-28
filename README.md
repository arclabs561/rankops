# rankops

[![crates.io](https://img.shields.io/crates/v/rankops.svg)](https://crates.io/crates/rankops)
[![Documentation](https://docs.rs/rankops/badge.svg)](https://docs.rs/rankops)
[![CI](https://github.com/arclabs561/rankops/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rankops/actions/workflows/ci.yml)

Operations on ranked lists: fuse multiple retrievers, then rerank. Pairs with [**rankfns**](https://crates.io/crates/rankfns) (scoring kernels).

`rankops` covers the post-retrieval pipeline:

- **Fusion** -- combine ranked lists from heterogeneous retrievers (BM25, dense, sparse)
- **Reranking** -- MaxSim/ColBERT late interaction, MMR/DPP diversity, Matryoshka two-stage
- **Evaluation** -- NDCG, MAP, MRR, Precision@k, recall@k, Hit Rate, fusion parameter optimization
- **Diagnostics** -- complementarity analysis, score distribution stats, fusion recommendations

## Quickstart

```toml
[dependencies]
rankops = "0.1.4"
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

| Function | Uses scores | Description |
|----------|:-----------:|-------------|
| `rrf` | No | Reciprocal Rank Fusion: $\text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}$ |
| `isr` | No | Inverse Square Root: $\sum_i \frac{1}{\sqrt{\text{rank}_i(d)}}$ |
| `borda` | No | Borda count: $\sum_i (N - \text{rank}_i(d))$ |
| `condorcet` | No | Pairwise Condorcet voting -- outlier-robust |
| `copeland` | No | Copeland voting -- net pairwise wins, more discriminative than Condorcet |
| `median_rank` | No | Median rank across lists -- outlier-robust aggregation |
| `combsum` | Yes | $\sum_i \hat{s}_i(d)$ (min-max normalized) |
| `combmnz` | Yes | $\lvert\{i : d \in L_i\}\rvert \cdot \sum_i \hat{s}_i(d)$ |
| `combmax` | Yes | Max score across lists |
| `combmin` | Yes | Min score -- conservative, requires all retrievers to agree |
| `combmed` | Yes | Median score -- robust to outliers |
| `combanz` | Yes | Average of non-zero scores |
| `weighted` | Yes | Weighted combination with per-list weights |
| `dbsf` | Yes | Distribution-Based Score Fusion (z-score normalization) |
| `standardized` | Yes | ERANK-style z-score fusion with clipping |

All two-list functions have `*_multi` variants for 3+ lists. Explainability variants (`rrf_explain`, `combsum_explain`, etc.) return full provenance.

### Normalization

Score normalization for cross-retriever fusion via `Normalization` enum:

| Variant | Range | Notes |
|---------|-------|-------|
| `MinMax` | [0, 1] | Default. Sensitive to outliers |
| `ZScore` | ~[-3, 3] | Robust to different distributions |
| `Quantile` | [0, 1] | Percentile ranks. Most robust to non-Gaussian scores |
| `Sigmoid` | (0, 1) | Logistic squash. Handles unbounded scores (cross-encoder logits) |
| `Sum` | [0, 1] | Relative magnitudes. For probability-like scores |
| `Rank` | [0, 1] | Ignores magnitudes entirely |

## Evaluation

| Function | Description |
|----------|-------------|
| `ndcg_at_k` | $\text{DCG}@k / \text{IDCG}@k$ where $\text{DCG} = \sum_{i=1}^{k} \frac{\text{rel}\_{i}}{\log\_{2}(i+1)}$ |
| `map` / `map_at_k` | $\frac{1}{\lvert R \rvert}\sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$ |
| `mrr` | $\frac{1}{\lvert Q \rvert}\sum_{q} \frac{1}{\text{rank}_q}$ (first relevant result) |
| `precision_at_k` | $\frac{\lvert\\{\text{relevant}\\} \cap \text{top-}k\rvert}{k}$ |
| `recall_at_k` | $\frac{\lvert\\{\text{relevant}\\} \cap \text{top-}k\rvert}{\lvert\\{\text{relevant}\\}\rvert}$ |
| `hit_rate` | Binary: any relevant doc in top-k? |
| `evaluate_metric` | Dispatch by `OptimizeMetric` enum |
| `optimize_fusion` | Grid search over fusion parameters |

## Diagnostics

The `diagnostics` module helps decide whether fusion is beneficial:

| Function | Description |
|----------|-------------|
| `score_stats` | Distribution analysis (mean, std, median, percentiles) |
| `overlap_ratio` | Jaccard overlap between document sets |
| `complementarity` | Fraction of relevant docs unique to one retriever |
| `rank_correlation` | Kendall's tau-b on shared documents |
| `diagnose` | Full report with fusion recommendation |

Based on Louis et al., "Know When to Fuse" (2024): high complementarity (>0.5) predicts fusion benefit; low correlation between rankers predicts fusion benefit.

## Adapters

The `adapt` module converts retriever outputs to rankops format:

| Function | Input | Conversion |
|----------|-------|------------|
| `from_distances` | L2/Euclidean (lower=closer) | `1/(1+d)` to (0, 1] |
| `from_similarities` | Cosine sim (higher=better) | Sort descending |
| `from_logits` | Cross-encoder logits (unbounded) | Sigmoid to (0, 1) |
| `from_inner_product` | Dot product (higher=better) | Sort descending |

All have `_mapped` variants for ID type conversion (e.g., `u32` doc index to `&str` doc name).

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
cargo run --example fusion            # All fusion methods side-by-side
cargo run --example hybrid_search     # End-to-end: BM25 + dense -> fuse -> rerank -> eval
cargo run --example evaluate          # Metrics, diagnostics, pipeline, method comparison
cargo run --example rerank_maxsim     # ColBERT MaxSim scoring
cargo run --example diversity         # MMR and DPP diversity reranking
```

## See also

- [**rankfns**](https://crates.io/crates/rankfns) -- scoring kernels (BM25, TF-IDF, cosine) that pair with `rankops`
- [**innr**](https://crates.io/crates/innr) -- SIMD dot product and MaxSim primitives used by the `rerank` feature
- [**vicinity**](https://crates.io/crates/vicinity) -- ANN vector search (HNSW, IVF-PQ) that feeds ranked candidates to `rankops`
- [**rankit**](https://crates.io/crates/rankit) -- learning-to-rank training (LTR losses, differentiable sorting, evaluation)

## License

MIT OR Apache-2.0
