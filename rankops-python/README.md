# rankops-python

Python bindings for **rankops** (Rust) -- fusion and reranking for hybrid search.

## Installation

```bash
pip install rankops
```

### From source

```bash
# Using uv (recommended)
cd rankops-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv

# Or using pip
pip install maturin
maturin develop --release
```

## Quick Start

```python
import rankops

# BM25 results (keyword search)
bm25 = [("doc_1", 87.5), ("doc_2", 82.3), ("doc_3", 78.1)]

# Dense embedding results (semantic search)
dense = [("doc_2", 0.92), ("doc_1", 0.88), ("doc_4", 0.85)]

# RRF finds consensus: doc_2 appears high in both lists
fused = rankops.rrf(bm25, dense, k=60)
# [("doc_2", 0.033), ("doc_1", 0.032), ("doc_3", 0.016), ("doc_4", 0.016)]
```

**Why RRF?** BM25 scores are 0-100, dense scores are 0-1. RRF ignores scores and uses only rank positions, so no normalization needed.

## API Reference

### Rank-based Fusion (ignores scores)

These methods work with any score scale because they only use rank positions:

#### RRF (Reciprocal Rank Fusion)

```python
# Two lists
fused = rankops.rrf(bm25, dense, k=60, top_k=10)

# Multiple lists
fused = rankops.rrf_multi([bm25, dense, sparse], k=60, top_k=10)
```

**When to use**: Different score scales (BM25: 0-100, dense: 0-1), zero-configuration needs.

#### ISR (Inverse Square Rank)

```python
fused = rankops.isr(bm25, dense, k=1, top_k=10)
fused = rankops.isr_multi([bm25, dense], k=1, top_k=10)
```

**When to use**: When lower ranks should contribute more relative to top positions.

#### Borda Count

```python
fused = rankops.borda(bm25, dense, top_k=10)
fused = rankops.borda_multi([bm25, dense], top_k=10)
```

**When to use**: Simple voting-based fusion.

### Score-based Fusion (uses scores)

These methods require scores on compatible scales:

#### CombSUM

```python
fused = rankops.combsum(bm25, dense, top_k=10)
fused = rankops.combsum_multi([bm25, dense], top_k=10)
```

**When to use**: Scores on the same scale (both 0-1, both cosine similarity).

#### CombMNZ

```python
fused = rankops.combmnz(bm25, dense, top_k=10)
fused = rankops.combmnz_multi([bm25, dense], top_k=10)
```

**When to use**: Same scale as CombSUM, but want to reward documents appearing in multiple lists.

#### Weighted Fusion

```python
fused = rankops.weighted(
    bm25, dense,
    weight_a=0.7,  # Weight for first list
    weight_b=0.3,  # Weight for second list
    normalize=True,  # Normalize scores to [0,1] before combining
    top_k=10
)
```

**When to use**: Trust one retriever more than another.

#### DBSF (Distribution-Based Score Fusion)

```python
fused = rankops.dbsf(bm25, dense, top_k=10)
fused = rankops.dbsf_multi([bm25, dense], top_k=10)
```

**When to use**: Score distributions differ significantly between retrievers.

#### Standardized Fusion (ERANK-style)

```python
# Two lists with default clipping [-3.0, 3.0]
fused = rankops.standardized(bm25, dense, clip_range=(-3.0, 3.0), top_k=10)

# Multiple lists
fused = rankops.standardized_multi(
    [bm25, dense],
    clip_min=-3.0,
    clip_max=3.0,
    top_k=10
)
```

**When to use**: Different score distributions, need configurable outlier handling, have negative scores.

#### Additive Multi-Task Fusion (ResFlow-style)

```python
# E-commerce example: CTR + CTCVR with 1:20 weight ratio
fused = rankops.additive_multi_task(
    ctr_scores,      # Click-through rate predictions
    ctcvr_scores,    # Click-to-conversion rate predictions
    weight_a=1.0,    # Weight for first task
    weight_b=20.0,   # Weight for second task (20x more important)
    normalization="minmax",  # Normalization method: "zscore", "minmax", "sum", "rank", "none"
    top_k=10
)
```

**When to use**: Combining multiple ranking tasks (e.g., CTR + CTCVR in e-commerce), tasks have different scales but you know relative importance.

### Explainability

Get detailed provenance information showing which retrievers contributed to each result:

```python
import rankops

bm25 = [("doc_1", 87.5), ("doc_2", 82.3)]
dense = [("doc_2", 0.92), ("doc_3", 0.88)]

# Get results with full provenance
explained = rankops.rrf_explain(
    [bm25, dense],
    ["bm25", "dense"],
    k=60
)

# Inspect first result
result = explained[0]
print(f"Document: {result.id}")
print(f"Score: {result.score}")
print(f"Rank: {result.rank}")
print(f"Consensus: {result.explanation.consensus_score}")

# See which retrievers contributed
for source in result.explanation.sources:
    print(f"  {source.retriever_id.id}: rank={source.original_rank}, contribution={source.contribution}")
```

Available explainability functions:
- `rrf_explain()` - RRF with explainability
- `combsum_explain()` - CombSUM with explainability
- `combmnz_explain()` - CombMNZ with explainability
- `dbsf_explain()` - DBSF with explainability

### Result Validation

Validate fusion results to ensure they meet expected properties:

```python
import rankops

fused = rankops.rrf(bm25, dense, k=60)

# Comprehensive validation
result = rankops.validate(fused, check_non_negative=False, max_results=10)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")
if result.warnings:
    print(f"Warnings: {result.warnings}")

# Individual checks
sorted_check = rankops.validate_sorted(fused)
dup_check = rankops.validate_no_duplicates(fused)
finite_check = rankops.validate_finite_scores(fused)
non_neg_check = rankops.validate_non_negative_scores(fused)
bounds_check = rankops.validate_bounds(fused, max_results=10)
```

Validation checks:
- **Sorted**: Results are sorted by score (descending)
- **No duplicates**: Each document ID appears only once
- **Finite scores**: No NaN or Infinity values
- **Non-negative** (optional): Warns on negative scores when `check_non_negative=True`
- **Max results** (optional): Warns if result count exceeds `max_results`

### Configuration Objects

For advanced usage, you can use configuration objects:

```python
config = rankops.RrfConfigPy(k=100)

# Other config types available:
# - FusionConfigPy (for CombSUM/CombMNZ/Borda/DBSF)
# - WeightedConfigPy (for weighted fusion)
# - StandardizedConfigPy (for standardized fusion)
# - AdditiveMultiTaskConfigPy (for additive multi-task fusion)
```

## Complete Example: RAG Pipeline

```python
import rankops

def rag_pipeline(query: str):
    # Step 1: Retrieve from multiple sources
    bm25_results = [
        ("doc_123", 87.5),
        ("doc_456", 82.3),
        ("doc_789", 78.1),
    ]

    dense_results = [
        ("doc_456", 0.92),
        ("doc_123", 0.88),
        ("doc_999", 0.85),
    ]

    # Step 2: Fuse results (RRF finds consensus)
    fused = rankops.rrf_multi(
        [bm25_results, dense_results],
        k=60,
        top_k=100  # Top 100 for reranking
    )

    # Step 3: Rerank with cross-encoder (expensive, so only top 100)
    # reranked = cross_encoder_rerank([id for id, _ in fused], query)

    # Step 4: Return top 10 for LLM context
    return [id for id, _ in fused[:10]]
```

## Type Hints and IDE Support

The package includes type stubs (`.pyi` files) for full type checking support with:
- **mypy**: `mypy your_code.py`
- **pyright**: Built into VS Code and other editors
- **PyCharm**: Automatic type checking

All functions are fully typed:

```python
from typing import List, Tuple

RankedList = List[Tuple[str, float]]
fused: RankedList = rankops.rrf(bm25, dense, k=60)
```

## Error Handling

```python
# Most functions don't error - they return empty list for edge cases
fused = rankops.rrf([], [])  # Returns []

# Validation functions return ValidationResult with errors/warnings
result = rankops.validate(fused, check_non_negative=False, max_results=10)
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")
```

## Performance

Python bindings add minimal overhead over native Rust:

- **RRF (100 items)**: ~15us (vs 13us in Rust)
- **RRF (1000 items)**: ~180us (vs 159us in Rust)

Overhead comes from Python object conversion and result serialization (~1-2us per call).

## See Also

- **[rankops (Rust) README](../README.md)** -- Algorithm overview and quickstart
- **[API Documentation](https://docs.rs/rankops)** -- Full Rust API reference

## License

MIT OR Apache-2.0
