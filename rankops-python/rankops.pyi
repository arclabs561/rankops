"""Type stubs for rankops Python bindings (PyPI package: rankops)."""

from typing import List, Tuple, Optional, Literal

__version__: str

RankedList = List[Tuple[str, float]]
MultiRankedLists = List[RankedList]
NormalizationType = Literal["zscore", "minmax", "sum", "rank", "none"]

# Configuration Classes

class RrfConfig:
    """Configuration for Reciprocal Rank Fusion."""

    k: int

    def __init__(self, k: int = 60) -> None:
        """Create an RRF configuration.

        Args:
            k: Smoothing constant. Higher values reduce the influence of
                high-ranked items. Default: 60.
        """
        ...

    def with_k(self, k: int) -> "RrfConfig":
        """Return a new config with a different k value.

        Args:
            k: Smoothing constant.

        Returns:
            New RrfConfig with updated k.
        """
        ...

class FusionConfig:
    """Configuration for score-based fusion methods (CombSUM, CombMNZ, Borda, DBSF)."""

    def __init__(self) -> None:
        """Create a default fusion configuration."""
        ...

    def with_top_k(self, top_k: int) -> "FusionConfig":
        """Return a new config that limits output to top_k results.

        Args:
            top_k: Maximum number of results to return.

        Returns:
            New FusionConfig with updated top_k.
        """
        ...

class WeightedConfig:
    """Configuration for weighted score fusion."""

    weight_a: float
    weight_b: float

    def __init__(self, weight_a: float, weight_b: float) -> None:
        """Create a weighted fusion configuration.

        Args:
            weight_a: Weight for the first ranked list.
            weight_b: Weight for the second ranked list.
        """
        ...

    def with_normalize(self, normalize: bool) -> "WeightedConfig":
        """Return a new config with normalization enabled or disabled.

        Args:
            normalize: If True, normalize scores to [0, 1] before weighting.

        Returns:
            New WeightedConfig with updated normalization setting.
        """
        ...

    def with_top_k(self, top_k: int) -> "WeightedConfig":
        """Return a new config that limits output to top_k results.

        Args:
            top_k: Maximum number of results to return.

        Returns:
            New WeightedConfig with updated top_k.
        """
        ...

class StandardizedConfig:
    """Configuration for standardized (z-score) fusion."""

    clip_range: Tuple[float, float]

    def __init__(self, clip_range: Tuple[float, float] = (-3.0, 3.0)) -> None:
        """Create a standardized fusion configuration.

        Args:
            clip_range: Min and max bounds for clipping z-scores.
                Default: (-3.0, 3.0).
        """
        ...

    def with_top_k(self, top_k: int) -> "StandardizedConfig":
        """Return a new config that limits output to top_k results.

        Args:
            top_k: Maximum number of results to return.

        Returns:
            New StandardizedConfig with updated top_k.
        """
        ...

class AdditiveMultiTaskConfig:
    """Configuration for additive multi-task fusion (ResFlow-style)."""

    weights: Tuple[float, float]

    def __init__(self, weights: Tuple[float, float] = (1.0, 1.0), normalization: str = "minmax") -> None:
        """Create an additive multi-task fusion configuration.

        Args:
            weights: Weight for each task as (weight_a, weight_b).
                Default: (1.0, 1.0).
            normalization: Score normalization method. One of "zscore",
                "minmax", "sum", "rank", "none". Default: "minmax".
        """
        ...

    def with_normalization(self, normalization: NormalizationType) -> "AdditiveMultiTaskConfig":
        """Return a new config with a different normalization method.

        Args:
            normalization: One of "zscore", "minmax", "sum", "rank", "none".

        Returns:
            New AdditiveMultiTaskConfig with updated normalization.
        """
        ...

    def with_top_k(self, top_k: int) -> "AdditiveMultiTaskConfig":
        """Return a new config that limits output to top_k results.

        Args:
            top_k: Maximum number of results to return.

        Returns:
            New AdditiveMultiTaskConfig with updated top_k.
        """
        ...

# Explainability Classes

class SourceContribution:
    """Contribution of a single retriever to a fused result."""

    retriever_id: str
    """Identifier of the retriever that produced this contribution."""
    original_rank: Optional[int]
    """Rank of the document in the original retriever's list, if present."""
    original_score: Optional[float]
    """Score from the original retriever, if present."""
    normalized_score: Optional[float]
    """Score after normalization, if applicable."""
    contribution: float
    """Contribution of this retriever to the final fused score."""

class Explanation:
    """Provenance information for a fused result."""

    sources: List[SourceContribution]
    """Per-retriever contributions to this result."""
    method: str
    """Fusion method used (e.g., "rrf", "combsum")."""
    consensus_score: float
    """Consensus score across retrievers (0.0 = single source, 1.0 = all agree)."""

class FusedResult:
    """A single result from an explainable fusion operation."""

    id: str
    """Document identifier."""
    score: float
    """Fused score."""
    rank: int
    """Position in the fused result list (0-indexed)."""
    explanation: Explanation
    """Provenance details for this result."""

class RetrieverId:
    """Identifier for a retriever in explainability operations."""

    id: str

    def __init__(self, id: str) -> None:
        """Create a retriever identifier.

        Args:
            id: String identifier for the retriever.
        """
        ...

class RetrieverStats:
    """Aggregate statistics for a retriever across fused results."""

    top_k_count: int
    """Number of this retriever's documents in the top-k."""
    avg_contribution: float
    """Average contribution score across results."""
    unique_docs: int
    """Number of unique documents from this retriever."""

class ConsensusReport:
    """Report on retriever agreement across fused results."""

    high_consensus: List[str]
    """Document IDs that appear in most retrievers."""
    single_source: List[str]
    """Document IDs that appear in only one retriever."""
    rank_disagreement: List[Tuple[str, List[Tuple[str, int]]]]
    """Documents with large rank differences across retrievers.
    Each entry is (doc_id, [(retriever_id, rank), ...])."""

class ValidationResult:
    """Result of validating a fused ranked list."""

    is_valid: bool
    """True if no errors were found."""
    errors: List[str]
    """List of error messages (failures)."""
    warnings: List[str]
    """List of warning messages (non-fatal issues)."""

# Rank-based Fusion

def rrf(results_a: RankedList, results_b: RankedList, k: int = 60, top_k: Optional[int] = None) -> RankedList:
    """Reciprocal Rank Fusion of two ranked lists.

    Fuses results using rank positions only (ignores scores), so no
    normalization is needed across different score scales.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        k: Smoothing constant. Higher values reduce the influence of
            high-ranked items. Default: 60.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def rrf_multi(lists: MultiRankedLists, k: int = 60, top_k: Optional[int] = None) -> RankedList:
    """Reciprocal Rank Fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        k: Smoothing constant. Default: 60.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def isr(results_a: RankedList, results_b: RankedList, k: int = 1, top_k: Optional[int] = None) -> RankedList:
    """Inverse Square Rank fusion of two ranked lists.

    Like RRF but uses 1/(k + rank)^2 weighting, giving lower ranks
    more relative contribution compared to top positions.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        k: Smoothing constant. Default: 1.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def isr_multi(lists: MultiRankedLists, k: int = 1, top_k: Optional[int] = None) -> RankedList:
    """Inverse Square Rank fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        k: Smoothing constant. Default: 1.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def borda(results_a: RankedList, results_b: RankedList, top_k: Optional[int] = None) -> RankedList:
    """Borda count fusion of two ranked lists.

    Assigns points based on rank position (highest rank gets most points)
    and sums across lists.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def borda_multi(lists: MultiRankedLists, top_k: Optional[int] = None) -> RankedList:
    """Borda count fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

# Score-based Fusion

def combsum(results_a: RankedList, results_b: RankedList, top_k: Optional[int] = None) -> RankedList:
    """CombSUM fusion of two ranked lists.

    Sums scores for documents appearing in both lists. Requires scores
    on compatible scales.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def combsum_multi(lists: MultiRankedLists, top_k: Optional[int] = None) -> RankedList:
    """CombSUM fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def combmnz(results_a: RankedList, results_b: RankedList, top_k: Optional[int] = None) -> RankedList:
    """CombMNZ fusion of two ranked lists.

    Like CombSUM but multiplies the sum by the number of lists containing
    each document, rewarding documents that appear in multiple retrievers.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def combmnz_multi(lists: MultiRankedLists, top_k: Optional[int] = None) -> RankedList:
    """CombMNZ fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def weighted(results_a: RankedList, results_b: RankedList, weight_a: float, weight_b: float, normalize: bool = True, top_k: Optional[int] = None) -> RankedList:
    """Weighted score fusion of two ranked lists.

    Computes weight_a * score_a + weight_b * score_b for each document.
    Optionally normalizes scores to [0, 1] before weighting.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        weight_a: Weight for the first list's scores.
        weight_b: Weight for the second list's scores.
        normalize: If True, normalize scores to [0, 1] before weighting.
            Default: True.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def dbsf(results_a: RankedList, results_b: RankedList, top_k: Optional[int] = None) -> RankedList:
    """Distribution-Based Score Fusion of two ranked lists.

    Normalizes scores based on their distribution (mean and standard
    deviation) within each list before combining.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def dbsf_multi(lists: MultiRankedLists, top_k: Optional[int] = None) -> RankedList:
    """Distribution-Based Score Fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def standardized(results_a: RankedList, results_b: RankedList, clip_range: Tuple[float, float] = (-3.0, 3.0), top_k: Optional[int] = None) -> RankedList:
    """Standardized (ERANK-style) fusion of two ranked lists.

    Applies z-score normalization with configurable clipping to handle
    different score distributions, including negative scores.

    Args:
        results_a: List of (id, score) tuples from the first retriever.
        results_b: List of (id, score) tuples from the second retriever.
        clip_range: Min and max for clipping z-scores. Default: (-3.0, 3.0).
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def standardized_multi(lists: MultiRankedLists, clip_range: Tuple[float, float] = (-3.0, 3.0), top_k: Optional[int] = None) -> RankedList:
    """Standardized (ERANK-style) fusion of multiple ranked lists.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        clip_range: Min and max for clipping z-scores. Default: (-3.0, 3.0).
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

def additive_multi_task(results_a: RankedList, results_b: RankedList, weights: Tuple[float, float] = (1.0, 1.0), normalization: str = "minmax", top_k: Optional[int] = None) -> RankedList:
    """Additive multi-task fusion (ResFlow-style) of two ranked lists.

    Combines scores from multiple tasks with configurable weights and
    normalization. Designed for e-commerce ranking (e.g., CTR + CTCVR).

    Args:
        results_a: List of (id, score) tuples from the first task.
        results_b: List of (id, score) tuples from the second task.
        weights: Weight for each task as (weight_a, weight_b).
            Default: (1.0, 1.0).
        normalization: Score normalization method. One of "zscore",
            "minmax", "sum", "rank", "none". Default: "minmax".
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by fused score descending.
    """
    ...

# Explainability

def rrf_explain(lists: MultiRankedLists, retriever_ids: List[str], k: int = 60, top_k: Optional[int] = None) -> List[FusedResult]:
    """RRF fusion with per-result provenance information.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        retriever_ids: String identifier for each list (same length as lists).
        k: Smoothing constant. Default: 60.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of FusedResult objects with explanation details.
    """
    ...

def combsum_explain(lists: MultiRankedLists, retriever_ids: List[str], top_k: Optional[int] = None) -> List[FusedResult]:
    """CombSUM fusion with per-result provenance information.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        retriever_ids: String identifier for each list (same length as lists).
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of FusedResult objects with explanation details.
    """
    ...

def combmnz_explain(lists: MultiRankedLists, retriever_ids: List[str], top_k: Optional[int] = None) -> List[FusedResult]:
    """CombMNZ fusion with per-result provenance information.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        retriever_ids: String identifier for each list (same length as lists).
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of FusedResult objects with explanation details.
    """
    ...

def dbsf_explain(lists: MultiRankedLists, retriever_ids: List[str], top_k: Optional[int] = None) -> List[FusedResult]:
    """DBSF fusion with per-result provenance information.

    Args:
        lists: List of ranked lists, each containing (id, score) tuples.
        retriever_ids: String identifier for each list (same length as lists).
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of FusedResult objects with explanation details.
    """
    ...

# Validation

def validate_sorted(results: RankedList) -> ValidationResult:
    """Check that results are sorted by score in descending order.

    Args:
        results: List of (id, score) tuples.

    Returns:
        ValidationResult with errors if not sorted.
    """
    ...

def validate_no_duplicates(results: RankedList) -> ValidationResult:
    """Check that no document ID appears more than once.

    Args:
        results: List of (id, score) tuples.

    Returns:
        ValidationResult with errors if duplicates found.
    """
    ...

def validate_finite_scores(results: RankedList) -> ValidationResult:
    """Check that all scores are finite (no NaN or Infinity).

    Args:
        results: List of (id, score) tuples.

    Returns:
        ValidationResult with errors if non-finite scores found.
    """
    ...

def validate_non_negative_scores(results: RankedList) -> ValidationResult:
    """Check that all scores are non-negative (warning only).

    Args:
        results: List of (id, score) tuples.

    Returns:
        ValidationResult with warnings if negative scores found.
    """
    ...

def validate_bounds(results: RankedList, max_results: Optional[int] = None) -> ValidationResult:
    """Check that result count is within expected bounds.

    Args:
        results: List of (id, score) tuples.
        max_results: Maximum expected number of results. None skips this check.

    Returns:
        ValidationResult with warnings if bounds exceeded.
    """
    ...

def validate(results: RankedList, check_non_negative: bool = False, max_results: Optional[int] = None) -> ValidationResult:
    """Run all validation checks on a ranked list.

    Checks sorting, duplicates, finite scores, and optionally
    non-negative scores and result count bounds.

    Args:
        results: List of (id, score) tuples.
        check_non_negative: If True, warn on negative scores. Default: False.
        max_results: Maximum expected number of results. None skips bound check.

    Returns:
        ValidationResult combining all check results.
    """
    ...

# Reranking: SIMD vector operations

TokenEmbeddings = List[List[float]]

def dot(a: List[float], b: List[float]) -> float:
    """Compute dot product of two vectors (SIMD-accelerated).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Dot product as a float.
    """
    ...

def cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (SIMD-accelerated).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0].
    """
    ...

def maxsim(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> float:
    """Compute MaxSim score between query and document token embeddings.

    For each query token, finds the maximum dot product with any document
    token, then sums across all query tokens.

    Args:
        query_tokens: List of token embedding vectors for the query.
        doc_tokens: List of token embedding vectors for the document.

    Returns:
        MaxSim score as a float.
    """
    ...

def maxsim_cosine(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> float:
    """Compute MaxSim score using cosine similarity instead of dot product.

    Args:
        query_tokens: List of token embedding vectors for the query.
        doc_tokens: List of token embedding vectors for the document.

    Returns:
        MaxSim-cosine score as a float.
    """
    ...

# Reranking: ColBERT / MaxSim

def maxsim_rank(query_tokens: TokenEmbeddings, docs: List[Tuple[str, TokenEmbeddings]], top_k: Optional[int] = None) -> RankedList:
    """Rank documents by MaxSim score against a query.

    Args:
        query_tokens: List of token embedding vectors for the query.
        docs: List of (id, token_embeddings) pairs for candidate documents.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by MaxSim score descending.
    """
    ...

def maxsim_refine(candidates: RankedList, query_tokens: TokenEmbeddings, docs: List[Tuple[str, TokenEmbeddings]], alpha: float = 0.5, top_k: Optional[int] = None) -> RankedList:
    """Refine first-stage candidates using MaxSim reranking.

    Blends original retrieval scores with MaxSim scores using alpha.

    Args:
        candidates: List of (id, score) from first-stage retrieval.
        query_tokens: List of token embedding vectors for the query.
        docs: List of (id, token_embeddings) for candidate documents.
        alpha: Blending weight. 0.0 = all MaxSim, 1.0 = all original.
            Default: 0.5.
        top_k: Maximum number of results to return. None returns all.

    Returns:
        List of (id, score) tuples sorted by blended score descending.
    """
    ...

def maxsim_alignments(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> List[Tuple[int, int, float]]:
    """Get token-level alignment details between query and document.

    For each query token, finds the best-matching document token.

    Args:
        query_tokens: List of token embedding vectors for the query.
        doc_tokens: List of token embedding vectors for the document.

    Returns:
        List of (query_token_idx, doc_token_idx, similarity) triples.
    """
    ...

# Reranking: Diversity

def mmr(candidates: RankedList, embeddings: List[List[float]], lambda_: float = 0.5, k: int = 10) -> RankedList:
    """Maximal Marginal Relevance diversity reranking.

    Iteratively selects documents that balance relevance with diversity
    relative to already-selected documents.

    Args:
        candidates: List of (id, score) pairs.
        embeddings: Embedding vector for each candidate (same order).
        lambda_: Trade-off parameter. 1.0 = pure relevance,
            0.0 = pure diversity. Default: 0.5.
        k: Number of results to select. Default: 10.

    Returns:
        List of (id, score) tuples for the selected diverse subset.
    """
    ...

def dpp(candidates: RankedList, embeddings: List[List[float]], k: int = 10, alpha: float = 1.0) -> RankedList:
    """Determinantal Point Process diversity selection.

    Selects a diverse subset by modeling repulsion between similar items
    using a DPP kernel.

    Args:
        candidates: List of (id, score) pairs.
        embeddings: Embedding vector for each candidate (same order).
        k: Number of results to select. Default: 10.
        alpha: Relevance weight. Higher values weight relevance more
            relative to diversity. Default: 1.0.

    Returns:
        List of (id, score) tuples for the selected diverse subset.
    """
    ...

# Reranking: Matryoshka

def matryoshka_refine(candidates: RankedList, query: List[float], docs: List[Tuple[str, List[float]]], head_dims: int, alpha: float = 0.5) -> RankedList:
    """Refine candidates using Matryoshka (nested) embeddings.

    Uses tail dimensions (beyond head_dims) to re-score candidates
    that were initially retrieved using only the head dimensions.

    Args:
        candidates: List of (id, score) from first-stage retrieval
            using head dimensions.
        query: Full query embedding vector (all dimensions).
        docs: List of (id, full_embedding) for candidate documents.
        head_dims: Number of head dimensions used in first-stage retrieval.
        alpha: Blending weight. 0.0 = tail-only, 1.0 = original-only.
            Default: 0.5.

    Returns:
        List of (id, score) tuples sorted by blended score descending.
    """
    ...

# Utilities

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize a list of scores to the [0, 1] range via min-max scaling.

    Args:
        scores: List of raw scores.

    Returns:
        List of scores scaled to [0.0, 1.0]. Returns all zeros if
        min equals max.
    """
    ...
