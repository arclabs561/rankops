"""Tests for rankops reranking Python bindings (SIMD ops, ColBERT, diversity, matryoshka)."""

import math
import pytest
import rankops


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vec_a():
    return [1.0, 0.0, 0.0]


@pytest.fixture
def vec_b():
    return [0.0, 1.0, 0.0]


@pytest.fixture
def vec_ab():
    """45-degree vector between a and b."""
    s = 1.0 / math.sqrt(2)
    return [s, s, 0.0]


@pytest.fixture
def query_tokens():
    """3 query tokens, each 4-dim."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]


@pytest.fixture
def doc_tokens_good():
    """Document with tokens similar to query."""
    return [
        [0.9, 0.1, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.1, 0.9],
    ]


@pytest.fixture
def doc_tokens_bad():
    """Document with tokens dissimilar to query."""
    return [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ── SIMD Vector Operations ──────────────────────────────────────────────────


class TestDot:
    def test_orthogonal(self, vec_a, vec_b):
        assert rankops.dot(vec_a, vec_b) == pytest.approx(0.0, abs=1e-6)

    def test_self(self, vec_a):
        assert rankops.dot(vec_a, vec_a) == pytest.approx(1.0, abs=1e-6)

    def test_scaled(self):
        assert rankops.dot([2.0, 3.0], [4.0, 5.0]) == pytest.approx(23.0, abs=1e-5)

    def test_empty(self):
        assert rankops.dot([], []) == pytest.approx(0.0, abs=1e-6)


class TestCosine:
    def test_identical(self, vec_a):
        assert rankops.cosine(vec_a, vec_a) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self, vec_a, vec_b):
        assert rankops.cosine(vec_a, vec_b) == pytest.approx(0.0, abs=1e-6)

    def test_angle(self, vec_a, vec_ab):
        cos_45 = 1.0 / math.sqrt(2)
        assert rankops.cosine(vec_a, vec_ab) == pytest.approx(cos_45, abs=1e-5)

    def test_magnitude_invariant(self):
        a = [1.0, 2.0, 3.0]
        b = [10.0, 20.0, 30.0]
        assert rankops.cosine(a, b) == pytest.approx(1.0, abs=1e-5)


class TestMaxSim:
    def test_identical_tokens(self, query_tokens):
        score = rankops.maxsim(query_tokens, query_tokens)
        assert score > 0

    def test_good_doc_scores_higher(
        self, query_tokens, doc_tokens_good, doc_tokens_bad
    ):
        good = rankops.maxsim(query_tokens, doc_tokens_good)
        bad = rankops.maxsim(query_tokens, doc_tokens_bad)
        assert good > bad

    def test_cosine_variant(self, query_tokens, doc_tokens_good):
        dot_score = rankops.maxsim(query_tokens, doc_tokens_good)
        cos_score = rankops.maxsim_cosine(query_tokens, doc_tokens_good)
        assert math.isfinite(dot_score)
        assert math.isfinite(cos_score)


# ── ColBERT / MaxSim Ranking ────────────────────────────────────────────────


class TestMaxSimRank:
    def test_rank_ordering(self, query_tokens, doc_tokens_good, doc_tokens_bad):
        docs = [
            ("good_doc", doc_tokens_good),
            ("bad_doc", doc_tokens_bad),
        ]
        result = rankops.maxsim_rank(query_tokens, docs)
        assert len(result) == 2
        assert result[0][0] == "good_doc"
        assert result[0][1] > result[1][1]

    def test_top_k(self, query_tokens, doc_tokens_good, doc_tokens_bad):
        docs = [
            ("good_doc", doc_tokens_good),
            ("bad_doc", doc_tokens_bad),
        ]
        result = rankops.maxsim_rank(query_tokens, docs, top_k=1)
        assert len(result) == 1
        assert result[0][0] == "good_doc"

    def test_empty_docs(self, query_tokens):
        result = rankops.maxsim_rank(query_tokens, [])
        assert result == []

    def test_single_doc(self, query_tokens, doc_tokens_good):
        result = rankops.maxsim_rank(query_tokens, [("only", doc_tokens_good)])
        assert len(result) == 1


class TestMaxSimRefine:
    def test_refine_reorders(self, query_tokens, doc_tokens_good, doc_tokens_bad):
        # Candidates have bad_doc ranked first (wrong order)
        candidates = [("bad_doc", 10.0), ("good_doc", 5.0)]
        docs = [
            ("good_doc", doc_tokens_good),
            ("bad_doc", doc_tokens_bad),
        ]
        # alpha=0 means pure MaxSim (ignore original scores)
        result = rankops.maxsim_refine(candidates, query_tokens, docs, alpha=0.0)
        assert result[0][0] == "good_doc"

    def test_alpha_one_preserves_order(
        self, query_tokens, doc_tokens_good, doc_tokens_bad
    ):
        candidates = [("bad_doc", 10.0), ("good_doc", 5.0)]
        docs = [
            ("good_doc", doc_tokens_good),
            ("bad_doc", doc_tokens_bad),
        ]
        # alpha=1 means original scores only
        result = rankops.maxsim_refine(candidates, query_tokens, docs, alpha=1.0)
        assert result[0][0] == "bad_doc"


class TestMaxSimAlignments:
    def test_alignments_structure(self, query_tokens, doc_tokens_good):
        alignments = rankops.maxsim_alignments(query_tokens, doc_tokens_good)
        assert len(alignments) == len(query_tokens)
        for qi, di, sim in alignments:
            assert isinstance(qi, int)
            assert isinstance(di, int)
            assert math.isfinite(sim)

    def test_self_alignments_diagonal(self):
        tokens = [[1.0, 0.0], [0.0, 1.0]]
        alignments = rankops.maxsim_alignments(tokens, tokens)
        # Each query token should best-match its own position
        for qi, di, sim in alignments:
            assert qi == di
            assert sim == pytest.approx(1.0, abs=1e-5)


# ── Diversity: MMR ───────────────────────────────────────────────────────────


class TestMMR:
    @pytest.fixture
    def candidates_and_embeddings(self):
        candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],  # very similar to a
            [0.0, 1.0, 0.0],  # orthogonal to a
            [0.0, 0.0, 1.0],  # orthogonal to both
        ]
        return candidates, embeddings

    def test_pure_relevance(self, candidates_and_embeddings):
        candidates, embeddings = candidates_and_embeddings
        result = rankops.mmr(candidates, embeddings, lambda_=1.0, k=3)
        assert len(result) == 3
        # Pure relevance should follow original score order
        assert result[0][0] == "a"

    def test_diversity_reorders(self, candidates_and_embeddings):
        candidates, embeddings = candidates_and_embeddings
        result = rankops.mmr(candidates, embeddings, lambda_=0.0, k=3)
        assert len(result) == 3
        ids = [r[0] for r in result]
        # With pure diversity, "b" (too similar to "a") should be demoted
        assert "c" in ids  # orthogonal gets promoted

    def test_k_limit(self, candidates_and_embeddings):
        candidates, embeddings = candidates_and_embeddings
        result = rankops.mmr(candidates, embeddings, k=2)
        assert len(result) == 2


# ── Diversity: DPP ───────────────────────────────────────────────────────────


class TestDPP:
    def test_basic(self):
        candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
        result = rankops.dpp(candidates, embeddings, k=2)
        assert len(result) == 2

    def test_selects_diverse(self):
        candidates = [("a", 0.9), ("b", 0.85), ("c", 0.7)]
        embeddings = [
            [1.0, 0.0],
            [0.99, 0.1],  # near-duplicate of a
            [0.0, 1.0],  # orthogonal
        ]
        result = rankops.dpp(candidates, embeddings, k=2)
        ids = [r[0] for r in result]
        # Should prefer a + c (diverse) over a + b (similar)
        assert "a" in ids
        assert "c" in ids


# ── Matryoshka Refinement ───────────────────────────────────────────────────


class TestMatryoshkaRefine:
    def test_basic(self):
        # 8-dim embeddings, head=4
        query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        candidates = [("a", 0.9), ("b", 0.8)]
        docs = [
            ("a", [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            ("b", [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
        ]
        result = rankops.matryoshka_refine(candidates, query, docs, head_dims=4)
        assert len(result) == 2

    def test_tail_can_reorder(self):
        # Tail dims [4:8] favor doc b
        query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        candidates = [("a", 0.9), ("b", 0.5)]
        docs = [
            ("a", [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # good head, bad tail
            ("b", [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # bad head, good tail
        ]
        # alpha=0 = tail only
        result = rankops.matryoshka_refine(
            candidates, query, docs, head_dims=4, alpha=0.0
        )
        assert result[0][0] == "b"

    def test_invalid_head_dims_raises(self):
        query = [1.0, 2.0]
        candidates = [("a", 1.0)]
        docs = [("a", [1.0, 2.0])]
        with pytest.raises(ValueError):
            rankops.matryoshka_refine(candidates, query, docs, head_dims=5)


# ── Utilities ────────────────────────────────────────────────────────────────


class TestNormalizeScores:
    def test_basic(self):
        result = rankops.normalize_scores([1.0, 2.0, 3.0])
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)

    def test_single(self):
        result = rankops.normalize_scores([5.0])
        assert len(result) == 1

    def test_empty(self):
        result = rankops.normalize_scores([])
        assert result == []
