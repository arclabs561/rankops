"""Tests for rankops Python bindings."""

import math
import pytest
import rankops


# Test data fixtures
@pytest.fixture
def bm25_results():
    return [("doc_A", 12.5), ("doc_B", 11.0), ("doc_C", 9.0)]


@pytest.fixture
def dense_results():
    return [("doc_B", 0.9), ("doc_D", 0.8), ("doc_A", 0.7)]


@pytest.fixture
def keyword_results():
    return [("doc_C", 0.95), ("doc_E", 0.85), ("doc_A", 0.75)]


@pytest.fixture
def multi_lists(bm25_results, dense_results, keyword_results):
    return [bm25_results, dense_results, keyword_results]


def _is_sorted_desc(results):
    """Check results are sorted by score descending."""
    return all(results[i][1] >= results[i + 1][1] for i in range(len(results) - 1))


def _ids(results):
    """Extract just the IDs from results."""
    return [r[0] for r in results]


def _no_duplicate_ids(results):
    ids = _ids(results)
    return len(ids) == len(set(ids))


# ── RRF ──────────────────────────────────────────────────────────────────────


class TestRRF:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.rrf(bm25_results, dense_results, k=60)
        assert len(result) == 4  # union of {A,B,C} and {B,D,A}
        assert _is_sorted_desc(result)
        assert _no_duplicate_ids(result)

    def test_default_k(self, bm25_results, dense_results):
        result = rankops.rrf(bm25_results, dense_results)
        assert len(result) > 0

    def test_custom_k(self, bm25_results, dense_results):
        r60 = rankops.rrf(bm25_results, dense_results, k=60)
        r1 = rankops.rrf(bm25_results, dense_results, k=1)
        # Different k values produce different score magnitudes
        assert r60[0][1] != r1[0][1]

    def test_top_k(self, bm25_results, dense_results):
        result = rankops.rrf(bm25_results, dense_results, k=60, top_k=2)
        assert len(result) == 2
        assert _is_sorted_desc(result)

    def test_k_zero_raises(self, bm25_results, dense_results):
        with pytest.raises(ValueError, match="k must be >= 1"):
            rankops.rrf(bm25_results, dense_results, k=0)

    def test_empty_lists(self):
        assert rankops.rrf([], []) == []

    def test_one_empty(self, bm25_results):
        result = rankops.rrf(bm25_results, [])
        assert len(result) == 3
        assert _is_sorted_desc(result)

    def test_overlapping_docs_ranked_higher(self):
        a = [("shared", 10.0), ("only_a", 5.0)]
        b = [("shared", 0.9), ("only_b", 0.5)]
        result = rankops.rrf(a, b)
        assert result[0][0] == "shared"

    def test_multi(self, multi_lists):
        result = rankops.rrf_multi(multi_lists, k=60)
        assert len(result) == 5  # union of A,B,C,D,E
        assert _is_sorted_desc(result)
        assert _no_duplicate_ids(result)

    def test_multi_top_k(self, multi_lists):
        result = rankops.rrf_multi(multi_lists, k=60, top_k=3)
        assert len(result) == 3

    def test_multi_empty(self):
        assert rankops.rrf_multi([]) == []

    def test_multi_single_list(self, bm25_results):
        result = rankops.rrf_multi([bm25_results])
        assert len(result) == 3


# ── ISR ──────────────────────────────────────────────────────────────────────


class TestISR:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.isr(bm25_results, dense_results, k=1)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_default_k(self, bm25_results, dense_results):
        result = rankops.isr(bm25_results, dense_results)
        assert len(result) > 0

    def test_top_k(self, bm25_results, dense_results):
        result = rankops.isr(bm25_results, dense_results, k=1, top_k=2)
        assert len(result) <= 2

    def test_multi(self, multi_lists):
        result = rankops.isr_multi(multi_lists, k=1)
        assert len(result) > 0
        assert _is_sorted_desc(result)


# ── CombSUM ──────────────────────────────────────────────────────────────────


class TestCombSUM:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.combsum(bm25_results, dense_results)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_overlap_scores_higher(self):
        a = [("x", 3.0), ("y", 1.0)]
        b = [("x", 2.0), ("z", 4.0)]
        result = rankops.combsum(a, b)
        scores = {r[0]: r[1] for r in result}
        # x appears in both lists, should have highest combined score
        assert scores["x"] >= scores["y"]

    def test_top_k(self, bm25_results, dense_results):
        result = rankops.combsum(bm25_results, dense_results, top_k=2)
        assert len(result) == 2

    def test_multi(self, multi_lists):
        result = rankops.combsum_multi(multi_lists)
        assert len(result) > 0
        assert _is_sorted_desc(result)


# ── CombMNZ ──────────────────────────────────────────────────────────────────


class TestCombMNZ:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.combmnz(bm25_results, dense_results)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_multi(self, multi_lists):
        result = rankops.combmnz_multi(multi_lists)
        assert len(result) > 0

    def test_mnz_boosts_overlap(self):
        a = [("x", 3.0), ("y", 1.0)]
        b = [("x", 2.0), ("z", 5.0)]
        result = rankops.combmnz(a, b)
        scores = {r[0]: r[1] for r in result}
        # x appears in both lists: (3+2)*2 = 10
        # z appears in one: 5*1 = 5
        assert scores["x"] > scores["z"]


# ── Borda ────────────────────────────────────────────────────────────────────


class TestBorda:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.borda(bm25_results, dense_results)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_multi(self, multi_lists):
        result = rankops.borda_multi(multi_lists)
        assert len(result) > 0


# ── DBSF ─────────────────────────────────────────────────────────────────────


class TestDBSF:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.dbsf(bm25_results, dense_results)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_multi(self, multi_lists):
        result = rankops.dbsf_multi(multi_lists)
        assert len(result) > 0


# ── Weighted ─────────────────────────────────────────────────────────────────


class TestWeighted:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.weighted(
            bm25_results, dense_results, weight_a=0.7, weight_b=0.3, normalize=True
        )
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_no_normalize(self, bm25_results, dense_results):
        result = rankops.weighted(
            bm25_results, dense_results, weight_a=0.6, weight_b=0.4, normalize=False
        )
        assert len(result) > 0

    def test_zero_weights_raises(self, bm25_results, dense_results):
        with pytest.raises(ValueError, match="weights cannot both be zero"):
            rankops.weighted(bm25_results, dense_results, weight_a=0.0, weight_b=0.0)

    def test_infinite_weight_raises(self, bm25_results, dense_results):
        with pytest.raises(ValueError, match="weights must be finite"):
            rankops.weighted(
                bm25_results, dense_results, weight_a=float("inf"), weight_b=0.5
            )

    def test_weight_asymmetry(self):
        a = [("x", 10.0)]
        b = [("x", 10.0)]
        r1 = rankops.weighted(a, b, weight_a=0.9, weight_b=0.1, normalize=False)
        r2 = rankops.weighted(a, b, weight_a=0.1, weight_b=0.9, normalize=False)
        # Same input, symmetric weights -> same score
        assert r1[0][1] == pytest.approx(r2[0][1], abs=1e-5)


# ── Standardized ─────────────────────────────────────────────────────────────


class TestStandardized:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.standardized(bm25_results, dense_results)
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_custom_clip(self, bm25_results, dense_results):
        result = rankops.standardized(
            bm25_results, dense_results, clip_range=(-2.0, 2.0)
        )
        assert len(result) > 0

    def test_multi(self, multi_lists):
        result = rankops.standardized_multi(multi_lists)
        assert len(result) > 0


# ── Additive Multi-Task ──────────────────────────────────────────────────────


class TestAdditiveMultiTask:
    def test_basic(self, bm25_results, dense_results):
        result = rankops.additive_multi_task(
            bm25_results, dense_results, weights=(1.0, 1.0), normalization="minmax"
        )
        assert len(result) > 0
        assert _is_sorted_desc(result)

    def test_all_normalizations(self, bm25_results, dense_results):
        for norm in ["zscore", "minmax", "sum", "rank", "none"]:
            result = rankops.additive_multi_task(
                bm25_results, dense_results, weights=(1.0, 1.0), normalization=norm
            )
            assert len(result) > 0, f"normalization={norm} returned empty"

    def test_invalid_normalization_raises(self, bm25_results, dense_results):
        with pytest.raises(ValueError, match="normalization must be one of"):
            rankops.additive_multi_task(
                bm25_results, dense_results, weights=(1.0, 1.0), normalization="invalid"
            )


# ── Config Classes ───────────────────────────────────────────────────────────


class TestConfigClasses:
    def test_rrf_config(self):
        config = rankops.RrfConfig(k=100)
        assert config.k == 100

    def test_fusion_config(self):
        config = rankops.FusionConfig()
        # with_top_k returns a new config (builder pattern)
        config_with_top_k = config.with_top_k(10)
        assert config_with_top_k is not None

    def test_weighted_config(self):
        config = rankops.WeightedConfig(weight_a=0.7, weight_b=0.3)
        assert config.weight_a == pytest.approx(0.7, abs=1e-5)
        assert config.weight_b == pytest.approx(0.3, abs=1e-5)

    def test_standardized_config(self):
        config = rankops.StandardizedConfig(clip_range=(-2.0, 2.0))
        lo, hi = config.clip_range
        assert lo == pytest.approx(-2.0, abs=1e-5)
        assert hi == pytest.approx(2.0, abs=1e-5)

    def test_additive_multi_task_config(self):
        config = rankops.AdditiveMultiTaskConfig(weights=(1.0, 1.0))
        assert config.weights == pytest.approx((1.0, 1.0), abs=1e-5)


# ── Explainability ───────────────────────────────────────────────────────────


class TestExplainability:
    @pytest.fixture
    def retriever_ids(self):
        return ["BM25", "Dense", "Keyword"]

    def test_rrf_explain(self, multi_lists, retriever_ids):
        result = rankops.rrf_explain(multi_lists, retriever_ids, k=60)
        assert len(result) > 0
        item = result[0]
        assert hasattr(item, "id")
        assert hasattr(item, "score")
        assert hasattr(item, "rank")
        assert hasattr(item, "explanation")
        assert len(item.explanation.sources) > 0

    def test_rrf_explain_top_k(self, multi_lists, retriever_ids):
        result = rankops.rrf_explain(multi_lists, retriever_ids, k=60, top_k=2)
        assert len(result) == 2

    def test_combsum_explain(self, multi_lists, retriever_ids):
        result = rankops.combsum_explain(multi_lists, retriever_ids)
        assert len(result) > 0

    def test_combmnz_explain(self, multi_lists, retriever_ids):
        result = rankops.combmnz_explain(multi_lists, retriever_ids)
        assert len(result) > 0

    def test_dbsf_explain(self, multi_lists, retriever_ids):
        result = rankops.dbsf_explain(multi_lists, retriever_ids)
        assert len(result) > 0

    def test_explain_source_fields(self, multi_lists, retriever_ids):
        result = rankops.rrf_explain(multi_lists, retriever_ids)
        for item in result:
            for src in item.explanation.sources:
                assert hasattr(src, "retriever_id")
                assert hasattr(src, "original_rank")
                assert hasattr(src, "contribution")

    def test_explain_mismatched_ids_raises(self, multi_lists):
        with pytest.raises(ValueError, match="same length"):
            rankops.rrf_explain(multi_lists, ["only_one"])


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_validate_sorted_pass(self):
        result = rankops.validate_sorted([("a", 3.0), ("b", 2.0), ("c", 1.0)])
        assert result.is_valid

    def test_validate_sorted_fail(self):
        result = rankops.validate_sorted([("a", 1.0), ("b", 3.0)])
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_no_duplicates_pass(self):
        result = rankops.validate_no_duplicates([("a", 1.0), ("b", 2.0)])
        assert result.is_valid

    def test_validate_no_duplicates_fail(self):
        result = rankops.validate_no_duplicates([("a", 1.0), ("a", 2.0)])
        assert not result.is_valid

    def test_validate_finite_scores_pass(self):
        result = rankops.validate_finite_scores([("a", 1.0), ("b", 2.0)])
        assert result.is_valid

    def test_validate_comprehensive(self, bm25_results, dense_results):
        fused = rankops.rrf(bm25_results, dense_results)
        result = rankops.validate(fused)
        assert result.is_valid

    def test_validate_bounds(self):
        data = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        result = rankops.validate_bounds(data, max_results=2)
        assert len(result.warnings) > 0


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_invalid_input_type(self):
        with pytest.raises((TypeError, ValueError)):
            rankops.rrf("not a list", [("doc", 1.0)])

    def test_malformed_tuples(self):
        with pytest.raises((TypeError, ValueError)):
            rankops.rrf([("doc",)], [("doc", 1.0)])

    def test_negative_scores(self):
        result = rankops.rrf([("doc", -1.0)], [("doc", -0.5)])
        assert len(result) == 1

    def test_very_large_k(self):
        result = rankops.rrf(
            [("a", 12.5), ("b", 11.0)], [("b", 0.9), ("a", 0.8)], k=10000
        )
        assert len(result) == 2

    def test_single_item_lists(self):
        result = rankops.rrf([("a", 1.0)], [("b", 0.9)])
        assert len(result) == 2

    def test_identical_lists(self):
        lst = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        result = rankops.rrf(lst, lst)
        assert len(result) == 3
        assert _is_sorted_desc(result)

    def test_large_list(self):
        big = [(f"doc_{i}", float(1000 - i)) for i in range(1000)]
        result = rankops.rrf(big, big, top_k=10)
        assert len(result) == 10
        assert _is_sorted_desc(result)

    def test_all_same_score(self):
        lst = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        result = rankops.rrf(lst, lst)
        assert len(result) == 3

    def test_unicode_ids(self):
        a = [("cafe\u0301", 1.0), ("nai\u0308ve", 0.5)]
        b = [("cafe\u0301", 0.8)]
        result = rankops.rrf(a, b)
        assert any("caf" in r[0] for r in result)


# ── Cross-method consistency ─────────────────────────────────────────────────


class TestCrossMethod:
    """Verify that all fusion methods produce valid output on the same input."""

    def test_all_methods_produce_valid_output(self, bm25_results, dense_results):
        methods = [
            lambda: rankops.rrf(bm25_results, dense_results),
            lambda: rankops.isr(bm25_results, dense_results),
            lambda: rankops.combsum(bm25_results, dense_results),
            lambda: rankops.combmnz(bm25_results, dense_results),
            lambda: rankops.borda(bm25_results, dense_results),
            lambda: rankops.dbsf(bm25_results, dense_results),
            lambda: rankops.weighted(
                bm25_results, dense_results, weight_a=0.5, weight_b=0.5
            ),
            lambda: rankops.standardized(bm25_results, dense_results),
            lambda: rankops.additive_multi_task(bm25_results, dense_results),
        ]
        for method in methods:
            result = method()
            assert len(result) > 0, f"empty result from {method}"
            assert _is_sorted_desc(result), f"not sorted from {method}"
            assert _no_duplicate_ids(result), f"duplicate ids from {method}"
            assert all(math.isfinite(s) for _, s in result), (
                f"non-finite score from {method}"
            )

    def test_all_multi_methods(self, multi_lists):
        methods = [
            lambda: rankops.rrf_multi(multi_lists),
            lambda: rankops.isr_multi(multi_lists),
            lambda: rankops.combsum_multi(multi_lists),
            lambda: rankops.combmnz_multi(multi_lists),
            lambda: rankops.borda_multi(multi_lists),
            lambda: rankops.dbsf_multi(multi_lists),
            lambda: rankops.standardized_multi(multi_lists),
        ]
        for method in methods:
            result = method()
            assert len(result) > 0
            assert _is_sorted_desc(result)
            assert _no_duplicate_ids(result)

    def test_top_k_respected_everywhere(self, bm25_results, dense_results):
        top_k = 2
        methods = [
            lambda: rankops.rrf(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.isr(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.combsum(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.combmnz(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.borda(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.dbsf(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.weighted(
                bm25_results, dense_results, weight_a=0.5, weight_b=0.5, top_k=top_k
            ),
            lambda: rankops.standardized(bm25_results, dense_results, top_k=top_k),
            lambda: rankops.additive_multi_task(
                bm25_results, dense_results, top_k=top_k
            ),
        ]
        for method in methods:
            result = method()
            assert len(result) <= top_k
