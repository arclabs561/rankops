"""Tests for the explicit-contract options: id tie-break, weighted RRF, MMR over a similarity matrix."""

import random

import pytest
import rankops


def _ids(results):
    return [item[0] for item in results]


class TestTieBreak:
    def test_id_orders_equal_scores_ascending(self):
        # "b" and "a" have identical fused scores: each is rank 1 in one list.
        lists = [[("b", 1.0)], [("a", 1.0)]]
        assert _ids(rankops.rrf_multi(lists, tie_break="id")) == ["a", "b"]

    def test_id_is_applied_before_top_k(self):
        # Three tied ids; the cut must keep the smallest, not an arbitrary one.
        lists = [[("c", 1.0)], [("a", 1.0)], [("b", 1.0)]]
        assert _ids(rankops.rrf_multi(lists, top_k=2, tie_break="id")) == ["a", "b"]

    def test_default_is_unchanged(self):
        lists = [[("a", 1.0), ("b", 0.5)], [("b", 1.0), ("c", 0.5)]]
        assert rankops.rrf_multi(lists) == rankops.rrf_multi(lists, tie_break=None)

    def test_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="tie_break"):
            rankops.rrf_multi([[("a", 1.0)]], tie_break="hash")


class TestRrfWeighted:
    def test_scores_follow_the_formula(self):
        k = 60
        lists = [[("x", 0.0), ("y", 0.0)], [("y", 0.0), ("z", 0.0)]]
        out = dict(rankops.rrf_weighted(lists, [0.25, 0.75], k=k))
        # Weights are normalized by the core, so compare ratios, not raw values.
        raw = {
            "x": 0.25 / (k + 1),
            "y": 0.25 / (k + 2) + 0.75 / (k + 1),
            "z": 0.75 / (k + 2),
        }
        assert out["y"] / out["x"] == pytest.approx(raw["y"] / raw["x"], rel=1e-4)
        assert out["z"] / out["x"] == pytest.approx(raw["z"] / raw["x"], rel=1e-4)

    def test_equal_weights_match_unweighted_order(self):
        lists = [[("a", 0.0), ("b", 0.0), ("c", 0.0)], [("c", 0.0), ("b", 0.0)]]
        weighted = rankops.rrf_weighted(lists, [1.0, 1.0], tie_break="id")
        plain = rankops.rrf_multi(lists, tie_break="id")
        assert _ids(weighted) == _ids(plain)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rankops.rrf_weighted([[("a", 1.0)], [("b", 1.0)]], [1.0])

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError):
            rankops.rrf_weighted([[("a", 1.0)]], [0.0])


class TestMmrSimilarity:
    CANDIDATES = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    SIMILARITY = [
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]

    def test_prefers_the_diverse_second_pick(self):
        out = rankops.mmr_similarity(self.CANDIDATES, self.SIMILARITY, lambda_=0.5, k=2)
        assert _ids(out) == ["doc1", "doc3"]

    def test_pure_relevance_ignores_similarity(self):
        out = rankops.mmr_similarity(self.CANDIDATES, self.SIMILARITY, lambda_=1.0, k=3)
        assert _ids(out) == ["doc1", "doc2", "doc3"]

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="3 x 3"):
            rankops.mmr_similarity(self.CANDIDATES, [[1.0, 0.0], [0.0, 1.0]])

    def test_matches_the_unnormalized_penalty_form(self):
        # For relevance already in [0, 1] and spanning that range, selecting by
        # `rel - p * max_sim` equals selecting by `lam * rel - (1 - lam) * max_sim`
        # with p = (1 - lam) / lam, because the two scores differ by the positive
        # factor lam. This is the reparametrization needed to move a
        # `relevance - penalty * similarity` caller onto this function.
        rng = random.Random(7)
        for _ in range(50):
            n = 8
            rel = [rng.random() for _ in range(n)]
            rel[0], rel[1] = 0.0, 1.0  # pin the min-max range to [0, 1]
            sim = [[0.0] * n for _ in range(n)]
            for i in range(n):
                sim[i][i] = 1.0
                for j in range(i):
                    sim[i][j] = sim[j][i] = rng.random()
            lam = rng.choice([0.3, 0.5, 0.7, 0.9])
            penalty = (1 - lam) / lam
            expected = self._penalty_form(rel, sim, penalty, k=4)
            cands = [(str(i), r) for i, r in enumerate(rel)]
            got = _ids(rankops.mmr_similarity(cands, sim, lambda_=lam, k=4))
            assert got == [str(i) for i in expected]

    @staticmethod
    def _penalty_form(rel, sim, penalty, k):
        selected, remaining = [], list(range(len(rel)))
        while len(selected) < k and remaining:
            def score(c):
                max_sim = max((sim[c][s] for s in selected), default=0.0)
                return rel[c] - penalty * max_sim
            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
        return selected
