"""TREC evaluation tests for the Python binding."""

import pytest
import rankops


QRELS = "q1 0 d1 2\nq1 0 d2 0\n"
RUN = "q1 Q0 d1 1 10.0 rankops\nq1 Q0 d2 2 5.0 rankops\n"


def test_evaluate_trec_returns_collection_metrics():
    result = rankops.evaluate_trec(QRELS, RUN, 2)

    assert result["num_queries"] == 1
    assert result["k"] == 2
    assert result["ndcg_at_k"] == pytest.approx(1.0)
    assert result["map"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["precision_at_k"] == pytest.approx(0.5)


def test_evaluate_trec_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="score must be a finite number"):
        rankops.evaluate_trec(QRELS, "q1 Q0 d1 1 NaN rankops\n", 10)
