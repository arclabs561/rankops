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


def test_evaluate_trec_detailed_includes_per_query_diagnostics():
    result = rankops.evaluate_trec_detailed(
        "q1 0 d1 1\nq2 0 d2 1\n",
        "q1 Q0 d1 1 1.0 rankops\n",
        10,
    )

    assert result["num_queries"] == 2
    assert result["queries"] == [
        {
            "query_id": "q1",
            "num_retrieved": 1,
            "num_relevant": 1,
            "ndcg_at_k": 1.0,
            "average_precision": 1.0,
            "reciprocal_rank": 1.0,
            "recall_at_k": 1.0,
            "precision_at_k": 0.1,
        },
        {
            "query_id": "q2",
            "num_retrieved": 0,
            "num_relevant": 1,
            "ndcg_at_k": 0.0,
            "average_precision": 0.0,
            "reciprocal_rank": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
        },
    ]
