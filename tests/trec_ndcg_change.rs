use rankops::trec::{evaluate_detailed, explain_ndcg_change, parse_qrels, parse_run};

#[test]
fn changes_sum_to_the_evaluator_difference_at_every_cutoff() {
    let qrels = parse_qrels("q 0 a 3\nq 0 b 2\nq 0 c 1\nq 0 d 0\nq 0 neg -1\n".as_bytes()).unwrap();
    // The submitted rank is advisory. At tied scores, z precedes c by document ID.
    let before = parse_run(
        "q Q0 b 9 3 before\nq Q0 c 8 1 before\nq Q0 z 7 1 before\nq Q0 d 6 0 before\n".as_bytes(),
    )
    .unwrap();
    let after = parse_run(
        "q Q0 c 9 5 after\nq Q0 a 8 4 after\nq Q0 neg 7 3 after\nq Q0 b 6 2 after\n".as_bytes(),
    )
    .unwrap();
    for k in 0..=6 {
        let changes = explain_ndcg_change(&before, &after, &qrels, "q", k).unwrap();
        let score = |run| evaluate_detailed(run, &qrels, k).queries[0].ndcg_at_k;
        let delta: f64 = changes.iter().map(|c| c.ndcg_delta).sum();
        assert!((delta - (score(&after) - score(&before))).abs() < 1e-12);
    }
    let changes = explain_ndcg_change(&before, &after, &qrels, "q", 2).unwrap();
    let rows: Vec<_> = changes
        .iter()
        .map(|c| {
            (
                c.document_id.as_str(),
                c.relevance,
                c.before_rank,
                c.after_rank,
            )
        })
        .collect();
    assert_eq!(
        rows,
        [
            ("b", 2, Some(1), Some(4)),
            ("a", 3, None, Some(2)),
            ("c", 1, Some(3), Some(1))
        ]
    );
    assert!(changes[0].ndcg_delta < 0.);
    assert!(changes[1].ndcg_delta > 0.);
    assert_eq!(
        explain_ndcg_change(&before, &before, &qrels, "q", 2),
        Some(vec![])
    );
}

#[test]
fn a_missing_run_query_scores_zero_and_an_unknown_judgment_query_is_undefined() {
    let qrels = parse_qrels("q 0 a 1\nempty-gold 0 a 0\n".as_bytes()).unwrap();
    let before = parse_run("q Q0 a 1 1 tag\nempty-gold Q0 a 1 1 tag\n".as_bytes()).unwrap();
    let after = parse_run("".as_bytes()).unwrap();
    let rows = explain_ndcg_change(&before, &after, &qrels, "q", 10).unwrap();
    assert_eq!(rows[0].ndcg_delta, -1.);
    assert_eq!(rows[0].after_rank, None);
    assert_eq!(
        explain_ndcg_change(&before, &after, &qrels, "empty-gold", 10),
        Some(vec![])
    );
    assert_eq!(
        explain_ndcg_change(&before, &after, &qrels, "other", 10),
        None
    );
}

#[test]
fn equal_gains_can_cancel_while_both_changes_remain_visible() {
    let qrels = parse_qrels("q 0 x 1\nq 0 y 1\n".as_bytes()).unwrap();
    let first = parse_run("q Q0 y 1 1 first\n".as_bytes()).unwrap();
    let second = parse_run("q Q0 x 1 1 second\n".as_bytes()).unwrap();
    let changes = explain_ndcg_change(&first, &second, &qrels, "q", 1).unwrap();
    assert_eq!(
        changes
            .iter()
            .map(|r| (r.document_id.as_str(), r.ndcg_delta))
            .collect::<Vec<_>>(),
        [("x", 1.0), ("y", -1.0)]
    );
}
