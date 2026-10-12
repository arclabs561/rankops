//! Adaptive MMR must retain the original candidate-to-matrix mapping.
use rankops::rerank::diversity::tuning::mmr_adaptive;

#[test]
fn adaptive_mmr_selects_two_candidates_without_losing_matrix_shape() {
    let candidates = [("first", 1.0), ("second", 0.5)];
    let similarity = [1.0, 0.2, 0.2, 1.0];
    assert_eq!(
        mmr_adaptive(&candidates, &similarity, 0.9, 0.3, 2),
        candidates
    );
}

#[test]
fn adaptive_mmr_keeps_original_indices_and_penalizes_previous_selections() {
    let candidates = [
        ("low", 0.5),
        ("best", 1.0),
        ("duplicate", 0.9),
        ("diverse", 0.8),
    ];
    let similarity = [
        1.0, 0.2, 0.1, 0.2, 0.2, 1.0, 0.99, 0.1, 0.1, 0.99, 1.0, 0.1, 0.2, 0.1, 0.1, 1.0,
    ];
    assert_eq!(
        mmr_adaptive(&candidates, &similarity, 0.9, 0.0, 3),
        [("best", 1.0), ("diverse", 0.8), ("low", 0.5)]
    );
    assert_eq!(
        mmr_adaptive(&candidates, &similarity, 1.0, 1.0, 3),
        [("best", 1.0), ("duplicate", 0.9), ("diverse", 0.8)]
    );
}

#[test]
fn adaptive_mmr_preserves_empty_limits_and_duplicate_id_semantics() {
    let candidates = [("same", 1.0), ("same", 0.9), ("other", 0.5)];
    let similarity = [1.0; 9];
    assert!(mmr_adaptive(&candidates, &[], 0.9, 0.3, 0).is_empty());
    assert!(mmr_adaptive::<&str>(&[], &[], 0.9, 0.3, 2).is_empty());
    assert_eq!(
        mmr_adaptive(&candidates, &similarity, 0.9, 0.3, 1),
        [candidates[0]]
    );
    assert_eq!(
        mmr_adaptive(&candidates, &similarity, 0.9, 0.3, 5),
        [candidates[0], candidates[2]]
    );
}
