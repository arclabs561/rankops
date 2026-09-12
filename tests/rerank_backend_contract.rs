//! Public-contract checks for the reranking SIMD backend.

use rankops::rerank::simd::{cosine, dot, maxsim};

#[test]
fn core_operations_handle_valid_empty_zero_and_nan_inputs() {
    assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    assert!(dot(&[f32::NAN], &[1.0]).is_nan());
    assert_eq!(cosine(&[], &[]), 0.0);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);

    let empty: [&[f32]; 0] = [];
    assert_eq!(maxsim(&empty, &[&[1.0, 0.0]]), 0.0);
    assert_eq!(maxsim(&[&[1.0, 0.0]], &empty), 0.0);
}

#[cfg(feature = "rerank")]
#[test]
#[should_panic(expected = "innr::dot: slice length mismatch")]
fn rerank_backend_rejects_mismatched_dimensions() {
    let _ = dot(&[1.0, 2.0], &[1.0]);
}
