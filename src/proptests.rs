//! Property tests for rankops algorithms.
//!
//! These tests verify invariants that should always hold:
//! - Output is sorted descending
//! - No duplicate IDs
//! - Commutativity (order of input lists doesn't matter)
//! - Bounds (output size, score ranges)
//! - Edge cases (empty lists, NaN, Infinity)

#[cfg(test)]
mod tests {
    use super::super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn arb_results(max_len: usize) -> impl Strategy<Value = Vec<(u32, f32)>> {
        proptest::collection::vec((0u32..100, 0.0f32..1.0), 0..max_len)
    }

    proptest! {
        #[test]
        fn rrf_output_bounded(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(&a, &b);
            prop_assert!(result.len() <= a.len() + b.len());
        }

        #[test]
        fn rrf_scores_positive(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(&a, &b);
            for (_, score) in &result {
                prop_assert!(*score > 0.0);
            }
        }

        #[test]
        fn rrf_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = rrf(&a, &b);
            let ba = rrf(&b, &a);

            prop_assert_eq!(ab.len(), ba.len());

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6);
            }
        }

        #[test]
        fn rrf_sorted_descending(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(&a, &b);
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        #[test]
        fn rrf_top_k_respected(a in arb_results(50), b in arb_results(50), k in 1usize..20) {
            let result = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(k));
            prop_assert!(result.len() <= k);
        }

        #[test]
        fn borda_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = borda(&a, &b);
            let ba = borda(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();
            prop_assert_eq!(ab_map, ba_map);
        }

        #[test]
        fn combsum_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combsum(&a, &b);
            let ba = combsum(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).unwrap();
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn combmnz_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combmnz(&a, &b);
            let ba = combmnz(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5,
                    "CombMNZ not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn dbsf_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = dbsf(&a, &b);
            let ba = dbsf(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5,
                    "DBSF not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn isr_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = isr(&a, &b);
            let ba = isr(&b, &a);

            prop_assert_eq!(ab.len(), ba.len());

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6);
            }
        }

        #[test]
        fn isr_sorted_descending(a in arb_results(50), b in arb_results(50)) {
            let result = isr(&a, &b);
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        #[test]
        fn all_methods_sorted_descending(a in arb_results(20), b in arb_results(20)) {
            for result in [
                rrf(&a, &b),
                combsum(&a, &b),
                combmnz(&a, &b),
                combmax(&a, &b),
                combmin(&a, &b),
                combmed(&a, &b),
                combanz(&a, &b),
                borda(&a, &b),
                isr(&a, &b),
                dbsf(&a, &b),
                condorcet(&a, &b),
                copeland(&a, &b),
                median_rank(&a, &b),
                rbc(&a, &b),
            ] {
                for window in result.windows(2) {
                    prop_assert!(
                        window[0].1.total_cmp(&window[1].1) != std::cmp::Ordering::Less,
                        "Not sorted: {} < {}", window[0].1, window[1].1
                    );
                }
            }
        }

        #[test]
        fn all_methods_have_unique_ids(a in arb_results(20), b in arb_results(20)) {
            for result in [
                rrf(&a, &b),
                combsum(&a, &b),
                combmnz(&a, &b),
                combmax(&a, &b),
                combmin(&a, &b),
                combmed(&a, &b),
                combanz(&a, &b),
                borda(&a, &b),
                isr(&a, &b),
                dbsf(&a, &b),
                condorcet(&a, &b),
                copeland(&a, &b),
                median_rank(&a, &b),
                rbc(&a, &b),
            ] {
                let mut seen = std::collections::HashSet::new();
                for (id, _) in &result {
                    prop_assert!(seen.insert(id), "Duplicate ID in output: {:?}", id);
                }
            }
        }

        #[test]
        fn standardized_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = standardized(&a, &b);
            let ba = standardized(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5,
                    "Standardized not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn copeland_commutative(a in arb_results(15), b in arb_results(15)) {
            let ab = copeland(&a, &b);
            let ba = copeland(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6,
                    "Copeland not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn median_rank_commutative(a in arb_results(15), b in arb_results(15)) {
            let ab = median_rank(&a, &b);
            let ba = median_rank(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6,
                    "MedianRank not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn combmax_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combmax(&a, &b);
            let ba = combmax(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn combmin_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combmin(&a, &b);
            let ba = combmin(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn combmed_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combmed(&a, &b);
            let ba = combmed(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn combanz_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combanz(&a, &b);
            let ba = combanz(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn rbc_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = rbc(&a, &b);
            let ba = rbc(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn condorcet_commutative(a in arb_results(15), b in arb_results(15)) {
            let ab = condorcet(&a, &b);
            let ba = condorcet(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6);
            }
        }

        #[test]
        fn fusion_method_dispatch_matches_direct(a in arb_results(15), b in arb_results(15)) {
            // Verify FusionMethod dispatch produces same results as direct calls
            let methods: Vec<(FusionMethod, Vec<(u32, f32)>)> = vec![
                (FusionMethod::rrf(), rrf(&a, &b)),
                (FusionMethod::CombSum, combsum(&a, &b)),
                (FusionMethod::CombMnz, combmnz(&a, &b)),
                (FusionMethod::CombMax, combmax(&a, &b)),
                (FusionMethod::CombMin, combmin(&a, &b)),
                (FusionMethod::CombMed, combmed(&a, &b)),
                (FusionMethod::CombAnz, combanz(&a, &b)),
                (FusionMethod::Borda, borda(&a, &b)),
                (FusionMethod::Dbsf, dbsf(&a, &b)),
            ];

            for (method, direct) in &methods {
                let via_enum = method.fuse(&a, &b);
                let direct_map: HashMap<_, _> = direct.iter().cloned().collect();
                let enum_map: HashMap<_, _> = via_enum.into_iter().collect();

                prop_assert_eq!(direct_map.len(), enum_map.len(),
                    "Length mismatch for {:?}", method);
                for (id, score_d) in &direct_map {
                    let score_e = enum_map.get(id).expect("same keys");
                    prop_assert!((score_d - score_e).abs() < 1e-5,
                        "Score mismatch for {:?}, id {:?}: {} vs {}", method, id, score_d, score_e);
                }
            }
        }

        #[test]
        fn additive_multi_task_sorted_descending(a in arb_results(20), b in arb_results(20)) {
            let config = AdditiveMultiTaskConfig::default();
            let result = additive_multi_task_with_config(&a, &b, config);
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }
    }
}
