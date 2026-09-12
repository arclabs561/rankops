use std::cmp::Ordering;
use std::hash::Hash;

use crate::FusionMethod;

impl FusionMethod {
    /// Fuse two ranked lists, ordering equal fused scores with `compare_ids`.
    ///
    /// `compare_ids` must define a total order. It is applied after
    /// [`f32::total_cmp`] and before a caller takes a prefix of the returned
    /// results. This leaves [`Self::fuse`] and its identifier-hash tie behavior
    /// unchanged.
    ///
    /// ```
    /// use rankops::FusionMethod;
    ///
    /// let a = [("b", 1.0)];
    /// let b = [("a", 1.0)];
    /// let fused = FusionMethod::CombSum.fuse_by(&a, &b, Ord::cmp);
    ///
    /// assert_eq!(fused[0].0, "a");
    /// ```
    #[must_use]
    pub fn fuse_by<I, F>(&self, a: &[(I, f32)], b: &[(I, f32)], compare_ids: F) -> Vec<(I, f32)>
    where
        I: Clone + Eq + Hash,
        F: Fn(&I, &I) -> Ordering,
    {
        let mut results = self.fuse(a, b);
        sort_scored_desc_by(&mut results, &compare_ids);
        results
    }

    /// Fuse multiple ranked lists, ordering equal fused scores with `compare_ids`.
    ///
    /// `compare_ids` must define a total order. It is applied after
    /// [`f32::total_cmp`] and before a caller takes a prefix of the returned
    /// results. This leaves [`Self::fuse_multi`] and its identifier-hash tie
    /// behavior unchanged.
    #[must_use]
    pub fn fuse_multi_by<I, L, F>(&self, lists: &[L], compare_ids: F) -> Vec<(I, f32)>
    where
        I: Clone + Eq + Hash,
        L: AsRef<[(I, f32)]>,
        F: Fn(&I, &I) -> Ordering,
    {
        let mut results = self.fuse_multi(lists);
        sort_scored_desc_by(&mut results, &compare_ids);
        results
    }
}

#[inline]
fn sort_scored_desc_by<I, F>(results: &mut [(I, f32)], compare_ids: &F)
where
    F: Fn(&I, &I) -> Ordering,
{
    results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| compare_ids(&a.0, &b.0)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct CollidingId(&'static str);

    impl Hash for CollidingId {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            state.write_u8(0);
        }
    }

    #[test]
    fn fusion_tie_comparator_orders_hash_collisions_before_cutoff() {
        let a = vec![(CollidingId("b"), 1.0)];
        let b = vec![(CollidingId("a"), 1.0)];
        let fused = FusionMethod::CombSum.fuse_by(&a, &b, CollidingId::cmp);

        assert_eq!(
            fused.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            fused
                .into_iter()
                .take(1)
                .map(|(id, _)| id.0)
                .collect::<Vec<_>>(),
            ["a"]
        );
    }

    #[test]
    fn multi_fusion_tie_comparator_orders_hash_collisions() {
        let lists = [vec![(CollidingId("b"), 1.0)], vec![(CollidingId("a"), 1.0)]];
        let fused = FusionMethod::CombSum.fuse_multi_by(&lists, CollidingId::cmp);

        assert_eq!(
            fused.into_iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}
