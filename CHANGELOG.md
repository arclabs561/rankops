# Changelog

## Source history (release attribution not established)

These source changes are listed without assigning them to a published release.

- Adaptive MMR preserves original candidate indices and diversity penalties from all earlier selections, avoiding the second-selection panic while retaining original relevance scores.
  ([source change](https://github.com/arclabs561/rankops/commit/cd0f282dd75008b4449ebbcb8e9cf2c6a29dcb4b))

## [Unreleased]

### Changed
- RRF (`rrf`, `rrf_with_config`, `rrf_multi`, `rrf_weighted`, `rrf_explain`, `FusionMethod::Rrf`) counts a document repeated within one list once, at its first (best) rank, as in Cormack et al. 2009. It previously added a contribution per occurrence; fused scores change for inputs with repeated ids.
- `k == 0` follows one documented rule (see `RrfConfig::k`): `RrfConfig::new` and `with_k` panic, infallible fusion functions return an empty list, and `rrf_weighted` returns `FusionError::InvalidConfig`. `rrf_explain` and `rrf_weighted` previously accepted `k == 0` and returned scores.

### Fixed
- `Pipeline` docs say that a pipeline without `.fuse(..)` uses `FusionMethod::default()` (RRF, k = 60), matching `execute`; they said it returned an empty list.

## [0.3.0] - 2026-10-30

- `adapt::from_distances` and `from_distances_mapped` keep negative distances
  in best-first order. They mapped every distance through `1 / (1 + d)`,
  which reverses order across `d = -1` and divides by zero at it, so
  vicinity's inner-product distances (`-dot`) came back scrambled and an
  exact unit-vector match scored infinity, then NaN after CombSUM. Negative
  distances now map to `1 - d`, which meets `1 / (1 + d)` at zero with the
  same slope; scores for non-negative distances are unchanged.
- `dbsf`, `dbsf_with_config`, `dbsf_multi` and `dbsf_explain` map each list to
  `(s - (mu - 3 sigma)) / (6 sigma)` clipped to [0, 1] before summing, as in
  Mazzeschi's DBSF, so a document missing from a list counts as that list's
  floor rather than its mean. Fused scores and orderings change; the doc
  example now ranks d2, d3, d1, d4.
- `dp_topk::dp_knapsack` returns the correct selection marginals. The backward
  pass propagated the "pick" transition the wrong way, so marginals could
  exceed one in total under a unit-capacity constraint.
- Hybrid-search example now reports fixture-specific metrics and includes a
  synthetic two-query comparison of best-single, qrels-informed per-query
  oracle, and RRF scores. No library API changed.
- Evaluation helpers now use linear-gain nDCG, pad P@k for short runs, and use the full relevant set as the MAP@k denominator; `pipeline::compare` evaluates the exact requested metric/cutoff.
- Fusion output has a deterministic fallback order for equal scores.

### Changed
- RRF (`rrf`, `rrf_multi`, `rrf_weighted`, `rrf_explain`, `FusionMethod::Rrf`) uses 1-based ranks as in Cormack et al. 2009, so the top document scores `1/(k+1)`; fused scores change for every caller.
- Tie ordering hashes ids with FNV-1a instead of std's unspecified `DefaultHasher`, so it no longer depends on the Rust version.
- `trec` now stores validated qrels and runs in opaque, deterministic containers; add `TrecQrels::from_records` and `TrecRun::from_records` for in-memory input.
- Add `TrecEvalConfig` and `evaluate_with_config` for relevance thresholds, qrels-only-query handling, and per-query result limits. TREC calculations now use `f64` scores and aggregates.

### Added
- TREC parity fixtures and an opt-in `scripts/verify-trec-parity.sh` comparison against a locally installed `trec_eval`.

## [0.2.0] - 2026-07-09

### Added
- `rerank::colbert::rank_flat` and `maxsim_with_top_k_flat` for row-major token-vector buffers, including borrowed document buffers without nested token-vector allocation.

### Changed
- Marked public error enums as `#[non_exhaustive]`; downstream exhaustive matches need a wildcard arm.

## [0.1.10] - 2026-06-13

### Added
- `trec` module: `parse_qrels` / `parse_run` read TREC-format qrels and run files, and `evaluate` reports collection-level mean metrics (nDCG@k, MAP, MRR, recall@k, precision@k) over all judged queries, the BEIR / `trec_eval` / ranx workflow. New `trec_eval` example.

## [0.1.8] - 2026-06-10

### Changed
- Bumped `innr` to 0.4 (re-exported SIMD fns in `rerank::simd` unchanged in signature).
- rankops-wasm marked standalone workspace to fix CI; README and CONTRIBUTING polish.

Earlier releases predate this changelog; see git history.
