# Changelog

## [0.1.10] - 2026-06-13

### Added
- `trec` module: `parse_qrels` / `parse_run` read TREC-format qrels and run files, and `evaluate` reports collection-level mean metrics (nDCG@k, MAP, MRR, recall@k, precision@k) over all judged queries, the BEIR / `trec_eval` / ranx workflow. New `trec_eval` example.

## [0.1.8] - 2026-06-10

### Changed
- Bumped `innr` to 0.4 (re-exported SIMD fns in `rerank::simd` unchanged in signature).
- rankops-wasm marked standalone workspace to fix CI; README and CONTRIBUTING polish.

Earlier releases predate this changelog; see git history.
