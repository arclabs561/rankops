#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trec_eval_bin="${TREC_EVAL_BIN:-trec_eval}"

if ! command -v "$trec_eval_bin" >/dev/null 2>&1; then
    printf 'trec_eval is required; set TREC_EVAL_BIN or install trec_eval.\n' >&2
    exit 2
fi

cd "$repo_root"
RANKOPS_TREC_EVAL="$(command -v "$trec_eval_bin")" \
    cargo test --test trec_parity optional_reference_binary_matches_corpus
