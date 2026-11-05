#!/usr/bin/env bash
# Query/gallery records may come from any compatible live-model extraction.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ "${1:-}" == --help || $# == 0 ]]; then
    echo 'Usage: bash scripts/retrieval/rank_cached_features.sh QUERY_RECORD GALLERY_RECORD OUTPUT_JSONL [retrieval tool options]'
    echo 'Optional --config YOUR.yaml selects search and reranking; TOPK defaults to 20.'
    exit 0
fi
if (( $# < 3 )); then
    echo 'Expected query record, gallery record and output JSONL' >&2
    exit 2
fi
query_record="$1"
gallery_record="$2"
output="$3"
shift 3
run "$python_bin" tools/retrieval/retrieve.py \
    --query-record "$query_record" --gallery-record "$gallery_record" \
    --topk "${TOPK:-20}" --output "$output" "$@"
