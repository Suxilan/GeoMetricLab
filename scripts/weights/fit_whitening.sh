#!/usr/bin/env bash
# Preview compatible caches first; fitting is an explicit separate invocation.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ $# == 0 || "$1" == --help ]]; then
    echo 'Usage: bash scripts/weights/fit_whitening.sh {preview|fit} [whitening tool options]'
    echo 'FEATURE_LIST overrides config/whitening/example.yaml. Record paths in it are YAML-relative.'
    exit 0
fi
case "$1" in
    preview) mode=--preview ;;
    fit) mode=--fit ;;
    *) echo "Expected preview or fit: $1" >&2; exit 2 ;;
esac
shift
run "$python_bin" tools/weights/assemble_whitening_model.py \
    --feature-list "${FEATURE_LIST:-config/whitening/example.yaml}" "$mode" "$@"
