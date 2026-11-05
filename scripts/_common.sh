#!/usr/bin/env bash
# Small helpers for editable checkout examples; stable behavior lives in tools/gmlab.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
