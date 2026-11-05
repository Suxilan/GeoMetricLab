#!/usr/bin/env bash
# Compose feature extraction and pair export; COLMAP matching is a separate step.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ "${1:-}" == --help || $# == 0 ]]; then
    echo 'Usage: bash scripts/retrieval/extract_and_build_pairs.sh IMAGE_ROOT EXTRACTION_CONFIG OUTPUT_DIR [extraction tool options]'
    echo 'Optional FEATURE_DEVICE, FEATURE_BATCH_SIZE, FEATURE_WORKERS, DATA_NAME and TOPK.'
    echo 'Named outputs require FEATURE_STREAM. Optional RERANKING_CONFIG is a method mapping.'
    exit 0
fi
if (( $# < 3 )); then
    echo 'Expected image root, extraction config and output directory' >&2
    exit 2
fi
input_root="$1"
recipe="$2"
output_dir="$3"
shift 3
run "$python_bin" tools/features/extract_image_features.py \
    --config "$recipe" --input-root "$input_root" --data-name "${DATA_NAME:-pair_images}" \
    --device "${FEATURE_DEVICE:-cuda:0}" \
    --batch-size "${FEATURE_BATCH_SIZE:-4}" --num-workers "${FEATURE_WORKERS:-4}" \
    --record-output "$output_dir/features.json" "$@"
pair_options=()
if [[ -n "${FEATURE_STREAM:-}" ]]; then
    pair_options+=(--stream "$FEATURE_STREAM")
fi
if [[ -n "${RERANKING_CONFIG:-}" ]]; then
    pair_options+=(--reranking-config "$RERANKING_CONFIG")
fi
run "$python_bin" tools/retrieval/build_retrieval_pairs.py \
    --root "$input_root" --extraction-record "$output_dir/features.json" \
    --topk "${TOPK:-30}" --output "$output_dir/pairs.txt" "${pair_options[@]}"
