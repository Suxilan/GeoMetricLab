#!/usr/bin/env bash
# Restore the two-dataset extraction example using one maintained producer.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/features/run_whitening_feature_extraction.sh [extraction tool options]'
    echo 'EXTRACTION_CONFIG selects the same pre-whitening model for both datasets.'
    echo 'GL3D_ROOT, U1652_ROOT, FEATURE_DEVICE, FEATURE_BATCH_SIZE and FEATURE_WORKERS are optional.'
    echo 'Writes results/extraction/{gl3d,u1652}_raw.json; does not fit whitening.'
    exit 0
fi
for dataset in gl3d u1652; do
    case "$dataset" in
        gl3d) input_root="${GL3D_ROOT:-data/GL3D/train}" ;;
        u1652) input_root="${U1652_ROOT:-data/university1652/train}" ;;
    esac
    run "$python_bin" tools/features/extract_image_features.py \
        --config "${EXTRACTION_CONFIG:-config/extraction/peft_dinov2_scpp_gl3d_pretrain.yaml}" \
        --input-root "$input_root" --data-name "${dataset}_train" \
        --device "${FEATURE_DEVICE:-cuda:0}" \
        --batch-size "${FEATURE_BATCH_SIZE:-4}" --num-workers "${FEATURE_WORKERS:-4}" \
        --record-output "results/extraction/${dataset}_raw.json" "$@"
done
