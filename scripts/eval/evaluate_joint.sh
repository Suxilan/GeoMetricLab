#!/usr/bin/env bash
# Evaluate independent protocols already declared by the joint recipe.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ $# == 0 || "$1" == --help ]]; then
    echo 'Usage: bash scripts/eval/evaluate_joint.sh RUN_DIR [gmlab.evaluate options]'
    echo 'Evaluates both best and last checkpoints. Optional TRAIN_CONFIG overrides the recipe.'
    exit 0
fi
run_dir="$1"
shift
for selection in best last; do
    run "$python_bin" -m gmlab.evaluate \
        --config "${TRAIN_CONFIG:-config/gmlab/joint/gl3d_u1652_joint.yaml}" \
        --run-dir "$run_dir" --weights "$selection" --split valid \
        --precision 32-true --no-matmul-tf32 --no-cudnn-tf32 \
        --output "$run_dir/evaluation_$selection.json" "$@"
done
