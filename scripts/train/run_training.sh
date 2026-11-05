#!/usr/bin/env bash
# Replace a preset or edit its YAML to define a downstream experiment.
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
if [[ $# == 0 || "$1" == --help ]]; then
    echo 'Usage: bash scripts/train/run_training.sh {metric|relation|multimodal|joint|joint_multimodal} [gmlab.train options]'
    echo 'Example: DRY_RUN=1 bash scripts/train/run_training.sh joint --dev 2'
    exit 0
fi
case "$1" in
    metric) recipe=config/gmlab/metric/u1652_labels.yaml ;;
    relation) recipe=config/gmlab/relation/gl3d_supscene_peft_dinov2_scpp.yaml ;;
    multimodal) recipe=config/gmlab/multimodal/u1652_multimodal.yaml ;;
    joint) recipe=config/gmlab/joint/gl3d_u1652_joint.yaml ;;
    joint_multimodal) recipe=config/gmlab/joint/multisource_multimodal.yaml ;;
    *) echo "Unknown training preset: $1" >&2; exit 2 ;;
esac
shift
run "$python_bin" -m gmlab.train --config "$recipe" "$@"
