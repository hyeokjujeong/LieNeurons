#!/usr/bin/env bash
# Run §8.3--§8.5 of pointwise_pipeline.md.
#
# Activate the `lieneurons` conda environment before invoking this script:
#   conda activate lieneurons
#   bash experiment/pc_se3_congruence/run_pointwise_gpu_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda}"
WANDB_MODE="${WANDB_MODE:-online}"
VERIFY_OUT="${VERIFY_OUT:-experiment/pc_se3_congruence/pointwise_verify_results.json}"

COMMON_GRAPH_ARGS=(
    --pw-radius-mode density_scaled
    --pw-radius-alpha 1.15
    --pw-target-k 16
    --pw-candidates 64
)

echo "===== Phase 1/3: structural verification ====="
python experiment/pc_se3_congruence/verify_pointwise.py \
    --device "${DEVICE}" \
    --full \
    --n-points 128 \
    --candidates 64 \
    --radius-mode density_scaled \
    --radius-alpha 1.15 \
    --target-k 16 \
    --out "${VERIFY_OUT}"

echo "===== Phase 2/3: realizability (teacher target) ====="
python experiment/pc_se3_congruence/run_pointwise_suite.py \
    --phase teacher \
    --recipe full \
    --device "${DEVICE}" \
    --wandb-mode "${WANDB_MODE}" \
    "${COMMON_GRAPH_ARGS[@]}"

echo "===== Phase 3/3: analytic target ====="
python experiment/pc_se3_congruence/run_pointwise_suite.py \
    --phase analytic \
    --recipe full \
    --device "${DEVICE}" \
    --wandb-mode "${WANDB_MODE}" \
    "${COMMON_GRAPH_ARGS[@]}"
