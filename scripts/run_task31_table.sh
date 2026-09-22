#!/usr/bin/env bash
set -euo pipefail

CONFIG="/src/config/config_model.yaml"
SPLIT="random"
SEED="23"
VALIDATE_EVERY="100"

MODELS=(
  cnn
  lstm
  jepa_rollout
  jepa_direct_mse
  jepa_direct
)

for model in "${MODELS[@]}"; do
  uv run python run_training.py \
    --task task_3-1 \
    --config "${CONFIG}" \
    --model "${model}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --validate_every "${VALIDATE_EVERY}"

  uv run python run_evaluation.py \
    --task task_3-1 \
    --config "${CONFIG}" \
    --model "${model}" \
    --split "${SPLIT}" \
    --seed "${SEED}"
done

uv run python scripts/summarize_task31.py \
  --split "${SPLIT}" \
  --seed "${SEED}"

