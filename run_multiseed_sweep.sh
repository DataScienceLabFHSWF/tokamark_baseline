#!/bin/bash
# Multi-seed experiment sweep for PLUME Task 3-1
# Runs training and evaluation for all 8 models across 10 seeds (23-32)
# Seeds: 23, 24, 25, 26, 27, 28, 29, 30, 31, 32

set -u

# Configuration
TASK="task_3-1"
CONFIG="/src/config/config_model.yaml"
SPLIT="random"
SEEDS=(23 24 25 26 27 28 29 30 31 32)
MODELS=("plume_direct_mse" "plume_vanilla_mse" "plume_direct_jepa" "plume_koopman_mse" "plume_koopman_jepa" "plume_vanilla_jepa" "cnn" "lstm")

# Logging
SWEEP_DIR="results/sweep_multiseed_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="${SWEEP_DIR}/sweep_summary.txt"
mkdir -p "$SWEEP_DIR"

echo "Starting multi-seed sweep at $(date)"
echo "Task: $TASK, Config: $CONFIG, Split: $SPLIT"
echo "Seeds: ${SEEDS[@]}"
echo "Models: ${MODELS[@]}"
echo "Results directory: $SWEEP_DIR"
echo ""
tee -a "$SUMMARY_FILE" <<< "Multi-seed sweep started at $(date)"
tee -a "$SUMMARY_FILE" <<< "Task: $TASK, Config: $CONFIG, Split: $SPLIT"
tee -a "$SUMMARY_FILE" <<< "Seeds: ${SEEDS[@]}"
tee -a "$SUMMARY_FILE" <<< "Models: ${MODELS[@]}"
tee -a "$SUMMARY_FILE" <<< ""

# Track progress
TOTAL_JOBS=$((${#SEEDS[@]} * ${#MODELS[@]}))
COMPLETED=0
FAILED=0
FAILED_JOBS=""

# Run training and evaluation for each model and seed
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    JOB_START=$(date +%s)
    
    echo ""
    echo "=========================================="
    echo "Job $COMPLETED/$TOTAL_JOBS: $model (seed=$seed)"
    echo "Start: $(date)"
    echo "=========================================="
    tee -a "$SUMMARY_FILE" <<< ""
    tee -a "$SUMMARY_FILE" <<< "Job $COMPLETED/$TOTAL_JOBS: $model (seed=$seed)"
    tee -a "$SUMMARY_FILE" <<< "Start: $(date)"
    
    # Training phase
    echo "[TRAIN] Running training for $model with seed $seed..."
    if timeout 600 uv run python run_training.py \
      --task "$TASK" \
      --config "$CONFIG" \
      --model "$model" \
      --split "$SPLIT" \
      --seed "$seed" \
      2>&1 | tee -a "${SWEEP_DIR}/${model}_seed${seed}_train.log"; then
      TRAIN_EXIT=0
      echo "[TRAIN] ✓ Training succeeded for $model (seed=$seed)"
      tee -a "$SUMMARY_FILE" <<< "[TRAIN] ✓ Training succeeded for $model (seed=$seed)"
    else
      TRAIN_EXIT=$?
      FAILED=$((FAILED + 1))
      FAILED_JOBS="${FAILED_JOBS}\n  $model (seed=$seed) - training failed (exit=$TRAIN_EXIT)"
      echo "[TRAIN] ✗ Training failed for $model (seed=$seed) with exit code $TRAIN_EXIT"
      tee -a "$SUMMARY_FILE" <<< "[TRAIN] ✗ Training failed for $model (seed=$seed) with exit code $TRAIN_EXIT"
      JOB_END=$(date +%s)
      JOB_DURATION=$((JOB_END - JOB_START))
      echo "Duration: ${JOB_DURATION}s"
      tee -a "$SUMMARY_FILE" <<< "Duration: ${JOB_DURATION}s"
      continue
    fi
    
    # Evaluation phase
    echo "[EVAL] Running evaluation for $model with seed $seed..."
    if timeout 300 uv run python run_evaluation.py \
      --task "$TASK" \
      --config "$CONFIG" \
      --model "$model" \
      --split "$SPLIT" \
      --seed "$seed" \
      2>&1 | tee -a "${SWEEP_DIR}/${model}_seed${seed}_eval.log"; then
      EVAL_EXIT=0
      echo "[EVAL] ✓ Evaluation succeeded for $model (seed=$seed)"
      tee -a "$SUMMARY_FILE" <<< "[EVAL] ✓ Evaluation succeeded for $model (seed=$seed)"
    else
      EVAL_EXIT=$?
      FAILED=$((FAILED + 1))
      FAILED_JOBS="${FAILED_JOBS}\n  $model (seed=$seed) - eval failed (exit=$EVAL_EXIT)"
      echo "[EVAL] ✗ Evaluation failed for $model (seed=$seed) with exit code $EVAL_EXIT"
      tee -a "$SUMMARY_FILE" <<< "[EVAL] ✗ Evaluation failed for $model (seed=$seed) with exit code $EVAL_EXIT"
    fi
    
    JOB_END=$(date +%s)
    JOB_DURATION=$((JOB_END - JOB_START))
    echo "Duration: ${JOB_DURATION}s"
    tee -a "$SUMMARY_FILE" <<< "Duration: ${JOB_DURATION}s"
  done
done

echo ""
echo "=========================================="
echo "Sweep completed at $(date)"
echo "Completed jobs: $((TOTAL_JOBS - FAILED))/$TOTAL_JOBS"
echo "Failed jobs: $FAILED"
if [[ -n "$FAILED_JOBS" ]]; then
  echo -e "Failed job details:$FAILED_JOBS"
fi
echo "Results directory: $SWEEP_DIR"
echo "=========================================="
tee -a "$SUMMARY_FILE" <<< ""
tee -a "$SUMMARY_FILE" <<< "Sweep completed at $(date)"
tee -a "$SUMMARY_FILE" <<< "Completed jobs: $((TOTAL_JOBS - FAILED))/$TOTAL_JOBS"
tee -a "$SUMMARY_FILE" <<< "Failed jobs: $FAILED"
if [[ -n "$FAILED_JOBS" ]]; then
  tee -a "$SUMMARY_FILE" <<< -e "Failed job details:$FAILED_JOBS"
fi

exit $([[ $FAILED -eq 0 ]] && echo 0 || echo 1)
