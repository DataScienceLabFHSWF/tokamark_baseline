#!/bin/bash
# Run CNN Q.ANT for remaining seeds (24-32)
# Seed 23 was already completed in the nohup session

set -u

# Configuration
TASK="task_3-1"
CONFIG="/src/config/config_model.yaml"
SPLIT="random"
SEEDS=(24 25 26 27 28 29 30 31 32)
MODEL="cnn_qant"

# Logging
RESULTS_DIR="results/random/${MODEL}"
mkdir -p "$RESULTS_DIR"

echo "Starting CNN Q.ANT training for remaining seeds"
echo "Task: $TASK, Config: $CONFIG, Split: $SPLIT"
echo "Seeds: ${SEEDS[@]}"
echo "Model: $MODEL"
echo ""

# Track progress
TOTAL_JOBS=${#SEEDS[@]}
COMPLETED=0
FAILED=0
FAILED_JOBS=""

# Run training and evaluation for each seed
for seed in "${SEEDS[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    JOB_START=$(date +%s)
    
    echo ""
    echo "=========================================="
    echo "Job $COMPLETED/$TOTAL_JOBS: $MODEL (seed=$seed)"
    echo "Start: $(date)"
    echo "=========================================="
    
    # Training phase
    echo "[TRAIN] Running training for $MODEL with seed $seed..."
    if timeout 900 uv run python run_training.py \
      --task "$TASK" \
      --config "$CONFIG" \
      --model "$MODEL" \
      --split "$SPLIT" \
      --seed "$seed" \
      2>&1 | tee "${RESULTS_DIR}/${MODEL}_seed${seed}_train.log"; then
      TRAIN_EXIT=0
      echo "[TRAIN] ✓ Training succeeded for $MODEL (seed=$seed)"
    else
      TRAIN_EXIT=$?
      FAILED=$((FAILED + 1))
      FAILED_JOBS="${FAILED_JOBS}\n  $MODEL (seed=$seed) - training failed (exit=$TRAIN_EXIT)"
      echo "[TRAIN] ✗ Training failed for $MODEL (seed=$seed) with exit code $TRAIN_EXIT"
      JOB_END=$(date +%s)
      JOB_DURATION=$((JOB_END - JOB_START))
      echo "Duration: ${JOB_DURATION}s"
      continue
    fi
    
    # Evaluation phase
    echo "[EVAL] Running evaluation for $MODEL with seed $seed..."
    if timeout 300 uv run python run_evaluation.py \
      --task "$TASK" \
      --config "$CONFIG" \
      --model "$MODEL" \
      --split "$SPLIT" \
      --seed "$seed" \
      2>&1 | tee "${RESULTS_DIR}/${MODEL}_seed${seed}_eval.log"; then
      EVAL_EXIT=0
      echo "[EVAL] ✓ Evaluation succeeded for $MODEL (seed=$seed)"
    else
      EVAL_EXIT=$?
      FAILED=$((FAILED + 1))
      FAILED_JOBS="${FAILED_JOBS}\n  $MODEL (seed=$seed) - eval failed (exit=$EVAL_EXIT)"
      echo "[EVAL] ✗ Evaluation failed for $MODEL (seed=$seed) with exit code $EVAL_EXIT"
    fi
    
    JOB_END=$(date +%s)
    JOB_DURATION=$((JOB_END - JOB_START))
    echo "Duration: ${JOB_DURATION}s"
done

echo ""
echo "=========================================="
echo "CNN Q.ANT sweep completed at $(date)"
echo "Completed jobs: $((TOTAL_JOBS - FAILED))/$TOTAL_JOBS"
echo "Failed jobs: $FAILED"
if [[ -n "$FAILED_JOBS" ]]; then
  echo -e "Failed job details:$FAILED_JOBS"
fi
echo "=========================================="

exit $([[ $FAILED -eq 0 ]] && echo 0 || echo 1)
