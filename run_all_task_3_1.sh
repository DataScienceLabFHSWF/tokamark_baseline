#!/usr/bin/env bash
# Run from the directory containing run_training.py and run_evaluation.py.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

task=task_3-1
config=/src/config/config_model.yaml
split=random
seed=23
models=(cnn lstm plume_vanilla_mse plume_vanilla_jepa plume_koopman_mse plume_koopman_jepa plume_direct_mse plume_direct_jepa)

log_dir="logs/${task}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$log_dir" || exit 1
summary="$log_dir/summary.tsv"
printf 'model\tstage\tstatus\texit_code\tlog\n' > "$summary"
failures=0

run_stage() {
    local model=$1 stage=$2 script=$3
    local log_file="$log_dir/${model}_${stage}.log"
    local code
    printf '\n[%s] %s %s (log: %s)\n' "$(date -u +%FT%TZ)" "$model" "$stage" "$log_file"
    {
        printf 'Started: %s\n' "$(date -u +%FT%TZ)"
        printf 'Command: uv run python %s --task %s --config %s --model %s --split %s --seed %s\n' \
            "$script" "$task" "$config" "$model" "$split" "$seed"
        uv run python "$script" --task "$task" --config "$config" \
            --model "$model" --split "$split" --seed "$seed"
    } 2>&1 | tee "$log_file"
    code=${PIPESTATUS[0]}
    printf 'Finished: %s | exit code: %s\n' "$(date -u +%FT%TZ)" "$code" | tee -a "$log_file"
    if (( code == 0 )); then
        printf '%s\t%s\tOK\t0\t%s\n' "$model" "$stage" "$log_file" >> "$summary"
    else
        printf '%s\t%s\tFAILED\t%s\t%s\n' "$model" "$stage" "$code" "$log_file" >> "$summary"
        ((failures += 1))
    fi
    return "$code"
}

for model in "${models[@]}"; do
    if run_stage "$model" training run_training.py; then
        run_stage "$model" evaluation run_evaluation.py || true
    else
        printf '%s\tevaluation\tSKIPPED (training failed)\t-\t-\n' "$model" >> "$summary"
    fi
done

printf '\nSummary: %s\n' "$summary"
cat "$summary"
printf 'Failed commands: %s\n' "$failures"
(( failures == 0 ))
