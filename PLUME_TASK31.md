# PLUME Task 3-1 integration

All commands below use the canonical TokaMark loader, preprocessing, split,
trainer, checkpoint layout, and evaluator. Run them from the repository root.

## Model matrix

| CLI model | Latent dynamics | Forecast | Objective |
| --- | --- | --- | --- |
| `plume_vanilla_mse` | Original PLUME gated vanilla predictor | 10-step latent rollout | Masked profile MSE |
| `plume_vanilla_jepa` | Original PLUME gated vanilla predictor | 10-step latent rollout | Profile + latent + reconstruction |
| `plume_koopman_mse` | Original PLUME `Kz + Bu` transition | 10-step latent rollout | Masked profile MSE |
| `plume_koopman_jepa` | Original PLUME `Kz + Bu` transition | 10-step latent rollout | Profile + latent + reconstruction |
| `plume_direct_mse` | Direct-window PLUME extension | All 10 latents in one pass | Masked profile MSE |
| `plume_direct_jepa` | Direct-window PLUME extension | All 10 latents in one pass | Profile + latent + reconstruction |

The direct-window entries are extensions, not architectures present in the
original PLUME source. They reuse PLUME's exact profile encoder, decoder, and
Q.ANT primitives and introduce only `TokaMarkDirectWindowPredictor`.

## Training

```bash
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_vanilla_mse --split random --seed 23
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_vanilla_jepa --split random --seed 23
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_koopman_mse --split random --seed 23
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_koopman_jepa --split random --seed 23
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_direct_mse --split random --seed 23
uv run python run_training.py --task task_3-1 --config /src/config/config_model.yaml --model plume_direct_jepa --split random --seed 23
```

## Evaluation

Run the matching evaluation after each training command:

```bash
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_vanilla_mse --split random --seed 23
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_vanilla_jepa --split random --seed 23
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_koopman_mse --split random --seed 23
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_koopman_jepa --split random --seed 23
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_direct_mse --split random --seed 23
uv run python run_evaluation.py --task task_3-1 --config /src/config/config_model.yaml --model plume_direct_jepa --split random --seed 23
```

For the temporal split, replace `--split random` with `--split temporal`.

## Losses

`*_mse` always sets the auxiliary weights to zero. `*_jepa` reads the three
weights under `plume.loss` in the selected YAML:

```yaml
plume:
  loss:
    profile_weight: 1.0
    latent_weight: 1.0
    reconstruction_weight: 0.3
```

The target encoder is the same PLUME encoder with stop-gradient; there is no
EMA target encoder. Missing target bins are zero-filled only for construction
of the latent target, while the reported profile loss retains the baseline's
NaN mask.

## Q.ANT backend

The five Q.ANT primitives required by these models are vendored in
`src/plume_qant_layers.py`. `qant_backend: torch` is the default and remains
active after `model.eval()`. Native execution is opt-in via `qant_backend:
qant` and requires Q.ANT SDK 2.3.0 plus `ml-dtypes==0.5.3` on a CPU host.
