# TwoRoom Training Configs

The TwoRoom pipeline is split into two stages, but both stages now use the
paper-compatible LeWM data window:

- `history_size: 3`
- `num_preds: 1`
- `num_steps: 4`
- `frameskip: 5`
- `action_dim: 2`, so every model-step action block is 10-dimensional
- SIGReg `weight: 0.09`, `knots: 17`, `num_proj: 1024`
- DeepSpeed global batch size: `32 per GPU * 4 GPUs = 128`

## Stage 1: `tworoom_stage1.yaml`

Warmup:

- `num_steps: 4`
- DeepSpeed `batch_size: 32` per GPU
- `max_epochs: 2`
- output: `outputs/tworoom_stage1_ds`

This stage verifies image/action loading and produces:

```text
outputs/tworoom_stage1_ds/epoch_2
```

## Stage 2: `tworoom_stage2.yaml`

Continuation:

- resumes from stage 1 checkpoint
- `num_steps: 4`
- DeepSpeed `batch_size: 32` per GPU
- `max_epochs: 8`
- `num_proj: 1024`
- output: `outputs/tworoom_stage2_ds`

Both stages optimize the same loss:

```text
loss = MSE(z_hat_{t+1}, stopgrad(z_{t+1})) + 0.09 * SIGReg(z)
```

## Commands

```bash
export STABLEWM_HOME=/path/to/stable-wm-data
bash scripts/train_tworoom_full.sh all
```

Override Hydra fields by appending arguments:

```bash
bash scripts/train_tworoom_full.sh stage1 trainer.max_epochs=1 loader.batch_size=8
bash scripts/train_tworoom_full.sh stage2 optimizer.lr=1e-5
```

## Current Result

Paper-compatible full-dataset MPC evaluation:

```bash
STABLEWM_HOME=/path/to/stable-wm-data \
  bash eval_tworoom.sh outputs/tworoom_stage2_ds/epoch_8
```

Result:

```text
checkpoint: outputs/tworoom_stage2_ds/epoch_8/mp_rank_00_model_states.pt
dataset_name: tworoom
num_eval: 50
seed: 42
goal_offset_steps: 25
eval_budget: 50
horizon: 5
receding_horizon: 5
action_block: 5
cem_samples: 300
cem_steps: 30
cem_topk: 30
success_rate: 92.0%  # 46 / 50
```
