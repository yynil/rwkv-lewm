# PushT Training And Evaluation

This mirrors the original LeWM PushT protocol while replacing the Transformer
dynamics with RWKV-v7 dynamics.

## Original LeWM Parameters

Training data config from the upstream `le-wm/config/train/data/pusht.yaml`:

```yaml
dataset:
  name: pusht_expert_train
  num_steps: ${wm.num_preds + wm.history_size}
  frameskip: 5
  keys_to_load:
    - pixels
    - action
    - proprio
    - state
  keys_to_cache:
    - action
    - proprio
    - state
```

Base LeWM training config from the upstream `le-wm/config/train/lewm.yaml`:

```yaml
loader.batch_size: 128
trainer.max_epochs: 10
optimizer.lr: 5e-5
optimizer.weight_decay: 1e-3
wm.history_size: 3
wm.num_preds: 1
wm.embed_dim: 192
loss.sigreg.weight: 0.09
loss.sigreg.kwargs.knots: 17
loss.sigreg.kwargs.num_proj: 1024
```

Therefore PushT uses:

```text
num_steps = 3 + 1 = 4
frameskip = 5
raw action dim = 2
model-step action dim = 5 * 2 = 10
```

Evaluation config from the upstream `le-wm/config/eval/pusht.yaml`:

```yaml
world.env_name: swm/PushT-v1
world.history_size: 1
world.frame_skip: 1
eval.dataset_name: pusht_expert_train
eval.num_eval: 50
eval.goal_offset_steps: 25
eval.eval_budget: 50
plan_config.horizon: 5
plan_config.receding_horizon: 5
plan_config.action_block: 5
solver.num_samples: 300
solver.n_steps: 30
solver.topk: 30
solver.var_scale: 1.0
```

State setup during evaluation:

```text
_set_state uses state
_set_goal_state uses goal_state
```

## RWKV-LeWM PushT Configs

DeepSpeed configs:

```text
config/train/pusht_stage1_ds.yaml
config/train/pusht_stage2_ds.yaml
```

Both stages use the original LeWM data/loss protocol:

```text
data.name: pusht_expert_train
data.num_steps: 4
data.frameskip: 5
model.residual_prediction: false
model.action_encoder_type: lewm
loss.mode: lewm_context
loader.batch_size: 32 per GPU
global batch size: 32 * 4 = 128
optimizer.lr: 5e-5
SIGReg: weight 0.09, knots 17, num_proj 1024
```

Stage 1 runs 2 epochs and writes:

```text
outputs/pusht_lewmalign_stage1_ds/epoch_2
```

Stage 2 resumes from Stage 1, runs 8 epochs, and writes:

```text
outputs/pusht_lewmalign_stage2_ds/epoch_8
```

## Commands

Download data:

```bash
export STABLEWM_HOME=/path/to/stable-wm-data
python scripts/download_lewm_data.py pusht --output "$STABLEWM_HOME"
```

Train both stages on 4 GPUs with DeepSpeed ZeRO-2:

```bash
STABLEWM_HOME=/path/to/stable-wm-data \
  bash scripts/train_pusht_deepspeed_zero2.sh all
```

Run only one stage:

```bash
bash scripts/train_pusht_deepspeed_zero2.sh stage1
bash scripts/train_pusht_deepspeed_zero2.sh stage2
```

Evaluate final checkpoint:

```bash
STABLEWM_HOME=/path/to/stable-wm-data \
  bash eval_pusht.sh outputs/pusht_lewmalign_stage2_ds/epoch_8
```

The result JSON is written to:

```text
outputs/pusht_lewmalign_stage2_ds/epoch_8/mpc_eval_50_seed42_cem30.json
```
