# RWKV-LeWM

RWKV-LeWM is a research implementation of a pixel-based latent world model. It
keeps the end-to-end visual representation learning, SIGReg regularization, and
latent-space planning interface of
[LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm), while replacing its
trajectory Transformer with an RWKV-v7 dynamics model.

The project asks a concrete question: can recurrent RWKV dynamics retain LeWM's
planning quality while eventually providing linear-time sequence processing and
a fixed-size rollout state instead of a growing Transformer KV cache?

> [!IMPORTANT]
> This is an experimental research codebase, not an official LeWM or RWKV
> release. The current CUDA adapter trains with RWKV-v7 sequence kernels, but
> MPC evaluation still re-runs the latest three-token context at each predicted
> step. A true stateful O(1)-memory rollout kernel remains future work.

## Project goals

- Preserve LeWM's compact, end-to-end latent planning pipeline.
- Study RWKV-v7 as the temporal dynamics backbone for visual world models.
- Compare action-conditioning and action-fusion designs under a fixed LeWM
  training and MPC evaluation protocol.
- Reproduce results on TwoRoom and improve action-sensitive planning on PushT.

The repository currently contains complete training and evaluation paths for
TwoRoom and PushT. Download support is also provided for the official LeWM Cube
and Reacher datasets, but no RWKV-LeWM results are reported for those tasks yet.

## Method

### Core world model

Each RGB observation is encoded into one compact latent token, and each block
of environment actions is encoded into the same latent width:

```text
z_t = normalize(Projector(ViT(pixels_t)))
u_t = ActionEncoder(action_block_t)
x_t = Fuse(z_t, u_t, recent_actions)
z_hat_{t+1} = PredictionHead(RWKV-v7(x_{<=t}))
```

The model is trained end-to-end from pixels. The ViT is initialized from
scratch; no pretrained visual features or proprioceptive inputs are used by the
world model. Proprioceptive/state columns are loaded only by the environment
evaluation machinery to initialize simulator states and goals.

The current PushT configuration (`v22`) is:

| Component | Configuration |
| --- | --- |
| Image encoder | ViT, 224x224 input, 14x14 patches, width 192, 12 layers, 3 heads |
| Latent projector | 2048-hidden MLP with BatchNorm, output width 192 |
| Action input | 5 raw 2-D actions concatenated into one 10-D action block |
| Action encoder | LeWM-compatible Conv1d + SiLU MLP, output width 192 |
| Dynamics | 11 RWKV-v7 blocks, width 192, head size 64, FFN width 2048 |
| Prediction head | LeWM-style 2048-hidden MLP, direct next-latent prediction |
| Positional encoding | None; temporal order is handled by RWKV recurrence |
| Backend | Vendored RWKV-v7 CUDA sequence kernels; PyTorch fallback for debugging |

### Current v22 action-history fusion

The strongest current PushT model augments the stable additive input path with
a small causal summary of recent action embeddings:

```text
x_t = z_t + u_t + alpha * MLP([u_{t-k+1}, ..., u_t])
```

The history branch uses three action blocks, has a zero-initialized output, and
starts with scalar `alpha = 0.1`. Two variants are included:

- `current`: summarizes `[u_{t-2}, u_{t-1}, u_t]`.
- `past`: summarizes `[u_{t-3}, u_{t-2}, u_{t-1}]`, so the current action is
  not duplicated in the history branch.

The corresponding configs are
`pusht_rwkv_v22_actionhist_{current,past}_{stage1,stage2}_ds.yaml`.

### Objective

Training uses four model steps: three context steps and one prediction step.
The model makes dense next-latent predictions over the context, and targets are
detached to prevent the predictor from changing the target branch directly:

```text
L = mean(||z_hat_{t+1} - stopgrad(z_{t+1})||^2)
    + 0.09 * SIGReg(z)
```

SIGReg uses 17 knots and 1,024 random projections to keep the learned latent
distribution from collapsing.

### Latent MPC planning

At evaluation time, the current and goal images are encoded once. Cross-Entropy
Method (CEM) planning samples action sequences, rolls each candidate forward in
latent space, and minimizes terminal latent distance:

```text
cost(actions) = ||z_hat_H(actions) - z_goal||^2
```

The reported protocol uses 300 candidates, 30 CEM refinement steps, top-30
elite selection, a five-step model horizon, five raw actions per action block,
and an environment budget of 50 raw steps.

## Data

The loaders use the official LeWM HDF5 datasets hosted on Hugging Face. Data is
stored outside this repository under `$STABLEWM_HOME`.

| Name | Hugging Face dataset | Expected file | Compressed size | Used here |
| --- | --- | --- | ---: | --- |
| PushT | [`quentinll/lewm-pusht`](https://huggingface.co/datasets/quentinll/lewm-pusht) | `pusht_expert_train.h5` | 13.1 GB | Yes |
| TwoRoom | [`quentinll/lewm-tworooms`](https://huggingface.co/datasets/quentinll/lewm-tworooms) | `tworoom.h5` | 3.43 GB | Yes |
| OGBench Cube | [`quentinll/lewm-cube`](https://huggingface.co/datasets/quentinll/lewm-cube) | `cube_single_expert.h5` | 46.2 GB | Download only |
| Reacher | [`quentinll/lewm-reacher`](https://huggingface.co/datasets/quentinll/lewm-reacher) | `reacher.h5` | 23.8 GB | Download only |

For the reported TwoRoom and PushT runs:

```text
pixels: (batch, 4, 3, 224, 224)
action: (batch, 4, 10)  # raw action dim 2 x frameskip 5
```

The non-image columns (`action`, `proprio`, and, for PushT, `state`) are
standardized with statistics computed from the dataset. Images are normalized
with ImageNet statistics.

Download and extract data with:

```bash
export STABLEWM_HOME=/path/to/stable-wm-data

python scripts/download_lewm_data.py tworoom --output "$STABLEWM_HOME"
python scripts/download_lewm_data.py pusht --output "$STABLEWM_HOME"
```

`zstd` and `tar` must be installed on the host. Downloading all four archives
requires roughly 86.5 GB before extraction.

## Installation

The package requires Python 3.12 or newer. The default backend requires Linux,
an NVIDIA GPU, a working CUDA toolchain, and `ninja` to compile the custom
RWKV-v7 operator.

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e '.[dev,swm]'
uv pip install deepspeed
```

Run the lightweight CPU test with the PyTorch fallback backend:

```bash
pytest -q
python scripts/smoke_test.py --device cpu --backend fallback
```

Validate the default CUDA backend with:

```bash
python scripts/smoke_test.py --device cuda --backend rwkv7_cuda
```

## Training

Hydra configs live in `config/train/`. The main entry points are `train.py` for
single-process experiments and `train_deepspeed.py` for distributed training.

### Current PushT v22 training

The reported runs use four GPUs with DeepSpeed ZeRO-2 and bfloat16. The
per-GPU batch size is 32, giving a global batch size of 128. Optimization uses
AdamW with learning rate `5e-5`, weight decay `1e-3`, betas `(0.9, 0.95)`, and
gradient clipping at 1.0.

Stage 1 trains for two epochs. Stage 2 resumes the complete DeepSpeed state from
`epoch_2` and trains for ten more epochs. Run either v22 variant with:

```bash
export STABLEWM_HOME=/path/to/stable-wm-data

# Train both stages, then evaluate epochs 1, 2, 3, 4, 8, 9, and 10.
bash scripts/train_eval_pusht_rwkv_v22.sh current all
bash scripts/train_eval_pusht_rwkv_v22.sh past all
```

The second argument can be `stage1`, `stage2`, or `eval` to run only one part.
Additional Hydra overrides may be appended after it.

### TwoRoom training

The paper-aligned TwoRoom run uses four GPUs, DeepSpeed ZeRO-2, global batch
size 128, two warm-up epochs, and an eight-epoch continuation:

```bash
export STABLEWM_HOME=/path/to/stable-wm-data
bash scripts/train_tworoom_deepspeed_zero2.sh all

bash eval_tworoom.sh outputs/tworoom_stage2_ds/epoch_8
```

## Results

All numbers below use 50 evaluation episodes, seed 42, goal offset 25, budget
50, horizon 5, action block 5, 300 CEM candidates, 30 CEM steps, and top-k 30.
They are read from the generated evaluation JSON files in the local `outputs/`
directory.

| Task | Model / variant | Selected checkpoint | Successes | Success rate |
| --- | --- | --- | ---: | ---: |
| TwoRoom | RWKV-LeWM baseline | stage 2, epoch 8 | 46 / 50 | **92%** |
| PushT | v22 action history, current included | stage 2, epoch 9 | 41 / 50 | **82%** |
| PushT | v22 action history, past only | stage 2, epoch 8 | 41 / 50 | **82%** |
| PushT | v22 action history, current included | stage 2, epoch 10 | 35 / 50 | 70% |
| PushT | v22 action history, past only | stage 2, epoch 10 | 35 / 50 | 70% |

Selected PushT development milestones under the same evaluation protocol:

| Version | Main change | Best measured success rate |
| --- | --- | ---: |
| v17 | Additive `z_t + u_t`, no absolute position embedding | 80% |
| v20 | Learned additive action scale | 80% |
| v21 | Per-channel scaled action injection | 80% |
| v22 | Causal three-block action-history summary | **82%** |

These are checkpoint-selection results from one evaluation seed, not confidence
intervals or multi-seed averages. The final epoch is not always the best MPC
checkpoint, as the v22 results make explicit. Checkpoints, raw logs, datasets,
and `outputs/` are intentionally excluded from Git because they are large.

## Repository layout

```text
config/                         Hydra train/eval and DeepSpeed configs
src/rwkv_lewm/                  Model, losses, encoders, and dynamics backends
src/rwkv_lewm/backends/         Vendored RWKV-v7 model implementation
scripts/                        Data, training, evaluation, and diagnostics
tests/                          CPU smoke test
third_party/rwkv_v7_reference/  Upstream RWKV-v7 training reference
train.py                        Single-process trainer
train_deepspeed.py              Four-GPU DeepSpeed trainer
```

## Known limitations

- The best PushT checkpoint was selected on one 50-episode, seed-42 evaluation;
  a multi-seed test set is still needed.
- The CUDA backend currently exposes sequence-oriented kernels. MPC keeps a
  three-token sliding context instead of using a true one-step recurrent kernel.
- Only TwoRoom and PushT have completed result tables.
- No pretrained weights or datasets are distributed in this repository.
- The implementation is optimized for the local four-GPU research workflow;
  other CUDA/toolchain combinations may require adjustment.

## Acknowledgements

This project builds on the official
[LeWorldModel code and protocol](https://github.com/lucas-maes/le-wm),
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel),
[stable-pretraining](https://github.com/galilai-group/stable-pretraining), and
the official [RWKV-v7 implementation](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7).

Please cite the upstream projects when using this code in research.
