# Cosmos Retained-Context 2-Minute Training

## Goal

Train the Cosmos DV8x8x8 TinyWorlds dynamics model with a two-minute retained context.

## Setup

- Dataset: `PICODOOM`
- Cached data: `data/picodoom_frames.h5`
- Frame resolution: `64x64`
- FPS: `30`
- Context length: `3601` frames, about `120` seconds
- Cosmos tokenizer: `nvidia/Cosmos-0.1-Tokenizer-DV8x8x8`
- Latent context: `451` latent time steps
- Tokens per latent step: `64`
- Tokens per clip: about `28,864`
- Mask strategy: `target_block`
- Target latent steps: `1`
- Base checkpoint: `results/2026_05_07_18_51_50/dynamics/checkpoints/dynamics_step_9500`
- Batch size: `1`
- Learning rate: `3e-4`
- Updates: `1000`

## Smoke Test

- One-step smoke passed with `context_length=3601`.
- Smoke loss: `11.1577`
- No OOM.

## Training Result

- Remote run directory: `/root/tinyworlds-cosmos/results/2026_05_08_02_11_21/dynamics`
- Latest checkpoint: `dynamics_step_975`
- GPU memory during training: about `33.8 GiB` on GPU0
- Throughput: roughly `4 it/s`
- Final logged loss: `9.4706 @ step 975`

Selected logged losses:

| Step | Loss |
| ---: | ---: |
| 250 | 9.6562 |
| 500 | 9.6833 |
| 750 | 9.5314 |
| 825 | 9.2321 |
| 900 | 9.4497 |
| 975 | 9.4706 |

## Notes

- This run verifies that full two-minute retained contexts are feasible with the current ST-Transformer dynamics at `64x64`.
- The loss is noisy because batch size is `1` and only the final target latent block is masked.
- No large checkpoint files or raw logs are committed.
