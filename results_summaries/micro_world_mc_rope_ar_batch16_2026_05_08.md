# Micro-World MC RoPE AR Batch-16 Continuation

## Setup

- Dataset: `MICRO_WORLD_MC`
- Resolution: `256x256`
- Tokenizer: `Cosmos-0.1-Tokenizer-DV8x8x8`
- Context length: `33` frames
- Cosmos latent steps: `5`
- Objective: clean history blocks predict the next masked block
- `mask_strategy`: `target_block`
- `use_temporal_rope`: `true`
- Starting checkpoint: `/root/tinyworlds-cosmos/results/2026_05_08_03_45_35/dynamics/checkpoints/dynamics_step_2900`
- Batch size: `16`
- Updates: `3000`

## Result

- Remote run directory: `/root/tinyworlds-cosmos/results/2026_05_08_03_59_16/dynamics`
- Latest checkpoint: `dynamics_step_2900`
- No traceback or OOM.
- GPU memory during training: about `131 GiB` on GPU0.

Selected losses:

| Step | Loss |
| ---: | ---: |
| 1800 | 9.5361 |
| 2000 | 9.4918 |
| 2200 | 9.4618 |
| 2400 | 9.4562 |
| 2600 | 9.4412 |
| 2800 | 9.4394 |
| 2900 | 9.4307 |

## Visualization

- Checkpoint: `dynamics_step_2900`
- Context: `33` frames
- Generation: `5` Cosmos latent steps, about `40` frames
- Output frames: `73`
- MSE: `0.023631`
- Local outputs:
  - `video_mc_rope_ar_batch16_ctx33_gen5.mp4`
  - `vis_mc_rope_ar_batch16_ctx33_gen5.png`

## Notes

- Increasing the batch size from `2` to `16` substantially stabilizes the target-block objective and continues reducing the loss.
- The rollout MSE is not directly comparable across random sampled clips, but the training objective clearly improves.
- No raw data, checkpoints, or HF tokens are committed.
