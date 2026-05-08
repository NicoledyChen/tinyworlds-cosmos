# Micro-World MC RoPE AR Short-Context Run

## Goal

Test temporal RoPE with an AR-style objective on Micro-World Minecraft.

## Setup

- Dataset: `MICRO_WORLD_MC`
- Cached data: `/root/tinyworlds-cosmos/data/micro_world_mc_frames.h5`
- Frame resolution: `256x256`
- Tokenizer: `nvidia/Cosmos-0.1-Tokenizer-DV8x8x8`
- Raw context length: `33` frames
- Cosmos latent steps: `5`
- Objective: keep previous Cosmos blocks clean and mask only the final target block
- `mask_strategy`: `target_block`
- `target_token_steps`: `1`
- `use_temporal_rope`: `true`
- Batch size: `2`
- Updates: `3000`
- Learning rate: `1e-3`

## Training Result

- Remote run directory: `/root/tinyworlds-cosmos/results/2026_05_08_03_45_35/dynamics`
- Latest checkpoint: `dynamics_step_2900`
- Smoke loss: `11.3067`
- Final logged loss: `10.1009 @ step 2900`

Selected losses:

| Step | Loss |
| ---: | ---: |
| 1800 | 10.1546 |
| 2100 | 10.1331 |
| 2300 | 10.1239 |
| 2600 | 10.1069 |
| 2700 | 10.0954 |
| 2800 | 10.0532 |
| 2900 | 10.1009 |

## Visualization

- Checkpoint: `dynamics_step_2900`
- Context: `33` frames
- Generation: `5` Cosmos latent steps, about `40` frames
- Output frames: `73`
- MSE: `0.017853`
- Local outputs:
  - `video_mc_rope_ar_ctx33_gen5.mp4`
  - `vis_mc_rope_ar_ctx33_gen5.png`

## Notes

- This run uses a different objective than the earlier random-mask MC smoke run, so the losses are not directly comparable.
- The objective matches the intended AR-style setup: n clean Cosmos blocks predict the next masked block.
- No raw data, checkpoints, or HF tokens are committed.
