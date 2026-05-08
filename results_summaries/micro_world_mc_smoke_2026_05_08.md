# Micro-World MC Cosmos Smoke Run

## Goal

Run the TinyWorlds + Cosmos pipeline on Micro-World Minecraft data before returning to long-context experiments.

## Data Preparation

- Dataset source: `amd/Micro-World-MC-Dataset`
- Files used: first `32` videos under `video/`
- Frames per video: `81`
- Resolution: `256x256`
- Output cache: `/root/tinyworlds-cosmos/data/micro_world_mc_frames.h5`
- Cached shape: `(2592, 256, 256, 3)`

## Training Setup

- Tokenizer: `nvidia/Cosmos-0.1-Tokenizer-DV8x8x8`
- Dataset key: `MICRO_WORLD_MC`
- Context length: `9` frames
- Cosmos latent steps: `2`
- Tokens per latent step: `1024`
- Batch size: `2`
- Updates: `500`
- Learning rate: `1e-3`
- Mask strategy: `random`
- Remote run directory: `/root/tinyworlds-cosmos/results/2026_05_08_03_25_47/dynamics`
- Latest checkpoint: `dynamics_step_450`

## Result

Training completed successfully.

Selected losses:

| Step | Loss |
| ---: | ---: |
| 0 | 11.2481 |
| 50 | 11.0823 |
| 100 | 10.6563 |
| 150 | 10.4941 |
| 250 | 10.3855 |
| 350 | 10.4065 |
| 450 | 10.3494 |

Inference:

- Checkpoint: `dynamics_step_450`
- Context: `9` frames
- Generation: `5` Cosmos latent steps, about `40` frames
- Output frames: `49`
- MSE: `0.012189`
- Local outputs:
  - `video_mc_smoke_ctx9_gen5.mp4`
  - `vis_mc_smoke_ctx9_gen5.png`

## Notes

- This verifies that Micro-World MC data can be converted to TinyWorlds HDF5 format, loaded by the training pipeline, tokenized by Cosmos, and used for dynamics training/inference.
- No raw data, checkpoints, or HF tokens are committed.
