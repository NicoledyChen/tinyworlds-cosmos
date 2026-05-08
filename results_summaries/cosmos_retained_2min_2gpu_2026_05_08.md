# Cosmos Retained-Context 2-Minute 2-GPU Training

## Goal

Run a normal two-GPU training pass for the Cosmos DV8x8x8 retained-context dynamics model.

## Setup

- Dataset: `PICODOOM`
- Frame resolution: `64x64`
- FPS: `30`
- Context length: `3601` frames, about `120` seconds
- Cosmos latent steps: `451`
- Tokens per clip: about `28,864`
- Mask strategy: `target_block`
- Target latent steps: `1`
- Starting checkpoint: `results/2026_05_08_02_11_21/dynamics/checkpoints/dynamics_step_975`
- GPUs: `2 x B200`
- Distributed mode: DDP
- `find_unused_parameters: true`
- Batch size per GPU: `2`
- Effective batch size: `4`
- Learning rate: `3e-4`
- Updates: `5000`

## Result

- Remote run directory: `/root/tinyworlds-cosmos/results/2026_05_08_02_29_02/dynamics`
- Latest checkpoint: `dynamics_step_4900`
- GPU memory during training: about `74.5 GiB` per GPU
- GPU utilization during training: roughly `75-97%`
- No traceback or OOM.
- Training completed; GPUs were idle afterward.

Selected logged losses:

| Step | Rank 0 Loss | Rank 1 Loss |
| ---: | ---: | ---: |
| 100 | 9.5603 | 9.4973 |
| 200 | 9.5538 | 9.5377 |
| 3800 | 9.3919 | 9.3791 |
| 4000 | 9.3653 | 9.3603 |
| 4400 | 9.3497 | 9.4259 |
| 4600 | 9.3952 | 9.3883 |
| 4800 | 9.3773 | 9.3517 |
| 4900 | 9.3763 | 9.5103 |

## Notes

- DDP initially failed without `find_unused_parameters=True` because `target_block` masking can leave some parameters unused in a given iteration.
- Enabling `find_unused_parameters` fixed the issue.
- No checkpoint files, raw logs, or datasets are committed.
