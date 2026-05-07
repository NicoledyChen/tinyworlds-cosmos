# Cosmos retained-context experiment summary - 2026-05-07

## Branches and commits

- Local branch: `retained-context-cosmos`
- Local commits:
  - `050e426` Add retained-context Cosmos dynamics controls.
  - `b7cc2e0` Set Cosmos inference frame size.
- Remote branch: `retained-context-cosmos`
- Remote commits:
  - `8d3361d` Add retained-context Cosmos dynamics controls.
  - `2fc5790` Set Cosmos inference frame size.

## Tuned sliding baseline

- Remote run: `/root/tinyworlds-cosmos/results/2026_05_07_18_51_50/dynamics`
- Command log: `/root/tinyworlds-cosmos/tuned_long_train.log`
- Config: Cosmos DV8x8x8, `context_length=9` frames, random MaskGIT masking, batch 256, 10k updates.
- Status: completed `10000/10000` with no Traceback, RuntimeError, or CUDA OOM.
- Speed: progress bar completed in about `21:31`, approximately `7.7 it/s` steady state.
- GPU: B200 GPU0 observed around `31.7 GiB / 183 GiB` during training.
- Loss checkpoints from log: `9.0931 @ step4000`, `8.9548 @ step6000`, `8.9170 @ step7000`, `8.8978 @ step8000`, `8.8864 @ step9000`, `8.8832 @ step9500`.
- Outputs: checkpoints and PNG visualizations saved every 500 steps through `dynamics_step_9500`. No final step-10000 checkpoint was emitted by the current save cadence.

## Retained-context implementation

- Inference adds `retain_full_context`; default remains `false` so sliding-window behavior is unchanged.
- With `retain_full_context=true`, autoregressive inference feeds all generated history (`context_frames = generated_frames`) and fully re-tokenizes it each step. No KV cache is used.
- Training adds `mask_strategy: target_block` and `target_token_steps`; default remains random masking.
- Retained training config uses `context_length=33` frames for Cosmos DV8x8x8, mapping to 5 latent time steps: 4 clean history steps plus 1 target latent block.
- Inference config now has `frame_size` and passes `resolution=(frame_size, frame_size)` for Cosmos so PicoDoom inference matches 64x64 dynamics checkpoints.

## Retained-context checks

- Syntax/import smoke passed on remote conda `torch` env.
- Target-block backward smoke passed: loss `4.9662`, `97` masked tokens, historical token mask count verified as zero.
- Cosmos tokenizer training smoke (`batch_size=4`, `n_updates=2`, `context_length=33`, GPU1) passed: `11.2771 -> 11.2006` loss.
- Full-context inference smoke using the 2-step checkpoint passed on PicoDoom 64x64: 49 generated/GT frames, MSE `0.017466`, PNG `inference_results/inference_results_gt_vs_pred_no_actions_20260507_190500.png`.
- Sliding inference smoke with the same checkpoint also passed: MSE `0.017135`, PNG `inference_results/inference_results_gt_vs_pred_no_actions_20260507_190524.png`.
- OpenCV reported H.264 hardware encoder warnings for MP4 output; PNG outputs were produced, but MP4 files should be treated as suspect until the codec path is fixed.

## Retained-context small run

- Remote run: `/root/tinyworlds-cosmos/results/retained_context_small_2026_05_07_1906/dynamics`
- Command log: `/root/tinyworlds-cosmos/retained_context_small_train.log`
- Config: Cosmos DV8x8x8, `context_length=33`, `mask_strategy=target_block`, `target_token_steps=1`, batch 64, 200 updates, GPU1.
- Status: completed with no Traceback, RuntimeError, or CUDA OOM.
- Speed: progress bar completed in about `31s`; final displayed average `6.28 it/s`, steady-state mostly `6.5-7.7 it/s`.
- GPU: GPU1 observed around `20.3 GiB / 183 GiB` during training.
- Loss: `11.2624 @ step0`, `10.9267 @ step50`, `10.4194 @ step100`, `10.2698 @ step150`.
- Outputs: checkpoints and PNG visualizations at steps `0`, `50`, `100`, and `150`.

## Next steps

- Run a longer retained-context training run, likely 5k-10k updates, using batch 128 or 256 after confirming desired memory headroom.
- Add a final checkpoint save at the end of training so `n_updates=10000` produces a step-10000 checkpoint.
- Add explicit inference logging for latent context length per generation step to make sliding vs retained comparisons easier.
- Fix MP4 writing by switching to an available codec or making video export optional when `VideoWriter` fails.