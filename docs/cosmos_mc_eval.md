# Cosmos Minecraft Tokenizer Evaluation

This experiment checks whether NVIDIA Cosmos DV8x8x8 can tokenize and reconstruct
Minecraft clips from `amd/Micro-World-MC-Dataset` before wiring the tokenizer into
TinyWorlds dynamics training.

## Data

The script loads a small number of samples from Hugging Face:

```bash
python scripts/eval_cosmos_tokenizer_mc.py \
  --dataset_name amd/Micro-World-MC-Dataset \
  --split train \
  --max_samples 4 \
  --frames_per_clip 9 \
  --resize 256 \
  --out_dir cosmos_mc_eval_results
```

`frames_per_clip=9` is intentional for Cosmos causal video tokenizers: frame 0 is
kept as the first latent step, and the remaining frames are compressed by the
temporal factor. For DV8x8x8, `(T - 1)` should be divisible by 8.

## Tensor Conventions

TinyWorlds uses `[B, T, C, H, W]` tensors in `[-1, 1]`. Cosmos uses
`[B, C, T, H, W]`. `CosmosTokenizerAdapter` handles the layout conversion and
returns flattened token indices `[B, Tz, P]`, where `P = (H / 8) * (W / 8)`.

## Outputs

The script writes:

- `sample_XXXX_comparison.png`: top row original frames, bottom row reconstructions.
- `metrics.json`: per-clip and aggregate `MSE`, `MAE`, `PSNR`, token shape, and
  unique-index ratio.

Metrics are computed after mapping frames back to `[0, 1]`.
