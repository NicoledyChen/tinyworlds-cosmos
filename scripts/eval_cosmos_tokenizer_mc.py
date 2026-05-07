import argparse
import json
import math
import os
import sys
from typing import Any, Iterable, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image

from models.cosmos_tokenizer_adapter import CosmosTokenizerAdapter, CosmosVideoShape


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cosmos tokenizer on Micro-World MC clips.")
    parser.add_argument("--dataset_name", default="amd/Micro-World-MC-Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--frames_column", default=None)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--frames_per_clip", type=int, default=9)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--out_dir", default="cosmos_mc_eval_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cosmos_model", default="nvidia/Cosmos-0.1-Tokenizer-DV8x8x8")
    parser.add_argument("--cosmos_checkpoint_dir", default=None)
    parser.add_argument("--cosmos_codebook_size", type=int, default=65536)
    return parser.parse_args()


def image_to_tensor(image: Any, resize: int) -> Optional[torch.Tensor]:
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255 if arr.max() > 1 else 1)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    elif isinstance(image, dict) and "array" in image:
        return image_to_tensor(image["array"], resize)
    elif isinstance(image, dict) and "path" in image and image["path"]:
        return image_to_tensor(Image.open(image["path"]), resize)
    else:
        return None

    pil = pil.resize((resize, resize), Image.Resampling.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def video_path_to_frames(path: str, resize: int, limit: int) -> list[torch.Tensor]:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = image_to_tensor(frame, resize)
        if tensor is not None:
            frames.append(tensor)
    cap.release()
    return frames


def iter_candidate_values(sample: dict[str, Any], preferred: Optional[str]) -> Iterable[Any]:
    if preferred and preferred in sample:
        yield sample[preferred]
    for key in ("frames", "images", "video", "clip", "image", "path", "mp4"):
        if key in sample and key != preferred:
            yield sample[key]
    for key, value in sample.items():
        if key not in {"caption", "text", "actions", "metadata"}:
            yield value


def extract_clip(sample: dict[str, Any], frames_column: Optional[str], frames_per_clip: int, resize: int):
    for value in iter_candidate_values(sample, frames_column):
        frames = []
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = image_to_tensor(item, resize)
                if tensor is not None:
                    frames.append(tensor)
                if len(frames) >= frames_per_clip:
                    break
        elif isinstance(value, np.ndarray):
            if value.ndim == 4:
                for frame in value[:frames_per_clip]:
                    tensor = image_to_tensor(frame, resize)
                    if tensor is not None:
                        frames.append(tensor)
            else:
                tensor = image_to_tensor(value, resize)
                if tensor is not None:
                    frames.append(tensor)
        elif isinstance(value, str) and os.path.exists(value):
            if value.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                frames = video_path_to_frames(value, resize, frames_per_clip)
            else:
                tensor = image_to_tensor(Image.open(value), resize)
                if tensor is not None:
                    frames.append(tensor)
        else:
            tensor = image_to_tensor(value, resize)
            if tensor is not None:
                frames.append(tensor)

        if len(frames) >= 1:
            while len(frames) < frames_per_clip:
                frames.append(frames[-1].clone())
            return torch.stack(frames[:frames_per_clip], dim=0)
    return None


def to_zero_one(frames: torch.Tensor) -> torch.Tensor:
    return torch.clamp((frames + 1.0) / 2.0, 0.0, 1.0)


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    original_01 = to_zero_one(original)
    reconstructed_01 = to_zero_one(reconstructed)
    mse = F.mse_loss(reconstructed_01, original_01).item()
    mae = F.l1_loss(reconstructed_01, original_01).item()
    psnr = 99.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)
    return {"mse": mse, "mae": mae, "psnr": psnr}


def save_comparison(original: torch.Tensor, reconstructed: torch.Tensor, path: str, max_frames: int = 8):
    original_01 = to_zero_one(original[0, :max_frames])
    reconstructed_01 = to_zero_one(reconstructed[0, :max_frames])
    grid = make_grid(
        torch.cat([original_01, reconstructed_01], dim=0),
        nrow=max_frames,
        padding=2,
    )
    save_image(grid, path)


def load_hf_dataset(dataset_name: str, split: str):
    # The repository has a local package named "datasets", so temporarily remove
    # the repo root from sys.path before importing the Hugging Face package.
    repo_root = os.getcwd()
    removed = []
    for entry in ("", repo_root):
        if entry in sys.path:
            sys.path.remove(entry)
            removed.append(entry)
    try:
        from datasets import load_dataset
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)
    return load_dataset(dataset_name, split=split)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_hf_dataset(args.dataset_name, args.split)
    adapter = CosmosTokenizerAdapter(
        model_name=args.cosmos_model,
        checkpoint_dir=args.cosmos_checkpoint_dir,
        device=args.device,
        codebook_size=args.cosmos_codebook_size,
    )

    rows = []
    for sample_idx, sample in enumerate(dataset):
        if len(rows) >= args.max_samples:
            break
        clip = extract_clip(sample, args.frames_column, args.frames_per_clip, args.resize)
        if clip is None:
            continue

        frames = clip.unsqueeze(0).to(args.device)
        output_shape = CosmosVideoShape(
            frames=frames.shape[1],
            height=frames.shape[-2],
            width=frames.shape[-1],
        )
        indices = adapter.tokenize(frames)
        reconstructed = adapter.detokenize(indices, output_shape=output_shape)
        metrics = compute_metrics(frames.detach().cpu(), reconstructed.detach().cpu())
        unique_ratio = torch.unique(indices).numel() / float(indices.numel())
        row = {
            "sample_idx": sample_idx,
            "frames": int(frames.shape[1]),
            "height": int(frames.shape[-2]),
            "width": int(frames.shape[-1]),
            "token_shape": list(indices.shape),
            "codebook_size": adapter.codebook_size,
            "num_token_classes": adapter.num_token_classes,
            "unique_index_ratio": unique_ratio,
            **metrics,
        }
        rows.append(row)
        save_comparison(
            frames.detach().cpu(),
            reconstructed.detach().cpu(),
            os.path.join(args.out_dir, f"sample_{sample_idx:04d}_comparison.png"),
        )

    if not rows:
        raise RuntimeError("No usable clips were found in the dataset. Try setting --frames_column.")

    aggregate = {}
    for key in ("mse", "mae", "psnr", "unique_index_ratio"):
        aggregate[key] = float(np.mean([row[key] for row in rows]))
    result = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "cosmos_model": args.cosmos_model,
        "resize": args.resize,
        "frames_per_clip": args.frames_per_clip,
        "clips": rows,
        "aggregate": aggregate,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
