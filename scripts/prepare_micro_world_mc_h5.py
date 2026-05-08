import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Micro-World MC videos as a TinyWorlds HDF5 frame cache.")
    parser.add_argument("--dataset_name", default="amd/Micro-World-MC-Dataset")
    parser.add_argument("--video_prefix", default="video/")
    parser.add_argument("--max_videos", type=int, default=32)
    parser.add_argument("--frames_per_video", type=int, default=81)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--read_step", type=int, default=1)
    parser.add_argument("--out", default="data/micro_world_mc_frames.h5")
    return parser.parse_args()


def list_video_files(dataset_name: str, video_prefix: str):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith(video_prefix) and f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )


def read_video_frames(path: str, resize: int, frames_per_video: int, read_step: int):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_idx = 0
    while len(frames) < frames_per_video:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % read_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def main():
    args = parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_files = list_video_files(args.dataset_name, args.video_prefix)
    selected = video_files[: args.max_videos]
    if not selected:
        raise RuntimeError(f"No videos found under {args.video_prefix!r} in {args.dataset_name}")

    all_frames = []
    for filename in tqdm(selected, desc="Downloading and decoding MC videos"):
        local_path = hf_hub_download(
            repo_id=args.dataset_name,
            repo_type="dataset",
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
        all_frames.extend(read_video_frames(local_path, args.resize, args.frames_per_video, args.read_step))

    if not all_frames:
        raise RuntimeError("No frames were decoded from the selected videos.")

    frames = np.asarray(all_frames, dtype=np.uint8)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("frames", data=frames, compression="lzf")

    print(f"Saved {frames.shape[0]} frames to {output_path}")
    print(f"Frame shape: {frames.shape[1:]}")


if __name__ == "__main__":
    main()
