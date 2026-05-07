import torch
from datasets.data_utils import load_data_and_data_loaders
import matplotlib.pyplot as plt
import time
import os
import random
import glob
import re
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint, load_dynamics_from_checkpoint, find_latest_checkpoint
from utils.config import InferenceConfig, load_config
from utils.inference_utils import load_models, visualize_inference, sample_random_action, get_action_latent
from models.cosmos_tokenizer_adapter import CosmosTokenizerAdapter, CosmosVideoShape
from einops import repeat
from typing import Optional


def main():
    # load inference config
    args: InferenceConfig = load_config(InferenceConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml'))

    # enable tf32 if requested
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer_backend = getattr(args, 'tokenizer_backend', 'fsq')
    # whether any setting requires using action tokens
    use_latent_actions = (args.use_actions or args.use_gt_actions or args.use_interactive_mode)
    if tokenizer_backend == 'cosmos' and use_latent_actions:
        print("[WARN] Cosmos inference currently ignores latent actions; set action flags false for controlled runs.")
        use_latent_actions = False

    # check if any path is missing
    def missing(path: Optional[str]) -> bool:
        return (path is None) or (not os.path.exists(path))

    # resolve latest checkpoints if requested or any path missing
    base_dir = os.getcwd()
    if tokenizer_backend != 'cosmos' and (args.use_latest_checkpoints or missing(args.video_tokenizer_path)):
        vt_ckpt = find_latest_checkpoint(base_dir, "video_tokenizer")
        args.video_tokenizer_path = vt_ckpt
    if (args.use_latest_checkpoints or missing(args.latent_actions_path)) and use_latent_actions:
        lam_ckpt = find_latest_checkpoint(base_dir, "latent_actions")
        args.latent_actions_path = lam_ckpt
    if args.use_latest_checkpoints or missing(args.dynamics_path):
        dyn_ckpt = find_latest_checkpoint(base_dir, "dynamics")
        args.dynamics_path = dyn_ckpt
    
    # confirm which ckpts are being used
    if tokenizer_backend != 'cosmos':
        print(f"Using video_tokenizer checkpoint: {args.video_tokenizer_path}")
    else:
        print(f"Using Cosmos tokenizer: {args.cosmos_model}")
    if use_latent_actions:
        print(f"Using latent_actions checkpoint: {args.latent_actions_path}")
    print(f"Using dynamics checkpoint: {args.dynamics_path}")

    # validate required paths
    if tokenizer_backend != 'cosmos' and missing(args.video_tokenizer_path):
        raise FileNotFoundError("video_tokenizer_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints with available runs.")
    if use_latent_actions and missing(args.latent_actions_path):
        raise FileNotFoundError("latent_actions_path is not set or not a file while actions are requested. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
    if missing(args.dynamics_path):
        raise FileNotFoundError("dynamics_path is not set or not a file. Set it in configs/inference.yaml or enable use_latest_checkpoints.")
 
    # load models, optionally compile
    cosmos_tokenizer = None
    if tokenizer_backend == 'cosmos':
        video_tokenizer = None
        latent_action_model = None
        cosmos_tokenizer = CosmosTokenizerAdapter(
            model_name=args.cosmos_model,
            checkpoint_dir=args.cosmos_checkpoint_dir,
            device=args.device,
            temporal_compression=args.cosmos_temporal_compression,
            spatial_compression=args.cosmos_spatial_compression,
            codebook_size=args.cosmos_codebook_size,
        )
        dynamics_model, _ = load_dynamics_from_checkpoint(args.dynamics_path, args.device)
    else:
        video_tokenizer, latent_action_model, dynamics_model = load_models(args.video_tokenizer_path, args.latent_actions_path, args.dynamics_path, args.device, use_actions=use_latent_actions)
    if args.compile:
        if video_tokenizer is not None:
            video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead", fullgraph=False, dynamic=True)
        if use_latent_actions:
            latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Compiled all models for inference.")

    # determine how many ground-truth frames we need in each batch: context + generation steps + prediction horizon
    frame_step = args.prediction_horizon
    if tokenizer_backend == 'cosmos':
        frame_step *= args.cosmos_temporal_compression
        frames_to_load = args.context_window + args.generation_steps * frame_step
    else:
        frames_to_load = args.context_window + args.generation_steps * frame_step

    # dataloader
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides = {'preload_ratio': args.preload_ratio}
    else:
        data_overrides = {}
    if tokenizer_backend == 'cosmos':
        data_overrides['resolution'] = (args.frame_size, args.frame_size)
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, batch_size=1, num_frames=frames_to_load, **data_overrides)

    # sample random batch
    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    og_ground_truth_frames = data_loader.dataset[random_idx][0]  # full sequence
    og_ground_truth_frames = og_ground_truth_frames.unsqueeze(0).to(args.device)  # [1, frames_to_load, C, H, W]

    ground_truth_frames = og_ground_truth_frames[:, :frames_to_load, :, :, :]  # [1, frames_to_load, C, H, W]

    # start with initial context (first context_window GT frames)
    context_frames = ground_truth_frames[:, :args.context_window, :, :, :]
    generated_frames = context_frames.clone()

    # initialize actions
    n_actions = None
    inferred_actions = []
    if use_latent_actions:
        n_actions = latent_action_model.quantizer.codebook_size        

    # ensure we don’t exceed available GT frames if in teacher-forced mode
    max_possible_steps = (ground_truth_frames.shape[1] - args.context_window) // frame_step
    if args.teacher_forced and args.generation_steps > max_possible_steps:
        print(f"[WARN] Requested {args.generation_steps} generation steps but only {max_possible_steps} are possible with teacher-forced context. Clamping.")
    effective_steps = args.generation_steps if not args.teacher_forced else min(args.generation_steps, max_possible_steps)
    if args.retain_full_context:
        print("[INFO] Retaining full generated/teacher-forced history for dynamics context.")

    for i in range(effective_steps):
        print(f"Inferring frame {i+1}/{effective_steps}")
        # select context depending on teacher-forced flag
        if args.teacher_forced:
            if args.retain_full_context:
                context_end = args.context_window + i * frame_step
                context_frames = ground_truth_frames[:, :context_end, :, :, :]
            else:
                context_start = i * frame_step  # shift window along ground truth
                context_frames = ground_truth_frames[:, context_start:context_start+args.context_window, :, :, :] # [1, context_window, C, H, W]
        else:
            if args.retain_full_context:
                context_frames = generated_frames
            else:
                # autoregressive: last context_window frames from generated sequence
                context_frames = generated_frames[:, -args.context_window:, :, :, :]  # [1, context_window, C, H, W]

        # encode context frames each iteration
        if tokenizer_backend == 'cosmos':
            video_indices = cosmos_tokenizer.tokenize(context_frames)
            video_latents = video_indices
        else:
            video_indices = video_tokenizer.tokenize(context_frames)
            video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

        sampled_action_index, action_latent = get_action_latent(args, inferred_actions, n_actions, context_frames, latent_action_model, i)

        # dynamics forward inference needs idx -> latents fun
        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        # autocast for inference if amp enabled (bfloat16 on CUDA by default)
        autocast_dtype = torch.bfloat16 if args.amp else None
        with torch.amp.autocast('cuda', enabled=args.amp, dtype=autocast_dtype):
            if tokenizer_backend == 'cosmos':
                next_video_latents = dynamics_model.forward_inference_indices(
                    context_indices=video_latents,
                    prediction_horizon=args.prediction_horizon,
                    num_steps=10,
                    conditioning=action_latent,
                    temperature=args.temperature,
                )
            else:
                next_video_latents = dynamics_model.forward_inference(
                    context_latents=video_latents,
                    prediction_horizon=args.prediction_horizon,
                    num_steps=10,
                    index_to_latents_fn=idx_to_latents,
                    conditioning=action_latent,
                    temperature=args.temperature,
                )

        # decode next video tokens to frames
        if tokenizer_backend == 'cosmos':
            output_shape = CosmosVideoShape(
                frames=context_frames.shape[1] + args.prediction_horizon * args.cosmos_temporal_compression,
                height=context_frames.shape[-2],
                width=context_frames.shape[-1],
            )
            next_frames = cosmos_tokenizer.detokenize(next_video_latents, output_shape=output_shape)
            generated_frames = torch.cat([generated_frames, next_frames[:, context_frames.shape[1]:, :, :]], dim=1)
        else:
            next_frames = video_tokenizer.detokenize(next_video_latents)  # [1, T, C, H, W]
            generated_frames = torch.cat([generated_frames, next_frames[:, -args.prediction_horizon:, :, :]], dim=1)
        # TODO: if using interactive mode, visualize next_frames[:, -1] (recently inferred frame) every time, probably with matplotlib is easiest
        # point is for user to be able to interact with it in real time

    # visualize inference
    visualize_inference(generated_frames, ground_truth_frames, inferred_actions, args.fps, use_actions=use_latent_actions)


if __name__ == "__main__":
    main()
