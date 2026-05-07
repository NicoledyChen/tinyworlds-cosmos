import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class CosmosVideoShape:
    frames: int
    height: int
    width: int


class CosmosTokenizerAdapter:
    """Thin wrapper around NVIDIA Cosmos discrete video tokenizers.

    TinyWorlds uses videos as [B, T, C, H, W] tensors in the [-1, 1] range.
    Cosmos expects [B, C, T, H, W]. The adapter keeps the TinyWorlds-facing
    interface stable and hides checkpoint download / tensor layout details.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-0.1-Tokenizer-DV8x8x8",
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        temporal_compression: int = 8,
        spatial_compression: int = 8,
        codebook_size: int = 65536,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.temporal_compression = temporal_compression
        self.spatial_compression = spatial_compression
        self.codebook_size = codebook_size

        checkpoint_dir = self._resolve_checkpoint_dir(model_name, checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        encoder_path = os.path.join(checkpoint_dir, "encoder.jit")
        decoder_path = os.path.join(checkpoint_dir, "decoder.jit")
        if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
            raise FileNotFoundError(
                "Cosmos checkpoint is missing encoder.jit or decoder.jit. "
                f"Expected files under: {checkpoint_dir}"
            )

        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Could not import cosmos_tokenizer. Install NVIDIA Cosmos-Tokenizer "
                "in the active environment, for example: "
                "git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git && "
                "cd Cosmos-Tokenizer && pip install -e ."
            ) from exc

        self.encoder = CausalVideoTokenizer(checkpoint_enc=encoder_path)
        self.decoder = CausalVideoTokenizer(checkpoint_dec=decoder_path)

    def _resolve_checkpoint_dir(self, model_name: str, checkpoint_dir: Optional[str]) -> str:
        if checkpoint_dir:
            return checkpoint_dir

        repo_id = model_name
        local_name = model_name.split("/")[-1]
        local_dir = os.path.join("pretrained_ckpts", local_name)
        if os.path.isfile(os.path.join(local_dir, "encoder.jit")):
            return local_dir

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download Cosmos checkpoints automatically."
            ) from exc

        os.makedirs(local_dir, exist_ok=True)
        return snapshot_download(repo_id=repo_id, local_dir=local_dir)

    def _pad_to_supported_shape(self, frames: torch.Tensor) -> Tuple[torch.Tensor, CosmosVideoShape]:
        # Cosmos causal video tokenizer maps frame 0 to latent 0, then compresses
        # the remaining frames by temporal_compression. Pad by repeating the last
        # frame so (T - 1) is divisible by the compression factor.
        B, T, C, H, W = frames.shape
        original = CosmosVideoShape(frames=T, height=H, width=W)

        if T < 1:
            raise ValueError("Cosmos tokenizer requires at least one frame.")
        rem_t = (T - 1) % self.temporal_compression
        if rem_t:
            pad_t = self.temporal_compression - rem_t
            last = frames[:, -1:].expand(B, pad_t, C, H, W)
            frames = torch.cat([frames, last], dim=1)

        pad_h = (-H) % self.spatial_compression
        pad_w = (-W) % self.spatial_compression
        if pad_h or pad_w:
            frames = torch.nn.functional.pad(frames, (0, pad_w, 0, pad_h))

        return frames, original

    def tokenize(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode [B, T, C, H, W] frames in [-1, 1] into [B, Tz, P] indices."""
        frames, _ = self._pad_to_supported_shape(frames)
        cosmos_input = frames.permute(0, 2, 1, 3, 4).to(self.device, self.dtype)
        with torch.no_grad():
            indices, _codes = self.encoder.encode(cosmos_input)
        # Cosmos returns [B, Tz, Hz, Wz]. Flatten spatial dims for TinyWorlds.
        B, Tz, Hz, Wz = indices.shape
        return indices.long().reshape(B, Tz, Hz * Wz)

    def detokenize(
        self,
        indices: torch.Tensor,
        output_shape: Optional[CosmosVideoShape] = None,
    ) -> torch.Tensor:
        """Decode [B, Tz, P] or [B, Tz, Hz, Wz] indices to [B, T, C, H, W]."""
        if indices.ndim == 3:
            B, Tz, P = indices.shape
            spatial = int(P**0.5)
            if spatial * spatial != P:
                raise ValueError(f"Flattened Cosmos token count must be square, got {P}.")
            indices = indices.reshape(B, Tz, spatial, spatial)
        indices = indices.to(self.device).long()
        with torch.no_grad():
            reconstructed = self.decoder.decode(indices)
        frames = reconstructed.permute(0, 2, 1, 3, 4).to(torch.float32)
        if output_shape is not None:
            frames = frames[:, : output_shape.frames, :, : output_shape.height, : output_shape.width]
        return frames

    @property
    def num_token_classes(self) -> int:
        # Cosmos discrete indices are documented as [1..64K], leaving class 0 unused.
        return self.codebook_size + 1
