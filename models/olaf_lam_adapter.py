import os
import sys
from typing import Optional

import torch


class OlafLAMAdapter:
    """Adapter for Olaf-World's frozen LAM encoder.

    TinyWorlds training batches use frames shaped [B, T, C, H, W] in [-1, 1].
    Olaf LAM expects [B, T, H, W, C] in [0, 1] and returns [B, T-1, D].
    """

    def __init__(
        self,
        checkpoint_path: str,
        olaf_root: Optional[str] = None,
        variant: str = "align",
        device: str = "cuda",
    ) -> None:
        if olaf_root:
            olaf_root = os.path.abspath(olaf_root)
            if olaf_root not in sys.path:
                sys.path.insert(0, olaf_root)

        try:
            from lam.inference import load_lam_encoder
        except ImportError as exc:
            raise ImportError(
                "Could not import Olaf-World LAM. Clone https://github.com/showlab/Olaf-World "
                "and pass olaf_lam_root=/path/to/Olaf-World."
            ) from exc

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Olaf LAM checkpoint not found: {checkpoint_path}")

        self.encoder = load_lam_encoder(checkpoint_path, variant=variant, device=device)
        self.encoder.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # [B, T, C, H, W] in [-1, 1] -> [B, T, H, W, C] in [0, 1]
        videos = (frames.detach().float() + 1.0) * 0.5
        videos = videos.clamp(0.0, 1.0).permute(0, 1, 3, 4, 2).contiguous()
        return self.encoder(videos).to(frames.device)
