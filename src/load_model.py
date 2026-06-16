"""
Minimal CPU loader for the mixed-domain SupCon R(2+1)D checkpoint.

Checkpoint: supcon/final_best.pt
  - Backbone: torchvision r2plus1d_18 (trained from scratch; NOT Kinetics-init)
  - Classifier head: fc = Linear(512 -> 2)   [supervised baseline head, carried along]
  - Projection head: proj_head = Sequential(Linear(512->512), ReLU, Linear(512->128))
    Used at eval; its L2-normalised 128-d output is the prototype/inference space.

The checkpoint is a dict with keys:
  epoch, model_state_dict, optimizer_state_dict, best_selection_score, args
We load only `model_state_dict` (strict=True against the module built below).

This file performs NO inference. The __main__ block is a load-only smoke test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18

BACKBONE_DIM = 512
PROJ_HIDDEN_DIM = 512
PROJ_DIM = 128
NUM_CLASSES = 2


def build_model() -> nn.Module:
    """Build the exact module layout stored in the checkpoint."""
    model = r2plus1d_18(weights=None)            # from scratch, no Kinetics weights
    model.fc = nn.Linear(BACKBONE_DIM, NUM_CLASSES)
    model.proj_head = nn.Sequential(
        nn.Linear(BACKBONE_DIM, PROJ_HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(PROJ_HIDDEN_DIM, PROJ_DIM),
    )
    return model


def load_model(checkpoint_path: str, map_location: str = "cpu") -> nn.Module:
    """Build the module and load the checkpoint state dict on CPU (strict)."""
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print("WARNING missing keys:", missing)
    if unexpected:
        print("WARNING unexpected keys:", unexpected)
    model.eval()
    return model


def extract_backbone_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """512-d pooled backbone feature. Input x: (B, 3, T, 224, 224), pixels in [0,1]."""
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x.flatten(1)


def extract_projected_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """L2-normalised 128-d projection embedding (the prototype/inference space)."""
    x = extract_backbone_embedding(model, x)
    x = model.proj_head(x)
    return F.normalize(x, dim=1)


if __name__ == "__main__":
    import sys

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "supcon/final_best.pt"
    model = load_model(ckpt_path, map_location="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK, {n_params} params")
