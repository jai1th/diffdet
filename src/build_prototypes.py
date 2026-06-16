#!/usr/bin/env python3
"""
build_prototypes.py

Build fixed real/fake prototypes from a labeled support JSONL for mixed SupCon inference.

This script is matched to training done with:
- frames in [0,1]
- NO Kinetics mean/std normalization
- R(2+1)D backbone + projection head
- projection embeddings normalized with L2 norm

Critical detail
---------------
The training script you showed loads checkpoints in a bad order for projection inference.
This builder fixes that by:
1. creating model
2. attaching proj_head
3. loading checkpoint

Expected support JSONL
----------------------
Each line must contain at least:
{
  "cache_path": "/path/to/cached_uint8_tensor.pt",
  "y": 0 or 1
}

Optional fields like fps / decoded_fps are supported.

Output
------
Saves a .pt file containing:
{
  "real_proto": Tensor[D],
  "fake_proto": Tensor[D],
  "embedding_source": "projection" or "backbone",
  "checkpoint": "...",
  "support_jsonl": "...",
  "clip_len": ...,
  "target_fps": ...,
  "proj_hidden_dim": ...,
  "proj_dim": ...,
  "n_support": ...,
  "n_real": ...,
  "n_fake": ...
}

Example
-------
python build_prototypes.py \
  --support-jsonl <PATH>/support_bank.jsonl \
  --checkpoint <PATH>/supcon_best_model.pt \
  --output <PATH>/prototypes.pt \
  --embedding-source projection \
  --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r2plus1d_18


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build fixed real/fake prototypes from a labeled support JSONL."
    )
    p.add_argument("--support-jsonl", required=True, help="Labeled support JSONL")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--output", required=True, help="Output .pt file for prototypes")

    p.add_argument("--clip-len", type=int, default=24)
    p.add_argument("--target-fps", type=float, default=24.0)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument(
        "--embedding-source",
        type=str,
        default="projection",
        choices=["backbone", "projection"],
        help="Embedding space to use for prototype building",
    )
    p.add_argument("--proj-hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# LOGGING / IO
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("build_prototypes")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# =============================================================================
# DATA
# =============================================================================

def load_frames(
    cache_path: str,
    clip_len: int,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    """
    Matched to training:
    - NO Kinetics normalization
    - frames returned in [0,1]
    """
    try:
        frames = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed loading cache tensor: {cache_path}") from e

    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(frames)} at {cache_path}")
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(frames.shape)} at {cache_path}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {frames.dtype} at {cache_path}")

    if frames.shape[-1] == 3:
        pass
    elif frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Unrecognized frame layout {tuple(frames.shape)} at {cache_path}")

    T = int(frames.shape[0])
    if T < 1:
        raise ValueError(f"Empty video tensor at {cache_path}")

    if source_fps <= 0:
        source_fps = target_fps
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    window = int(round(clip_len * (source_fps / target_fps)))
    window = max(window, clip_len)
    window = min(window, T)

    start = max(0, (T - window) // 2)
    end = start + window

    if window >= clip_len:
        indices = torch.linspace(start, end - 1, clip_len).long()
    else:
        indices = torch.arange(start, end)
        pad = clip_len - len(indices)
        indices = torch.cat([indices, indices[-1].repeat(pad)])

    frames = frames[indices]
    frames = frames.permute(3, 0, 1, 2).float() / 255.0
    return frames


class VideoDataset(Dataset):
    def __init__(self, records: List[Dict], clip_len: int, target_fps: float):
        self.records = records
        self.clip_len = clip_len
        self.target_fps = target_fps

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        source_fps = float(rec.get("decoded_fps", rec.get("fps", self.target_fps)))
        frames = load_frames(
            rec["cache_path"],
            self.clip_len,
            source_fps,
            self.target_fps,
        )
        label = int(rec["y"])
        return frames, label


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger,
    proj_hidden_dim: int,
    proj_dim: int,
) -> nn.Module:
    """
    Correct loading order for projection inference:
    1. create backbone
    2. replace fc
    3. attach proj_head
    4. load checkpoint
    """
    model = r2plus1d_18(weights=None)
    backbone_dim = model.fc.in_features
    model.fc = nn.Linear(backbone_dim, 2)

    model.proj_head = nn.Sequential(
        nn.Linear(backbone_dim, proj_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(proj_hidden_dim, proj_dim),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_backbone_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = x.flatten(1)
    return F.normalize(x, dim=1)


@torch.no_grad()
def extract_projection_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = x.flatten(1)
    x = model.proj_head(x)
    return F.normalize(x, dim=1)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    records: List[Dict],
    clip_len: int,
    target_fps: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_source: str,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = VideoDataset(records, clip_len, target_fps)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_embs = []
    all_labels = []

    n_batches = len(loader)
    logger.info(f"Extracting embeddings for {len(ds)} support videos across {n_batches} batches")

    for batch_idx, (frames, labels) in enumerate(loader):
        logger.info(f"Embedding batch {batch_idx + 1}/{n_batches}")
        frames = frames.to(device, non_blocking=(device.type == "cuda"))

        if embedding_source == "projection":
            embs = extract_projection_embedding(model, frames)
        elif embedding_source == "backbone":
            embs = extract_backbone_embedding(model, frames)
        else:
            raise ValueError(f"Unknown embedding_source: {embedding_source}")

        all_embs.append(embs.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Building prototypes from support JSONL")
    logger.info("=" * 70)
    logger.info(f"support_jsonl     : {args.support_jsonl}")
    logger.info(f"checkpoint        : {args.checkpoint}")
    logger.info(f"output            : {args.output}")
    logger.info(f"embedding_source  : {args.embedding_source}")
    logger.info(f"clip_len          : {args.clip_len}")
    logger.info(f"target_fps        : {args.target_fps}")
    logger.info(f"batch_size        : {args.batch_size}")
    logger.info(f"num_workers       : {args.num_workers}")
    logger.info(f"device            : {args.device}")
    logger.info("")

    support_records = load_jsonl(args.support_jsonl)
    if len(support_records) == 0:
        raise ValueError("Support JSONL is empty.")

    labels = [int(r["y"]) for r in support_records]
    n_real = sum(1 for y in labels if y == 0)
    n_fake = sum(1 for y in labels if y == 1)

    logger.info(f"Loaded support bank: {len(support_records)} videos | real={n_real} fake={n_fake}")

    if n_real == 0 or n_fake == 0:
        raise ValueError("Support bank must contain both real and fake samples.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(
        checkpoint_path=args.checkpoint,
        device=device,
        logger=logger,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_dim=args.proj_dim,
    )

    embs, labels_t = extract_embeddings(
        model=model,
        records=support_records,
        clip_len=args.clip_len,
        target_fps=args.target_fps,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_source=args.embedding_source,
        logger=logger,
    )

    logger.info(f"Embeddings shape: {tuple(embs.shape)}")

    real_proto = F.normalize(embs[labels_t == 0].mean(0), dim=0)
    fake_proto = F.normalize(embs[labels_t == 1].mean(0), dim=0)

    out = {
        "real_proto": real_proto.cpu(),
        "fake_proto": fake_proto.cpu(),
        "embedding_source": args.embedding_source,
        "checkpoint": args.checkpoint,
        "support_jsonl": args.support_jsonl,
        "clip_len": args.clip_len,
        "target_fps": args.target_fps,
        "proj_hidden_dim": args.proj_hidden_dim,
        "proj_dim": args.proj_dim,
        "n_support": len(support_records),
        "n_real": n_real,
        "n_fake": n_fake,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    logger.info(f"Saved prototypes -> {out_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()