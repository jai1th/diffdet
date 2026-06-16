#!/usr/bin/env python3
"""
preprocess.py — the raw-video → model-ready-clip transform for this project.

This is the single source of truth for preprocessing. The training, prototype,
and inference scripts in this repo all consume clips produced by exactly this
math, and the Hugging Face demo replicates it byte-for-byte. There is NO Kinetics
mean/std normalisation anywhere — the backbone was trained from scratch on pixels
in [0, 1].

Two stages
----------
1. Spatial normalisation (deterministic, done once and cached as uint8):

       raw video file
         -> decode all frames           -> uint8 [T, H, W, 3] RGB
         -> float [T, C, H, W] / 255
         -> resize short side to 224     (bilinear, align_corners=False, antialias=True)
         -> center-crop square           -> [T, C, 224, 224]
         -> clamp + quantize             -> uint8 [T, 224, 224, 3] RGB

   The cached tensor is uint8 [T, 224, 224, 3] (aspect preserved, square, no
   temporal subsampling). This keeps caches compact and lossless to re-sample.

2. Temporal sampling (done per clip at train/eval time), identical to the
   `load_frames` used by train_supcon.py / build_prototypes.py / prototype_inference.py:

       uint8 [T, 224, 224, 3]
         -> center temporal window scaled by source_fps / target_fps
         -> linspace to exactly clip_len frames (24 @ 24 fps)
         -> [C, T, 224, 224] float in [0, 1]

Note on scope
-------------
The production pipeline cached thousands of clips with a batched, resumable,
fault-tolerant cacher (ThreadPool + atomic writes + integrity verification).
That orchestration is intentionally not shipped here — this module is the exact
per-video transform it wrapped, so the preprocessing is fully documented and
reproducible on a single file without the cluster machinery.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SHORT_SIDE = 224
CLIP_LEN = 24
TARGET_FPS = 24.0
CHUNK_T = 256  # frames per spatial-norm chunk, to bound peak RAM


# =============================================================================
# STAGE 0 — DECODE
# =============================================================================

def decode_video_cv2(video_path: str) -> tuple[torch.Tensor, float]:
    """
    Decode every frame of a raw video with cv2.

    Returns (frames, fps):
      frames : uint8 tensor [T, H, W, 3] in RGB order
      fps    : source frames-per-second (cv2.CAP_PROP_FPS)

    cv2 decodes BGR, so each frame is converted to RGB to match the cached
    tensors the model was trained on.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    arr = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)  # [T, H, W, 3]
    return arr, fps


# =============================================================================
# STAGE 1 — SPATIAL NORMALISATION
# =============================================================================

def resize_short_side(video_tchw: torch.Tensor, short_side: int) -> torch.Tensor:
    """Resize [T,C,H,W] float so the short spatial side == short_side (aspect kept)."""
    T, C, H, W = video_tchw.shape
    if H <= 0 or W <= 0:
        raise ValueError(f"Bad spatial size H={H} W={W}")

    if H < W:                                   # landscape: H is short side
        new_h = short_side
        new_w = int(round(W * (short_side / H)))
    else:                                        # portrait / square: W is short
        new_w = short_side
        new_h = int(round(H * (short_side / W)))

    return F.interpolate(
        video_tchw,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def center_crop_square(video_tchw: torch.Tensor) -> torch.Tensor:
    """Center-crop [T,C,H,W] to [T,C,S,S] where S = min(H, W)."""
    T, C, H, W = video_tchw.shape
    S = min(H, W)
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    return video_tchw[:, :, y0:y0 + S, x0:x0 + S]


def apply_spatial_norm(
    frames_thwc: torch.Tensor,
    short_side: int = SHORT_SIDE,
    chunk_t: int = CHUNK_T,
) -> torch.Tensor:
    """
    uint8 [T,H,W,3] RGB -> uint8 [T,short_side,short_side,3] RGB.

    Processed in temporal chunks to bound peak RAM. Output is the cacheable
    spatially-normalised tensor.
    """
    if frames_thwc.dtype != torch.uint8 or frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
        raise ValueError(
            f"Expected uint8 [T,H,W,3], got {frames_thwc.dtype} {tuple(frames_thwc.shape)}"
        )

    T_total = int(frames_thwc.shape[0])
    out_chunks = []

    for t0 in range(0, T_total, chunk_t):
        t1 = min(T_total, t0 + chunk_t)
        chunk = frames_thwc[t0:t1]                                    # [t,H,W,3] uint8

        chunk = chunk.to(torch.float32).permute(0, 3, 1, 2) / 255.0   # [t,C,H,W]
        chunk = resize_short_side(chunk, short_side)
        chunk = center_crop_square(chunk)                             # [t,C,S,S]

        if chunk.shape[2] != short_side or chunk.shape[3] != short_side:
            raise ValueError(
                f"Post-crop size {chunk.shape[2]}x{chunk.shape[3]} != {short_side} "
                f"(chunk t0={t0}) — off-by-one in aspect-ratio rounding"
            )

        chunk = (chunk.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        chunk = chunk.permute(0, 2, 3, 1).contiguous()               # [t,S,S,3]
        out_chunks.append(chunk)

    return torch.cat(out_chunks, dim=0)                               # [T,S,S,3]


# =============================================================================
# STAGE 2 — TEMPORAL SAMPLING  (identical to load_frames in the trainers)
# =============================================================================

def load_frames(
    frames_thwc: torch.Tensor,
    clip_len: int = CLIP_LEN,
    source_fps: float = TARGET_FPS,
    target_fps: float = TARGET_FPS,
) -> torch.Tensor:
    """
    Sample a fixed-length clip from a spatially-normalised uint8 [T,224,224,3]
    tensor. Mirrors `load_frames` in train_supcon.py exactly:

      - window = round(clip_len * source_fps / target_fps), clamped to [clip_len, T]
      - center the window, then linspace to exactly clip_len indices
      - return [C, T, 224, 224] float in [0, 1]

    `source_fps` is the decoded fps of the clip; `target_fps` is the sampling rate
    the model expects (24).
    """
    frames = frames_thwc
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {frames.dtype}")

    if frames.shape[-1] == 3:
        pass
    elif frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"Unrecognized frame layout {tuple(frames.shape)}")

    T = int(frames.shape[0])
    if T < 1:
        raise ValueError("Empty video tensor")

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
    frames = frames.permute(3, 0, 1, 2).float() / 255.0              # [C, T, H, W]
    return frames


def video_to_clip(
    video_path: str,
    clip_len: int = CLIP_LEN,
    target_fps: float = TARGET_FPS,
    short_side: int = SHORT_SIDE,
) -> torch.Tensor:
    """
    Convenience end-to-end: raw video file -> model-ready clip [C, T, 224, 224]
    float in [0, 1]. This is exactly what the demo feeds to the backbone.
    """
    raw, fps = decode_video_cv2(video_path)
    cached = apply_spatial_norm(raw, short_side=short_side)
    return load_frames(cached, clip_len=clip_len, source_fps=fps, target_fps=target_fps)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Spatially normalise a raw video into a cached uint8 tensor, "
                    "and/or produce a model-ready clip."
    )
    p.add_argument("--video", required=True, help="Path to a raw video file")
    p.add_argument("--out-cache", default=None,
                   help="Optional .pt path to save the spatially-normalised uint8 [T,224,224,3] tensor")
    p.add_argument("--out-clip", default=None,
                   help="Optional .pt path to save the sampled clip [C,T,224,224] float")
    p.add_argument("--clip-len", type=int, default=CLIP_LEN)
    p.add_argument("--target-fps", type=float, default=TARGET_FPS)
    p.add_argument("--short-side", type=int, default=SHORT_SIDE)
    args = p.parse_args()

    raw, fps = decode_video_cv2(args.video)
    print(f"decoded: {tuple(raw.shape)} uint8, source_fps={fps:.3f}")

    cached = apply_spatial_norm(raw, short_side=args.short_side)
    print(f"spatial cache: {tuple(cached.shape)} uint8")
    if args.out_cache:
        torch.save(cached, args.out_cache)
        print(f"saved cache -> {args.out_cache}")

    clip = load_frames(cached, clip_len=args.clip_len, source_fps=fps, target_fps=args.target_fps)
    print(f"clip: {tuple(clip.shape)} float in [{clip.min():.3f}, {clip.max():.3f}]")
    if args.out_clip:
        torch.save(clip, args.out_clip)
        print(f"saved clip -> {args.out_clip}")


if __name__ == "__main__":
    main()
