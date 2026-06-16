#!/usr/bin/env python3
"""
prototype_inference.py

Prototype-based inference for mixed SupCon checkpoints using PREBUILT prototypes.

What it does
------------
1. Loads a query JSONL
2. Loads a mixed SupCon checkpoint
3. Loads prebuilt real/fake prototypes from a .pt file
4. Extracts embeddings for all query videos
5. Scores each query with:
       score = sim_fake - sim_real
6. Makes a decision:
       fake if score >= threshold else real
7. Saves per-sample predictions
8. If query labels exist, computes accuracy-first metrics

Why use this version
--------------------
Use this if you have already built prototypes with:
    build_prototypes.py

This avoids recomputing support embeddings and rebuilding prototypes every time.

Expected prototypes file
------------------------
A torch-saved dict containing at least:
    {
      "real_proto": Tensor[D],
      "fake_proto": Tensor[D],
      "embedding_source": "projection" or "backbone",   # optional but recommended
      ...
    }

Example
-------
python prototype_inference.py \
  --query-jsonl /path/to/query.jsonl \
  --checkpoint /path/to/best_model.pt \
  --prototypes-path /path/to/prototypes.pt \
  --output-dir ./infer_out \
  --embedding-source projection \
  --threshold 0.0 \
  --num-workers 0
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import roc_auc_score, roc_curve
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
        description="Prototype-based inference using prebuilt prototypes."
    )
    p.add_argument("--query-jsonl", required=True, help="Query JSONL (labels optional)")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--prototypes-path", required=True, help="Saved prototypes .pt path")
    p.add_argument("--output-dir", required=True, help="Output directory")

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
        help="Embedding space to use at inference time. Must match prototype space.",
    )
    p.add_argument("--proj-hidden-dim", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=128)

    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Prediction threshold on score = sim_fake - sim_real",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save query embeddings to output-dir/query_embeddings.pt",
    )

    return p.parse_args()


# =============================================================================
# LOGGING / IO
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("saved_proto_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            pass
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    Matched to training + new prototype builder:
    - NO Kinetics normalization
    - frames returned in [0,1]
    """
    try:
        frames = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load cache tensor: {cache_path}") from e

    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in cache, got {type(frames)} at {cache_path}")
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(frames.shape)} at {cache_path}")
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
        label = rec.get("y", None)
        meta = {
            "path": rec.get("path", rec.get("video")),
            "cache_path": rec.get("cache_path"),
            "generator": rec.get("generator"),
            "label": label,
        }
        return frames, (-1 if label is None else int(label)), meta


def collate_fn(batch):
    frames = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return frames, labels, metas


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
        logger.warning(f"Missing keys ({len(missing)}): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected}")

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
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    ds = VideoDataset(records, clip_len, target_fps)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    all_embs = []
    all_labels = []
    all_meta = []

    n_batches = len(loader)
    logger.info(f"Starting embedding extraction for {len(ds)} videos across {n_batches} batches")

    for batch_idx, (frames, labels, metas) in enumerate(loader):
        logger.info(f"Embedding batch {batch_idx + 1}/{n_batches}")
        frames = frames.to(device, non_blocking=(device.type == "cuda"))

        if embedding_source == "backbone":
            embs = extract_backbone_embedding(model, frames)
        elif embedding_source == "projection":
            embs = extract_projection_embedding(model, frames)
        else:
            raise ValueError(f"Unknown embedding_source: {embedding_source}")

        all_embs.append(embs.cpu())
        all_labels.append(labels.cpu())
        all_meta.extend(metas)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0), all_meta


# =============================================================================
# PROTOTYPES / SCORING
# =============================================================================

def load_prototypes(
    prototypes_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    obj = torch.load(prototypes_path, map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        raise ValueError("Prototype file must be a dict.")

    if "real_proto" not in obj or "fake_proto" not in obj:
        raise ValueError("Prototype file must contain 'real_proto' and 'fake_proto'.")

    real_proto = obj["real_proto"]
    fake_proto = obj["fake_proto"]

    if not isinstance(real_proto, torch.Tensor) or not isinstance(fake_proto, torch.Tensor):
        raise TypeError("real_proto and fake_proto must be torch.Tensor.")

    real_proto = F.normalize(real_proto.float(), dim=0)
    fake_proto = F.normalize(fake_proto.float(), dim=0)
    return real_proto, fake_proto, obj


def compute_similarity_scores(
    query_embs: torch.Tensor,
    real_proto: torch.Tensor,
    fake_proto: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_proto = real_proto.cpu()
    fake_proto = fake_proto.cpu()

    sim_real = (query_embs * real_proto).sum(1).cpu().numpy()
    sim_fake = (query_embs * fake_proto).sum(1).cpu().numpy()
    scores = sim_fake - sim_real
    return sim_real, sim_fake, scores


# =============================================================================
# METRICS
# =============================================================================

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx])
    return eer, eer_thresh


def compute_metrics_from_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    preds = (scores >= threshold).astype(np.int32)

    out = {
        "n": int(len(labels)),
        "accuracy": float((labels == preds).mean()) if len(labels) > 0 else None,
        "threshold_used": float(threshold),
        "n_true_real": int((labels == 0).sum()),
        "n_true_fake": int((labels == 1).sum()),
        "n_pred_real": int((preds == 0).sum()),
        "n_pred_fake": int((preds == 1).sum()),
        "n_correct": int((labels == preds).sum()),
        "n_incorrect": int((labels != preds).sum()),
        "score_mean": float(np.mean(scores)) if len(scores) > 0 else None,
        "score_std": float(np.std(scores)) if len(scores) > 0 else None,
        "auroc": None,
        "eer": None,
        "eer_threshold": None,
    }

    if len(np.unique(labels)) == 2:
        out["auroc"] = float(roc_auc_score(labels, scores))
        eer, eer_thresh = compute_eer(labels, scores)
        out["eer"] = eer
        out["eer_threshold"] = eer_thresh

    return out


def compute_per_generator_metrics(rows: List[Dict]) -> Dict[str, Dict]:
    grouped_labels: Dict[str, List[int]] = {}
    grouped_scores: Dict[str, List[float]] = {}

    for row in rows:
        gt = row.get("ground_truth", None)
        if gt is None:
            continue
        gen = row.get("generator")
        if gen is None:
            gen = "<no_generator>"
        grouped_labels.setdefault(str(gen), []).append(int(gt))
        grouped_scores.setdefault(str(gen), []).append(float(row["score"]))

    per_generator = {}
    for gen in sorted(grouped_labels.keys()):
        labels = np.array(grouped_labels[gen], dtype=np.int32)
        scores = np.array(grouped_scores[gen], dtype=np.float32)
        threshold = float(rows[0]["threshold_used"]) if rows else 0.0
        per_generator[gen] = compute_metrics_from_scores(labels, scores, threshold)

    return per_generator


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "pipeline.log")

    logger.info("═" * 70)
    logger.info("  Prototype Inference From Saved Prototypes")
    logger.info(f"  Query JSONL         : {args.query_jsonl}")
    logger.info(f"  Checkpoint          : {args.checkpoint}")
    logger.info(f"  Prototypes          : {args.prototypes_path}")
    logger.info(f"  Device              : {args.device}")
    logger.info(f"  Clip len            : {args.clip_len}")
    logger.info(f"  Target fps          : {args.target_fps}")
    logger.info(f"  Batch size          : {args.batch_size}")
    logger.info(f"  Num workers         : {args.num_workers}")
    logger.info(f"  Threshold           : {args.threshold}")
    logger.info(f"  Embedding source    : {args.embedding_source}")
    logger.info(f"  Output dir          : {out_dir}")
    logger.info("═" * 70)

    device = torch.device(args.device)

    query_records = load_jsonl(args.query_jsonl)
    if len(query_records) == 0:
        raise ValueError("Query JSONL is empty.")
    logger.info(f"Loaded {len(query_records)} query records")

    n_query_labeled = sum(r.get("y", None) is not None for r in query_records)
    logger.info(f"Query labeled count   : {n_query_labeled}/{len(query_records)}")

    model = build_model(
        checkpoint_path=args.checkpoint,
        device=device,
        logger=logger,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_dim=args.proj_dim,
    )

    real_proto, fake_proto, proto_meta = load_prototypes(args.prototypes_path)
    logger.info(f"Loaded prototypes: real_proto={tuple(real_proto.shape)}, fake_proto={tuple(fake_proto.shape)}")

    proto_emb_src = proto_meta.get("embedding_source", None)
    if proto_emb_src is not None and proto_emb_src != args.embedding_source:
        raise ValueError(
            f"Embedding source mismatch: prototypes were built with '{proto_emb_src}', "
            f"but inference requested '{args.embedding_source}'."
        )

    proto_clip_len = proto_meta.get("clip_len", None)
    if proto_clip_len is not None and int(proto_clip_len) != int(args.clip_len):
        raise ValueError(
            f"clip_len mismatch: prototypes were built with clip_len={proto_clip_len}, "
            f"but inference requested clip_len={args.clip_len}."
        )

    proto_target_fps = proto_meta.get("target_fps", None)
    if proto_target_fps is not None and float(proto_target_fps) != float(args.target_fps):
        raise ValueError(
            f"target_fps mismatch: prototypes were built with target_fps={proto_target_fps}, "
            f"but inference requested target_fps={args.target_fps}."
        )

    proto_hidden = proto_meta.get("proj_hidden_dim", None)
    if proto_hidden is not None and int(proto_hidden) != int(args.proj_hidden_dim):
        raise ValueError(
            f"proj_hidden_dim mismatch: prototypes were built with proj_hidden_dim={proto_hidden}, "
            f"but inference requested proj_hidden_dim={args.proj_hidden_dim}."
        )

    proto_dim = proto_meta.get("proj_dim", None)
    if proto_dim is not None and int(proto_dim) != int(args.proj_dim):
        raise ValueError(
            f"proj_dim mismatch: prototypes were built with proj_dim={proto_dim}, "
            f"but inference requested proj_dim={args.proj_dim}."
        )

    query_embs, query_labels, query_meta = extract_embeddings(
        model=model,
        records=query_records,
        clip_len=args.clip_len,
        target_fps=args.target_fps,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_source=args.embedding_source,
        logger=logger,
    )
    logger.info(f"Query embeddings shape: {tuple(query_embs.shape)}")

    emb_dim = int(query_embs.shape[1])
    if real_proto.ndim != 1 or fake_proto.ndim != 1:
        raise ValueError(
            f"Expected 1D prototypes, got real_proto={tuple(real_proto.shape)}, fake_proto={tuple(fake_proto.shape)}"
        )
    if real_proto.shape[0] != emb_dim or fake_proto.shape[0] != emb_dim:
        raise ValueError(
            f"Prototype dimension mismatch: embedding_dim={emb_dim}, "
            f"real_proto={tuple(real_proto.shape)}, fake_proto={tuple(fake_proto.shape)}"
        )

    sim_real, sim_fake, scores = compute_similarity_scores(query_embs, real_proto, fake_proto)
    preds = (scores >= args.threshold).astype(np.int32)

    prediction_rows = []
    for i, meta in enumerate(query_meta):
        gt_val = int(query_labels[i].item())
        ground_truth = None if gt_val < 0 else gt_val
        pred_label = int(preds[i])
        decision = "fake" if pred_label == 1 else "real"

        prediction_rows.append({
            "index": int(i),
            "path": meta.get("path"),
            "cache_path": meta.get("cache_path"),
            "generator": meta.get("generator"),
            "ground_truth": ground_truth,
            "predicted_label": pred_label,
            "decision": decision,
            "score": float(scores[i]),
            "sim_real": float(sim_real[i]),
            "sim_fake": float(sim_fake[i]),
            "threshold_used": float(args.threshold),
            "embedding_source": args.embedding_source,
            "sim_rule": "score = sim_fake - sim_real",
            "correct": None if ground_truth is None else bool(ground_truth == pred_label),
        })

    preds_jsonl_path = out_dir / "predictions.jsonl"
    preds_csv_path = out_dir / "predictions.csv"
    write_jsonl(prediction_rows, preds_jsonl_path)
    write_csv(prediction_rows, preds_csv_path)

    metrics_payload: Dict = {
        "query_jsonl": args.query_jsonl,
        "checkpoint": args.checkpoint,
        "prototypes_path": args.prototypes_path,
        "embedding_source": args.embedding_source,
        "threshold": args.threshold,
        "overall": None,
        "per_generator": None,
    }

    valid_mask = np.array([int(v.item()) >= 0 for v in query_labels], dtype=bool)
    metrics_path = out_dir / "metrics.json"

    if valid_mask.any():
        gt = query_labels.numpy()[valid_mask]
        sc = scores[valid_mask]

        overall_metrics = compute_metrics_from_scores(gt, sc, args.threshold)
        labeled_rows = [row for row in prediction_rows if row["ground_truth"] is not None]
        per_generator_metrics = compute_per_generator_metrics(labeled_rows)

        metrics_payload["overall"] = overall_metrics
        metrics_payload["per_generator"] = per_generator_metrics

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        logger.info("─" * 70)
        logger.info("  QUERY SUMMARY")
        logger.info(f"  n            : {overall_metrics['n']}")
        logger.info(f"  accuracy     : {overall_metrics['accuracy']:.4f}" if overall_metrics["accuracy"] is not None else "  accuracy     : N/A")
        logger.info(f"  correct      : {overall_metrics['n_correct']}")
        logger.info(f"  incorrect    : {overall_metrics['n_incorrect']}")
        logger.info(f"  true_real    : {overall_metrics['n_true_real']}")
        logger.info(f"  true_fake    : {overall_metrics['n_true_fake']}")
        logger.info(f"  pred_real    : {overall_metrics['n_pred_real']}")
        logger.info(f"  pred_fake    : {overall_metrics['n_pred_fake']}")
        logger.info(f"  score_mean   : {overall_metrics['score_mean']:.6f}" if overall_metrics["score_mean"] is not None else "  score_mean   : N/A")
        logger.info(f"  score_std    : {overall_metrics['score_std']:.6f}" if overall_metrics["score_std"] is not None else "  score_std    : N/A")
        logger.info(f"  AUROC        : {overall_metrics['auroc']:.4f}" if overall_metrics["auroc"] is not None else "  AUROC        : N/A")
        logger.info(f"  EER          : {overall_metrics['eer']:.4f}" if overall_metrics["eer"] is not None else "  EER          : N/A")
        logger.info("─" * 70)

        logger.info("  GENERATOR SUMMARY TABLE")
        logger.info("  Generator                  n   Accuracy   PredReal   PredFake")
        logger.info("  ------------------------------------------------------------")
        for gen, m in per_generator_metrics.items():
            gen_str = str(gen) if gen is not None else "<no_generator>"
            acc_str = "N/A" if m["accuracy"] is None else f"{m['accuracy']:.4f}"
            logger.info(
                f"  {gen_str:<24s} {m['n']:>4d}  {acc_str:>8s}  "
                f"{m['n_pred_real']:>8d}  {m['n_pred_fake']:>8d}"
            )
        logger.info("─" * 70)

        logger.info(f"Metrics saved     -> {metrics_path}")
    else:
        logger.info("No query labels present. Skipping metric computation.")

    if args.save_embeddings:
        query_emb_path = out_dir / "query_embeddings.pt"
        torch.save(
            {
                "embeddings": query_embs,
                "labels": query_labels,
                "embedding_source": args.embedding_source,
                "jsonl": args.query_jsonl,
            },
            query_emb_path,
        )
        logger.info(f"Query embeddings saved -> {query_emb_path}")

    logger.info(f"Predictions JSONL -> {preds_jsonl_path}")
    logger.info(f"Predictions CSV   -> {preds_csv_path}")
    logger.info(f"Done: {len(prediction_rows)} query predictions written")
    logger.info(f"All outputs in    -> {out_dir}")


if __name__ == "__main__":
    main()