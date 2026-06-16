#!/usr/bin/env python3
"""
train_supcon.py

3-domain mixed SupCon fine-tuning of R(2+1)D with:

1. Generator-balanced fake train/val/test splits for domains that have generator labels
   - target (Aegis)
   - retain_genvid (GenVid)

2. Generator-equal fake sampling across the epoch for those domains

3. Plain class-balanced sampling for DVF, since DVF generator labels are unavailable

Domains
-------
- target         : Aegis
- retain_dvf     : DVF
- retain_genvid  : GenVid

Default batch layout for batch_size=24
--------------------------------------
- target real       4
- target fake       4
- dvf real          4
- dvf fake          4
- genvid real       4
- genvid fake       4

Checkpoint selection
--------------------
score = 0.5 * target_val_auroc + 0.3 * dvf_val_auroc + 0.2 * genvid_val_auroc
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models.video import r2plus1d_18


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-domain mixed SupCon fine-tuning with strict generator-balanced fake handling."
    )

    parser.add_argument("--target-jsonl", required=True, help="Target-domain JSONL (e.g. Aegis)")
    parser.add_argument("--retain-dvf-jsonl", required=True, help="DVF retention JSONL")
    parser.add_argument("--retain-genvid-jsonl", required=True, help="GenVid retention JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--target-train-per-class", type=int, default=25)
    parser.add_argument("--target-val-per-class", type=int, default=25)

    parser.add_argument("--retain-dvf-train-per-class", type=int, default=100)
    parser.add_argument("--retain-dvf-val-per-class", type=int, default=100)

    parser.add_argument("--retain-genvid-train-per-class", type=int, default=100)
    parser.add_argument("--retain-genvid-val-per-class", type=int, default=100)

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the DVF-trained R(2+1)D base checkpoint to transfer from "
             "(supervised baseline, e.g. produced by train_baseline.py). "
             "The mixed-domain SupCon stage is initialised from this backbone.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--clip-len", type=int, default=24)
    parser.add_argument("--target-fps", type=float, default=24.0)

    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--unfreeze-policy",
        type=str,
        default="layer4_all",
        choices=["layer4_all", "layer4_last_block", "layer4_last_block_plus_fc"],
    )

    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=128)

    parser.add_argument(
        "--eval-embedding-source",
        type=str,
        default="projection",
        choices=["backbone", "projection"],
    )
    parser.add_argument("--eval-support-per-class", type=int, default=20)
    parser.add_argument("--eval-n-trials", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=123)

    parser.add_argument("--target-weight", type=float, default=0.5)
    parser.add_argument("--dvf-weight", type=float, default=0.3)
    parser.add_argument("--genvid-weight", type=float, default=0.2)

    parser.add_argument("--save-last", action="store_true")
    return parser.parse_args()


# =============================================================================
# IO / DATA
# =============================================================================

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tag_records(records: List[Dict], domain: str) -> List[Dict]:
    out = []
    for r in records:
        rr = dict(r)
        rr["domain"] = domain
        out.append(rr)
    return out


def _group_fake_by_generator(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        if int(r["y"]) != 1:
            continue
        gen = r.get("generator", None)
        if gen is None or str(gen).strip() == "":
            gen = "__unknown__"
        groups[str(gen)].append(r)
    return groups


def stratified_split_plain(
    records: List[Dict],
    train_per_class: int,
    val_per_class: int,
    seed: int,
    domain_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)

    real = [r for r in records if int(r["y"]) == 0]
    fake = [r for r in records if int(r["y"]) == 1]

    rng.shuffle(real)
    rng.shuffle(fake)

    need = train_per_class + val_per_class
    if len(real) <= need or len(fake) <= need:
        raise ValueError(
            f"[{domain_name}] Not enough samples per class for split. "
            f"Need > {need} per class, got real={len(real)}, fake={len(fake)}."
        )

    train = real[:train_per_class] + fake[:train_per_class]
    val = real[train_per_class:train_per_class + val_per_class] + fake[train_per_class:train_per_class + val_per_class]
    test = real[train_per_class + val_per_class:] + fake[train_per_class + val_per_class:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def stratified_split_generator_balanced_fake(
    records: List[Dict],
    train_per_class: int,
    val_per_class: int,
    seed: int,
    domain_name: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Real class is split normally.
    Fake class is split within each generator, then merged back.

    This produces generator-balanced fake train/val/test as much as possible.
    """
    rng = random.Random(seed)

    real = [r for r in records if int(r["y"]) == 0]
    fake_groups = _group_fake_by_generator(records)

    rng.shuffle(real)
    need_real = train_per_class + val_per_class
    if len(real) <= need_real:
        raise ValueError(
            f"[{domain_name}] Not enough real samples. Need > {need_real}, got real={len(real)}."
        )

    real_train = real[:train_per_class]
    real_val = real[train_per_class:train_per_class + val_per_class]
    real_test = real[train_per_class + val_per_class:]

    # Allocate fake counts per generator proportionally but fairly.
    train_fake: List[Dict] = []
    val_fake: List[Dict] = []
    test_fake: List[Dict] = []

    # First shuffle each generator bucket
    groups = {}
    for g, arr in fake_groups.items():
        aa = arr.copy()
        rng.shuffle(aa)
        groups[g] = aa

    total_fake = sum(len(v) for v in groups.values())
    need_fake = train_per_class + val_per_class
    if total_fake <= need_fake:
        raise ValueError(
            f"[{domain_name}] Not enough fake samples. Need > {need_fake}, got fake={total_fake}."
        )

    # Target per-generator counts using largest remainder for train and val separately
    gens = sorted(groups.keys())
    sizes = {g: len(groups[g]) for g in gens}

    def allocate_counts(total_needed: int, available_sizes: Dict[str, int]) -> Dict[str, int]:
        total_avail = sum(available_sizes.values())
        raw = {g: total_needed * available_sizes[g] / total_avail for g in gens}
        base = {g: int(math.floor(raw[g])) for g in gens}
        remainder = total_needed - sum(base.values())

        order = sorted(gens, key=lambda g: (raw[g] - base[g]), reverse=True)
        for g in order[:remainder]:
            base[g] += 1

        # Ensure we do not exceed available
        overflow = True
        while overflow:
            overflow = False
            for g in gens:
                if base[g] > available_sizes[g]:
                    overflow = True
                    extra = base[g] - available_sizes[g]
                    base[g] = available_sizes[g]
                    # redistribute
                    for h in order:
                        if h == g:
                            continue
                        room = available_sizes[h] - base[h]
                        if room > 0:
                            add = min(room, extra)
                            base[h] += add
                            extra -= add
                            if extra == 0:
                                break
                    if extra > 0:
                        raise ValueError(
                            f"[{domain_name}] Could not allocate balanced fake counts; dataset too small/skewed."
                        )
        return base

    train_alloc = allocate_counts(train_per_class, sizes)

    remaining_after_train = {g: sizes[g] - train_alloc[g] for g in gens}
    val_alloc = allocate_counts(val_per_class, remaining_after_train)

    for g in gens:
        arr = groups[g]
        nt = train_alloc[g]
        nv = val_alloc[g]
        train_fake.extend(arr[:nt])
        val_fake.extend(arr[nt:nt + nv])
        test_fake.extend(arr[nt + nv:])

    train = real_train + train_fake
    val = real_val + val_fake
    test = real_test + test_fake

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_frames(
    cache_path: str,
    clip_len: int,
    source_fps: float,
    target_fps: float,
) -> torch.Tensor:
    # Backbone trained from scratch on pixels in [0, 1]; no Kinetics mean/std.
    frames = torch.load(cache_path, map_location="cpu", weights_only=False)
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
        label = int(rec["y"])
        domain = str(rec["domain"])
        return frames, label, domain


# =============================================================================
# EPOCH-BALANCED GENERATOR POOL
# =============================================================================

class StrictBalancedGeneratorEpochPool:
    """
    Makes each generator contribute as equally as possible across an epoch.

    For a per-batch fake count k and n_batches B, the epoch needs k*B fake picks.
    This pool creates an epoch plan with near-equal contribution per generator.
    """

    def __init__(self, groups: Dict[str, List[int]], n_batches: int, per_batch: int, seed: int = 42):
        if not groups:
            raise ValueError("StrictBalancedGeneratorEpochPool requires non-empty groups.")
        self.groups = {g: list(v) for g, v in groups.items()}
        self.n_batches = n_batches
        self.per_batch = per_batch
        self.seed = seed
        self.epoch = 0
        self.epoch_plan: List[List[int]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.epoch_plan = self._build_epoch_plan()

    def _build_epoch_plan(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        gens = sorted(self.groups.keys())
        total_needed = self.n_batches * self.per_batch
        n_gens = len(gens)

        base = total_needed // n_gens
        rem = total_needed % n_gens

        target_counts = {g: base for g in gens}
        order = gens.copy()
        rng.shuffle(order)
        for g in order[:rem]:
            target_counts[g] += 1

        # Build generator-specific cyclic shuffled queues
        gen_queues: Dict[str, deque] = {}
        for g in gens:
            arr = self.groups[g].copy()
            rng.shuffle(arr)
            q = deque(arr)
            gen_queues[g] = q

        # Materialize equalized fake picks
        picks: List[Tuple[str, int]] = []
        for g in gens:
            need = target_counts[g]
            for _ in range(need):
                q = gen_queues[g]
                if not q:
                    arr = self.groups[g].copy()
                    rng.shuffle(arr)
                    q.extend(arr)
                idx = q.popleft()
                picks.append((g, idx))

        # Shuffle generator order globally so batches are mixed
        rng.shuffle(picks)

        # Repack into batches, trying to maximize generator diversity within each batch
        by_gen: Dict[str, deque] = defaultdict(deque)
        for g, idx in picks:
            by_gen[g].append(idx)

        batches: List[List[int]] = []
        for _ in range(self.n_batches):
            batch: List[int] = []
            gen_order = list(by_gen.keys())
            rng.shuffle(gen_order)

            # First pass: distinct generators
            for g in gen_order:
                if len(batch) >= self.per_batch:
                    break
                if by_gen[g]:
                    batch.append(by_gen[g].popleft())

            # Fill if still short
            if len(batch) < self.per_batch:
                gen_order2 = list(by_gen.keys())
                rng.shuffle(gen_order2)
                for g in gen_order2:
                    while by_gen[g] and len(batch) < self.per_batch:
                        batch.append(by_gen[g].popleft())
                    if len(batch) >= self.per_batch:
                        break

            if len(batch) != self.per_batch:
                raise RuntimeError("Failed to build balanced generator batch plan.")

            batches.append(batch)

        return batches

    def batch_indices(self, batch_idx: int) -> List[int]:
        return self.epoch_plan[batch_idx]


# =============================================================================
# 6-BUCKET SAMPLER
# =============================================================================

def default_6bucket_counts(batch_size: int) -> Dict[str, int]:
    if batch_size != 24:
        raise ValueError("This script currently hardcodes the 6-bucket layout for batch_size=24 only.")
    return {
        "target_real": 6,
        "target_fake": 4,
        "dvf_real": 4,
        "dvf_fake": 4,
        "genvid_real": 4,
        "genvid_fake": 4,
    }


class Mixed3DomainStrictGenBatchSampler(Sampler[List[int]]):
    def __init__(self, records: List[Dict], batch_size: int, seed: int = 42):
        self.records = records
        self.seed = seed
        self.epoch = 0
        self.counts = default_6bucket_counts(batch_size)

        self.target_real = [i for i, r in enumerate(records) if r["domain"] == "target" and int(r["y"]) == 0]
        self.dvf_real = [i for i, r in enumerate(records) if r["domain"] == "retain_dvf" and int(r["y"]) == 0]
        self.dvf_fake = [i for i, r in enumerate(records) if r["domain"] == "retain_dvf" and int(r["y"]) == 1]
        self.genvid_real = [i for i, r in enumerate(records) if r["domain"] == "retain_genvid" and int(r["y"]) == 0]

        self.target_fake_by_gen = self._group_domain_fake(records, "target")
        self.genvid_fake_by_gen = self._group_domain_fake(records, "retain_genvid")

        c = self.counts
        if len(self.target_real) < c["target_real"]:
            raise ValueError("Not enough target real samples.")
        if len(self.dvf_real) < c["dvf_real"]:
            raise ValueError("Not enough DVF real samples.")
        if len(self.dvf_fake) < c["dvf_fake"]:
            raise ValueError("Not enough DVF fake samples.")
        if len(self.genvid_real) < c["genvid_real"]:
            raise ValueError("Not enough GenVid real samples.")

        self.n_batches = min(
            len(self.target_real) // c["target_real"],
            len(self.dvf_real) // c["dvf_real"],
            len(self.dvf_fake) // c["dvf_fake"],
            len(self.genvid_real) // c["genvid_real"],
        )

        self.target_fake_epoch_pool = StrictBalancedGeneratorEpochPool(
            self.target_fake_by_gen,
            n_batches=self.n_batches,
            per_batch=c["target_fake"],
            seed=seed + 111,
        )
        self.genvid_fake_epoch_pool = StrictBalancedGeneratorEpochPool(
            self.genvid_fake_by_gen,
            n_batches=self.n_batches,
            per_batch=c["genvid_fake"],
            seed=seed + 222,
        )

    @staticmethod
    def _group_domain_fake(records: List[Dict], domain_name: str) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(records):
            if r["domain"] != domain_name:
                continue
            if int(r["y"]) != 1:
                continue
            gen = r.get("generator", None)
            if gen is None or str(gen).strip() == "":
                gen = "__unknown__"
            groups[str(gen)].append(i)
        if not groups:
            raise ValueError(f"No generator-labelled fake groups found for domain {domain_name}.")
        return dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.target_fake_epoch_pool.set_epoch(epoch)
        self.genvid_fake_epoch_pool.set_epoch(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        c = self.counts

        tr = self.target_real.copy()
        dr = self.dvf_real.copy()
        df = self.dvf_fake.copy()
        gr = self.genvid_real.copy()

        rng.shuffle(tr)
        rng.shuffle(dr)
        rng.shuffle(df)
        rng.shuffle(gr)

        for i in range(self.n_batches):
            target_real_batch = tr[i * c["target_real"]:(i + 1) * c["target_real"]]
            dvf_real_batch = dr[i * c["dvf_real"]:(i + 1) * c["dvf_real"]]
            dvf_fake_batch = df[i * c["dvf_fake"]:(i + 1) * c["dvf_fake"]]
            genvid_real_batch = gr[i * c["genvid_real"]:(i + 1) * c["genvid_real"]]

            target_fake_batch = self.target_fake_epoch_pool.batch_indices(i)
            genvid_fake_batch = self.genvid_fake_epoch_pool.batch_indices(i)

            batch = (
                target_real_batch
                + target_fake_batch
                + dvf_real_batch
                + dvf_fake_batch
                + genvid_real_batch
                + genvid_fake_batch
            )
            rng.shuffle(batch)
            yield batch


# =============================================================================
# MODEL
# =============================================================================

def apply_unfreeze_policy(model: nn.Module, policy: str) -> None:
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.proj_head.parameters():
        p.requires_grad_(True)

    if policy == "layer4_all":
        for p in model.layer4.parameters():
            p.requires_grad_(True)
    elif policy == "layer4_last_block":
        for p in model.layer4[-1].parameters():
            p.requires_grad_(True)
    elif policy == "layer4_last_block_plus_fc":
        for p in model.layer4[-1].parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown unfreeze policy: {policy}")


def set_frozen_modules_eval(model: nn.Module, unfreeze_policy: str) -> None:
    model.stem.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    if unfreeze_policy == "layer4_all":
        model.layer4.train()
        model.fc.eval()
    elif unfreeze_policy == "layer4_last_block":
        for block in model.layer4[:-1]:
            block.eval()
        model.layer4[-1].train()
        model.fc.eval()
    elif unfreeze_policy == "layer4_last_block_plus_fc":
        for block in model.layer4[:-1]:
            block.eval()
        model.layer4[-1].train()
        model.fc.train()
    else:
        raise ValueError(f"Unknown unfreeze policy: {unfreeze_policy}")

    model.proj_head.train()


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = r2plus1d_18(weights=None)
    backbone_dim = model.fc.in_features
    model.fc = nn.Linear(backbone_dim, 2)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        logging.warning(f"[Model] Missing keys: {missing}")
    if unexpected:
        logging.warning(f"[Model] Unexpected keys: {unexpected}")

    model.proj_head = nn.Sequential(
        nn.Linear(backbone_dim, args.proj_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(args.proj_hidden_dim, args.proj_dim),
    )

    apply_unfreeze_policy(model, args.unfreeze_policy)
    model = model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    logging.info(
        f"[Model] Trainable params: {n_train:,} / {n_total:,} "
        f"({100.0 * n_train / n_total:.2f}%) — policy={args.unfreeze_policy}"
    )
    return model


def extract_backbone_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = x.flatten(1)
    return x


def extract_projected_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = extract_backbone_embedding(model, x)
    x = model.proj_head(x)
    return F.normalize(x, dim=1)


# =============================================================================
# SUPCON LOSS
# =============================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = torch.matmul(features, features.T) / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T) & (~self_mask)

        n_pos = pos_mask.sum(dim=1)
        if (n_pos == 0).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = sim - log_denom

        loss_per_anchor = -torch.where(
            pos_mask, log_prob, torch.zeros_like(log_prob)
        ).sum(dim=1) / n_pos

        return loss_per_anchor.mean()


# =============================================================================
# FEW-SHOT PROTOTYPE EVALUATION
# =============================================================================

def compute_eer_from_scores(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx])
    return eer, eer_thresh


@torch.no_grad()
def extract_embeddings_for_records(
    model: nn.Module,
    records: List[Dict],
    clip_len: int,
    target_fps: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    embedding_source: str = "projection",
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    loader = DataLoader(
        VideoDataset(records, clip_len, target_fps),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_embs = []
    all_labels = []

    for frames, labels, _domains in loader:
        frames = frames.to(device, non_blocking=(device.type == "cuda"))

        if embedding_source == "backbone":
            embs = extract_backbone_embedding(model, frames)
            embs = F.normalize(embs, dim=1)
        elif embedding_source == "projection":
            embs = extract_projected_embedding(model, frames)
        else:
            raise ValueError(f"Unknown embedding_source: {embedding_source}")

        all_embs.append(embs.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def fewshot_prototype_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    support_per_class: int,
    n_trials: int,
    seed: int,
) -> Dict[str, float]:
    real_idx = (labels == 0).nonzero(as_tuple=True)[0].tolist()
    fake_idx = (labels == 1).nonzero(as_tuple=True)[0].tolist()

    if len(real_idx) <= support_per_class or len(fake_idx) <= support_per_class:
        raise ValueError(
            f"Not enough held-out samples for {support_per_class}-shot evaluation. "
            f"Got real={len(real_idx)}, fake={len(fake_idx)}."
        )

    rng = random.Random(seed)
    aurocs = []
    eers = []

    for _ in range(n_trials):
        sup_real = rng.sample(real_idx, support_per_class)
        sup_fake = rng.sample(fake_idx, support_per_class)
        sup_idx = set(sup_real + sup_fake)

        qry_idx = [i for i in range(len(labels)) if i not in sup_idx]
        qry_idx_t = torch.tensor(qry_idx, dtype=torch.long)
        sup_real_t = torch.tensor(sup_real, dtype=torch.long)
        sup_fake_t = torch.tensor(sup_fake, dtype=torch.long)

        real_proto = F.normalize(embeddings[sup_real_t].mean(0), dim=0)
        fake_proto = F.normalize(embeddings[sup_fake_t].mean(0), dim=0)

        qry_embs = embeddings[qry_idx_t]
        qry_labels = labels[qry_idx_t].numpy()

        scores = ((qry_embs * fake_proto).sum(1) - (qry_embs * real_proto).sum(1)).numpy()

        aurocs.append(float(roc_auc_score(qry_labels, scores)))
        eer, _ = compute_eer_from_scores(qry_labels, scores)
        eers.append(eer)

    aurocs = np.array(aurocs, dtype=np.float32)
    eers = np.array(eers, dtype=np.float32)

    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_std": float(aurocs.std()),
        "eer_mean": float(eers.mean()),
        "eer_std": float(eers.std()),
    }


@torch.no_grad()
def evaluate_records_fewshot(
    model: nn.Module,
    records: List[Dict],
    clip_len: int,
    target_fps: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    support_per_class: int,
    n_trials: int,
    seed: int,
    embedding_source: str = "projection",
) -> Dict[str, float]:
    embs, labels = extract_embeddings_for_records(
        model=model,
        records=records,
        clip_len=clip_len,
        target_fps=target_fps,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_source=embedding_source,
    )
    return fewshot_prototype_metrics(
        embeddings=embs,
        labels=labels,
        support_per_class=support_per_class,
        n_trials=n_trials,
        seed=seed,
    )


# =============================================================================
# CHECKPOINT SELECTION
# =============================================================================

def compute_selection_score(
    target_val: Dict[str, float],
    dvf_val: Dict[str, float],
    genvid_val: Dict[str, float],
    args: argparse.Namespace,
) -> float:
    return (
        args.target_weight * target_val["auroc_mean"]
        + args.dvf_weight * dvf_val["auroc_mean"]
        + args.genvid_weight * genvid_val["auroc_mean"]
    )


def is_better(curr_score: float, best_score: float | None) -> bool:
    if best_score is None:
        return True
    return curr_score > best_score


# =============================================================================
# TRAIN
# =============================================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "train.log", mode="w"),
        ],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info("=" * 70)
    logging.info("3-Domain Mixed SupCon Fine-tuning — Strict Generator Balanced")
    logging.info("=" * 70)
    for k, v in vars(args).items():
        logging.info(f"  {k:<30}: {v}")
    logging.info("")
    logging.info(f"[BatchMix] {default_6bucket_counts(args.batch_size)}")
    logging.info("")

    target_all = tag_records(load_jsonl(args.target_jsonl), "target")
    dvf_all = tag_records(load_jsonl(args.retain_dvf_jsonl), "retain_dvf")
    genvid_all = tag_records(load_jsonl(args.retain_genvid_jsonl), "retain_genvid")

    target_train, target_val, target_test = stratified_split_generator_balanced_fake(
        target_all,
        train_per_class=args.target_train_per_class,
        val_per_class=args.target_val_per_class,
        seed=args.seed,
        domain_name="target",
    )
    dvf_train, dvf_val, dvf_test = stratified_split_plain(
        dvf_all,
        train_per_class=args.retain_dvf_train_per_class,
        val_per_class=args.retain_dvf_val_per_class,
        seed=args.seed,
        domain_name="retain_dvf",
    )
    genvid_train, genvid_val, genvid_test = stratified_split_generator_balanced_fake(
        genvid_all,
        train_per_class=args.retain_genvid_train_per_class,
        val_per_class=args.retain_genvid_val_per_class,
        seed=args.seed,
        domain_name="retain_genvid",
    )

    train_records = target_train + dvf_train + genvid_train
    random.Random(args.seed).shuffle(train_records)

    logging.info(
        f"[Target] Total={len(target_all)} | Train={len(target_train)} | Val={len(target_val)} | Test={len(target_test)}"
    )
    logging.info(
        f"[DVF]    Total={len(dvf_all)} | Train={len(dvf_train)} | Val={len(dvf_val)} | Test={len(dvf_test)}"
    )
    logging.info(
        f"[GenVid] Total={len(genvid_all)} | Train={len(genvid_train)} | Val={len(genvid_val)} | Test={len(genvid_test)}"
    )
    logging.info("")

    sampler = Mixed3DomainStrictGenBatchSampler(
        train_records,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        VideoDataset(train_records, args.clip_len, args.target_fps),
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args, device)
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def eval_domain(records: List[Dict], seed_offset: int) -> Dict[str, float]:
        return evaluate_records_fewshot(
            model=model,
            records=records,
            clip_len=args.clip_len,
            target_fps=args.target_fps,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            support_per_class=args.eval_support_per_class,
            n_trials=args.eval_n_trials,
            seed=seed_offset,
            embedding_source=args.eval_embedding_source,
        )

    baseline_target_val = eval_domain(target_val, args.eval_seed + 0)
    baseline_dvf_val = eval_domain(dvf_val, args.eval_seed + 1)
    baseline_genvid_val = eval_domain(genvid_val, args.eval_seed + 2)
    baseline_score = compute_selection_score(
        baseline_target_val, baseline_dvf_val, baseline_genvid_val, args
    )

    logging.info(
        f"[Baseline] TARGET VAL AUROC: {baseline_target_val['auroc_mean']:.4f}"
    )
    logging.info(
        f"[Baseline] DVF VAL AUROC   : {baseline_dvf_val['auroc_mean']:.4f}"
    )
    logging.info(
        f"[Baseline] GENVID VAL AUROC: {baseline_genvid_val['auroc_mean']:.4f}"
    )
    logging.info(f"[Baseline] Selection score: {baseline_score:.6f}")
    logging.info("")

    best_epoch = 0
    best_score = baseline_score

    history = [{
        "epoch": 0,
        "loss": None,
        "target_val": baseline_target_val,
        "dvf_val": baseline_dvf_val,
        "genvid_val": baseline_genvid_val,
        "selection_score": baseline_score,
    }]

    for epoch in range(1, args.epochs + 1):
        model.train()
        set_frozen_modules_eval(model, args.unfreeze_policy)
        sampler.set_epoch(epoch)

        t0 = time.time()
        epoch_loss = 0.0

        for frames, labels, _domains in train_loader:
            frames = frames.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            embs = extract_projected_embedding(model, frames)
            loss = criterion(embs, labels)

            if not torch.isfinite(loss):
                logging.error("[Train] Non-finite loss encountered. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(train_loader), 1)

        target_val_metrics = eval_domain(target_val, args.eval_seed + epoch * 10 + 0)
        dvf_val_metrics = eval_domain(dvf_val, args.eval_seed + epoch * 10 + 1)
        genvid_val_metrics = eval_domain(genvid_val, args.eval_seed + epoch * 10 + 2)

        curr_score = compute_selection_score(
            target_val_metrics, dvf_val_metrics, genvid_val_metrics, args
        )

        dt = time.time() - t0

        logging.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"target_val_auroc={target_val_metrics['auroc_mean']:.4f} | "
            f"dvf_val_auroc={dvf_val_metrics['auroc_mean']:.4f} | "
            f"genvid_val_auroc={genvid_val_metrics['auroc_mean']:.4f} | "
            f"score={curr_score:.6f} | time={dt:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "target_val": target_val_metrics,
            "dvf_val": dvf_val_metrics,
            "genvid_val": genvid_val_metrics,
            "selection_score": curr_score,
        })

        if is_better(curr_score, best_score):
            best_epoch = epoch
            best_score = curr_score

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_selection_score": best_score,
                    "args": vars(args),
                },
                out_dir / "best_model.pt",
            )
            logging.info(
                f"  -> New best: score={best_score:.6f} | "
                f"target={target_val_metrics['auroc_mean']:.4f} | "
                f"dvf={dvf_val_metrics['auroc_mean']:.4f} | "
                f"genvid={genvid_val_metrics['auroc_mean']:.4f}"
            )

        if args.save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                out_dir / "last_model.pt",
            )

    if best_epoch == 0:
        logging.info("[Info] No epoch beat baseline. Saving current model as best_model.pt")
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_selection_score": best_score,
                "args": vars(args),
            },
            out_dir / "best_model.pt",
        )

    logging.info("")
    logging.info("[Eval] Loading best checkpoint selected on validation...")
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    final_target_test = eval_domain(target_test, args.eval_seed + 9991)
    final_dvf_test = eval_domain(dvf_test, args.eval_seed + 9992)
    final_genvid_test = eval_domain(genvid_test, args.eval_seed + 9993)

    results = {
        "args": vars(args),
        "batch_mix_counts": default_6bucket_counts(args.batch_size),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "final_target_test_metrics": final_target_test,
        "final_dvf_test_metrics": final_dvf_test,
        "final_genvid_test_metrics": final_genvid_test,
        "history": history,
    }

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("")
    logging.info("=" * 70)
    logging.info(
        f"DONE | best_epoch={best_epoch} | "
        f"target_test_auroc={final_target_test['auroc_mean']:.4f}+/-{final_target_test['auroc_std']:.4f} | "
        f"dvf_test_auroc={final_dvf_test['auroc_mean']:.4f}+/-{final_dvf_test['auroc_std']:.4f} | "
        f"genvid_test_auroc={final_genvid_test['auroc_mean']:.4f}+/-{final_genvid_test['auroc_std']:.4f}"
    )
    logging.info(f"Best checkpoint -> {out_dir / 'best_model.pt'}")
    logging.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    train(args)