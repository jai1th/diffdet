#!/usr/bin/env python3
"""
R(2+1)D Training - Uses PRE-SPATIAL-CACHED .pt files (no resize/square in training)

Trains from scratch — no pretrained weights, no normalization.
Raw [0,1] floats fed directly to the model.
"""
import os
import json
import random
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision

CONFIG = {
    # Spatially-cached split manifests (see preprocess.py). Each JSONL line:
    #   {"cache_path": "<PATH>/clip.pt", "y": 0|1, "fps": <float>, ...}
    "TRAIN_JSONL": "<PATH>/train_spatial.jsonl",
    "VAL_JSONL":   "<PATH>/val_spatial.jsonl",
    "TEST_JSONL":  "<PATH>/test_spatial.jsonl",
    "OUT_DIR": "<PATH>/dvf_baseline_run",
    "LOG_FILE": None,
    # Optional held-out probe manifest; set to None to disable.
    "META_JSONL": None,
    "EPOCHS": 25,
    "BATCH_SIZE": 16,
    "ACCUMULATION_STEPS": 1,
    "LR": 1e-4,           # higher than fine-tuning — training from scratch
    "WEIGHT_DECAY": 1e-3,
    "DROPOUT": 0.4,
    "LABEL_SMOOTHING": 0.2,
    "PATIENCE": 3,
    "USE_AUGMENTATION": True,
    "COLOR_JITTER": True,
    "BRIGHTNESS": 0.4,
    "CONTRAST": 0.3,
    "SATURATION": 0.3,
    "HUE": 0.3,
    "CLIP_LEN": 24,
    "SPATIAL_SIZE": 224,
    "SHORT_SIDE_EXPECTED": 224,
    "TARGET_FPS": 24.0,
    "NUM_WORKERS": 2,
    "PREFETCH_FACTOR": 3,
    "USE_SCHEDULER": True,
    "SCHEDULER_PATIENCE": 3,
    "SCHEDULER_FACTOR": 0.5,
    "SCHEDULER_MIN_LR": 1e-7,
    "SEED": 42,
    "AMP": True,
    "PIN_MEMORY": True,
    "FAKE_WEIGHT": 1.5,
    "ON_CORRUPT": "skip",
}

import os, sys
print("RUNNING FILE:", os.path.abspath(__file__), flush=True)
print("PYTHON:", sys.executable, flush=True)
print("CWD:", os.getcwd(), flush=True)
print("TRAIN_JSONL:", CONFIG["TRAIN_JSONL"], flush=True)
print("VAL_JSONL:", CONFIG["VAL_JSONL"], flush=True)
print("TEST_JSONL:", CONFIG["TEST_JSONL"], flush=True)

def setup_logging(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    def log(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
    return log, log_file

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def time_based_indices(n_frames, fps0, target_fps, target_frames, training, rng):
    if fps0 <= 0 or not np.isfinite(fps0):
        raise ValueError(f"Invalid fps={fps0}")
    duration = n_frames / fps0
    window = target_frames / target_fps
    if duration <= window:
        t0 = 0.0
    else:
        max_t0 = duration - window
        t0 = float(rng.uniform(0.0, max_t0)) if training else float(max_t0 / 2.0)
    times = t0 + (np.arange(target_frames, dtype=np.float32) / target_fps)
    idx = np.rint(times * fps0).astype(np.int64)
    idx = np.clip(idx, 0, n_frames - 1)
    return idx

def apply_color_jitter(video_tchw, cfg, training, rng):
    if not training:
        return video_tchw
    if not (cfg.get("USE_AUGMENTATION", True) and cfg.get("COLOR_JITTER", True)):
        return video_tchw
    b = cfg.get("BRIGHTNESS", 0.0)
    c = cfg.get("CONTRAST", 0.0)
    s = cfg.get("SATURATION", 0.0)
    h = cfg.get("HUE", 0.0)
    if b > 0:
        bf = float(rng.uniform(1.0 - b, 1.0 + b))
        video_tchw = video_tchw * bf
    if c > 0:
        cf = float(rng.uniform(1.0 - c, 1.0 + c))
        mean = video_tchw.mean(dim=(0, 2, 3), keepdim=True)
        video_tchw = (video_tchw - mean) * cf + mean
    if (s > 0) or (h > 0):
        r, g, bch = video_tchw[:, 0:1], video_tchw[:, 1:2], video_tchw[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * bch
        u = bch - y
        v = r - y
        if s > 0:
            sf = float(rng.uniform(1.0 - s, 1.0 + s))
            u = u * sf
            v = v * sf
        if h > 0:
            ang = float(rng.uniform(-h, h)) * math.pi
            ca, sa = math.cos(ang), math.sin(ang)
            u2 = ca * u - sa * v
            v2 = sa * u + ca * v
            u, v = u2, v2
        r2 = y + v
        b2 = y + u
        g2 = (y - 0.299 * r2 - 0.114 * b2) / 0.587
        video_tchw = torch.cat([r2, g2, b2], dim=1)
    return video_tchw.clamp(0.0, 1.0)


class CachedSpatialVideoDataset(Dataset):
    """
    Cached .pt is uint8 [T, 224, 224, 3].
    Pipeline per sample:
      1. time-based sampling  -> [CLIP_LEN, 224, 224, 3] uint8
      2. /255                 -> [T, C, H, W] float32 in [0, 1]
      3. optional flip + color jitter
      4. permute              -> [C, T, H, W]
    No normalization applied.
    """

    def __init__(self, videos, clip_len, spatial_size, training, config):
        self.videos = videos
        self.clip_len = int(clip_len)
        self.spatial_size = int(spatial_size)
        self.training = bool(training)
        self.cfg = config
        self.short_side_expected = int(config["SHORT_SIDE_EXPECTED"])
        self.target_fps = float(config.get("TARGET_FPS", 24.0))
        self._rng = None
        self.fail_count = 0
        self.fail_examples = []

    def __len__(self):
        return len(self.videos)

    def _get_rng(self):
        if self._rng is None:
            seed = torch.initial_seed() % 2**32
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _handle_corrupt(self, msg):
        self.fail_count += 1
        if len(self.fail_examples) < 10:
            self.fail_examples.append(msg)
        if self.cfg.get("ON_CORRUPT", "skip") == "raise":
            raise RuntimeError(msg)
        return None

    def __getitem__(self, idx):
        v = self.videos[idx]
        cache_path = v.get("cache_path", None)
        if cache_path is None:
            return self._handle_corrupt("Missing cache_path")

        if "y" in v:
            label = int(v["y"])
        else:
            lab = v.get("label", None)
            if lab is None:
                return self._handle_corrupt(f"Missing y/label for {cache_path}")
            label = 0 if str(lab).lower() == "real" else 1

        if "fps" not in v:
            return self._handle_corrupt(f"Missing fps for {cache_path}")
        fps0 = float(v["fps"])

        try:
            frames = torch.load(cache_path, map_location="cpu")
        except Exception as e:
            return self._handle_corrupt(f"torch.load failed for {cache_path}: {repr(e)}")

        if (not torch.is_tensor(frames)) or frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
            return self._handle_corrupt(f"Bad cache tensor shape/dtype for {cache_path}")

        T_total = int(frames.shape[0])
        if T_total <= 0:
            return self._handle_corrupt(f"Empty tensor for {cache_path}")

        H, W = int(frames.shape[1]), int(frames.shape[2])
        S = self.short_side_expected
        if H != S or W != S:
            return self._handle_corrupt(f"Unexpected spatial size {H}x{W}, expected {S}x{S}: {cache_path}")

        rng = self._get_rng()

        try:
            idxs = time_based_indices(T_total, fps0, self.target_fps, self.clip_len, self.training, rng)
        except Exception as e:
            return self._handle_corrupt(f"time_based_indices failed for {cache_path}: {repr(e)}")

        clip = frames[torch.from_numpy(idxs)]          # [T, H, W, 3] uint8

        # Step 1: float [T, C, H, W] in [0, 1]
        clip = clip.to(torch.float32).permute(0, 3, 1, 2) / 255.0

        # Step 2: augmentation
        if self.training and self.cfg.get("USE_AUGMENTATION", True):
            if rng.random() < 0.5:
                clip = torch.flip(clip, dims=[3])
            clip = apply_color_jitter(clip, self.cfg, training=True, rng=rng)

        # Step 3: [C, T, H, W] — no normalization
        clip = clip.permute(1, 0, 2, 3).contiguous()
        return clip, torch.tensor(label, dtype=torch.long)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    clips, labels = zip(*batch)
    return torch.stack(clips), torch.stack(labels)

def compute_eer(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler, accumulation_steps):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0

    for batch in loader:
        if batch is None:
            continue
        clips, labels = batch
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_amp = (scaler is not None) and scaler.is_enabled() and (device.type == "cuda")
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(clips)
                loss = loss_fn(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(clips)
            loss = loss_fn(logits, labels) / accumulation_steps
            loss.backward()

        accum_counter += 1

        with torch.no_grad():
            total_loss += float(loss.item()) * clips.size(0) * accumulation_steps
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total   += int(labels.size(0))

        if accum_counter == accumulation_steps:
            if use_amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

    if accum_counter != 0:
        if use_amp:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs  = []

    for batch in loader:
        if batch is None:
            continue
        clips, labels = batch
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss   = loss_fn(logits, labels)
        probs  = torch.softmax(logits, dim=1)

        total_loss += loss.item() * clips.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += clips.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs[:, 1].detach().cpu().tolist())

    if total == 0:
        return {k: float("nan") for k in ["loss","accuracy","auroc","eer","eer_threshold"]}

    metrics = {"loss": total_loss / total, "accuracy": correct / total}
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
    except Exception:
        metrics["auroc"] = float("nan")
    try:
        eer, thr = compute_eer(all_labels, all_probs)
        metrics["eer"] = eer; metrics["eer_threshold"] = thr
    except Exception:
        metrics["eer"] = float("nan"); metrics["eer_threshold"] = float("nan")
    return metrics


def main():
    cfg = CONFIG
    seed_everything(cfg["SEED"])

    Path(cfg["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    log_path = cfg["LOG_FILE"] or (Path(cfg["OUT_DIR"]) / "training.log")
    log, log_file = setup_logging(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 80)
    log("R(2+1)D TRAINING (from scratch — no pretrained weights, no normalization)")
    log("=" * 80)
    log(f"Device: {device}")
    log(f"Output directory: {cfg['OUT_DIR']}")
    log("")

    log("Configuration:")
    for k, v in cfg.items():
        log(f"  {k}: {v}")
    log("")

    log("Loading manifests...")
    with open(cfg["TRAIN_JSONL"]) as f:
        train_videos = [json.loads(l) for l in f if l.strip()]
    with open(cfg["VAL_JSONL"]) as f:
        val_videos = [json.loads(l) for l in f if l.strip()]

    def count_real_fake(items):
        r = sum(int(v["y"]) == 0 if "y" in v else str(v.get("label","")).lower() == "real" for v in items)
        return int(r), int(len(items) - r)

    tr_real, tr_fake = count_real_fake(train_videos)
    va_real, va_fake = count_real_fake(val_videos)
    log(f"Train: {len(train_videos)} videos  ({tr_real} real, {tr_fake} fake, {tr_fake/max(1,len(train_videos))*100:.1f}% fake)")
    log(f"Val:   {len(val_videos)} videos  ({va_real} real, {va_fake} fake, {va_fake/max(1,len(val_videos))*100:.1f}% fake)")
    log("")

    # -------------------------------------------------------------------------
    # Model — random init, no pretrained weights
    # -------------------------------------------------------------------------
    log("Building model (random init, no pretrained weights)...")
    model = torchvision.models.video.r2plus1d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(cfg["DROPOUT"]), nn.Linear(in_features, 2))
    model = model.to(device)
    log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    log("")

    class_weights = torch.tensor([1.0, float(cfg["FAKE_WEIGHT"])], dtype=torch.float32).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg["LABEL_SMOOTHING"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    log(f"Class weights: real=1.0, fake={cfg['FAKE_WEIGHT']}")

    scheduler = None
    if cfg["USE_SCHEDULER"]:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg["SCHEDULER_FACTOR"],
            patience=cfg["SCHEDULER_PATIENCE"],
            min_lr=cfg["SCHEDULER_MIN_LR"],
        )
        log(f"LR Scheduler: ReduceLROnPlateau (patience={cfg['SCHEDULER_PATIENCE']}, factor={cfg['SCHEDULER_FACTOR']})")
    log("")

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["AMP"] and device.type == "cuda"))

    train_dataset = CachedSpatialVideoDataset(train_videos, cfg["CLIP_LEN"], cfg["SPATIAL_SIZE"], training=True,  config=cfg)
    val_dataset   = CachedSpatialVideoDataset(val_videos,   cfg["CLIP_LEN"], cfg["SPATIAL_SIZE"], training=False, config=cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=True,
                              num_workers=cfg["NUM_WORKERS"], pin_memory=cfg["PIN_MEMORY"],
                              prefetch_factor=cfg["PREFETCH_FACTOR"], worker_init_fn=seed_worker,
                              persistent_workers=False, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["BATCH_SIZE"], shuffle=False,
                              num_workers=cfg["NUM_WORKERS"], pin_memory=cfg["PIN_MEMORY"],
                              prefetch_factor=cfg["PREFETCH_FACTOR"], worker_init_fn=seed_worker,
                              persistent_workers=False, collate_fn=collate_skip_none)

    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches:   {len(val_loader)}")
    log("")

    meta_loader = None
    if cfg.get("META_JSONL"):
        try:
            with open(cfg["META_JSONL"]) as f:
                meta_videos = [json.loads(l) for l in f if l.strip()]
            meta_dataset = CachedSpatialVideoDataset(meta_videos, cfg["CLIP_LEN"], cfg["SPATIAL_SIZE"], training=False, config=cfg)
            meta_loader  = DataLoader(meta_dataset, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                                      num_workers=0, pin_memory=False, collate_fn=collate_skip_none)
            me_real = sum(1 for v in meta_videos if int(v.get("y", 0)) == 0)
            log(f"Meta PE: {len(meta_videos)} videos ({me_real} real, {len(meta_videos)-me_real} fake)")
        except Exception as e:
            log(f"WARNING: Could not load Meta PE — {e}")
    log("")

    best_val_acc = -1.0
    best_epoch   = 0
    patience_counter = 0
    ckpt_path = Path(cfg["OUT_DIR"]) / "best_model.pt"

    log("=" * 80)
    log("TRAINING START")
    log("=" * 80)

    for epoch in range(1, cfg["EPOCHS"] + 1):
        t0 = time.time()
        log("")
        log("=" * 80)
        log(f"Epoch {epoch}/{cfg['EPOCHS']}")
        log("=" * 80)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, cfg["ACCUMULATION_STEPS"])

        val_m  = evaluate(model, val_loader,  loss_fn, device)
        me_auc = float("nan"); me_eer = float("nan")
        if meta_loader is not None:
            meta_m = evaluate(model, meta_loader, loss_fn, device)
            me_auc = meta_m["auroc"]; me_eer = meta_m["eer"]

        cur_lr = optimizer.param_groups[0]["lr"]
        log(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f}")
        log(f"Val:   loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} auroc={val_m['auroc']:.4f} eer={val_m['eer']:.4f}")
        log(f"Meta:  auroc={me_auc:.4f} eer={me_eer:.4f}")
        log(f"Time:  {(time.time()-t0)/60:.1f} min | LR={cur_lr:.2e}")
        log(f"Dataset failures (skipped so far): train={train_dataset.fail_count} val={val_dataset.fail_count}")

        if scheduler is not None:
            old_lr = cur_lr
            scheduler.step(val_m["accuracy"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                log(f"LR reduced: {old_lr:.2e} -> {new_lr:.2e}")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
                "val_auroc": val_m["auroc"], "val_eer": val_m["eer"],
                "meta_auroc": me_auc, "meta_eer": me_eer,
                "config": cfg,
            }, ckpt_path)
            log(f"✓ New best saved: val_acc={best_val_acc:.4f}")
        else:
            patience_counter += 1
            log(f"No improvement (patience {patience_counter}/{cfg['PATIENCE']})")
            if patience_counter >= cfg["PATIENCE"]:
                log("=" * 80)
                log(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} val_acc={best_val_acc:.4f}")
                log("=" * 80)
                break

    log("")
    log("=" * 80)
    log("TRAINING COMPLETE")
    log("=" * 80)
    log(f"Best: epoch {best_epoch} val_acc={best_val_acc:.4f}")
    log(f"Saved: {ckpt_path}")
    log_file.close()

if __name__ == "__main__":
    main()