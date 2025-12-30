#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, argparse, csv, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- SHM/Worker güvenliği ----
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# ---- Hugging Face Transformers (SegFormer) ----
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# -----------------------
# Dataset
# -----------------------
VALID_IMG_EXT  = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
VALID_MASK_EXT = [".png",".jpg",".jpeg",".tif",".tiff"]

def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in VALID_MASK_EXT:
        p = mask_dir / f"{stem}{ext}"
        if p.exists(): return p
    # alt klasör olasılığı:
    for p in mask_dir.rglob("*"):
        if p.is_file() and p.stem == stem and p.suffix.lower() in VALID_MASK_EXT:
            return p
    return None

class WaterSegDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, size: int = 512):
        img_dir = Path(img_dir); mask_dir = Path(mask_dir)
        raw_imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in VALID_IMG_EXT])

        self.img_paths, self.mask_paths = [], []
        skipped = 0
        for ip in raw_imgs:
            mpth = _find_mask(mask_dir, ip.stem)
            if mpth is None:
                skipped += 1; continue
            self.img_paths.append(ip)
            self.mask_paths.append(mpth)
        if skipped > 0:
            warnings.warn(f"{skipped} görüntü için maske bulunamadı ve atlandı.")

        if len(self.img_paths) == 0:
            raise RuntimeError("Hiç eşleşen (görüntü+maske) bulunamadı.")

        self.size = size
        self.img_tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.mask_tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img_t = self.img_tf(img)
        mask  = self.mask_tf(mask)
        m = np.array(mask, dtype=np.uint8)
        if m.max() > 1: m = (m > 127).astype(np.uint8)
        mask_t = torch.from_numpy(m).long()  # (H,W) {0,1}
        return img_t, mask_t, str(self.img_paths[idx])

# -----------------------
# Metrics
# -----------------------
def _tp_fp_fn(pred_bin: torch.Tensor, target: torch.Tensor):
    pred = pred_bin.bool(); tgt = target.bool()
    tp = (pred & tgt).sum().item()
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()
    return tp, fp, fn

def iou_score(pred_bin, target):
    tp, fp, fn = _tp_fp_fn(pred_bin, target)
    denom = tp + fp + fn
    return tp/denom if denom>0 else 0.0

def dice_score(pred_bin, target):
    tp, fp, fn = _tp_fp_fn(pred_bin, target)
    denom = 2*tp + fp + fn
    return (2*tp)/denom if denom>0 else 0.0

def precision_score(pred_bin, target):
    tp, fp, fn = _tp_fp_fn(pred_bin, target)
    denom = tp + fp
    return tp/denom if denom>0 else 0.0

def recall_score(pred_bin, target):
    tp, fp, fn = _tp_fp_fn(pred_bin, target)
    denom = tp + fn
    return tp/denom if denom>0 else 0.0

def boundary_map(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim()==3: mask = mask.unsqueeze(1)
    inv = 1 - mask
    eroded_inv = F.max_pool2d(inv.float(), kernel_size=3, stride=1, padding=1)
    eroded = 1 - eroded_inv
    boundary = (mask - eroded).clamp(min=0)
    return boundary

def boundary_f1(pred_bin: torch.Tensor, target: torch.Tensor, tolerance: int = 2) -> float:
    if pred_bin.dim()==3: pred_bin = pred_bin.unsqueeze(1)
    if target.dim()==3:   target   = target.unsqueeze(1)
    pb = boundary_map(pred_bin); tb = boundary_map(target)
    k = 2*tolerance + 1
    pb_d = F.max_pool2d(pb, kernel_size=k, stride=1, padding=tolerance)
    tb_d = F.max_pool2d(tb, kernel_size=k, stride=1, padding=tolerance)
    tp_p = (pb * tb_d).sum().item(); pred_b = pb.sum().item()
    prec = tp_p/pred_b if pred_b>0 else 0.0
    tp_r = (tb * pb_d).sum().item(); tgt_b = tb.sum().item()
    rec = tp_r/tgt_b if tgt_b>0 else 0.0
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def average_precision_from_probs(probs: torch.Tensor, target: torch.Tensor, n_thresh: int = 101) -> float:
    if probs.dim()==3:  probs  = probs.unsqueeze(1)
    if target.dim()==3: target = target.unsqueeze(1)
    thresholds = torch.linspace(0,1,n_thresh, device=probs.device)
    precisions, recalls = [], []
    for t in thresholds:
        pred_bin = (probs >= t).long()
        precisions.append(precision_score(pred_bin, target))
        recalls.append(recall_score(pred_bin, target))
    recalls = np.array(recalls); precisions = np.array(precisions)
    order = np.argsort(recalls)
    recalls, precisions = recalls[order], precisions[order]
    return float(np.trapz(precisions, recalls))

def ece_from_probs(probs: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    if probs.dim()==3:  probs  = probs.unsqueeze(1)
    if target.dim()==3: target = target.unsqueeze(1)
    conf = probs.flatten().detach().cpu().numpy()
    lab  = target.flatten().detach().cpu().numpy()
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins-1 else (conf >= lo) & (conf <= hi)
        if not np.any(sel): continue
        acc = lab[sel].mean()
        avg_conf = conf[sel].mean()
        ece += (sel.sum()/N) * abs(acc - avg_conf)
    return float(ece)

# -----------------------
# Model
# -----------------------
def build_model(cfg_dict: dict | None = None, num_labels: int = 2, ignore_index: int = 255):
    """
    cfg_dict verilirse ckpt ile birebir aynı mimariyi kurar.
    Verilmezse B3 (hidden_sizes=[64,128,320,512], decoder_hidden_size=256) kurar.
    """
    if cfg_dict is not None:
        config = SegformerConfig.from_dict(cfg_dict)
        config.num_labels = num_labels
    else:
        config = SegformerConfig(
            num_labels=num_labels,
            id2label={0: "background", 1: "water"},
            label2id={"background":0, "water":1},
            hidden_sizes=[64, 128, 320, 512],  # B3
            decoder_hidden_size=256,            # B3 decoder
            reshape_last_stage=True
        )
    model = SegformerForSemanticSegmentation(config)
    model.config.ignore_index = ignore_index
    return model

# -----------------------
# Viz
# -----------------------
def save_prediction_grid(img_t, mask_t, prob_t, out_path):
    """
    img_t:  (3,H,W) normalized
    mask_t: (H,W) 0/1
    prob_t: (1,H,W) or (H,W) in [0,1]
    """
    # Denorm
    img = img_t.permute(1,2,0).cpu().numpy()
    img = (img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1).astype(np.float32)

    gt = mask_t.squeeze().cpu().numpy().astype(np.uint8)
    if prob_t.dim() == 3: prob_t = prob_t.squeeze(0)
    pr = (prob_t >= 0.5).float().cpu().numpy().astype(np.uint8)

    H, W = gt.shape
    assert img.shape[:2] == (H,W)

    overlay = img.copy()
    overlay[gt==1] = overlay[gt==1]*0.5 + np.array([0.0,0.8,0.0], dtype=np.float32)*0.5
    overlay[pr==1] = overlay[pr==1]*0.5 + np.array([0.9,0.0,0.0], dtype=np.float32)*0.5

    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(1,4,1); ax1.imshow(img); ax1.set_title("Image"); ax1.axis("off")
    ax2 = plt.subplot(1,4,2); ax2.imshow(gt, cmap="gray"); ax2.set_title("GT"); ax2.axis("off")
    ax3 = plt.subplot(1,4,3); ax3.imshow(pr, cmap="gray"); ax3.set_title("Pred(0.5)"); ax3.axis("off")
    ax4 = plt.subplot(1,4,4); ax4.imshow(overlay); ax4.set_title("Overlay"); ax4.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def evaluate(model, loader, device, save_samples_dir=None, max_save=8) -> Dict[str,float]:
    model.eval()
    all_probs = []; all_targets = []
    saved = 0
    if save_samples_dir: Path(save_samples_dir).mkdir(parents=True, exist_ok=True)

    for imgs, masks, paths in loader:
        imgs = imgs.to(device); masks = masks.to(device)

        out = model(pixel_values=imgs)
        logits = out.logits
        # logits'u GT boyutuna büyüt
        logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits_up, dim=1)[:,1:2,:,:]  # (N,1,H,W)

        all_probs.append(probs.cpu())
        all_targets.append(masks.unsqueeze(1).cpu())

        if save_samples_dir and saved < max_save:
            for b in range(imgs.size(0)):
                if saved >= max_save: break
                out_path = Path(save_samples_dir)/f"sample_{saved:02d}.png"
                save_prediction_grid(imgs[b].cpu(), masks[b].cpu(), probs[b].cpu(), out_path)
                saved += 1

    probs_all   = torch.cat(all_probs,   dim=0)  # (N,1,H,W)
    targets_all = torch.cat(all_targets, dim=0)  # (N,1,H,W)
    preds_bin   = (probs_all >= 0.5).long()

    metrics = {}
    metrics["IoU"]        = iou_score(preds_bin, targets_all)
    metrics["Dice"]       = dice_score(preds_bin, targets_all)
    metrics["Precision"]  = precision_score(preds_bin, targets_all)
    metrics["Recall"]     = recall_score(preds_bin, targets_all)
    metrics["BoundaryF1"] = boundary_f1(preds_bin, targets_all, tolerance=2)
    metrics["AP"]         = average_precision_from_probs(probs_all, targets_all, n_thresh=101)
    metrics["ECE"]        = ece_from_probs(probs_all, targets_all, n_bins=15)
    return metrics

# -----------------------
# Load/Save Checkpoints
# -----------------------
def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": model.config.to_dict()}, path)

def load_checkpoint(model, path: Path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    # {'model': ...} | {'state_dict': ...} | doğrudan state_dict
    if isinstance(ckpt, dict):
        if "model" in ckpt:     sd = ckpt["model"]
        elif "state_dict" in ckpt: sd = ckpt["state_dict"]
        else:                   sd = ckpt
    else:
        sd = ckpt
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k,v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
    return ckpt  # config'e erişmek için

# -----------------------
# Train Loop
# -----------------------
def make_loader(ds, batch, shuffle, workers):
    pin = torch.cuda.is_available() and workers > 0
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=workers,
                  pin_memory=pin, persistent_workers=False)
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)

def train_and_eval(args):
    set_seed(args.seed); device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "pred_samples"), exist_ok=True)

    ds = WaterSegDataset(args.images, args.masks, size=args.size)
    if len(ds) < 10: raise RuntimeError("Dataset çok küçük (>=10 önerilir).")

    n_total = len(ds)
    n_train = int(0.7*n_total); n_val = int(0.15*n_total); n_test = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(args.seed))

    train_loader = make_loader(train_ds, args.batch_size, True,  args.workers)
    val_loader   = make_loader(val_ds,   args.batch_size, False, args.workers)
    test_loader  = make_loader(test_ds,  args.batch_size, False, args.workers)

    model = build_model(cfg_dict=None, num_labels=2).to(device)
    params_count = count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = Path(args.output_dir) / "best_model.pt"

    for epoch in range(1, args.epochs+1):
        model.train(); running = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=imgs).logits
            logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits_up, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        # quick val IoU
        with torch.no_grad():
            model.eval()
            vp, vt = [], []
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(pixel_values=imgs).logits
                logits_up = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                probs = torch.softmax(logits_up, dim=1)[:,1:2,:,:]
                vp.append((probs>=0.5).long().cpu())
                vt.append(masks.unsqueeze(1).cpu())
            pv = torch.cat(vp,0); tv = torch.cat(vt,0)
            val_iou = iou_score(pv, tv)

        print(f"[Epoch {epoch:02d}/{args.epochs}] train_loss={train_loss:.4f}  val_IoU={val_iou:.4f}")
        if val_iou > best_val:
            best_val = val_iou
            save_checkpoint(model, best_path)

    # Test (best)
    raw = torch.load(best_path, map_location="cpu")
    cfg_dict = raw["config"] if isinstance(raw, dict) and "config" in raw else None
    model = build_model(cfg_dict=cfg_dict, num_labels=2).to(device)
    load_checkpoint(model, best_path, map_location=device)

    metrics = evaluate(model, test_loader, device,
                       save_samples_dir=os.path.join(args.output_dir,"pred_samples"), max_save=8)

    metrics_out = {"Params": params_count, **metrics}
    _write_metrics(args.output_dir, metrics_out)
    print(f"\nSaved best model to: {best_path}")
    print(f"Sample predictions : {os.path.join(args.output_dir, 'pred_samples')}")

def _write_metrics(out_dir: str, metrics_out: Dict[str, float]):
    # CSV + JSON + Markdown
    csv_path = Path(out_dir)/"metrics.csv"
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f); w.writerow(["Metric","Value"])
        for k,v in metrics_out.items(): w.writerow([k,v])
    with open(Path(out_dir)/"metrics.json","w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(Path(out_dir)/"metrics.md","w") as f:
        f.write("| Metric | Value |\n|---|---|\n")
        for k,v in metrics_out.items():
            if k=="Params": f.write(f"| {k} | {v} |\n")
            else:           f.write(f"| {k} | {v:.4f} |\n")
    print(f"Metrics saved to: {csv_path}  (also metrics.json / metrics.md)")

def eval_only(args):
    set_seed(args.seed); device = get_device()
    ds = WaterSegDataset(args.images, args.masks, size=args.size)
    n_total = len(ds)
    n_train = int(0.7*n_total); n_val = int(0.15*n_total); n_test = n_total - n_train - n_val
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test],
                                 generator=torch.Generator().manual_seed(args.seed))
    test_loader = make_loader(test_ds, args.batch_size, False, args.workers)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.output_dir)/"best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = raw["config"] if isinstance(raw, dict) and "config" in raw else None
    model = build_model(cfg_dict=cfg_dict, num_labels=2).to(device)
    load_checkpoint(model, ckpt_path, map_location=device)

    params_count = count_params(model)
    metrics = evaluate(model, test_loader, device,
                       save_samples_dir=os.path.join(args.output_dir,"samples_evalonly"), max_save=args.num_samples_to_save)
    metrics_out = {"Params": params_count, **metrics}
    _write_metrics(args.output_dir, metrics_out)
    print(f"Eval-only done. Samples: {os.path.join(args.output_dir,'samples_evalonly')}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="SegFormer-B3 water segmentation (train/eval).")
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--masks",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs_b3")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=1, help="0/1 güvenli; yüksek değer için /dev/shm büyütün.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples_to_save", type=int, default=8)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None, help="Eval-only için .pt yolu (default: output_dir/best_model.pt)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        eval_only(args)
    else:
        train_and_eval(args)
