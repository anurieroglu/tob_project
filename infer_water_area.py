#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Görüntü ölçekleme / normalizasyon ----------

def robust_scale(img: np.ndarray, pmin=2, pmax=98, gamma=1.0):
    """
    RGB (H,W,3) -> [0,1]
    - Kanal başına yüzde değerlerine göre germe (pmin/pmax)
    - Opsiyonel gamma ( >1 daha aydınlık görünür )
    """
    x = img.astype(np.float32)
    mx = x.max()
    if img.dtype == np.uint16 or mx > 255:
        x = np.clip(x / 10000.0, 0, 1)
    elif mx > 1:
        x = np.clip(x / 255.0, 0, 1)
    else:
        x = np.clip(x, 0, 1)

    out = np.zeros_like(x)
    for c in range(3):
        lo, hi = np.percentile(x[..., c], [pmin, pmax])
        if hi - lo < 1e-6:
            out[..., c] = x[..., c]
        else:
            out[..., c] = np.clip((x[..., c] - lo) / (hi - lo), 0, 1)

    if abs(gamma - 1.0) > 1e-3:
        out = np.clip(out, 0, 1) ** (1.0 / gamma)
    return np.clip(out, 0, 1)

def blue_index_01(img01):
    """Basit mavi-dominant indeks (B-R)/(B+R), [0,1] normalize."""
    B = img01[..., 2].astype(np.float32)
    R = img01[..., 0].astype(np.float32)
    idx = (B - R) / (B + R + 1e-6)
    mn, mx = np.percentile(idx, [2, 98])
    return np.clip((idx - mn) / (mx - mn + 1e-6), 0, 1)

def to_tensor(batch_np):
    # (B,H,W,3) -> (B,3,H,W), ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    t = torch.from_numpy(batch_np).permute(0,3,1,2).float()
    return (t - mean) / std

# ---------- Tiling + COS blending + TTA ----------

def cosine_window_2d(tile):
    """Kenarları yumuşatma için 2D kosinüs ağırlık matrisi [0,1]."""
    h = w = tile
    y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(h) / max(h - 1, 1)))
    x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / max(w - 1, 1)))
    wy = y[:, None]
    wx = x[None, :]
    w2 = wy * wx
    # ortada ağırlık yüksek, kenarda düşür (tersini al)
    w2 = (w2 - w2.min()) / max(w2.ptp(), 1e-6)
    w2 = 1 - w2
    return w2.astype(np.float32)

def sliding_windows(H, W, tile, overlap):
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    return ys, xs

def tta_forward(model_forward, x):
    """id, hflip, vflip, hvflip ortalaması."""
    outs = []
    outs.append(model_forward(x))
    xf = torch.flip(x, dims=[-1]); outs.append(torch.flip(model_forward(xf), dims=[-1]))
    xf = torch.flip(x, dims=[-2]); outs.append(torch.flip(model_forward(xf), dims=[-2]))
    xf = torch.flip(x, dims=[-2, -1]); outs.append(torch.flip(model_forward(xf), dims=[-2, -1]))
    return torch.mean(torch.stack(outs, 0), 0)

def accumulate_probs(img, tile, overlap, device, model_forward, to_tensor_fn, batch, use_tta=True):
    """
    img: (H,W,3) [0,1]
    Dönen: probs (H,W) [0,1]
    """
    H, W, _ = img.shape
    ys, xs = sliding_windows(H, W, tile, overlap)
    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight   = np.zeros((H, W), dtype=np.float32)
    win = cosine_window_2d(tile)

    patches, coords = [], []

    def flush():
        nonlocal patches, coords, prob_acc, weight
        if not patches: return
        batch_np = np.stack(patches, axis=0)
        with torch.no_grad():
            tin = to_tensor_fn(batch_np).to(device)
            logits = tta_forward(model_forward, tin) if use_tta else model_forward(tin)
            probs = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        for (y, x), p in zip(coords, probs):
            w = win
            prob_acc[y:y+tile, x:x+tile] += p * w
            weight  [y:y+tile, x:x+tile] += w
        patches, coords = [], []

    for y in ys:
        for x in xs:
            patches.append(img[y:y+tile, x:x+tile, :])
            coords.append((y, x))
            if len(patches) >= batch:
                flush()
        flush()

    weight[weight == 0] = 1.0
    return np.clip(prob_acc / weight, 0, 1)

# ---------- Model yükleme ----------

def load_model(model_path: str, device: torch.device):
    """
    Eğitiminle uyumlu: HF SegFormer-B5 şablonu + 'net.' önek temizliği + loss param atımı
    """
    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        print("[load] TorchScript model yüklendi.")
        return m, "torchscript"
    except Exception:
        pass

    try:
        obj = torch.load(model_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Checkpoint okunamadı: {e}")

    from transformers import SegformerForSemanticSegmentation
    base = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    if hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        sd = obj

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("net."):
            cleaned[k[len("net."):]] = v
        elif k.startswith("dice") or k.startswith("bce"):
            continue
        else:
            cleaned[k] = v

    missing, unexpected = base.load_state_dict(cleaned, strict=False)
    print(f"[load] SegFormer-B5 yüklendi | missing={len(missing)} unexpected={len(unexpected)}")
    base.to(device).eval()
    return base, "hf"

def model_forward_factory(model, kind: str):
    def f(x):
        if kind == "hf":
            out = model(pixel_values=x)
            logits = out.logits
            logits = F.interpolate(logits, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
            return logits
        else:
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if out.shape[1] == 1:
                return out
            return out[:, 1:2, ...]
    return f

# ---------- Eşik seçimi / post-proc ----------

def pick_stable_threshold(prob, fallback=0.5):
    """Perimetre/alan oranını minimize eden eşik (0.3–0.7 aralığında tarar)."""
    try:
        cands = np.linspace(0.3, 0.7, 9)
        best_t, best_score = fallback, float("inf")
        for t in cands:
            m = prob >= t
            ar = m.sum()
            if ar < 500:
                continue
            per = find_boundaries(m, mode="inner").sum()
            score = per / (ar + 1e-6)
            if score < best_score:
                best_score, best_t = score, float(t)
        return best_t
    except Exception:
        return fallback

def apply_morphology(binmask, min_ha, px_area_m2, keep_top_k=0):
    """Küçük objeleri/oyukları temizle + opsiyonel en büyük K bileşeni tut."""
    px_per_ha = max(1, int(round(10000.0 / px_area_m2)))
    min_area_px = max(10, int(min_ha * px_per_ha))
    m = remove_small_objects(binmask.astype(bool), min_size=min_area_px)
    m = remove_small_holes(m, area_threshold=min_area_px)

    if keep_top_k and keep_top_k > 0:
        lab = label(m)
        if lab.max() > keep_top_k:
            props = regionprops(lab)
            areas = sorted([(p.label, p.area) for p in props], key=lambda x: x[1], reverse=True)
            keep = set([a[0] for a in areas[:keep_top_k]])
            m = np.isin(lab, list(keep))
    return m

# ---------- Ana akış ----------

def run(args):
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) GeoTIFF oku
    with rasterio.open(args.tif) as src:
        rgb = src.read(indexes=[1,2,3], out_shape=(3, src.height, src.width), resampling=Resampling.bilinear)
        rgb = np.transpose(rgb, (1,2,0))
        transform: Affine = src.transform
        crs = src.crs
        res_x = abs(transform.a); res_y = abs(transform.e)
        px_area_m2 = res_x * res_y
        print(f"[geo] size={rgb.shape[:2]}, res≈{res_x:.2f}m × {res_y:.2f}m, px_area={px_area_m2:.2f} m²")

    # 2) Sağlam normalize
    rgb01 = robust_scale(rgb, pmin=args.pmin, pmax=args.pmax, gamma=args.gamma)

    # 3) Tile pad
    H, W, _ = rgb01.shape
    step = args.tile - args.overlap
    need_h = (math.ceil((H - args.overlap) / step) * step + args.overlap) - H
    need_w = (math.ceil((W - args.overlap) / step) * step + args.overlap) - W
    if need_h > 0 or need_w > 0:
        rgb01 = np.pad(rgb01, ((0, need_h), (0, need_w), (0, 0)), mode="reflect")
    Ph, Pw, _ = rgb01.shape

    # 4) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kind = load_model(args.model, device)
    model_fwd = model_forward_factory(model, kind)

    # 5) Olasılık haritası (TTA + COS blend)
    probs = accumulate_probs(rgb01, args.tile, args.overlap, device, model_fwd, to_tensor, args.batch, use_tta=not args.no_tta)
    probs = probs[:H, :W]

    # 6) (opsiyonel) mavi-indeks harmanı
    if args.blue_blend > 0:
        bi = blue_index_01(robust_scale(rgb, pmin=args.pmin, pmax=args.pmax, gamma=1.0))
        probs = np.clip((1 - args.blue_blend) * probs + args.blue_blend * bi, 0, 1)

    # 7) Eşik
    if args.thr is not None:
        thr = float(args.thr)
    else:
        try:
            thr_otsu = threshold_otsu(probs[(probs > 0) & (probs < 1)])
        except Exception:
            thr_otsu = 0.5
        thr = pick_stable_threshold(probs, fallback=thr_otsu)

    binmask = probs >= thr
    binmask = apply_morphology(binmask, args.min_ha, px_area_m2, keep_top_k=args.keep_top_k)

    # 8) Alanlar
    water_px = int(binmask.sum())
    water_m2 = float(water_px * px_area_m2)
    water_ha = water_m2 / 10000.0
    water_km2 = water_m2 / 1e6

    metrics = {
        "image": str(args.tif),
        "model": str(args.model),
        "pixel_area_m2": px_area_m2,
        "water_pixels": water_px,
        "water_area_m2": water_m2,
        "water_area_ha": water_ha,
        "water_area_km2": water_km2,
        "threshold_used": float(thr),
        "pmin": args.pmin, "pmax": args.pmax, "gamma": args.gamma,
        "blue_blend": args.blue_blend, "keep_top_k": args.keep_top_k,
        "tile": args.tile, "overlap": args.overlap, "tta": (not args.no_tta)
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 9) Maske GeoTIFF
    mask_u8 = (binmask.astype(np.uint8) * 255)
    with rasterio.open(
        out_dir / "water_mask.tif", "w",
        driver="GTiff", height=mask_u8.shape[0], width=mask_u8.shape[1],
        count=1, dtype=mask_u8.dtype, crs=crs, transform=transform, compress="LZW"
    ) as dst:
        dst.write(mask_u8, 1)

    # 10) Görseller
    borders = find_boundaries(binmask, mode="inner").astype(np.uint8)

    # overlay
    overlay = (robust_scale(rgb, pmin=args.pmin, pmax=args.pmax, gamma=args.gamma) * 255).astype(np.uint8)
    blue = np.zeros_like(overlay); blue[..., 2] = 255
    alpha = 0.35
    overlay[binmask] = (overlay[binmask].astype(np.float32) * (1 - alpha) + blue[binmask].astype(np.float32) * alpha).astype(np.uint8)
    overlay[borders.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)

    plt.figure(figsize=(12, 12)); plt.imshow(overlay); plt.axis("off")
    plt.title(f"Overlay — Su Alanı: {water_ha:.2f} ha ({water_km2:.3f} km²)")
    plt.tight_layout(); plt.savefig(out_dir / "overlay.png", dpi=200); plt.close()

    # comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(np.clip(robust_scale(rgb, pmin=args.pmin, pmax=args.pmax, gamma=args.gamma), 0, 1)); axes[0].set_title("RGB"); axes[0].axis("off")
    im1 = axes[1].imshow(probs, vmin=0, vmax=1); axes[1].set_title(f"Olasılık (thr≈{thr:.2f})"); axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    show_mask = np.zeros_like(overlay, dtype=np.float32); show_mask[..., 2] = binmask.astype(np.float32)
    axes[2].imshow(np.clip(robust_scale(rgb, pmin=args.pmin, pmax=args.pmax, gamma=args.gamma)*0.7 + show_mask*0.3, 0, 1))
    axes[2].imshow(borders, cmap="Reds", alpha=0.8); axes[2].set_title(f"Maske + Sınır — {water_ha:.2f} ha"); axes[2].axis("off")
    plt.tight_layout(); plt.savefig(out_dir / "comparison.png", dpi=200); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="RGB GeoTIFF (ör. beysehir_rgb_10m_clean.tif)")
    parser.add_argument("--model", required=True, help=".pt model dosyası")
    parser.add_argument("--out", default="outputs_water", help="Çıktı klasörü")
    parser.add_argument("--tile", type=int, default=1024, help="Tile boyutu (>=512 önerilir)")
    parser.add_argument("--overlap", type=int, default=128, help="Bindirme")
    parser.add_argument("--batch", type=int, default=4, help="Batch")
    parser.add_argument("--min_ha", type=float, default=0.1, help="Temizlikte min alan (ha)")
    parser.add_argument("--keep_top_k", type=int, default=1, help="En büyük K su kütlesini tut (0=kapalı)")
    parser.add_argument("--pmin", type=int, default=2, help="Girdi germe alt yüzdeliği")
    parser.add_argument("--pmax", type=int, default=98, help="Girdi germe üst yüzdeliği")
    parser.add_argument("--gamma", type=float, default=1.25, help="Girdi gamma (aydınlık için 1.2–1.4)")
    parser.add_argument("--blue_blend", type=float, default=0.0, help="Olasılıkla mavi-indeks harmanı (0–0.3 tipik)")
    parser.add_argument("--thr", type=float, default=None, help="Sabit eşik (None=otomatik)")
    parser.add_argument("--no_tta", action="store_true", help="TTA kapat")
    args = parser.parse_args()
    run(args)
