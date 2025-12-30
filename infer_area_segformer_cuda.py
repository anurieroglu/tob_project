#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import Resampling
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def coord_starts(L, tile, stride):
    if L <= tile: return [0]
    starts = list(range(0, L - tile + 1, stride))
    if starts[-1] != L - tile:
        starts.append(L - tile)
    return starts

def norm_rgb(arr, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    arr = arr.astype(np.float32)
    vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if vmax > 1.5:
        arr = arr / (255.0 if vmax <= 255.0 else 10000.0)
        arr = np.clip(arr, 0.0, 1.0)
    chw = np.transpose(arr, (2, 0, 1))
    mean = np.array(mean, dtype=np.float32)[:, None, None]
    std  = np.array(std,  dtype=np.float32)[:, None, None]
    return (chw - mean) / std

def try_load_state(model, state):
    """state_dict formatlarını esnek yükle."""
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"state_dict loaded. missing={len(missing)}, unexpected={len(unexpected)}")

def build_model(weights_path, backbone, num_labels, device, compile_flag=False):
    cfg = SegformerConfig.from_pretrained(backbone, num_labels=num_labels)
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone, config=cfg, ignore_mismatched_sizes=True
    )
    state = torch.load(weights_path, map_location="cpu")
    try_load_state(model, state)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # CUDA iyileştirmeleri
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if compile_flag and hasattr(torch, "compile"):
            model = torch.compile(model)  # PyTorch 2+
    return model

def pixel_area_m2_from_transform(transform: Affine, crs, height, width, approx_lat=None):
    px_w = abs(transform.a); px_h = abs(transform.e)
    if crs is not None and getattr(crs, "is_projected", False):
        return float(px_w * px_h)
    if approx_lat is None:
        cy = (height / 2.0)
        lat_deg = transform.f + transform.e * cy
    else:
        lat_deg = approx_lat
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return float((px_w * meters_per_deg_lon) * (px_h * meters_per_deg_lat))

def infer_tiled(ds, model, device, tile=512, stride=512, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                water_channel=1, amp=True):
    H, W = ds.height, ds.width
    xs = coord_starts(W, tile, stride)
    ys = coord_starts(H, tile, stride)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight   = np.zeros((H, W), dtype=np.float32)

    pbar = tqdm(total=len(xs)*len(ys), desc=f"tiling {H}x{W}", unit="tile")
    for y0 in ys:
        for x0 in xs:
            w = min(tile, W - x0); h = min(tile, H - y0)
            win = Window(x0, y0, w, h)
            bands = [1,2,3]
            patch = ds.read(indexes=bands, window=win, out_dtype="float32")  # [3,h,w]
            if patch.shape[0] < 3:
                c, ph, pw = patch.shape
                temp = np.zeros((3, ph, pw), dtype=np.float32)
                temp[:c] = patch; patch = temp
            hwc = np.transpose(patch, (1,2,0))               # [h,w,3]
            chw = norm_rgb(hwc, mean=mean, std=std)          # [3,h,w]
            tens = torch.from_numpy(chw).unsqueeze(0).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp and device.type=="cuda"):
                out = model(tens)
                logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
                if logits.shape[1] == 2:
                    logits = logits[:, 1:2, :, :]
                else:
                    logits = logits[:, 0:1, :, :]
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                probs  = torch.sigmoid(logits).squeeze(0).squeeze(0).float().cpu().numpy()

            prob_sum[y0:y0+h, x0:x0+w] += probs
            weight[y0:y0+h, x0:x0+w]   += 1.0
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(1)
    pbar.close()

    weight[weight == 0.0] = 1.0
    return prob_sum / weight  # [H,W] float32

def save_mask_geotiff(path, mask01, ref_ds):
    profile = ref_ds.profile.copy()
    profile.update({"count": 1, "dtype": "uint8", "compress": "deflate", "predictor": 2})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask01.astype(np.uint8), 1)

def auto_display_rgb(ds, max_width=3000):
    H, W = ds.height, ds.width
    if max_width and W > max_width:
        scale = W / float(max_width)
        out_w = max_width
        out_h = max(1, int(round(H / scale)))
    else:
        out_w, out_h = W, H
    rgb = ds.read(indexes=[1,2,3], out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
    rgb = np.transpose(rgb, (1,2,0))
    vmax = float(np.nanmax(rgb)) if rgb.size else 1.0
    if vmax > 1.5:
        rgb = rgb / (255.0 if vmax<=255 else 10000.0)
        rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)

def resize_prob_to(prob, out_h, out_w):
    t = torch.from_numpy(prob)[None,None,:,:].float()
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()

def mask_outline(mask01):
    m = mask01.astype(np.uint8)
    pad = np.pad(m, 1, mode='edge')
    neigh_sum = (
      pad[0:-2,0:-2]+ pad[0:-2,1:-1]+ pad[0:-2,2:]+
      pad[1:-1,0:-2]+ pad[1:-1,1:-1]+ pad[1:-1,2:]+
      pad[2:,0:-2]+ pad[2:,1:-1]+ pad[2:,2:]
    )
    edge = (m==1) & (neigh_sum<9)
    return edge.astype(np.uint8)

def overlay_on_rgb(rgb_uint8, mask01, prob=None, alpha=0.45, color=(0,200,255), mode="fill"):
    overlay = rgb_uint8.astype(np.float32).copy()
    if mode == "edge":
        edges = mask_outline(mask01)
        overlay[edges==1] = np.array(color, dtype=np.float32)
    else:
        color_arr = np.zeros_like(overlay); color_arr[:,:,:]=np.array(color, dtype=np.float32)
        m = (mask01[...,None]==1).astype(np.float32)
        overlay = (1.0-alpha)*overlay + alpha*(m*color_arr + (1.0-m)*overlay)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def save_png(path, img_uint8):
    Image.fromarray(img_uint8).save(path)

def main():
    ap = argparse.ArgumentParser(description="CUDA/CPU SegFormer B5 ile alan tahmini + overlay")
    ap.add_argument("--image", required=True, help="Girdi GeoTIFF (RGB ilk 3 bant)")
    ap.add_argument("--weights", required=True, help=".pt state_dict (SegFormer)")
    ap.add_argument("--backbone", default="nvidia/mit-b5", help="HF backbone id")
    ap.add_argument("--device", default="cuda", help="'cuda', 'cuda:0', 'cpu' ...")
    ap.add_argument("--amp", type=lambda x: str(x).lower()!="false", default=True, help="CUDA autocast/FP16 (true/false)")
    ap.add_argument("--compile", action="store_true", help="PyTorch 2.0 torch.compile()")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN)
    ap.add_argument("--std",  type=float, nargs=3, default=IMAGENET_STD)
    ap.add_argument("--gt_area_km2", type=float, default=None)

    ap.add_argument("--save_mask", default=None, help="Maske GeoTIFF (0/1)")
    ap.add_argument("--save_overlay", default=None, help="Overlay PNG (RGB+mask)")
    ap.add_argument("--save_compare", default=None, help="Yan yana karşılaştırma PNG")
    ap.add_argument("--save_heatmap", default=None, help="Olasılık ısı haritası overlay PNG")
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--overlay_color", type=str, default="0,200,255")
    ap.add_argument("--overlay_mode", type=str, default="fill", choices=["fill","edge"])
    ap.add_argument("--max_overlay_width", type=int, default=3000)

    args = ap.parse_args()
    # device seçimi
    want = args.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA bulunamadı, CPU'ya düşüyorum.")
        device = torch.device("cpu")
        args.amp = False
    else:
        device = torch.device(want)

    color = tuple(int(c) for c in args.overlay_color.split(","))
    model = build_model(args.weights, args.backbone, num_labels=2, device=device, compile_flag=args.compile)
    print(f"Device: {device} | AMP: {args.amp}")

    with rasterio.open(args.image) as ds:
        prob = infer_tiled(
            ds, model, device,
            tile=args.tile, stride=args.stride,
            mean=args.mean, std=args.std,
            water_channel=1, amp=args.amp
        )
        px_area_m2 = pixel_area_m2_from_transform(ds.transform, ds.crs, ds.height, ds.width)
        water_mask = (prob >= args.thresh).astype(np.uint8)
        water_pixels = int(water_mask.sum())
        pred_area_km2 = (water_pixels * px_area_m2) / 1_000_000.0

        if args.save_mask:
            save_mask_geotiff(args.save_mask, water_mask, ds)

        wants_overlay = bool(args.save_overlay or args.save_compare or args.save_heatmap)
        if wants_overlay:
            vis_rgb = auto_display_rgb(ds, max_width=args.max_overlay_width)
            vh, vw = vis_rgb.shape[:2]
            prob_small = resize_prob_to(prob, vh, vw)
            mask_small = (prob_small >= args.thresh).astype(np.uint8)

            if args.save_overlay:
                over = overlay_on_rgb(vis_rgb, mask_small, prob_small, alpha=args.overlay_alpha, color=color, mode=args.overlay_mode)
                save_png(args.save_overlay, over)

            if args.save_compare:
                over = overlay_on_rgb(vis_rgb, mask_small, prob_small, alpha=args.overlay_alpha, color=color, mode=args.overlay_mode)
                comp = np.concatenate([vis_rgb, over], axis=1)
                save_png(args.save_compare, comp)

            if args.save_heatmap:
                heat = (prob_small * 255.0).astype(np.uint8)
                heat_rgb = np.stack([np.zeros_like(heat), np.zeros_like(heat), heat], axis=-1).astype(np.uint8)
                alpha = (prob_small[...,None] * (255*args.overlay_alpha)).astype(np.uint8)
                dst = ( (alpha/255.0)*heat_rgb.astype(np.float32) + (1.0 - alpha/255.0)*vis_rgb.astype(np.float32) )
                save_png(args.save_heatmap, np.clip(dst,0,255).astype(np.uint8))

    out = {
        "image": os.path.abspath(args.image),
        "weights": os.path.abspath(args.weights),
        "backbone": args.backbone,
        "device": str(device),
        "amp": bool(args.amp),
        "tile": args.tile,
        "stride": args.stride,
        "thresh": args.thresh,
        "pixel_area_m2": round(float(px_area_m2), 6),
        "pred_area_km2": round(float(pred_area_km2), 4),
    }
    if args.gt_area_km2 is not None:
        abs_err = abs(pred_area_km2 - args.gt_area_km2)
        rel_err = (abs_err / max(1e-9, args.gt_area_km2)) * 100.0
        out.update({
            "gt_area_km2": float(args.gt_area_km2),
            "abs_err_km2": round(float(abs_err), 4),
            "rel_err_percent": round(float(rel_err), 3),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # Parçalı/çok büyük img'lerde bellek parçalanmasını azaltır
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
