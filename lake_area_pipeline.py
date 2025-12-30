#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, math, argparse, warnings
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from pyproj import CRS
from pystac_client import Client
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

EARTHSEARCH = "https://earth-search.aws.element84.com/v1"

def geocode_bbox(place: str):
    """Nominatim ile bbox (minx,miny,maxx,maxy) getirir. İnternet gerek."""
    import requests
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1, "polygon_geojson": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "lake-area/1.0"})
    r.raise_for_status()
    js = r.json()
    if not js:
        raise SystemExit(f"Yer bulunamadı: {place}")
    b = js[0]["boundingbox"]  # [south, north, west, east] as strings
    south, north, west, east = map(float, b)
    # bbox in lon/lat
    return (west, south, east, north), js[0].get("geojson")

def stac_search(bbox, start, end, cloud, max_items):
    cat = Client.open(EARTHSEARCH)
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud}},
        limit=max_items,
    )
    items = list(search.items())
    if not items:
        raise SystemExit("STAC: Kayıt bulunamadı. Tarih aralığını ve bulut eşiğini gevşetin.")
    # Az bulut -> öncele
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 1000))
    return items

def build_visual_mosaic(items, bbox_ll, target_crs, res_m, out_rgb):
    """S2-L2A visual asset'lerinden RGB mozaik üret."""
    datasets = []
    try:
        for it in items:
            if "visual" not in it.assets:
                # Bazı kataloglarda anahtar 'visual' olabilir; değilse B04/B03/B02 alınabilir
                raise SystemExit("Bu STAC item 'visual' asset içermiyor. Basitlik için 'visual' bekleniyor.")
            href = it.assets["visual"].href
            datasets.append(rasterio.open(href))

        # dst_crs, resolution ve hedef bounds (target_crs'de) ile merge
        # bbox_ll: lon/lat -> target_crs bounds
        dst_crs = target_crs
        bounds_dst = transform_bounds("EPSG:4326", dst_crs, *bbox_ll, densify_pts=21)

        mosaic, out_trans = rio_merge(
            datasets, 
            bounds=bounds_dst, 
            res=res_m, 
            dst_crs=dst_crs, 
            nodata=0
        )
        # Profile
        profile = datasets[0].profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": mosaic.shape[0],
            "crs": dst_crs,
            "transform": out_trans,
            "dtype": mosaic.dtype,
            "compress": "deflate",
            "tiled": True,
            "predictor": 2
        })

        os.makedirs(os.path.dirname(out_rgb), exist_ok=True)
        with rasterio.open(out_rgb, "w", **profile) as dst:
            dst.write(mosaic)

        return out_rgb
    finally:
        for ds in datasets:
            try:
                ds.close()
            except:
                pass

def load_segformer_b5(weights_path, device):
    # 2 sınıf: background, water; biz channel-1'i (water) kullanacağız
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b5",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    # Ağırlıklarımız HF SegFormer formatında state_dict olmalı
    state = torch.load(weights_path, map_location="cpu")
    # Hem doğrudan 'state_dict', hem düz dict destekle
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[UYARI] state_dict uyuşmazlığı. missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model

def normalize_img(arr):
    """(3,H,W) uint8/uint16 -> float32, ImageNet normalize."""
    arr = arr.astype(np.float32)
    if arr.max() > 255:
        arr /= 10000.0  # Sentinel-2 uint16 tipik
    else:
        arr /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean[:, None, None]) / std[:, None, None]
    return arr

def sliding_windows(W, H, tile, stride):
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    if xs[-1] + tile < W: xs.append(W - tile)
    if ys[-1] + tile < H: ys.append(H - tile)
    for y in ys:
        for x in xs:
            yield x, y, min(tile, W - x), min(tile, H - y)

@torch.inference_mode()
def infer_prob_map(image_tif, model, device, amp, tile, stride, thresh, out_mask=None):
    with rasterio.open(image_tif) as src:
        H, W = src.height, src.width
        prob_acc = np.zeros((H, W), dtype=np.float32)
        cnt_acc  = np.zeros((H, W), dtype=np.float32)

        pbar = tqdm(total=sum(1 for _ in sliding_windows(W,H,tile,stride)),
                    desc=f"tiling {W}x{H}", unit="tile")
        for x, y, w, h in sliding_windows(W, H, tile, stride):
            # (C,h,w)
            tile_arr = src.read(window=rasterio.windows.Window(x, y, w, h))  # 3,h,w
            if tile_arr.shape[0] == 4:  # bazen alfa kanalı olabilir
                tile_arr = tile_arr[:3]
            # Tam kareye pad (model sabit boyut isterse)
            pad_w = tile - w
            pad_h = tile - h
            if pad_w or pad_h:
                pad = ((0,0),(0,pad_h),(0,pad_w))
                tile_arr = np.pad(tile_arr, pad, mode="edge")

            tile_norm = normalize_img(tile_arr)
            tens = torch.from_numpy(tile_norm).unsqueeze(0).to(device)  # 1,3,tile,tile

            if amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    out = model(pixel_values=tens).logits  # 1,2,tile,tile
            else:
                out = model(pixel_values=tens).logits

            # channel-1: water
            out = torch.nn.functional.interpolate(out, size=(tile, tile), mode="bilinear", align_corners=False)
            water = torch.softmax(out, dim=1)[:,1]  # 1,tile,tile
            prob = water.squeeze(0).detach().float().cpu().numpy()

            # pad'i kırp
            prob = prob[:h, :w]
            prob_acc[y:y+h, x:x+w] += prob
            cnt_acc[y:y+h, x:x+w]  += 1.0
            pbar.update(1)
        pbar.close()

        prob_map = (prob_acc / np.maximum(cnt_acc, 1e-6)).astype(np.float32)
        mask = (prob_map >= float(thresh)).astype(np.uint8)

        if out_mask:
            prof = src.profile.copy()
            prof.update({"count": 1, "dtype": "uint8", "compress":"deflate", "tiled":True})
            with rasterio.open(out_mask, "w", **prof) as dst:
                dst.write((mask*255).astype(np.uint8), 1)

        return prob_map, mask

def overlay_png(rgb_tif, mask, out_overlay, alpha=0.45, color=(0,200,255), mode="fill"):
    with rasterio.open(rgb_tif) as src:
        arr = src.read()  # 3,H,W
    img = np.moveaxis(arr, 0, 2)  # H,W,3
    img = np.clip(img, 0, 255).astype(np.uint8) if img.max()<=255 else (img/ (img.max()/255.)).astype(np.uint8)
    base = Image.fromarray(img)
    H, W = mask.shape
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    ov = np.zeros((H,W,4), dtype=np.uint8)

    if mode == "fill":
        r,g,b = color
        ov[mask==1] = (r,g,b, int(255*alpha))
    else:  # contour
        import cv2
        contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(ov, [c], -1, (color[0],color[1],color[2], int(255*alpha)), thickness=2)

    overlay = Image.fromarray(ov, mode="RGBA")
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    out.save(out_overlay)

def side_by_side(rgb_tif, mask, out_compare):
    with rasterio.open(rgb_tif) as src:
        arr = src.read()
    img = np.moveaxis(arr, 0, 2)
    img = np.clip(img, 0, 255).astype(np.uint8) if img.max()<=255 else (img/ (img.max()/255.)).astype(np.uint8)
    pred = (mask*255).astype(np.uint8)
    pred_rgb = np.stack([np.zeros_like(pred), pred, pred], axis=-1)  # cyan-ish
    h, w = pred.shape
    # resize pred_rgb to match
    Image.fromarray(np.concatenate([img, pred_rgb], axis=1)).save(out_compare)

def heatmap_png(prob_map, out_heatmap):
    pm = (np.clip(prob_map, 0, 1)*255).astype(np.uint8)
    Image.fromarray(pm).save(out_heatmap)

def compute_area_km2(mask_tif, mask=None):
    """mask_tif'in CRS'i metrik olmalı (örn EPSG:32638)."""
    with rasterio.open(mask_tif) as src:
        if mask is None:
            mask = (src.read(1) > 0).astype(np.uint8)
        tr = src.transform
        # piksel boyutu (m)
        px = abs(tr.a)
        py = abs(tr.e)
        area_m2 = float(mask.sum()) * px * py
        return area_m2 / 1e6

def parse_color(s):
    r,g,b = s.split(",")
    return (int(r), int(g), int(b))

def main():
    ap = argparse.ArgumentParser()
    # 1) Veri indirme/mozaik
    ap.add_argument("--place", type=str, help="Örn: 'Van Gölü, Turkey'")
    ap.add_argument("--bbox", type=str, help="minx,miny,maxx,maxy (lon/lat)")
    ap.add_argument("--start", type=str, default="2024-06-01")
    ap.add_argument("--end",   type=str, default="2024-10-01")
    ap.add_argument("--cloud", type=float, default=20.0)
    ap.add_argument("--max_items", type=int, default=6)
    ap.add_argument("--target_crs", type=str, default="EPSG:32638")
    ap.add_argument("--res_m", type=float, default=10.0)
    ap.add_argument("--out_rgb", type=str, required=True)

    # 2) İnfer
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", type=str, default="true")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=448)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out_mask", type=str, required=True)

    # 3) Görseller
    ap.add_argument("--out_overlay", type=str, required=True)
    ap.add_argument("--out_compare", type=str, required=True)
    ap.add_argument("--out_heatmap", type=str, required=True)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--overlay_color", type=str, default="0,200,255")
    ap.add_argument("--overlay_mode", type=str, default="fill", choices=["fill","contour"])

    # 4) Alan doğrulama
    ap.add_argument("--gt_area_km2", type=float, default=None)

    args = ap.parse_args()
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    # Bbox tespiti
    if args.place:
        bbox_ll, geojson = geocode_bbox(args.place)
    elif args.bbox:
        minx, miny, maxx, maxy = map(float, args.bbox.split(","))
        bbox_ll = (minx, miny, maxx, maxy)
    else:
        raise SystemExit("Either --place or --bbox verin.")

    # STAC araması ve mozaik
    items = stac_search(bbox_ll, args.start, args.end, args.cloud, args.max_items)
    print(f"STAC: {len(items)} item seçildi. En düşük bulut: {items[0].properties.get('eo:cloud_cover', 'NA')}")
    rgb_path = build_visual_mosaic(items, bbox_ll, args.target_crs, args.res_m, args.out_rgb)
    print(f"Mozaik hazır: {rgb_path}")

    # Cihaz/AMP
    amp = str(args.amp).lower() == "true"
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA bekleniyordu ama bulunamadı. nvidia-smi / PyTorch CUDA kurulumunu kontrol edin.")
        device = torch.device(args.device if ":" in args.device else "cuda")
    else:
        device = torch.device("cpu")
        amp = False
    print(f"Device: {device.type} | AMP: {amp}")

    # Model
    model = load_segformer_b5(args.weights, device)

    # İnfer
    prob_map, mask = infer_prob_map(rgb_path, model, device, amp, args.tile, args.stride, args.thresh, out_mask=args.out_mask)
    print(f"Mask kaydedildi: {args.out_mask}")

    # Overlay ve görseller
    color = parse_color(args.overlay_color)
    overlay_png(rgb_path, mask, args.out_overlay, alpha=args.overlay_alpha, color=color, mode=args.overlay_mode)
    side_by_side(rgb_path, mask, args.out_compare)
    heatmap_png(prob_map, args.out_heatmap)
    print(f"Overlay: {args.out_overlay}\nCompare: {args.out_compare}\nHeatmap: {args.out_heatmap}")

    # Alan
    area_km2 = compute_area_km2(args.out_mask, mask)
    print(f"TAHMİNİ ALAN: {area_km2:.2f} km²")
    if args.gt_area_km2:
        err = area_km2 - args.gt_area_km2
        rel = 100.0 * err / args.gt_area_km2
        print(f"GERÇEK: {args.gt_area_km2:.2f} km² | HATA: {err:+.2f} km² ({rel:+.2f}%)")

if __name__ == "__main__":
    main()
