#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, tempfile, glob, shutil
import ee, geemap
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

# -------- EE init (Service Account) --------
def ee_init_with_service_account():
    sa = os.environ.get("EE_SA_EMAIL")
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    proj = os.environ.get("EE_PROJECT")
    assert sa, "EE_SA_EMAIL set edilmeli"
    assert key and os.path.exists(key), "GOOGLE_APPLICATION_CREDENTIALS (JSON key) yok"
    assert proj, "EE_PROJECT set edilmeli"
    ee.Initialize(ee.ServiceAccountCredentials(sa, key), project=proj)
    print(f"[EE] SA login: {sa} (project={proj})")

# -------- AOI --------
def aoi_from_bbox(bbox_str: str):
    vals = [float(x.strip()) for x in bbox_str.split(",")]
    assert len(vals) == 4, "--bbox minLon,minLat,maxLon,maxLat"
    minx, miny, maxx, maxy = vals
    return ee.Geometry.BBox(minx, miny, maxx, maxy), (minx, miny, maxx, maxy)

# -------- Cloud prob (JOIN YOK, LOOKUP VAR) --------
def add_cloudprob(ic):
    s2c = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    def _attach(img):
        img_id = img.get('system:index')
        match = s2c.filter(ee.Filter.equals('system:index', img_id)).first()
        prob = ee.Image(ee.Algorithms.If(
            match, ee.Image(match).select('probability'),
            ee.Image.constant(0).toUint8().rename('probability')
        )).rename('cloud_prob')
        return img.addBands(prob)
    return ic.map(_attach)

def mask_s2_sr(img, cloud_thr=40):
    qa60 = img.select('QA60')
    cloud_bits = qa60.bitwiseAnd(1 << 10).neq(0).Or(qa60.bitwiseAnd(1 << 11).neq(0))
    cp_mask = img.select('cloud_prob').lt(cloud_thr)  # 0–100
    scl = img.select('SCL')
    not_bad = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(cp_mask.And(cloud_bits.Not()).And(not_bad))

def visualize_truecolor(img, minv=0, maxv=3000, gamma=1.3):
    return img.select(['B4','B3','B2']).visualize(min=minv, max=maxv, gamma=gamma)

def compute_percentiles(img, region, scale, pmin=2, pmax=98):
    reducer = ee.Reducer.percentile([pmin, pmax])
    stats = img.select(['B4','B3','B2']).reduceRegion(reducer, region, scale, maxPixels=1e9)
    b4_min = stats.getNumber(f'B4_p{pmin}'); b3_min = stats.getNumber(f'B3_p{pmin}'); b2_min = stats.getNumber(f'B2_p{pmin}')
    b4_max = stats.getNumber(f'B4_p{pmax}'); b3_max = stats.getNumber(f'B3_p{pmax}'); b2_max = stats.getNumber(f'B2_p{pmax}')
    vmin = ee.Number(b4_min).min(b3_min).min(b2_min)
    vmax = ee.Number(b4_max).max(b3_max).max(b2_max)
    return vmin.getInfo(), vmax.getInfo()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bbox', type=str, required=True, help='minLon,minLat,maxLon,maxLat')
    ap.add_argument('--start', type=str, default='2024-06-01')
    ap.add_argument('--end',   type=str, default='2024-10-01')
    ap.add_argument('--cloud', type=int, default=40)
    ap.add_argument('--crs',   type=str, default='EPSG:32636')
    ap.add_argument('--scale', type=float, default=10.0)
    ap.add_argument('--out',   type=str, required=True)
    ap.add_argument('--gamma', type=float, default=1.3)
    ap.add_argument('--pmin',  type=int, default=2)
    ap.add_argument('--pmax',  type=int, default=98)
    ap.add_argument('--nx', type=int, default=3, help='Fayans kolon sayısı')
    ap.add_argument('--ny', type=int, default=3, help='Fayans satır sayısı')
    ap.add_argument('--keep_tiles', action='store_true', help='Ara fayansları silme')
    args = ap.parse_args()

    ee_init_with_service_account()

    aoi, (minx, miny, maxx, maxy) = aoi_from_bbox(args.bbox)

    # S2 koleksiyon ve maske
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(args.start, args.end) \
            .filterBounds(aoi) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    s2 = add_cloudprob(s2).map(lambda im: mask_s2_sr(im, args.cloud))
    comp = s2.median().clip(aoi)

    # Dinamik aralık, tek sefer hesapla
    vmin, vmax = compute_percentiles(comp, aoi, args.scale, pmin=args.pmin, pmax=args.pmax)
    vis_img = visualize_truecolor(comp, minv=vmin, maxv=vmax, gamma=args.gamma)

    # Geçici klasör & grid parçaları
    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="gee_tiles_")
    print(f"[TILES] Geçici klasör: {tmpdir}")

    dx = (maxx - minx) / args.nx
    dy = (maxy - miny) / args.ny
    tile_paths = []

    for r in range(args.ny):
        for c in range(args.nx):
            tx0 = minx + c * dx
            tx1 = minx + (c + 1) * dx
            ty0 = miny + r * dy
            ty1 = miny + (r + 1) * dy
            tile_geom = ee.Geometry.BBox(tx0, ty0, tx1, ty1)
            tpath = os.path.join(tmpdir, f"tile_r{r}_c{c}.tif")
            print(f"[DL] Tile r{r} c{c} -> {tpath}")
            try:
                geemap.ee_export_image(
                    vis_img, filename=tpath, scale=args.scale, region=tile_geom,
                    file_per_band=False, crs=args.crs
                )
                if os.path.exists(tpath) and os.path.getsize(tpath) > 0:
                    tile_paths.append(tpath)
                else:
                    print(f"[WARN] Boş/eksik: {tpath}")
            except Exception as e:
                print(f"[ERR] r{r} c{c}: {e}")

    if not tile_paths:
        raise RuntimeError("Hiç fayans indirilemedi.")

    # Mozaik
    print(f"[MERGE] {len(tile_paths)} parça birleştiriliyor...")
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = rio_merge(srcs)
    out_meta = srcs[0].meta.copy()
    for s in srcs: s.close()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0],
        "dtype": mosaic.dtype
    })
    with rasterio.open(args.out, "w", **out_meta) as dst:
        dst.write(mosaic)

    print(f"[OK] Mozaik GeoTIFF yazıldı: {args.out}")

    if not args.keep_tiles:
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"[KEEP] Ara dosyalar: {tmpdir}")

    print(f"[INFO] min={vmin:.1f} max={vmax:.1f} gamma={args.gamma} scale={args.scale} CRS={args.crs}")

if __name__ == "__main__":
    main()
