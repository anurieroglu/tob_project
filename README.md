# TOB Projesi - Su Kütlesi Segmentasyonu ve Alan Analizi

Bu proje, uydu görüntülerinden su kütlelerini tespit etmek ve alanlarını hesaplamak için geliştirilmiş Python scriptlerini içermektedir. Proje, Google Earth Engine'den Sentinel-2 görüntülerini indirme, çeşitli derin öğrenme modelleriyle eğitim ve çıkarım yapma yeteneklerine sahiptir.

## 📋 İçindekiler

- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Dosya Yapısı](#dosya-yapısı)
- [Kullanım](#kullanım)
  - [Veri İndirme](#veri-indirme)
  - [Model Eğitimi](#model-eğitimi)
  - [İnferans ve Analiz](#inferans-ve-analiz)
- [Detaylı Açıklamalar](#detaylı-açıklamalar)

## 🔧 Gereksinimler

### Python Paketleri

```bash
pip install torch torchvision
pip install transformers
pip install rasterio geemap earthengine-api
pip install segmentation-models-pytorch
pip install albumentations
pip install scikit-image scikit-learn
pip install pandas matplotlib pillow
pip install tqdm numpy
pip install pystac-client
pip install shapely pyproj
```

### Google Earth Engine Kurulumu

1. Google Earth Engine hesabı oluşturun
2. Service Account anahtarı indirin
3. Ortam değişkenlerini ayarlayın:

```bash
export EE_SA_EMAIL="your-service-account@project.iam.gserviceaccount.com"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
export EE_PROJECT="your-project-id"
```

## 📁 Dosya Yapısı

### Veri İndirme Scriptleri

#### `download_s2_truecolor.py`
Sentinel-2 RGB görüntülerini Google Earth Engine'den indirir.

**Kullanım:**
```bash
python download_s2_truecolor.py \
    --bbox "minLon,minLat,maxLon,maxLat" \
    --start "2024-06-01" \
    --end "2024-10-01" \
    --out "output.tif" \
    --cloud 40 \
    --scale 10.0 \
    --crs "EPSG:32636"
```

**Parametreler:**
- `--bbox`: Bounding box koordinatları (minLon,minLat,maxLon,maxLat)
- `--point`: Alternatif olarak nokta koordinatları (lon,lat)
- `--buffer_km`: Nokta için tampon mesafesi (km)
- `--start/--end`: Tarih aralığı
- `--cloud`: Bulut eşiği (0-100)
- `--scale`: Çözünürlük (metre)
- `--crs`: Koordinat referans sistemi
- `--gamma`: Görüntü gamma değeri
- `--pmin/--pmax`: Percentile aralığı

#### `download_s2_truecolor_tiled.py`
Büyük alanlar için görüntüleri fayanslara bölerek indirir ve birleştirir.

**Kullanım:**
```bash
python download_s2_truecolor_tiled.py \
    --bbox "minLon,minLat,maxLon,maxLat" \
    --out "output.tif" \
    --nx 3 --ny 3 \
    --keep_tiles
```

**Ek Parametreler:**
- `--nx/--ny`: Fayans grid boyutları
- `--keep_tiles`: Ara fayansları silme

#### `download_sam_checkpoint.py`
Segment Anything Model (SAM) checkpoint'lerini indirir.

**Kullanım:**
```bash
python download_sam_checkpoint.py [vit_b|vit_l|vit_h]
```

### Model Eğitimi Scriptleri

#### `train_segformer_b3_water.py`
SegFormer-B3 modeli ile su segmentasyonu eğitimi.

**Kullanım:**
```bash
python train_segformer_b3_water.py \
    --images "dataset/Images" \
    --masks "dataset/Masks" \
    --output_dir "outputs_b3" \
    --epochs 25 \
    --batch_size 8 \
    --lr 1e-4 \
    --size 512
```

**Değerlendirme:**
```bash
python train_segformer_b3_water.py \
    --images "dataset/Images" \
    --masks "dataset/Masks" \
    --output_dir "outputs_b3" \
    --eval_only \
    --checkpoint "outputs_b3/best_model.pt"
```

#### `train_resnet50_unet.py`
ResNet50 + UNet mimarisi ile eğitim.

**Kullanım:**
```bash
python train_resnet50_unet.py \
    --images "dataset/Images" \
    --masks "dataset/Masks" \
    --output_dir "outputs_r50_unet" \
    --epochs 25 \
    --batch_size 8
```

#### `train_resnet101_unetpp.py`
ResNet101 + UNet++ mimarisi ile eğitim.

**Kullanım:**
```bash
python train_resnet101_unetpp.py \
    --images "dataset/Images" \
    --masks "dataset/Masks" \
    --output_dir "outputs_r101_unetpp" \
    --epochs 25
```

#### `train_custom_sam.py`
Özel SAM modeli eğitimi (overfitting önleme ile).

**Özellikler:**
- Image encoder dondurulmuş
- Düşük learning rate
- Data augmentation
- Balanced sampling
- Early stopping

### İnferans ve Analiz Scriptleri

#### `infer_area_segformer_cuda.py`
SegFormer modeli ile CUDA destekli alan tahmini.

**Kullanım:**
```bash
python infer_area_segformer_cuda.py \
    --image "input.tif" \
    --weights "model.pt" \
    --backbone "nvidia/mit-b5" \
    --device "cuda" \
    --tile 512 \
    --stride 512 \
    --thresh 0.5 \
    --save_mask "mask.tif" \
    --save_overlay "overlay.png" \
    --save_compare "compare.png" \
    --save_heatmap "heatmap.png"
```

**Parametreler:**
- `--image`: Girdi GeoTIFF dosyası
- `--weights`: Model checkpoint yolu
- `--backbone`: HuggingFace model ID
- `--device`: cuda/cpu
- `--amp`: Mixed precision (true/false)
- `--compile`: PyTorch 2.0 compile
- `--tile/--stride`: Tiling parametreleri
- `--thresh`: Eşik değeri
- `--gt_area_km2`: Gerçek alan (doğrulama için)

#### `infer_water_area.py`
Gelişmiş su alanı tahmini (TTA, post-processing ile).

**Kullanım:**
```bash
python infer_water_area.py \
    --tif "input.tif" \
    --model "model.pt" \
    --out "outputs_water" \
    --tile 1024 \
    --overlap 128 \
    --batch 4 \
    --min_ha 0.1 \
    --keep_top_k 1 \
    --pmin 2 --pmax 98 \
    --gamma 1.25 \
    --blue_blend 0.0
```

**Özellikler:**
- Test-Time Augmentation (TTA)
- Cosine blending
- Otomatik eşik seçimi
- Morfolojik temizlik
- Mavi-indeks harmanı

#### `lake_area_pipeline.py`
Tam otomatik göl alanı analizi pipeline'ı.

**Kullanım:**
```bash
python lake_area_pipeline.py \
    --place "Van Gölü, Turkey" \
    --start "2024-06-01" \
    --end "2024-10-01" \
    --weights "model.pt" \
    --out_rgb "rgb.tif" \
    --out_mask "mask.tif" \
    --out_overlay "overlay.png" \
    --out_compare "compare.png" \
    --out_heatmap "heatmap.png"
```

**Özellikler:**
- STAC API ile Sentinel-2 arama
- Otomatik mozaik oluşturma
- Model çıkarımı
- Görselleştirme
- Alan hesaplama

#### `test_trained_sam.py`
Eğitilmiş SAM modelini test eder.

**Kullanım:**
```bash
python test_trained_sam.py
```

Script içinde CONFIG bölümünden ayarları değiştirebilirsiniz.

#### `semseg_final(1).py`
Zamana göre göl alanı değişim analizi.

**Özellikler:**
- Klasördeki tüm görüntüleri analiz eder
- Tarih bilgisini dosya adından çıkarır
- CSV ve grafik çıktıları üretir

**Kullanım:**
Script içindeki `MODEL_PATH` ve `IMAGE_FOLDER` değişkenlerini ayarlayın.

### Diğer Eğitim Scriptleri

- `training_segformer.py`: SegFormer eğitimi (eski versiyon)
- `training_segformerb5.py`: SegFormer-B5 eğitimi
- `training_advanced.py`: Gelişmiş eğitim scripti (GPU seçimi, augmentation)

## 📊 Metrikler

Eğitim scriptleri şu metrikleri hesaplar:
- **IoU** (Intersection over Union)
- **Dice Score**
- **Precision**
- **Recall**
- **Boundary F1**
- **Average Precision (AP)**
- **Expected Calibration Error (ECE)**

Metrikler CSV, JSON ve Markdown formatlarında kaydedilir.

## 🎯 Örnek Kullanım Senaryoları

### Senaryo 1: Yeni Bir Göl İçin Analiz

```bash
# 1. Görüntü indir
python download_s2_truecolor.py \
    --point "30.2,40.7" \
    --buffer_km 15 \
    --start "2024-06-01" \
    --end "2024-10-01" \
    --out "gol_rgb.tif"

# 2. Model ile analiz
python infer_water_area.py \
    --tif "gol_rgb.tif" \
    --model "best_model.pt" \
    --out "analiz_sonuclari"
```

### Senaryo 2: Tam Otomatik Pipeline

```bash
python lake_area_pipeline.py \
    --place "Sapanca Gölü, Turkey" \
    --start "2024-06-01" \
    --end "2024-10-01" \
    --weights "segformer_b3/best_model.pt" \
    --out_rgb "sapanca_rgb.tif" \
    --out_mask "sapanca_mask.tif" \
    --gt_area_km2 45.0
```

### Senaryo 3: Model Eğitimi

```bash
# SegFormer-B3 eğitimi
python train_segformer_b3_water.py \
    --images "dataset/Images" \
    --masks "dataset/Masks" \
    --output_dir "my_model" \
    --epochs 30 \
    --batch_size 4 \
    --lr 5e-5
```

## ⚙️ Yapılandırma İpuçları

### GPU Kullanımı
- CUDA kullanılabilirliği otomatik kontrol edilir
- `--device cuda:0` ile belirli GPU seçilebilir
- Mixed precision (`--amp true`) bellek kullanımını azaltır

### Bellek Optimizasyonu
- Büyük görüntüler için `--tile` ve `--stride` parametrelerini ayarlayın
- Batch size'ı azaltın
- `--workers 0` veya `1` kullanın (çoklu işlem sorunları için)

### Model Seçimi
- **SegFormer-B3**: Dengeli performans/bellek
- **SegFormer-B5**: En yüksek doğruluk
- **ResNet50+UNet**: Hızlı eğitim
- **ResNet101+UNet++**: İyi segmentasyon kalitesi

## 📝 Notlar

- Tüm scriptler Python 3.8+ ile uyumludur
- GeoTIFF dosyaları UTM projeksiyonunda olmalıdır (alan hesaplamaları için)
- Sentinel-2 görüntüleri 10m çözünürlükte işlenir
- Model checkpoint'leri PyTorch formatında kaydedilir

## 🔍 Sorun Giderme

### Google Earth Engine Hatası
- Service account kimlik bilgilerini kontrol edin
- Ortam değişkenlerinin doğru ayarlandığından emin olun
- Quota limitlerini kontrol edin

### CUDA Hatası
- PyTorch CUDA sürümünü kontrol edin: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU bellek kullanımını kontrol edin: `nvidia-smi`
- Batch size'ı azaltın

### Model Yükleme Hatası
- Checkpoint formatını kontrol edin
- Model mimarisinin eğitimle uyumlu olduğundan emin olun
- `strict=False` ile yükleme yapılabilir

Proje geliştirme sürecinde kullanılan kütüphaneler:
- PyTorch
- Hugging Face Transformers
- Google Earth Engine
- Rasterio
- Segmentation Models PyTorch

