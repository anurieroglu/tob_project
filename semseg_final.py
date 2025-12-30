import os
import glob
import re
from datetime import datetime

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import rasterio
from PIL import Image
import albumentations as A
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# --- Yapılandırma (Configuration) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/tf/uygulamalar/kamag/tob/resnet50_unet/model_0.21060dice.pt'
IMAGE_FOLDER = '/tf/uygulamalar/kamag/tob/sapanca_collection_UTM_1' # Orijinal .tif dosyalarının olduğu klasör
ENCODER = 'resnet50'
IMG_HEIGHT = 512
IMG_WIDTH = 512

# --- Model Sınıfının Yeniden Tanımlanması (Eğitimdeki ile aynı olmalı) ---
class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.arc = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images):
        return self.arc(images)

# --- Fonksiyon Tanımlamaları ---
def load_model(model_path):
    """
    Eğitilmiş modeli yükler ve değerlendirme moduna ayarlar.
    """
    model = SegmentationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    print("Model başarıyla yüklendi!")
    return model

def predict_mask(model, image_path):
    """
    Bir .tif uydu görüntüsünü açar, ön işler ve maske tahmini yapar.
    """
    with rasterio.open(image_path) as src:
        original_height = src.height
        original_width = src.width
        image_data = src.read([1, 2, 3])

    rgb_image = np.transpose(image_data, (1, 2, 0)).astype(np.float32)
    
    p2, p98 = np.percentile(rgb_image[rgb_image > 0], (2, 98))
    img_stretched = np.clip((rgb_image - p2) / (p98 - p2), 0, 1)

    transform = A.Compose([A.Resize(IMG_HEIGHT, IMG_WIDTH)])
    img_resized = transform(image=img_stretched)['image']

    img_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).to(DEVICE, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = (torch.sigmoid(logits) > 0.5) * 1.0

    final_mask = pred_mask.squeeze().cpu().numpy()
    return final_mask, original_height, original_width

def calculate_area(mask, original_height, original_width):
    """
    Bir maskenin alanını kilometrekare cinsinden hesaplar.
    """
    area_per_pixel_m2 = 10 * 10
    total_image_area_m2 = original_height * original_width * area_per_pixel_m2
    
    if mask.size == 0:
        return 0
        
    water_ratio = np.sum(mask) / mask.size
    lake_area_m2 = total_image_area_m2 * water_ratio
    return lake_area_m2 / 1_000_000

def analyze_images(model, image_folder):
    """
    Klasördeki tüm görüntüler için analiz yapar ve sonuçları döndürür.
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.tif')))
    results = []

    print(f"{len(image_files)} adet görüntü bulundu. Analiz başlıyor...")
    for image_path in tqdm(image_files):
        filename = os.path.basename(image_path)
        match = re.search(r'(\d{8}T\d{6})', filename)
        if match:
            date_str = match.group(1).split('T')[0]
            image_date = datetime.strptime(date_str, '%Y%m%d')
            predicted_mask, orig_h, orig_w = predict_mask(model, image_path)
            area_km2 = calculate_area(predicted_mask, orig_h, orig_w)
            results.append({'date': image_date, 'area_km2': area_km2})

    return pd.DataFrame(results).sort_values(by='date').reset_index(drop=True)

def plot_and_save_results(df, output_filename="sapanca_golu_alan_grafiği.png"):
    """
    Analiz sonuçlarını bir grafikte gösterir ve dosyaya kaydeder.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(df['date'], df['area_km2'], marker='o', linestyle='-', color='b', label='Göl Alanı')
    ax.set_title('Zamana Göre Sapanca Gölü Alan Değişimi', fontsize=16)
    ax.set_xlabel('Tarih', fontsize=12)
    ax.set_ylabel('Alan ($km^2$)', fontsize=12)
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    # --- DEĞİŞİKLİK BURADA ---
    # Grafiği yüksek çözünürlükte bir dosyaya kaydet
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Grafiği ekranda göster
    plt.show()

# --- Ana Çalıştırma Bloğu ---
def main():
    """
    Ana fonksiyon: Modeli yükler, analiz yapar, sonuçları gösterir ve kaydeder.
    """
    model = load_model(MODEL_PATH)
    results_df = analyze_images(model, IMAGE_FOLDER)
    
    print("\n--- Analiz Tamamlandı ---")
    print(results_df.head())
    
    # Grafiği çiz ve kaydet 
    graph_filename = os.path.basename(IMAGE_FOLDER)+os.path.basename(MODEL_PATH).replace('.', '_')+"_area_graph.png"
    plot_and_save_results(results_df, graph_filename)
    
    # Sonuçları CSV dosyasına kaydet
    csv_filename = os.path.basename(IMAGE_FOLDER)+os.path.basename(MODEL_PATH).replace('.', '_')+'_area_results.csv'
    results_df.to_csv(csv_filename, index=False)
    
    # --- DEĞİŞİKLİK BURADA ---
    # Kullanıcıyı bilgilendir
    print(f"\nGrafik '{graph_filename}' olarak kaydedildi.")
    print(f"Sonuçlar '{csv_filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()