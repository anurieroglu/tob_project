import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from sklearn.model_selection import train_test_split
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import json
import pandas as pd
from datetime import datetime
import time

# --- Configuration and Hyperparameters ---
HEIGHT, WIDTH = (256, 256)  # Image size'ı küçült (512'den 256'ya)
EPOCHS = 20  # Daha uzun eğitim
BATCH_SIZE = 2  # Batch size'ı küçült (4'ten 2'ye)
LR = 0.0001  # Daha düşük learning rate
RATIO = 0.5
SAMPLE_NUM = 2
ENCODER = 'resnet101'  # Daha güçlü encoder
WEIGHTS = 'imagenet'
ARCHITECTURE = 'unetplusplus'  # U-Net++ mimarisi

# --- GPU Selection and Management ---
def show_gpu_status():
    """
    Tüm GPU'ların durumunu gösterir
    """
    if not torch.cuda.is_available():
        print("❌ CUDA kullanılamıyor, CPU kullanılacak")
        return 'cpu'
    
    gpu_count = torch.cuda.device_count()
    print(f"🔍 {gpu_count} adet GPU bulundu")
    
    for gpu_id in range(gpu_count):
        try:
            torch.cuda.set_device(gpu_id)
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - memory_reserved
            gpu_name = torch.cuda.get_device_name(gpu_id)
            
            print(f"   GPU {gpu_id}: {gpu_name}")
            print(f"      💾 Memory: {memory_free/1024**3:.1f}GB boş, {memory_allocated/1024**3:.1f}GB kullanılan")
            
        except Exception as e:
            print(f"   GPU {gpu_id}: Hata - {e}")
            continue

def select_manual_gpu():
    """
    Kullanıcıdan GPU seçimi alır
    """
    if not torch.cuda.is_available():
        print("❌ CUDA kullanılamıyor, CPU kullanılacak")
        return 'cpu'
    
    gpu_count = torch.cuda.device_count()
    
    # GPU durumlarını göster
    show_gpu_status()
    
    while True:
        try:
            print(f"\n🎯 Hangi GPU'yu kullanmak istiyorsunuz? (0-{gpu_count-1})")
            print("   CPU kullanmak için 'cpu' yazın")
            print("   Otomatik seçim için 'auto' yazın")
            
            choice = input("   Seçiminiz: ").strip().lower()
            
            if choice == 'cpu':
                print("✅ CPU seçildi")
                return 'cpu'
            elif choice == 'auto':
                # Otomatik seçim
                best_gpu = 0
                max_memory = 0
                
                for gpu_id in range(gpu_count):
                    try:
                        torch.cuda.set_device(gpu_id)
                        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)
                        if memory_free > max_memory:
                            max_memory = memory_free
                            best_gpu = gpu_id
                    except:
                        continue
                
                print(f"✅ GPU {best_gpu} otomatik seçildi (en boş)")
                torch.cuda.set_device(best_gpu)
                return f'cuda:{best_gpu}'
            else:
                gpu_id = int(choice)
                if 0 <= gpu_id < gpu_count:
                    # GPU'yu test et
                    try:
                        torch.cuda.set_device(gpu_id)
                        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)
                        print(f"✅ GPU {gpu_id} seçildi")
                        print(f"💾 GPU {gpu_id} Memory: {memory_free/1024**3:.1f}GB boş")
                        return f'cuda:{gpu_id}'
                    except Exception as e:
                        print(f"❌ GPU {gpu_id} kullanılamıyor: {e}")
                        continue
                else:
                    print(f"❌ Geçersiz GPU ID: {gpu_id}")
                    continue
                    
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin")
            continue
        except KeyboardInterrupt:
            print("\n\n⚠️ Kullanıcı tarafından iptal edildi")
            print("🔄 Otomatik seçim yapılıyor...")
            
            # Otomatik seçim
            best_gpu = 0
            max_memory = 0
            
            for gpu_id in range(gpu_count):
                try:
                    torch.cuda.set_device(gpu_id)
                    memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)
                    if memory_free > max_memory:
                        max_memory = memory_free
                        best_gpu = gpu_id
                except:
                    continue
            
            print(f"✅ GPU {best_gpu} otomatik seçildi (en boş)")
            torch.cuda.set_device(best_gpu)
            return f'cuda:{best_gpu}'

# GPU seçimi
DEVICE = select_manual_gpu()

# --- Logging Configuration ---
EXPERIMENT_NAME = f"{ENCODER}_{ARCHITECTURE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"training_logs/{EXPERIMENT_NAME}"
MODEL_SAVE_DIR = f"{ENCODER}_{ARCHITECTURE}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"🚀 Yeni Deney Başlatılıyor: {EXPERIMENT_NAME}")
print(f"📁 Log Dizini: {LOG_DIR}")
print(f"💾 Model Dizini: {MODEL_SAVE_DIR}")
print(f"🔧 Konfigürasyon: {ENCODER} + {ARCHITECTURE}")
print(f"⚙️ Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")

# --- Data Loading and Processing ---
class Load_Data(Dataset):
    """
    Custom Dataset class for loading satellite images and their masks.
    """
    def __init__(self, image_list, mask_list, is_training=True):
        super().__init__()
        self.images_list = image_list
        self.mask_list = mask_list
        self.len = len(image_list)
        self.is_training = is_training
        
        if is_training:
            self.transform = A.Compose([
                A.Resize(HEIGHT, WIDTH),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                # ShiftScaleRotate kaldırıldı (bus error nedeni)
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
                # A.OneOf([
                #     A.GaussNoise(p=0.2),
                #     A.GaussianBlur(p=0.2),
                # ], p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(HEIGHT, WIDTH),
            ])
        
    def __getitem__(self, idx):
        img = Image.open(self.images_list[idx])
        mask = Image.open(self.mask_list[idx]).convert('L')
        
        img, mask = np.array(img), np.array(mask)
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32)

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32)

        return img, mask
    
    def __len__(self):
        return self.len

# --- Model Definition ---
class SegmentationModel(nn.Module):
    """
    U-Net++ based segmentation model using segmentation_models_pytorch.
    """
    def __init__(self):
        super().__init__()
        if ARCHITECTURE == 'unetplusplus':
            self.arc = smp.UnetPlusPlus(
                encoder_name=ENCODER,
                encoder_weights=WEIGHTS,
                in_channels=3,
                classes=1,
                activation=None
            )
        elif ARCHITECTURE == 'deeplabv3plus':
            self.arc = smp.DeepLabV3Plus(
                encoder_name=ENCODER,
                encoder_weights=WEIGHTS,
                in_channels=3,
                classes=1,
                activation=None
            )
        else:  # Default U-Net
            self.arc = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights=WEIGHTS,
                in_channels=3,
                classes=1,
                activation=None
            )

    def forward(self, images, masks=None):
        logits = self.arc(images)
        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1, loss2
        return logits

# --- Training and Evaluation Functions ---
def train_fn(data_loader, model, optimizer, epoch):
    model.train()
    total_diceloss = 0.0
    total_bceloss = 0.0
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits, diceloss, bceloss = model(images, masks)
        loss = diceloss + bceloss
        loss.backward()
        optimizer.step()
        
        total_diceloss += diceloss.item()
        total_bceloss += bceloss.item()
        total_loss += loss.item()
        
        # Progress bar güncelleme
        progress_bar.set_postfix({
            'Dice': f'{diceloss.item():.4f}',
            'BCE': f'{bceloss.item():.4f}',
            'Total': f'{loss.item():.4f}'
        })
        
    avg_dice = total_diceloss / len(data_loader)
    avg_bce = total_bceloss / len(data_loader)
    avg_total = total_loss / len(data_loader)
    
    return avg_dice, avg_bce, avg_total

def eval_fn(data_loader, model, epoch):
    model.eval()
    total_diceloss = 0.0
    total_bceloss = 0.0
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
    
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, diceloss, bceloss = model(images, masks)
            loss = diceloss + bceloss
            
            total_diceloss += diceloss.item()
            total_bceloss += bceloss.item()
            total_loss += loss.item()
            
            # Progress bar güncelleme
            progress_bar.set_postfix({
                'Dice': f'{diceloss.item():.4f}',
                'BCE': f'{bceloss.item():.4f}',
                'Total': f'{loss.item():.4f}'
            })
            
    avg_dice = total_diceloss / len(data_loader)
    avg_bce = total_bceloss / len(data_loader)
    avg_total = total_loss / len(data_loader)
    
    return avg_dice, avg_bce, avg_total

def save_training_plot(train_metrics, val_metrics, save_path):
    """Eğitim metriklerini görselleştirir ve kaydeder."""
    epochs = range(1, len(train_metrics['dice']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dice Loss
    axes[0, 0].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice Loss')
    axes[0, 0].plot(epochs, val_metrics['dice'], 'r-', label='Validation Dice Loss')
    axes[0, 0].set_title('Dice Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # BCE Loss
    axes[0, 1].plot(epochs, train_metrics['bce'], 'b-', label='Train BCE Loss')
    axes[0, 1].plot(epochs, val_metrics['bce'], 'r-', label='Validation BCE Loss')
    axes[0, 1].set_title('BCE Loss Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('BCE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Total Loss
    axes[1, 0].plot(epochs, train_metrics['total'], 'b-', label='Train Total Loss')
    axes[1, 0].plot(epochs, val_metrics['total'], 'r-', label='Validation Total Loss')
    axes[1, 0].set_title('Total Loss Over Time')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(epochs, [LR] * len(epochs), 'g-', label=f'Learning Rate: {LR}')
    axes[1, 1].set_title('Learning Rate Over Time')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_predictions(model, data_loader, epoch, save_dir):
    """Model tahminlerini görselleştirir ve kaydeder."""
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(data_loader))
        image = images[SAMPLE_NUM].to(DEVICE)
        mask = masks[SAMPLE_NUM]

        logits_mask = model(image.unsqueeze(0))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > RATIO) * 1.0
        
        f, axarr = plt.subplots(1, 3, figsize=(15, 5))
        axarr[0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        axarr[0].set_title('Original Image')
        axarr[0].axis('off')
        
        axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
        axarr[1].set_title('True Mask')
        axarr[1].axis('off')
        
        axarr[2].imshow(np.transpose(pred_mask.detach().cpu().squeeze(0), (1, 2, 0)))
        axarr[2].set_title('Predicted Mask')
        axarr[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'prediction_epoch_{epoch+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Tahmin görselleştirmesi kaydedildi: {save_path}")

# --- Main Execution Block ---
def main():
    """
    Main function to orchestrate the data loading, training, and evaluation.
    """
    start_time = time.time()
    
    print(f"\n🔄 Veri yükleniyor...")
    
    # 1. Data Loading
    X = sorted(glob.glob('dataset/Images/*'))
    y = sorted(glob.glob('dataset/Masks/*'))
    
    # Check if data paths are valid
    if not X or not y:
        print("❌ Hata: Görüntü veya maske bulunamadı. Dosya yollarını kontrol edin.")
        return

    print(f"✅ {len(X)} adet görüntü ve {len(y)} adet maske bulundu.")

    # 2. Data Splitting and DataLoader setup
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = Load_Data(X_train, y_train, is_training=True)
    valid_dataset = Load_Data(X_val, y_val, is_training=False)

    # Batch size'ı memory'ye göre ayarla
    if 'cuda' in DEVICE:
        gpu_id = int(DEVICE.split(':')[1]) if ':' in DEVICE else 0
        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)
        
        # Daha agresif memory yönetimi
        if memory_free < 2 * 1024**3:  # 2GB'dan az
            BATCH_SIZE = 1
        elif memory_free < 4 * 1024**3:  # 4GB'dan az
            BATCH_SIZE = 2
        elif memory_free < 8 * 1024**3:  # 8GB'dan az
            BATCH_SIZE = 4
        else:
            BATCH_SIZE = 8
        
        print(f"📦 Memory'ye göre batch size: {BATCH_SIZE}")
        
        # Memory durumunu kontrol et
        if memory_free < 5 * 1024**3:  # 5GB'dan az
            print(f"⚠️ GPU {gpu_id} memory'si kritik düşük!")
            print(f"🔄 Daha küçük image size kullanılıyor...")
            global HEIGHT, WIDTH
            HEIGHT, WIDTH = 128, 128  # 256'dan 128'e düşür
            print(f"🖼️ Yeni image size: {HEIGHT}x{WIDTH}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Worker sayısını 0'a düşür
        pin_memory=False,  # Pin memory'yi kapat
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Worker sayısını 0'a düşür
        pin_memory=False,  # Pin memory'yi kapat
    )

    print(f"📊 Veri bölünmesi: Train: {len(X_train)}, Validation: {len(X_val)}")

    # 3. Model, Optimizer, and Loss Initialization
    print(f"\n🏗️ Model oluşturuluyor: {ENCODER} + {ARCHITECTURE}")
    model = SegmentationModel()
    model.to(DEVICE)
    
    # Memory optimizasyonu
    if 'cuda' in DEVICE:
        gpu_id = int(DEVICE.split(':')[1]) if ':' in DEVICE else 0
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        
        # Memory bilgilerini göster
        memory_allocated = torch.cuda.memory_allocated(gpu_id)
        memory_reserved = torch.cuda.memory_reserved(gpu_id)
        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - memory_reserved
        
        print(f"🧹 GPU {gpu_id} cache temizlendi")
        print(f"💾 GPU {gpu_id} Memory: {memory_free/1024**3:.1f}GB boş, {memory_allocated/1024**3:.1f}GB kullanılan")
        
        # Memory limiti kontrol et
        if memory_free < 2 * 1024**3:  # 2GB'dan az boş memory
            print(f"⚠️ GPU {gpu_id} memory'si düşük! Batch size küçültülüyor...")
            print(f"📦 Mevcut batch size: {BATCH_SIZE}")
        
        # Environment variable ayarla
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"🔧 PYTORCH_CUDA_ALLOC_CONF ayarlandı")
    
    # Model parametre sayısını hesapla
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Toplam parametre sayısı: {total_params:,}")
    print(f"🎯 Eğitilebilir parametre sayısı: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4. Training Loop
    print(f"\n🚀 Eğitim başlıyor...")
    
    best_val_dice_loss = np.inf
    best_val_total_loss = np.inf
    
    # Metrik takibi
    train_metrics = {'dice': [], 'bce': [], 'total': []}
    val_metrics = {'dice': [], 'bce': [], 'total': []}
    
    # Eğitim logları
    training_log = {
        'experiment_name': EXPERIMENT_NAME,
        'configuration': {
            'encoder': ENCODER,
            'architecture': ARCHITECTURE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'image_size': f"{HEIGHT}x{WIDTH}",
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'training_history': [],
        'best_model_info': {},
        'training_time': 0
    }

    for i in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Training
        train_dice_loss, train_bce_loss, train_total_loss = train_fn(train_loader, model, optimizer, i)
        
        # Validation
        valid_dice_loss, valid_bce_loss, valid_total_loss = eval_fn(valid_loader, model, i)
        
        # Learning rate scheduling
        scheduler.step(valid_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Metrikleri kaydet
        train_metrics['dice'].append(train_dice_loss)
        train_metrics['bce'].append(train_bce_loss)
        train_metrics['total'].append(train_total_loss)
        
        val_metrics['dice'].append(valid_dice_loss)
        val_metrics['bce'].append(valid_bce_loss)
        val_metrics['total'].append(valid_total_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Epoch sonuçlarını yazdır
        print(f'\n📊 Epoch {i+1}/{EPOCHS} ({epoch_time:.1f}s)')
        print(f'   Train  --> Dice: {train_dice_loss:.5f} | BCE: {train_bce_loss:.5f} | Total: {train_total_loss:.5f}')
        print(f'   Valid  --> Dice: {valid_dice_loss:.5f} | BCE: {valid_bce_loss:.5f} | Total: {valid_total_loss:.5f}')
        print(f'   LR: {current_lr:.6f}')
        
        # Epoch logunu kaydet
        epoch_log = {
            'epoch': i + 1,
            'train_dice_loss': train_dice_loss,
            'train_bce_loss': train_bce_loss,
            'train_total_loss': train_total_loss,
            'valid_dice_loss': valid_dice_loss,
            'valid_bce_loss': valid_bce_loss,
            'valid_total_loss': valid_total_loss,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        }
        training_log['training_history'].append(epoch_log)
        
        # En iyi modeli kaydet (Dice Loss'a göre)
        if valid_dice_loss < best_val_dice_loss:
            save_path = os.path.join(MODEL_SAVE_DIR, f"model_{valid_dice_loss:.5f}dice.pt")
            torch.save(model.state_dict(), save_path)
            print(f'💾 Model kaydedildi (Dice): {save_path}')
            best_val_dice_loss = valid_dice_loss
            
            # En iyi model bilgisini güncelle
            training_log['best_model_info'] = {
                'best_dice_loss': valid_dice_loss,
                'best_epoch': i + 1,
                'model_path': save_path,
                'save_criteria': 'dice_loss'
            }
        
        # En iyi modeli kaydet (Total Loss'a göre)
        if valid_total_loss < best_val_total_loss:
            save_path = os.path.join(MODEL_SAVE_DIR, f"model_{valid_total_loss:.5f}total.pt")
            torch.save(model.state_dict(), save_path)
            print(f'💾 Model kaydedildi (Total): {save_path}')
            best_val_total_loss = valid_total_loss
        
        # Her 5 epoch'ta tahmin görselleştirmesi
        if (i + 1) % 5 == 0:
            visualize_predictions(model, valid_loader, i, LOG_DIR)
        
        # Memory temizleme
        if 'cuda' in DEVICE:
            torch.cuda.empty_cache()
            if (i + 1) % 5 == 0:  # Her 5 epoch'ta memory durumu
                gpu_id = int(DEVICE.split(':')[1]) if ':' in DEVICE else 0
                memory_allocated = torch.cuda.memory_allocated(gpu_id)
                memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_reserved(gpu_id)
                print(f"💾 Epoch {i+1} sonrası GPU {gpu_id} Memory: {memory_free/1024**3:.1f}GB boş, {memory_allocated/1024**3:.1f}GB kullanılan")
    
    # 5. Final Results and Logging
    total_training_time = time.time() - start_time
    training_log['training_time'] = total_training_time
    
    print(f"\n🎉 Eğitim tamamlandı!")
    print(f"⏱️ Toplam eğitim süresi: {total_training_time/60:.1f} dakika")
    print(f"🏆 En iyi Dice Loss: {best_val_dice_loss:.5f}")
    print(f"🏆 En iyi Total Loss: {best_val_total_loss:.5f}")
    
    # Final görselleştirme
    print(f"\n📊 Final görselleştirmeler oluşturuluyor...")
    final_plot_path = os.path.join(LOG_DIR, 'training_metrics.png')
    save_training_plot(train_metrics, val_metrics, final_plot_path)
    print(f"📈 Eğitim metrikleri kaydedildi: {final_plot_path}")
    
    # Final tahmin görselleştirmesi
    visualize_predictions(model, valid_loader, EPOCHS-1, LOG_DIR)
    
    # Training logunu JSON olarak kaydet
    log_file_path = os.path.join(LOG_DIR, 'training_log.json')
    with open(log_file_path, 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f"📝 Training log kaydedildi: {log_file_path}")
    
    # CSV formatında da kaydet
    df = pd.DataFrame(training_log['training_history'])
    csv_file_path = os.path.join(LOG_DIR, 'training_metrics.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"📊 Training metrikleri CSV olarak kaydedildi: {csv_file_path}")
    
    # Özet rapor
    summary_path = os.path.join(LOG_DIR, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"=== {EXPERIMENT_NAME} EĞİTİM ÖZETİ ===\n\n")
        f.write(f"Konfigürasyon:\n")
        f.write(f"- Encoder: {ENCODER}\n")
        f.write(f"- Architecture: {ARCHITECTURE}\n")
        f.write(f"- Epochs: {EPOCHS}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Learning Rate: {LR}\n")
        f.write(f"- Image Size: {HEIGHT}x{WIDTH}\n\n")
        
        f.write(f"Model Bilgileri:\n")
        f.write(f"- Toplam Parametre: {total_params:,}\n")
        f.write(f"- Eğitilebilir Parametre: {trainable_params:,}\n\n")
        
        f.write(f"Eğitim Sonuçları:\n")
        f.write(f"- Toplam Süre: {total_training_time/60:.1f} dakika\n")
        f.write(f"- En İyi Dice Loss: {best_val_dice_loss:.5f}\n")
        f.write(f"- En İyi Total Loss: {best_val_total_loss:.5f}\n")
        f.write(f"- En İyi Epoch (Dice): {training_log['best_model_info'].get('best_epoch', 'N/A')}\n\n")
        
        f.write(f"Veri Bilgileri:\n")
        f.write(f"- Toplam Görüntü: {len(X)}\n")
        f.write(f"- Train Set: {len(X_train)}\n")
        f.write(f"- Validation Set: {len(X_val)}\n")
    
    print(f"📋 Eğitim özeti kaydedildi: {summary_path}")
    
    print(f"\n🎯 Tüm dosyalar {LOG_DIR} dizininde kaydedildi!")
    print(f"💾 Modeller {MODEL_SAVE_DIR} dizininde kaydedildi!")

# --- Standard Python entry point ---
if __name__ == "__main__":
    main()
