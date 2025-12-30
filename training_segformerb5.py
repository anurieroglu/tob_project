import os, glob, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# ---- Konfig ----
HEIGHT, WIDTH = (512, 512)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
RATIO = 0.5
SAMPLE_NUM = 2

ARCHITECTURE = 'segformer_b5'
EXPERIMENT_NAME = f"{ARCHITECTURE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"training_logs/{EXPERIMENT_NAME}"
MODEL_SAVE_DIR = f"{ARCHITECTURE}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"🚀 Yeni Deney Başlatılıyor: {EXPERIMENT_NAME}")
print(f"📁 Log Dizini: {LOG_DIR}")
print(f"💾 Model Dizini: {MODEL_SAVE_DIR}")
print(f"🔧 Konfigürasyon: {ARCHITECTURE}")
print(f"⚙️ Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")

# ---- Dataset ----
class Load_Data(Dataset):
    def __init__(self, image_list, mask_list, is_training=True):
        super().__init__()
        self.images_list = image_list
        self.mask_list = mask_list
        self.len = len(image_list)
        self.is_training = is_training

        # SegFormer için temel resize + hafif augment
        if is_training:
            self.transform = A.Compose([
                A.Resize(HEIGHT, WIDTH),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(0.05, 0.05, 15, p=0.3),
            ])
        else:
            self.transform = A.Compose([A.Resize(HEIGHT, WIDTH)])

        # ImageNet norm (SegFormer ön-eğitimine uygun)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __getitem__(self, idx):
        img = Image.open(self.images_list[idx]).convert("RGB")
        mask = Image.open(self.mask_list[idx]).convert("L")

        img, mask = np.array(img), np.array(mask)
        t = self.transform(image=img, mask=mask)
        img = t['image']
        mask = t['mask']

        # [H,W,C]→[C,H,W], 0..1 → normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        # Binary mask [1,H,W] in {0,1}
        mask = (mask.astype(np.float32) / 255.0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)

        return img, mask

    def __len__(self):
        return self.len

# ---- Model: SegFormer-B5 (HF Transformers) ----
# Kurulum: pip install transformers timm accelerate
from transformers import SegformerForSemanticSegmentation

class SegFormerB5(nn.Module):
    def __init__(self):
        super().__init__()
        # num_labels=1 → binary logits kanalı
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        # Dice + BCE (binary)
        from segmentation_models_pytorch.losses import DiceLoss
        self.dice = DiceLoss(mode='binary')
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, images, masks=None):
        # images: [B,3,H,W]
        outputs = self.net(pixel_values=images)
        logits = outputs.logits  # [B,1,h',w'] (HF: upsample edilmiş olabilir)
        if masks is not None and logits.shape[-2:] != masks.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        if masks is not None:
            loss1 = self.dice(logits, masks)
            loss2 = self.bce(logits, masks)
            return logits, loss1, loss2
        return logits

# ---- Train/Eval döngüsü (aynı yapı) ----
def train_fn(data_loader, model, optimizer, epoch):
    model.train()
    td, tb, tt = 0.0, 0.0, 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, diceloss, bceloss = model(images, masks)
        loss = diceloss + bceloss
        loss.backward()
        optimizer.step()

        td += diceloss.item(); tb += bceloss.item(); tt += loss.item()
        pbar.set_postfix(Dice=f"{diceloss.item():.4f}", BCE=f"{bceloss.item():.4f}", Total=f"{loss.item():.4f}")
    n = len(data_loader)
    return td/n, tb/n, tt/n

def eval_fn(data_loader, model, epoch):
    model.eval()
    td, tb, tt = 0.0, 0.0, 0.0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)
            logits, diceloss, bceloss = model(images, masks)
            loss = diceloss + bceloss
            td += diceloss.item(); tb += bceloss.item(); tt += loss.item()
            pbar.set_postfix(Dice=f"{diceloss.item():.4f}", BCE=f"{bceloss.item():.4f}", Total=f"{loss.item():.4f}")
    n = len(data_loader)
    return td/n, tb/n, tt/n

def save_training_plot(train_metrics, val_metrics, save_path):
    epochs = range(1, len(train_metrics['dice']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(epochs, train_metrics['dice'], 'b-', label='Train Dice'); axes[0,0].plot(epochs, val_metrics['dice'], 'r-', label='Val Dice'); axes[0,0].legend(); axes[0,0].grid(True)
    axes[0,1].plot(epochs, train_metrics['bce'], 'b-', label='Train BCE');  axes[0,1].plot(epochs, val_metrics['bce'],  'r-', label='Val BCE');  axes[0,1].legend(); axes[0,1].grid(True)
    axes[1,0].plot(epochs, train_metrics['total'], 'b-', label='Train Total'); axes[1,0].plot(epochs, val_metrics['total'], 'r-', label='Val Total'); axes[1,0].legend(); axes[1,0].grid(True)
    axes[1,1].plot(epochs, [LR]*len(epochs), 'g-', label=f'LR={LR}'); axes[1,1].legend(); axes[1,1].grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def visualize_predictions(model, data_loader, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        try:
            images, masks = next(iter(data_loader))
        except StopIteration:
            print("⚠️ Valid loader boş, görselleştirme atlandı."); return
        idx = min(SAMPLE_NUM, len(images)-1)
        image = images[idx].to(DEVICE)
        mask  = masks[idx].cpu().squeeze().numpy()  # HxW

        logits = model(image.unsqueeze(0))[0]  # [1,1,h,w]
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        pred = (prob > RATIO).astype(np.float32)

        f, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(np.transpose(image.cpu().numpy(), (1,2,0))); ax[0].set_title('Original'); ax[0].axis('off')
        ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1);          ax[1].set_title('True');     ax[1].axis('off')
        ax[2].imshow(pred, cmap='gray', vmin=0, vmax=1);          ax[2].set_title('Pred');     ax[2].axis('off')
        plt.tight_layout()
        path = os.path.join(save_dir, f'prediction_epoch_{epoch+1}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"📊 Tahmin görselleştirmesi kaydedildi: {path}")

def main():
    start = time.time()
    print("\n🔄 Veri yükleniyor...")

    # Mevcut yapıya uyumlu sabit path (gerekirse burayı parametreleştirebilirsin)
    X = sorted(glob.glob('dataset/Water Bodies Dataset/Images/*'))
    Y = sorted(glob.glob('dataset/Water Bodies Dataset/Masks/*'))
    if not X or not Y:
        print("❌ Hata: Görüntü veya maske bulunamadı. Dosya yollarını kontrol edin."); return

    print(f"✅ {len(X)} adet görüntü ve {len(Y)} adet maske bulundu.")
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = Load_Data(X_train, y_train, is_training=True)
    valid_dataset = Load_Data(X_val,   y_val,   is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    print(f"📊 Veri bölünmesi: Train: {len(X_train)}, Validation: {len(X_val)}")

    print("\n🏗️ Model oluşturuluyor: SegFormer-B5")
    model = SegFormerB5().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Toplam parametre: {total_params:,} | 🎯 Eğitilebilir: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_dice, best_val_total = float('inf'), float('inf')
    train_metrics = {'dice': [], 'bce': [], 'total': []}
    val_metrics   = {'dice': [], 'bce': [], 'total': []}
    training_log = {
        'experiment_name': EXPERIMENT_NAME,
        'configuration': {'arch': ARCHITECTURE, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR, 'image_size': f"{HEIGHT}x{WIDTH}",
                          'total_params': total_params, 'trainable_params': trainable_params},
        'training_history': [], 'best_model_info': {}, 'training_time': 0
    }

    print("\n🚀 Eğitim başlıyor...")
    for i in range(EPOCHS):
        t0 = time.time()
        tr_d, tr_b, tr_t = train_fn(train_loader, model, optimizer, i)
        va_d, va_b, va_t = eval_fn(valid_loader, model, i)
        scheduler.step(va_t)
        cur_lr = optimizer.param_groups[0]['lr']

        train_metrics['dice'].append(tr_d); train_metrics['bce'].append(tr_b); train_metrics['total'].append(tr_t)
        val_metrics['dice'].append(va_d);   val_metrics['bce'].append(va_b);   val_metrics['total'].append(va_t)

        print(f'\n📊 Epoch {i+1}/{EPOCHS} ({time.time()-t0:.1f}s)')
        print(f'   Train  --> Dice: {tr_d:.5f} | BCE: {tr_b:.5f} | Total: {tr_t:.5f}')
        print(f'   Valid  --> Dice: {va_d:.5f} | BCE: {va_b:.5f} | Total: {va_t:.5f}')
        print(f'   LR: {cur_lr:.6f}')

        training_log['training_history'].append({
            'epoch': i+1, 'train_dice_loss': tr_d, 'train_bce_loss': tr_b, 'train_total_loss': tr_t,
            'valid_dice_loss': va_d, 'valid_bce_loss': va_b, 'valid_total_loss': va_t,
            'learning_rate': cur_lr, 'epoch_time': time.time()-t0
        })

        if va_d < best_val_dice:
            p = os.path.join(MODEL_SAVE_DIR, f"segformer_b5_{va_d:.5f}dice.pt")
            torch.save(model.state_dict(), p)
            print(f'💾 Model kaydedildi (Dice): {p}')
            best_val_dice = va_d
            training_log['best_model_info'] = {'best_dice_loss': va_d, 'best_epoch': i+1, 'model_path': p, 'save_criteria': 'dice_loss'}

        if va_t < best_val_total:
            p = os.path.join(MODEL_SAVE_DIR, f"segformer_b5_{va_t:.5f}total.pt")
            torch.save(model.state_dict(), p)
            print(f'💾 Model kaydedildi (Total): {p}')
            best_val_total = va_t

        if (i+1) % 5 == 0:
            visualize_predictions(model, valid_loader, i, LOG_DIR)

    total_time = time.time()-start
    training_log['training_time'] = total_time
    print(f"\n🎉 Eğitim tamamlandı! ⏱️ {total_time/60:.1f} dk | 🏆 Best Dice: {best_val_dice:.5f} | 🏆 Best Total: {best_val_total:.5f}")

    final_plot = os.path.join(LOG_DIR, 'training_metrics.png')
    save_training_plot(train_metrics, val_metrics, final_plot)
    print(f"📈 Eğitim metrikleri kaydedildi: {final_plot}")

    # Son tahmin
    visualize_predictions(model, valid_loader, EPOCHS-1, LOG_DIR)

    # Loglar
    with open(os.path.join(LOG_DIR, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    print(f"📝 Training log kaydedildi: {os.path.join(LOG_DIR, 'training_log.json')}")

    import pandas as pd
    df = pd.DataFrame(training_log['training_history'])
    csvp = os.path.join(LOG_DIR, 'training_metrics.csv')
    df.to_csv(csvp, index=False)
    print(f"📊 CSV kaydedildi: {csvp}")

    with open(os.path.join(LOG_DIR, 'training_summary.txt'), 'w') as f:
        f.write(f"=== {EXPERIMENT_NAME} (SegFormer-B5) ===\n\n")
        f.write(f"Epochs: {EPOCHS} | BS: {BATCH_SIZE} | LR: {LR} | Image: {HEIGHT}x{WIDTH}\n")
        f.write(f"Params: total={total_params:,}, trainable={trainable_params:,}\n")
        f.write(f"Best Dice: {best_val_dice:.5f} | Best Total: {best_val_total:.5f}\n")
    print(f"📋 Özet kaydedildi: {os.path.join(LOG_DIR, 'training_summary.txt')}")

if __name__ == "__main__":
    main()
