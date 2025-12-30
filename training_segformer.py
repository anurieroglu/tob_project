import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, cv2, json, time, pandas as pd
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from segmentation_models_pytorch.losses import DiceLoss

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ========= Config =========
HEIGHT, WIDTH = (256, 256)
EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-4
RATIO = 0.5           # inference eşik
SAMPLE_NUM = 0        # güvenli
SEGFORMER_BACKBONE = "nvidia/segformer-b3-finetuned-ade-512-512"  # B5: "nvidia/segformer-b5-finetuned-ade-640-640"
NUM_LABELS = 1        # binary (tek kanal logit)
IGNORE_INDEX = 255

# ========= Cihaz seçimi (otomatik) =========
def pick_device():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"CUDA var. Serbest bellek: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")
        return "cuda"
    return "cpu"

DEVICE = pick_device()

# ========= Log dizinleri =========
EXPERIMENT_NAME = f"segformer_{os.path.basename(SEGFORMER_BACKBONE)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = f"training_logs/{EXPERIMENT_NAME}"
MODEL_SAVE_DIR = f"segformer_models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"🚀 Yeni Deney: {EXPERIMENT_NAME}")
print(f"Backbone: {SEGFORMER_BACKBONE} | Num Labels: {NUM_LABELS} | Image: {HEIGHT}x{WIDTH}")

# ========= Dataset =========
class WaterDataset(Dataset):
    def __init__(self, image_list, mask_list, processor, is_training=True):
        self.images_list = image_list
        self.mask_list = mask_list
        self.processor = processor
        self.is_training = is_training

        # Albumentations ile augment + boyutlandırma
        if is_training:
            self.aug = A.Compose([
                A.Resize(HEIGHT, WIDTH),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.aug = A.Compose([A.Resize(HEIGHT, WIDTH)])

        # Processor’ın mean/std’sini çekelim (pretrained ile uyum)
        self.mean = np.array(self.processor.image_mean)  # [3]
        self.std  = np.array(self.processor.image_std)   # [3]

    def __len__(self): return len(self.images_list)

    def __getitem__(self, idx):
        img = Image.open(self.images_list[idx]).convert("RGB")
        msk = Image.open(self.mask_list[idx]).convert("L")

        img, msk = np.array(img), np.array(msk)
        # maskeyi 0/1 yap
        if msk.max() > 1: msk = (msk > 127).astype(np.uint8)

        # albumentations
        aug = self.aug(image=img, mask=msk)
        img, msk = aug["image"], aug["mask"]

        # HF normalization (manuel): (img/255 - mean) / std
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))  # CHW

        # tensörler
        img_t = torch.tensor(img, dtype=torch.float32)
        msk_t = torch.tensor(msk[None, ...], dtype=torch.float32)  # [1,H,W]

        return img_t, msk_t

# ========= Model =========
class SegFormerBinary(nn.Module):
    def __init__(self, backbone, num_labels=1, ignore_index=255):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        # (opsiyonel) hafıza için:
        # self.model.enable_gradient_checkpointing()

        self.dice = DiceLoss(mode="binary")
        self.bce  = nn.BCEWithLogitsLoss()

        self.ignore_index = ignore_index
        self.num_labels   = num_labels

    def forward(self, images, masks=None):
        """
        images: [B,3,H,W], normalize edilmiş
        masks:  [B,1,H,W], 0/1
        """
        out = self.model(pixel_values=images)
        logits = out.logits  # [B, C, H/4, W/4] (C=1)

        # upsample -> mask boyutuna
        if masks is not None:
            target_h, target_w = masks.shape[-2], masks.shape[-1]
        else:
            target_h, target_w = images.shape[-2], images.shape[-1]

        logits_up = F.interpolate(logits, size=(target_h, target_w),
                                  mode="bilinear", align_corners=False)

        if masks is None:
            return logits_up

        loss_dice = self.dice(logits_up, masks)
        loss_bce  = self.bce(logits_up, masks)
        return logits_up, loss_dice, loss_bce

# ========= Yardımcılar =========
def iou_and_dice_from_logits(logits, masks, thr=0.5, eps=1e-7):
    """ Basit IoU ve Dice hesaplaması (binary) """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    intersection = (preds * masks).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + masks.sum(dim=(2,3)) - intersection
    iou = (intersection + eps) / (union + eps)

    dice = (2*intersection + eps) / (preds.sum(dim=(2,3)) + masks.sum(dim=(2,3)) + eps)
    return iou.mean().item(), dice.mean().item()

def save_training_plot(train_metrics, val_metrics, save_path):
    epochs = range(1, len(train_metrics['dice']) + 1)
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.plot(epochs, train_metrics['dice'], label='Train DiceLoss'); plt.plot(epochs, val_metrics['dice'], label='Val DiceLoss'); plt.title('Dice Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,2); plt.plot(epochs, train_metrics['bce'], label='Train BCE'); plt.plot(epochs, val_metrics['bce'], label='Val BCE'); plt.title('BCE Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,3); plt.plot(epochs, train_metrics['total'], label='Train Total'); plt.plot(epochs, val_metrics['total'], label='Val Total'); plt.title('Total Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2,2,4); plt.plot(epochs, train_metrics['iou'], label='Train IoU'); plt.plot(epochs, val_metrics['iou'], label='Val IoU'); plt.title('IoU'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

def visualize_predictions(model, data_loader, epoch, save_dir, thr=RATIO):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(data_loader))
        image = images[SAMPLE_NUM].to(DEVICE)
        mask = masks[SAMPLE_NUM]

        logits = model(image.unsqueeze(0))
        probs = torch.sigmoid(logits)
        pred  = (probs > thr).float()

        f, axarr = plt.subplots(1, 3, figsize=(15,5))
        axarr[0].imshow(np.transpose(image.cpu().numpy(), (1,2,0))); axarr[0].set_title('Image'); axarr[0].axis('off')
        axarr[1].imshow(mask.squeeze(0).cpu().numpy(), cmap='gray'); axarr[1].set_title('GT Mask'); axarr[1].axis('off')
        axarr[2].imshow(pred.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray'); axarr[2].set_title('Pred Mask'); axarr[2].axis('off')
        plt.tight_layout()
        outp = os.path.join(save_dir, f'pred_segformer_epoch_{epoch+1}.png')
        plt.savefig(outp, dpi=300, bbox_inches='tight'); plt.close()
        print(f"📊 Tahmin görseli kaydedildi: {outp}")

# ========= Train/Valid döngüleri =========
def train_fn(data_loader, model, optimizer, epoch):
    model.train()
    t_d, t_b, t_tot, t_iou, t_dc = 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        logits, dl, bl = model(images, masks)
        loss = 0.7*dl + 0.3*bl  # ağırlık örneği

        loss.backward()
        optimizer.step()

        iou, dice_metric = iou_and_dice_from_logits(logits.detach(), masks, thr=RATIO)

        t_d += dl.item(); t_b += bl.item(); t_tot += loss.item()
        t_iou += iou; t_dc += dice_metric

        pbar.set_postfix(DiceLoss=f'{dl.item():.4f}', BCE=f'{bl.item():.4f}', IoU=f'{iou:.3f}', DICE=f'{dice_metric:.3f}')
    n = len(data_loader)
    return t_d/n, t_b/n, t_tot/n, t_iou/n, t_dc/n

def eval_fn(data_loader, model, epoch):
    model.eval()
    t_d, t_b, t_tot, t_iou, t_dc = 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            logits, dl, bl = model(images, masks)
            loss = 0.7*dl + 0.3*bl

            iou, dice_metric = iou_and_dice_from_logits(logits, masks, thr=RATIO)

            t_d += dl.item(); t_b += bl.item(); t_tot += loss.item()
            t_iou += iou; t_dc += dice_metric

            pbar.set_postfix(DiceLoss=f'{dl.item():.4f}', BCE=f'{bl.item():.4f}', IoU=f'{iou:.3f}', DICE=f'{dice_metric:.3f}')
    n = len(data_loader)
    return t_d/n, t_b/n, t_tot/n, t_iou/n, t_dc/n

def main():
    start_time = time.time()

    # 1) Veri
    X = sorted(glob.glob('dataset/Images/*'))
    y = sorted(glob.glob('dataset/Masks/*'))
    if not X or not y:
        print("❌ Görsel/mask bulunamadı."); return
    print(f"✅ {len(X)} görüntü / {len(y)} maske bulundu.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_BACKBONE, do_reduce_labels=False)

    train_ds = WaterDataset(X_train, y_train, processor, is_training=True)
    val_ds   = WaterDataset(X_val, y_val, processor, is_training=False)

    # 2) DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=('cuda' in DEVICE))
    valid_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=('cuda' in DEVICE))
    print(f"📊 Split: Train={len(X_train)}, Val={len(X_val)}")

    # 3) Model/optim
    model = SegFormerBinary(SEGFORMER_BACKBONE, num_labels=NUM_LABELS, ignore_index=IGNORE_INDEX).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Parametreler: total={total_params:,} | trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 4) Eğitim döngüsü
    best_val_total = np.inf
    best_val_dice  = np.inf  # (loss)
    train_metrics = {'dice': [], 'bce': [], 'total': [], 'iou': [], 'dice_metric': []}
    val_metrics   = {'dice': [], 'bce': [], 'total': [], 'iou': [], 'dice_metric': []}

    training_log = {
        'experiment_name': EXPERIMENT_NAME,
        'configuration': {
            'backbone': SEGFORMER_BACKBONE,
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'image_size': f"{HEIGHT}x{WIDTH}",
            'total_params': total_params, 'trainable_params': trainable_params
        },
        'history': [],
        'best_model': {},
        'training_time': 0
    }

    for epoch in range(EPOCHS):
        e_start = time.time()

        tr_d, tr_b, tr_t, tr_iou, tr_dm = train_fn(train_loader, model, optimizer, epoch)
        va_d, va_b, va_t, va_iou, va_dm = eval_fn(valid_loader, model, epoch)

        scheduler.step(va_t)
        cur_lr = optimizer.param_groups[0]['lr']

        # log
        train_metrics['dice'].append(tr_d); train_metrics['bce'].append(tr_b); train_metrics['total'].append(tr_t)
        train_metrics['iou'].append(tr_iou); train_metrics['dice_metric'].append(tr_dm)

        val_metrics['dice'].append(va_d); val_metrics['bce'].append(va_b); val_metrics['total'].append(va_t)
        val_metrics['iou'].append(va_iou); val_metrics['dice_metric'].append(va_dm)

        e_time = time.time() - e_start
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} ({e_time:.1f}s)")
        print(f"   Train -> DiceLoss:{tr_d:.5f} BCE:{tr_b:.5f} Total:{tr_t:.5f} | IoU:{tr_iou:.3f} Dice(F1):{tr_dm:.3f}")
        print(f"   Valid -> DiceLoss:{va_d:.5f} BCE:{va_b:.5f} Total:{va_t:.5f} | IoU:{va_iou:.3f} Dice(F1):{va_dm:.3f}")
        print(f"   LR: {cur_lr:.6f}")

        training_log['history'].append({
            'epoch': epoch+1,
            'train_dice_loss': tr_d, 'train_bce_loss': tr_b, 'train_total_loss': tr_t,
            'train_iou': tr_iou, 'train_dice_metric': tr_dm,
            'val_dice_loss': va_d, 'val_bce_loss': va_b, 'val_total_loss': va_t,
            'val_iou': va_iou, 'val_dice_metric': va_dm,
            'lr': cur_lr, 'epoch_time': e_time
        })

        # En iyi modelleri kaydet
        if va_d < best_val_dice:
            path = os.path.join(MODEL_SAVE_DIR, f"segformer_best_dice_{va_d:.5f}.pt")
            torch.save(model.state_dict(), path)
            print(f"💾 Kaydedildi (DiceLoss): {path}")
            best_val_dice = va_d
            training_log['best_model']['by_dice_loss'] = {'path': path, 'epoch': epoch+1, 'val_dice_loss': va_d}

        if va_t < best_val_total:
            path = os.path.join(MODEL_SAVE_DIR, f"segformer_best_total_{va_t:.5f}.pt")
            torch.save(model.state_dict(), path)
            print(f"💾 Kaydedildi (TotalLoss): {path}")
            best_val_total = va_t
            training_log['best_model']['by_total_loss'] = {'path': path, 'epoch': epoch+1, 'val_total_loss': va_t}

        # Her 5 epoch’ta örnek görsel
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, valid_loader, epoch, LOG_DIR, thr=RATIO)
            if DEVICE == "cuda": torch.cuda.empty_cache()

    # 5) Final
    total_time = time.time() - start_time
    training_log['training_time'] = total_time
    print(f"\n🎉 Eğitim bitti. Süre: {total_time/60:.1f} dk | En iyi TotalLoss: {best_val_total:.5f}")

    # Plot
    plot_path = os.path.join(LOG_DIR, 'training_metrics.png')
    save_training_plot(train_metrics, val_metrics, plot_path)
    print(f"📈 Metrik grafikleri: {plot_path}")

    # Son örnek görsel
    visualize_predictions(model, valid_loader, EPOCHS-1, LOG_DIR, thr=RATIO)

    # JSON/CSV
    with open(os.path.join(LOG_DIR,'training_log.json'),'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    df = pd.DataFrame(training_log['history'])
    df.to_csv(os.path.join(LOG_DIR,'training_metrics.csv'), index=False)
    print(f"📝 Log/CSV kaydedildi: {LOG_DIR}")

if __name__ == "__main__":
    main()
