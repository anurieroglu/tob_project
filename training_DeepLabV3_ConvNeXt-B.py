# training_DeepLabV3_EfficientNetV2M.py
import os, glob, json, time, argparse, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- Argparse -----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="dataset/Images", help="Görüntü klasörü")
    p.add_argument("--masks",  type=str, default="dataset/Masks",  help="Maske klasörü")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--bs",     type=int, default=4)
    p.add_argument("--size",   type=int, default=512)
    p.add_argument("--arch",   type=str, default="deeplabv3plus", choices=["deeplabv3plus","unet","unetplusplus"])
    # NOT: ConvNeXt smp sürümünüzde yok. En sağlam seçeneği default verdim:
    p.add_argument("--encoder",type=str, default="timm-efficientnetv2-m")
    p.add_argument("--weights",type=str, default="imagenet")
    p.add_argument("--ratio",  type=float, default=0.5, help="Sigmoid threshold")
    p.add_argument("--val",    type=float, default=0.2, help="Validation split")
    p.add_argument("--workers",type=int, default=2)
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()

# ----------------- GPU seçimi -----------------
def pick_best_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA yok, CPU kullanılacak")
        return "cpu"
    n = torch.cuda.device_count()
    print(f"🔍 {n} GPU bulundu")
    free_list = []
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        torch.cuda.empty_cache()
        mem_free = prop.total_memory - torch.cuda.memory_reserved(i)
        print(f"  GPU {i}: {prop.name} | ~{mem_free/1024**3:.1f} GB boş")
        free_list.append((mem_free, i, prop.name))
    best = max(free_list)[1]
    print(f"✅ Otomatik seçim: cuda:{best}")
    return f"cuda:{best}"

# ----------------- Dataset -----------------
class WaterSet(Dataset):
    def __init__(self, img_paths, msk_paths, train=True, image_size=512):
        self.imgs = img_paths
        self.msks = msk_paths
        self.train = train
        self.size = image_size

        if train:
            # Albumentations v2: RandomResizedCrop -> size=(H,W), scale ve ratio 0-1 aralığında
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(self.size, self.size),
                                    scale=(0.7, 1.0),
                                    ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Affine(rotate=(-10,10), scale=(0.9,1.1),
                         shear=(-5,5), translate_percent=(0.0, 0.05), p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(height=self.size, width=self.size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.imgs[idx]).convert("RGB"))
        msk = np.array(Image.open(self.msks[idx]).convert("L"))
        # mask'i 0/1'e normalize et
        if msk.dtype != np.uint8:
            msk = (msk > 127).astype(np.uint8)
        else:
            msk = (msk > 127).astype(np.uint8)

        augmented = self.aug(image=img, mask=msk)
        img = augmented["image"]
        msk = augmented["mask"].unsqueeze(0).float()  # [1,H,W]
        return img, msk

# ----------------- Model -----------------
class DLv3pModel(nn.Module):
    def __init__(self, encoder_name="timm-efficientnetv2-m", encoder_weights="imagenet", classes=1):
        super().__init__()
        if args.arch == "deeplabv3plus":
            self.net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=classes,
                activation=None
            )
        elif args.arch == "unetplusplus":
            self.net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=classes,
                activation=None
            )
        else:
            self.net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=classes,
                activation=None
            )

    def forward(self, x, y=None):
        logits = self.net(x)  # [B,1,H,W]
        if y is not None:
            loss_dice = DiceLoss(mode="binary")(logits, y)
            loss_bce  = nn.BCEWithLogitsLoss()(logits, y)
            return logits, loss_dice, loss_bce
        return logits

# ----------------- Train / Valid -----------------
def train_one_epoch(loader, model, opt, device, epoch, epochs, scaler=None):
    model.train()
    td, tb, tt = 0.0, 0.0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for imgs, msks in pbar:
        imgs, msks = imgs.to(device), msks.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, ld, lb = model(imgs, msks)
                loss = ld + lb
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits, ld, lb = model(imgs, msks)
            loss = ld + lb
            loss.backward()
            opt.step()
        td += ld.item(); tb += lb.item(); tt += loss.item()
        pbar.set_postfix(dice=f"{ld.item():.4f}", bce=f"{lb.item():.4f}", total=f"{loss.item():.4f}")
    n = len(loader)
    return td/n, tb/n, tt/n

@torch.no_grad()
def validate(loader, model, device, epoch, epochs):
    model.eval()
    td, tb, tt = 0.0, 0.0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
    for imgs, msks in pbar:
        imgs, msks = imgs.to(device), msks.to(device)
        logits, ld, lb = model(imgs, msks)
        loss = ld + lb
        td += ld.item(); tb += lb.item(); tt += loss.item()
        pbar.set_postfix(dice=f"{ld.item():.4f}", bce=f"{lb.item():.4f}", total=f"{loss.item():.4f}")
    n = len(loader)
    return td/n, tb/n, tt/n

def save_curves(tr_hist, va_hist, out_png, lr):
    epochs = range(1, len(tr_hist["dice"])+1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, tr_hist["total"], label="Train Total")
    plt.plot(epochs, va_hist["total"], label="Valid Total")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss Curves (lr={lr})"); plt.legend()
    plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ----------------- Main -----------------
def main():
    global args
    args = get_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = pick_best_gpu()

    # dosyaları topla
    X = sorted(glob.glob(os.path.join(args.images, "*")))
    y = sorted(glob.glob(os.path.join(args.masks,  "*")))
    assert len(X) == len(y) and len(X) > 0, "Görüntü/maske sayıları uyuşmuyor ya da boş."

    print(f"✅ {len(X)} görüntü / {len(y)} maske")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=args.val, random_state=args.seed)

    tr_ds = WaterSet(X_tr, y_tr, train=True,  image_size=args.size)
    va_ds = WaterSet(X_va, y_va, train=False, image_size=args.size)

    bs = args.bs
    if "cuda" in device:
        # V100 32GB için bs=4 rahat; istersen arttır
        bs = args.bs
    print(f"📦 Batch size: {bs}")

    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=args.workers, pin_memory=("cuda" in device))
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=args.workers, pin_memory=("cuda" in device))

    exp = f"{args.encoder.replace('/','-')}_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("training_logs", exp)
    save_dir = os.path.join("saved_models", f"{args.encoder.replace('/','-')}_{args.arch}")
    os.makedirs(log_dir, exist_ok=True); os.makedirs(save_dir, exist_ok=True)
    print(f"🚀 Deney: {exp}\n📁 Log:   {log_dir}\n💾 Model: {save_dir}\n⚙️  {args.encoder} + {args.arch} | epochs={args.epochs} bs={bs} lr={args.lr} size={args.size}")

    model = DLv3pModel(encoder_name=args.encoder, encoder_weights=args.weights).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Parametreler: total={total_params:,} | trainable={train_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in device))

    best_val = float("inf")
    tr_hist = {"dice":[], "bce":[], "total":[]}
    va_hist = {"dice":[], "bce":[], "total":[]}
    log_json = {
        "experiment": exp,
        "config": vars(args),
        "params": {"total": int(total_params), "trainable": int(train_params)},
        "history": [],
        "best": {}
    }

    t0 = time.time()
    for ep in range(args.epochs):
        td, tb, tt = train_one_epoch(tr_ld, model, opt, device, ep, args.epochs, scaler)
        vd, vb, vt = validate(va_ld, model, device, ep, args.epochs)
        sch.step(vt)

        tr_hist["dice"].append(td); tr_hist["bce"].append(tb); tr_hist["total"].append(tt)
        va_hist["dice"].append(vd); va_hist["bce"].append(vb); va_hist["total"].append(vt)

        cur_lr = opt.param_groups[0]["lr"]
        print(f"\n📊 Epoch {ep+1}/{args.epochs} | Train: D={td:.4f} BCE={tb:.4f} T={tt:.4f} | Valid: D={vd:.4f} BCE={vb:.4f} T={vt:.4f} | LR={cur_lr:.6f}")

        log_json["history"].append({
            "epoch": ep+1, "train_dice": td, "train_bce": tb, "train_total": tt,
            "valid_dice": vd, "valid_bce": vb, "valid_total": vt, "lr": cur_lr
        })

        if vt < best_val:
            best_val = vt
            best_path = os.path.join(save_dir, f"model_{vt:.5f}_valtotal.pt")
            torch.save(model.state_dict(), best_path)
            print(f"💾 En iyi model kaydedildi: {best_path}")
            log_json["best"] = {"epoch": ep+1, "valid_total": vt, "path": best_path}

        # örnek görselleştirme
        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                imgs, msks = next(iter(va_ld))
                imgs = imgs.to(device)
                logits = model(imgs)
                pr = torch.sigmoid(logits).cpu().numpy()
                # basit tek görsel kaydı
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1,3, figsize=(12,4))
                ax[0].imshow(np.moveaxis(imgs[0].cpu().numpy(),0,2)); ax[0].set_title("Image"); ax[0].axis("off")
                ax[1].imshow(msks[0,0].cpu().numpy(), cmap="gray"); ax[1].set_title("Mask"); ax[1].axis("off")
                ax[2].imshow((pr[0,0]>args.ratio).astype(np.uint8), cmap="gray"); ax[2].set_title("Pred"); ax[2].axis("off")
                out_png = os.path.join(log_dir, f"pred_ep{ep+1}.png")
                plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
                print(f"🖼️ Görsel kaydedildi: {out_png}")

    total_min = (time.time()-t0)/60.0
    print(f"\n🎉 Eğitim bitti | Süre ≈ {total_min:.1f} dk | En iyi valid total={best_val:.5f}")

    # metrik grafik
    curves_png = os.path.join(log_dir, "loss_curves.png")
    save_curves(tr_hist, va_hist, curves_png, args.lr)
    print(f"📈 Loss eğrileri: {curves_png}")

    # log yaz
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(log_json, f, indent=2)

    # csv
    pd.DataFrame(log_json["history"]).to_csv(os.path.join(log_dir,"metrics.csv"), index=False)
    print(f"📝 Log ve metrikler kaydedildi: {log_dir}")

if __name__ == "__main__":
    main()
