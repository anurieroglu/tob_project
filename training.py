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

# --- Configuration and Hyperparameters ---
HEIGHT, WIDTH = (512, 512)
DEVICE = 'cuda'
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.001
RATIO = 0.5
SAMPLE_NUM = 2
ENCODER = 'resnet50'
WEIGHTS = 'imagenet'

# --- Data Loading and Processing ---
class Load_Data(Dataset):
    """
    Custom Dataset class for loading satellite images and their masks.
    """
    def __init__(self, image_list, mask_list):
        super().__init__()
        self.images_list = image_list
        self.mask_list = mask_list
        self.len = len(image_list)
        self.transform = A.Compose([
            A.Resize(HEIGHT, WIDTH),
            A.HorizontalFlip(),
            # A.RandomBrightnessContrast(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
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
    U-Net based segmentation model using segmentation_models_pytorch.
    """
    def __init__(self):
        super().__init__()
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
def train_fn(data_loader, model, optimizer):
    model.train()
    total_diceloss = 0.0
    total_bceloss = 0.0
    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits, diceloss, bceloss = model(images, masks)
        # Note: The original code had two backward calls, which can be problematic.
        # It's better to combine the losses before calling backward.
        loss = diceloss + bceloss
        loss.backward()
        optimizer.step()
        total_diceloss += diceloss.item()
        total_bceloss += bceloss.item()
        
    return total_diceloss / len(data_loader), total_bceloss / len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_diceloss = 0.0
    total_bceloss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, diceloss, bceloss = model(images, masks)
            total_diceloss += diceloss.item()
            total_bceloss += bceloss.item()
            
    # Visualization part, moved to a separate function or main loop for clarity
    # and to avoid repeated plotting within the evaluation loop.
            
    return total_diceloss / len(data_loader), total_bceloss / len(data_loader)

def visualize_predictions(model, data_loader):
    """
    Visualizes a single sample prediction from the model.
    """
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
        axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray')
        axarr[1].set_title('True Mask')
        axarr[2].imshow(np.transpose(pred_mask.detach().cpu().squeeze(0), (1, 2, 0)))
        axarr[2].set_title('Predicted Mask')
        plt.show()

# --- Main Execution Block ---
def main():
    """
    Main function to orchestrate the data loading, training, and evaluation.
    """
    # 1. Data Loading
    X = sorted(glob.glob('/tf/uygulamalar/kamag/tob/data/datasets/franciscoescobar/satellite-images-of-water-bodies/versions/2/Water Bodies Dataset/Images/*'))
    y = sorted(glob.glob('/tf/uygulamalar/kamag/tob/data/datasets/franciscoescobar/satellite-images-of-water-bodies/versions/2/Water Bodies Dataset/Masks/*'))
    
    # Check if data paths are valid
    if not X or not y:
        print("Error: No images or masks found. Please check the file paths.")
        return

    # 2. Data Splitting and DataLoader setup
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = Load_Data(X_train, y_train)
    valid_dataset = Load_Data(X_val, y_val)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use a better practice for num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 3. Model, Optimizer, and Loss Initialization
    model = SegmentationModel()
    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. Directory creation for saving models
    model_save_dir = f"/tf/uygulamalar/kamag/tob/{ENCODER}_unet"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 5. Training Loop
    best_val_dice_loss = np.inf

    for i in range(EPOCHS):
        train_dice_loss, train_bce_loss = train_fn(train_loader, model, optimizer)
        valid_dice_loss, valid_bce_loss = eval_fn(valid_loader, model)
        
        print(f'Epochs: {i+1}\nTrain Loss --> Dice: {train_dice_loss:.5f} BCE: {train_bce_loss:.5f} \nValid Loss --> Dice: {valid_dice_loss:.5f} BCE: {valid_bce_loss:.5f}')
        
        # Save the best model
        if valid_dice_loss < best_val_dice_loss:
            save_path = os.path.join(model_save_dir, f"model_{valid_dice_loss:.5f}dice.pt")
            torch.save(model.state_dict(), save_path)
            print(f'Model Saved to {save_path}')
            best_val_dice_loss = valid_dice_loss

    # 6. Final visualization
    print("\nTraining complete. Visualizing a sample prediction.")
    visualize_predictions(model, valid_loader)

# --- Standard Python entry point ---
if __name__ == "__main__":
    main()