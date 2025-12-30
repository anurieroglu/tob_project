#!/usr/bin/env python3
"""
SAM Overfitting Fix Script
Overfitting olan SAM modelini düzeltmek için yeniden eğitim

Usage: python fix_overfitting_sam.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedWaterBodiesDataset(Dataset):
    """Overfitting'i önlemek için improved dataset"""
    
    def __init__(self, image_paths, mask_paths, input_size=1024, 
                 augment=True, balanced_sampling=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = input_size
        self.augment = augment
        self.balanced_sampling = balanced_sampling
        
        self.transform = ResizeLongestSide(input_size)
        
        # Analyze dataset for class balance
        if balanced_sampling:
            self._analyze_class_balance()
        
        logger.info(f"Dataset initialized: {len(self.image_paths)} samples")
        logger.info(f"Augmentation: {augment}, Balanced sampling: {balanced_sampling}")
    
    def _analyze_class_balance(self):
        """Analyze class balance to prevent overfitting"""
        water_ratios = []
        
        for i in range(min(100, len(self.mask_paths))):  # Sample first 100
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                water_ratio = (mask > 127).mean()
                water_ratios.append(water_ratio)
        
        if water_ratios:
            avg_water_ratio = np.mean(water_ratios)
            logger.info(f"Average water ratio in dataset: {avg_water_ratio:.3f}")
            
            if avg_water_ratio > 0.7:
                logger.warning("⚠️  Dataset has high water ratio - risk of overfitting!")
            elif avg_water_ratio < 0.1:
                logger.warning("⚠️  Dataset has low water ratio - may underfit!")
            else:
                logger.info("✓ Dataset water ratio looks balanced")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _apply_augmentation(self, image, mask):
        """Apply data augmentation to prevent overfitting"""
        if not self.augment:
            return image, mask
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random rotation (small angles)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        
        # Random brightness/contrast (subtle)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image, mask
    
    def _generate_balanced_points(self, mask, num_points=5):
        """Generate balanced positive/negative points"""
        points = []
        labels = []
        
        # Find water and non-water regions
        water_mask = mask > 127
        non_water_mask = mask <= 127
        
        # Get water points
        water_coords = np.where(water_mask)
        if len(water_coords[0]) > 0:
            water_indices = np.random.choice(len(water_coords[0]), 
                                           min(num_points//2 + 1, len(water_coords[0])), 
                                           replace=False)
            for idx in water_indices:
                y, x = water_coords[0][idx], water_coords[1][idx]
                points.append([x, y])
                labels.append(1)
        
        # Get non-water points
        non_water_coords = np.where(non_water_mask)
        if len(non_water_coords[0]) > 0:
            non_water_indices = np.random.choice(len(non_water_coords[0]), 
                                                min(num_points//2, len(non_water_coords[0])), 
                                                replace=False)
            for idx in non_water_indices:
                y, x = non_water_coords[0][idx], non_water_coords[1][idx]
                points.append([x, y])
                labels.append(0)  # Negative points
        
        # If no points found, use random points
        if len(points) == 0:
            h, w = mask.shape
            for _ in range(num_points):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                points.append([x, y])
                labels.append(1 if mask[y, x] > 127 else 0)
        
        return points, labels
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                raise ValueError(f"Failed to load image or mask: {image_path}, {mask_path}")
            
            # Apply augmentation
            image, mask = self._apply_augmentation(image, mask)
            
            # Resize
            image_resized = cv2.resize(image, (self.input_size, self.input_size))
            mask_resized = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            
            # Generate balanced points
            points, labels = self._generate_balanced_points(mask_resized)
            
            # Convert to tensors (but keep on CPU for DataLoader compatibility)
            image_tensor = torch.as_tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask_tensor = torch.as_tensor(mask_resized, dtype=torch.float32) / 255.0
            points_tensor = torch.as_tensor(points, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'points': points_tensor,
                'labels': labels_tensor
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)
            dummy_mask = torch.zeros((self.input_size, self.input_size), dtype=torch.float32)
            dummy_points = torch.tensor([[self.input_size//2, self.input_size//2]], dtype=torch.float32)
            dummy_labels = torch.tensor([0], dtype=torch.int64)
            
            return {
                'image': dummy_image,
                'mask': dummy_mask,
                'points': dummy_points,
                'labels': dummy_labels
            }

class ImprovedSAMTrainer:
    """Improved SAM trainer with overfitting prevention"""
    
    def __init__(self, model_type='vit_b', device='cuda', input_size=1024):
        self.device = device
        self.model_type = model_type
        self.input_size = input_size
        
        # Load pretrained SAM
        checkpoint_path = f"sam_{model_type}.pth"
        if not os.path.exists(checkpoint_path):
            logger.error(f"Pretrained checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Please download {checkpoint_path}")
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        
        # Freeze image encoder to prevent overfitting
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        
        logger.info("✓ Image encoder frozen to prevent overfitting")
        
        # Setup optimizer with lower learning rate
        trainable_params = []
        for name, param in self.sam.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=1e-5,  # Much lower learning rate
            weight_decay=0.01  # Add weight decay for regularization
        )
        
        logger.info(f"✓ Optimizer setup with {len(trainable_params)} trainable parameters")
        
        # Loss function with class balancing
        self.criterion = self._get_balanced_loss()
        
    def _get_balanced_loss(self):
        """Get balanced loss function to prevent overfitting"""
        def combined_loss(pred_masks, gt_masks, pred_ious=None):
            # Binary cross entropy with logits
            bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='mean')
            
            # Dice loss to handle class imbalance
            pred_sigmoid = torch.sigmoid(pred_masks)
            dice_loss = self._dice_loss(pred_sigmoid, gt_masks)
            
            # Combine losses
            total_loss = 0.7 * bce_loss + 0.3 * dice_loss
            
            return total_loss
        
        return combined_loss
    
    def _dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for better handling of class imbalance"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with overfitting monitoring"""
        self.sam.train()
        
        # Only train mask decoder and prompt encoder
        self.sam.image_encoder.eval()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                images = batch['image'].to(self.device)
                gt_masks = batch['mask'].to(self.device)
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    image_embeddings = self.sam.image_encoder(images)
                
                # Process each sample in batch
                batch_loss = 0
                batch_size = images.shape[0]
                
                for i in range(batch_size):
                    # Get sample data
                    sample_points = points[i]
                    sample_labels = labels[i]
                    sample_gt = gt_masks[i]
                    
                    if len(sample_points) == 0:
                        continue
                    
                    # Prompt encoder
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=(sample_points.unsqueeze(0), sample_labels.unsqueeze(0)),
                        boxes=None,
                        masks=None,
                    )
                    
                    # Mask decoder
                    low_res_masks, iou_predictions = self.sam.mask_decoder(
                        image_embeddings=image_embeddings[i:i+1],
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # Upscale to full resolution
                    masks = F.interpolate(
                        low_res_masks,
                        size=(self.input_size, self.input_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Calculate loss
                    sample_loss = self.criterion(masks[0, 0], sample_gt)
                    batch_loss += sample_loss
                
                if batch_size > 0:
                    batch_loss = batch_loss / batch_size
                    batch_loss.backward()
                    
                    # Gradient clipping to prevent overfitting
                    torch.nn.utils.clip_grad_norm_(self.sam.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
                    
                    progress_bar.set_postfix({'loss': batch_loss.item()})
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, dataloader):
        """Validation with overfitting detection"""
        self.sam.eval()
        total_loss = 0
        num_batches = 0
        
        # Track prediction statistics
        water_ratios = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                try:
                    images = batch['image'].to(self.device)
                    gt_masks = batch['mask'].to(self.device)
                    points = batch['points'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    image_embeddings = self.sam.image_encoder(images)
                    
                    batch_loss = 0
                    batch_size = images.shape[0]
                    
                    for i in range(batch_size):
                        sample_points = points[i]
                        sample_labels = labels[i]
                        sample_gt = gt_masks[i]
                        
                        if len(sample_points) == 0:
                            continue
                        
                        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                            points=(sample_points.unsqueeze(0), sample_labels.unsqueeze(0)),
                            boxes=None,
                            masks=None,
                        )
                        
                        low_res_masks, _ = self.sam.mask_decoder(
                            image_embeddings=image_embeddings[i:i+1],
                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        masks = F.interpolate(
                            low_res_masks,
                            size=(self.input_size, self.input_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        sample_loss = self.criterion(masks[0, 0], sample_gt)
                        batch_loss += sample_loss
                        
                        # Track water prediction ratio
                        pred_binary = torch.sigmoid(masks[0, 0]) > 0.5
                        water_ratio = pred_binary.float().mean().item()
                        water_ratios.append(water_ratio)
                    
                    if batch_size > 0:
                        batch_loss = batch_loss / batch_size
                        total_loss += batch_loss.item()
                        num_batches += 1
                
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Analyze predictions for overfitting
        if water_ratios:
            avg_water_ratio = np.mean(water_ratios)
            logger.info(f"Validation - Average water prediction ratio: {avg_water_ratio:.3f}")
            
            if avg_water_ratio > 0.8:
                logger.warning("⚠️  Possible overfitting: Model predicts too much water")
            elif avg_water_ratio < 0.1:
                logger.warning("⚠️  Possible underfitting: Model predicts too little water")
            else:
                logger.info("✓ Water prediction ratio looks reasonable")
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader, epochs=50, save_dir="fixed_sam_checkpoints"):
        """Main training loop with early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        logger.info("=== Starting Improved SAM Training ===")
        logger.info(f"Epochs: {epochs}, Patience: {patience}")
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_dataloader)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.sam.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'input_size': self.input_size
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_fixed.pth'))
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break
        
        logger.info("=== Training Completed ===")
        return os.path.join(save_dir, 'best_fixed.pth')

def main():
    """Main function"""
    
    # Configuration with overfitting prevention
    CONFIG = {
        'data_dir': 'dataset',
        'epochs': 10,  # Fewer epochs
        'batch_size': 1,  # Small batch size
        'learning_rate': 1e-5,  # Lower learning rate
        'input_size': 1024,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'augmentation': True,  # Enable augmentation
        'balanced_sampling': True  # Enable balanced sampling
    }
    
    logger.info("=== SAM Overfitting Fix ===")
    logger.info(f"Configuration: {CONFIG}")
    
    # Check dataset
    data_dir = Path(CONFIG['data_dir'])
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return
    
    images_dir = data_dir / 'Images'
    masks_dir = data_dir / 'Masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        logger.error("Images or Masks directory not found")
        return
    
    # Get file lists
    image_files = sorted(list(images_dir.glob('*.jpg')))
    mask_files = sorted(list(masks_dir.glob('*.jpg')))
    
    if len(image_files) != len(mask_files):
        logger.error(f"Mismatch: {len(image_files)} images, {len(mask_files)} masks")
        return
    
    logger.info(f"Found {len(image_files)} image-mask pairs")
    
    # Split dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Create datasets
    train_dataset = FixedWaterBodiesDataset(
        train_images, train_masks, 
        input_size=CONFIG['input_size'],
        augment=CONFIG['augmentation'],
        balanced_sampling=CONFIG['balanced_sampling']
    )
    
    val_dataset = FixedWaterBodiesDataset(
        val_images, val_masks,
        input_size=CONFIG['input_size'],
        augment=False,  # No augmentation for validation
        balanced_sampling=CONFIG['balanced_sampling']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create trainer
    trainer = ImprovedSAMTrainer(
        model_type='vit_b',
        device=CONFIG['device'],
        input_size=CONFIG['input_size']
    )
    
    # Train
    best_checkpoint = trainer.train(
        train_loader, val_loader,
        epochs=CONFIG['epochs']
    )
    
    logger.info(f"✓ Training completed! Best model: {best_checkpoint}")
    
    # Test the fixed model
    logger.info("\nTesting fixed model...")
    from test_trained_sam import TrainedSAMPredictor
    
    try:
        predictor = TrainedSAMPredictor(best_checkpoint)
        
        # Find a test image
        test_dir = Path('sapanca_collection_UTM_1')
        if test_dir.exists():
            tif_files = list(test_dir.glob('*.tif'))
            if tif_files:
                result = predictor.test_on_image(
                    tif_files[0],
                    reference_area_km2=45,
                    pixel_size_m=10,
                    output_dir="fixed_sam_results"
                )
                
                if result:
                    water_area = result['analysis']['water_area_km2']
                    logger.info(f"✓ Fixed model result: {water_area:.2f} km²")
                    
                    if 0.1 < result['analysis']['water_percentage'] < 90:
                        logger.info("✓ Water prediction looks reasonable!")
                    else:
                        logger.warning("⚠️  Still may have issues...")
    
    except Exception as e:
        logger.error(f"Testing failed: {e}")

if __name__ == "__main__":
    main()
