#!/usr/bin/env python3
"""
Test Script for Trained Custom SAM Model
Test your trained water bodies detection model

Usage: python test_trained_sam.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from segment_anything import sam_model_registry
import rasterio

class TrainedSAMPredictor:
    """Predictor for trained custom SAM model"""
    
    def __init__(self, checkpoint_path, model_type="vit_b", device=None, input_size=1024):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.input_size = input_size
        
        print(f"Loading trained SAM model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load base SAM model
        self.sam = sam_model_registry[model_type](checkpoint=None)
        
        # Load trained weights
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.sam.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Trained weights loaded successfully!")
            print(f"✓ Training epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"✓ Validation loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.sam.to(self.device)
        self.sam.eval()
        
        print("Custom SAM model loaded and ready for testing!")
    
    def load_tif_image(self, tif_path):
        """Load TIF image and convert to RGB"""
        try:
            with rasterio.open(tif_path) as src:
                if src.count >= 3:
                    red = src.read(1)
                    green = src.read(2) 
                    blue = src.read(3)
                    
                    rgb_image = np.dstack((red, green, blue))
                    rgb_image = ((rgb_image - rgb_image.min()) / 
                               (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8)
                    
                    return rgb_image, src.transform, src.crs
                else:
                    print(f"Warning: {tif_path} doesn't have enough bands")
                    return None, None, None
                    
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            return None, None, None
    
    def generate_water_points(self, image, num_points=5):
        """Generate potential water points using color analysis"""
        # Convert to different color spaces for water detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Water detection heuristics
        # 1. Blue-ish colors in RGB
        blue_mask = (image[:,:,2] > image[:,:,0]) & (image[:,:,2] > image[:,:,1])
        
        # 2. Low saturation in HSV (water often appears grayish)
        low_sat_mask = hsv[:,:,1] < 100
        
        # 3. Specific hue ranges for water
        water_hue_mask = ((hsv[:,:,0] > 90) & (hsv[:,:,0] < 130))  # Blue-cyan range
        
        # Combine masks
        water_potential = blue_mask | (low_sat_mask & water_hue_mask)
        
        # Find contours
        contours, _ = cv2.findContours(water_potential.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small areas
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append([cx, cy])
        
        # If no points found, use grid-based sampling
        if len(points) < 3:
            h, w = image.shape[:2]
            grid_points = [
                [w//4, h//4], [w//2, h//4], [3*w//4, h//4],
                [w//4, h//2], [w//2, h//2], [3*w//4, h//2],
                [w//4, 3*h//4], [w//2, 3*h//4], [3*w//4, 3*h//4]
            ]
            points.extend(grid_points)
        
        # Limit number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = [points[i] for i in indices]
        
        return points
    
    def predict_water_mask(self, image, points=None, use_auto_points=True):
        """Predict water mask using trained SAM model"""
        
        original_h, original_w = image.shape[:2]
        
        if points is None and use_auto_points:
            points = self.generate_water_points(image)
        elif points is None:
            # Default center point
            points = [[original_w//2, original_h//2]]
        
        print(f"Using {len(points)} prompt points")
        
        # Resize image to training input size
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Scale points to resized image
        scale_x = self.input_size / original_w
        scale_y = self.input_size / original_h
        scaled_points = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]
        
        # Prepare input tensor
        input_tensor = torch.as_tensor(image_resized, dtype=torch.float32, device=self.device)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, 3, H, W]
        
        with torch.no_grad():
            # Encode image
            image_embedding = self.sam.image_encoder(input_tensor)
            
            # Convert points to tensor
            point_coords = torch.tensor(scaled_points, dtype=torch.float32, device=self.device)
            point_labels = torch.ones(len(scaled_points), dtype=torch.int64, device=self.device)
            
            # Add batch dimension
            point_coords = point_coords.unsqueeze(0)  # [1, N, 2]
            point_labels = point_labels.unsqueeze(0)  # [1, N]
            
            # Generate mask
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Upscale to original size
            masks_upscaled = F.interpolate(
                low_res_masks,
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to binary mask
            mask = (torch.sigmoid(masks_upscaled[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
            confidence = torch.sigmoid(masks_upscaled[0, 0]).cpu().numpy()
        
        return mask, confidence, points, iou_predictions.cpu().numpy()
    
    def analyze_water_area(self, mask, pixel_size_m=10, reference_area_km2=None):
        """Calculate water area from mask"""
        water_pixels = np.sum(mask)
        total_pixels = mask.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Real area calculation
        pixel_area_m2 = pixel_size_m * pixel_size_m
        water_area_m2 = water_pixels * pixel_area_m2
        water_area_km2 = water_area_m2 / 1_000_000
        
        results = {
            'water_pixels': int(water_pixels),
            'total_pixels': int(total_pixels),
            'water_percentage': water_percentage,
            'water_area_km2': water_area_km2,
            'water_area_m2': water_area_m2
        }
        
        if reference_area_km2:
            error_km2 = abs(water_area_km2 - reference_area_km2)
            error_percentage = (error_km2 / reference_area_km2) * 100
            results.update({
                'reference_area_km2': reference_area_km2,
                'error_km2': error_km2,
                'error_percentage': error_percentage
            })
        
        return results
    
    def visualize_results(self, image, mask, confidence, points, analysis, 
                         title="Trained SAM Results", save_path=None):
        """Visualize detection results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with points
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image with Prompt Points')
        for point in points:
            axes[0, 0].plot(point[0], point[1], 'ro', markersize=8)
        axes[0, 0].axis('off')
        
        # Predicted mask
        axes[0, 1].imshow(mask, cmap='Blues')
        axes[0, 1].set_title(f'Trained SAM Prediction\n{analysis["water_area_km2"]:.2f} km²')
        axes[0, 1].axis('off')
        
        # Confidence map
        axes[1, 0].imshow(confidence, cmap='viridis')
        axes[1, 0].set_title('Prediction Confidence')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[mask == 1] = [0, 100, 255]  # Blue for water
        axes[1, 1].imshow(overlay)
        error_text = f"Error: {analysis['error_percentage']:.1f}%" if 'error_percentage' in analysis else ""
        axes[1, 1].set_title(f'Water Areas Overlay\n{error_text}')
        axes[1, 1].axis('off')
        
        # Add analysis text
        analysis_text = f"""
        Water Area: {analysis['water_area_km2']:.2f} km²
        Water Percentage: {analysis['water_percentage']:.1f}%
        Total Pixels: {analysis['total_pixels']:,}
        Water Pixels: {analysis['water_pixels']:,}
        """
        
        if 'error_percentage' in analysis:
            analysis_text += f"""
        Reference Area: {analysis['reference_area_km2']} km²
        Error: {analysis['error_km2']:.2f} km² ({analysis['error_percentage']:.1f}%)
        """
        
        fig.suptitle(f'{title}\n{analysis_text}', fontsize=12, y=0.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def test_on_image(self, image_path, reference_area_km2=None, pixel_size_m=10, 
                     output_dir="trained_sam_results"):
        """Test trained model on single image"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Testing trained SAM on: {image_path}")
        
        # Load image
        if str(image_path).endswith('.tif'):
            image, transform, crs = self.load_tif_image(image_path)
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            print("Failed to load image!")
            return None
        
        print(f"Image shape: {image.shape}")
        
        # Predict
        start_time = time.time()
        mask, confidence, points, iou_scores = self.predict_water_mask(image)
        prediction_time = time.time() - start_time
        
        # Analyze
        analysis = self.analyze_water_area(mask, pixel_size_m, reference_area_km2)
        
        print(f"Prediction time: {prediction_time:.2f}s")
        print(f"Water area: {analysis['water_area_km2']:.2f} km²")
        if 'error_percentage' in analysis:
            print(f"Error: {analysis['error_percentage']:.1f}%")
        
        # Visualize
        filename = Path(image_path).stem
        save_path = os.path.join(output_dir, f"{filename}_trained_sam_results.png")
        
        self.visualize_results(
            image, mask, confidence, points, analysis,
            title=f"Trained SAM Results - {filename}",
            save_path=save_path
        )
        
        # Save detailed results
        results = {
            'filename': filename,
            'prediction_time': prediction_time,
            'iou_scores': iou_scores.tolist(),
            'num_points': len(points),
            'analysis': analysis,
            'model_info': {
                'checkpoint': 'trained_custom_sam',
                'input_size': self.input_size
            }
        }
        
        results_path = os.path.join(output_dir, f"{filename}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_with_pretrained(self, image_path, pretrained_checkpoint="sam_vit_b.pth"):
        """Compare trained model with pretrained SAM"""
        
        print("=== Comparing Trained vs Pretrained SAM ===")
        
        # Test with trained model
        print("\n1. Testing with TRAINED model:")
        trained_results = self.test_on_image(image_path, output_dir="comparison_results/trained")
        
        # Test with pretrained model
        print("\n2. Testing with PRETRAINED model:")
        pretrained_sam = sam_model_registry[self.model_type](checkpoint=pretrained_checkpoint)
        pretrained_sam.to(self.device)
        pretrained_sam.eval()
        
        # Temporarily replace model
        original_sam = self.sam
        self.sam = pretrained_sam
        
        pretrained_results = self.test_on_image(image_path, output_dir="comparison_results/pretrained")
        
        # Restore trained model
        self.sam = original_sam
        
        # Compare results
        if trained_results and pretrained_results:
            print("\n=== COMPARISON RESULTS ===")
            print(f"Trained Model   - Area: {trained_results['analysis']['water_area_km2']:.2f} km²")
            print(f"Pretrained Model - Area: {pretrained_results['analysis']['water_area_km2']:.2f} km²")
            
            if 'error_percentage' in trained_results['analysis']:
                print(f"Trained Model   - Error: {trained_results['analysis']['error_percentage']:.1f}%")
            if 'error_percentage' in pretrained_results['analysis']:
                print(f"Pretrained Model - Error: {pretrained_results['analysis']['error_percentage']:.1f}%")
        
        return trained_results, pretrained_results

def main():
    """Main testing function"""
    
    # Configuration
    CONFIG = {
        'checkpoint_path': 'final_sam_checkpoints/best.pth',  # Trained model
        'model_type': 'vit_b',
        'test_image_dir': 'sapanca_collection_UTM_1',
        'reference_area_km2': 45,  # Sapanca Lake area
        'pixel_size_m': 10,
        'max_test_images': 3
    }
    
    print("=== Testing Trained Custom SAM Model ===")
    print(f"Configuration: {CONFIG}")
    
    # Check if trained model exists
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"❌ Trained model not found: {CONFIG['checkpoint_path']}")
        print("Available checkpoints:")
        checkpoint_dir = Path(CONFIG['checkpoint_path']).parent
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob("*.pth"):
                print(f"  - {file}")
        else:
            print("  No checkpoint directory found")
        return
    
    # Create predictor
    predictor = TrainedSAMPredictor(
        checkpoint_path=CONFIG['checkpoint_path'],
        model_type=CONFIG['model_type']
    )
    
    # Find test images
    test_dir = Path(CONFIG['test_image_dir'])
    if test_dir.exists():
        # Test on TIF files
        tif_files = list(test_dir.glob("*.tif"))[:CONFIG['max_test_images']]
        
        if tif_files:
            print(f"\nTesting on {len(tif_files)} images from {test_dir}")
            
            for i, tif_file in enumerate(tif_files, 1):
                print(f"\n--- Test {i}/{len(tif_files)}: {tif_file.name} ---")
                
                try:
                    result = predictor.test_on_image(
                        tif_file,
                        reference_area_km2=CONFIG['reference_area_km2'],
                        pixel_size_m=CONFIG['pixel_size_m']
                    )
                    
                    if result:
                        print(f"✓ Success: {result['analysis']['water_area_km2']:.2f} km²")
                        if 'error_percentage' in result['analysis']:
                            print(f"✓ Error: {result['analysis']['error_percentage']:.1f}%")
                    else:
                        print("✗ Failed")
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            # Compare with pretrained on first image
            if tif_files:
                print(f"\n--- Comparison Test ---")
                try:
                    predictor.compare_with_pretrained(tif_files[0])
                except Exception as e:
                    print(f"Comparison failed: {e}")
        
        else:
            print(f"No TIF files found in {test_dir}")
    else:
        print(f"Test directory not found: {test_dir}")
    
    print("\n=== Testing Completed ===")

if __name__ == "__main__":
    main()
