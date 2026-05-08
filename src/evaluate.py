import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Handle imports
from src.utils import Config, ssim, post_process, get_sobel_map
from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN, ResNetBackbone

def evaluate(category):
    device = torch.device(Config.DEVICE)
    print(f"Evaluating category: {category} with Advanced DIP & Template-Aware Models...")
    
    dataset = MVTecDataset(category, split='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    ae = Autoencoder().to(device)
    predictor = PredictorFCN().to(device)
    backbone = ResNetBackbone().to(device).eval()
    
    ae_path = f"checkpoints/ae_{category}.pth"
    pred_path = f"checkpoints/pred_{category}.pth"
    temp_path = f"checkpoints/template_{category}.pth"
    
    if not os.path.exists(ae_path) or not os.path.exists(pred_path) or not os.path.exists(temp_path):
        print(f"Checkpoints or Template for {category} not found. Please train first.")
        return

    ae.load_state_dict(torch.load(ae_path, map_location=device))
    predictor.load_state_dict(torch.load(pred_path, map_location=device))
    template = torch.load(temp_path, map_location=device)
    
    ae.eval()
    predictor.eval()
    
    all_preds = []
    all_masks = []
    
    os.makedirs("results", exist_ok=True)
    
    print("Running enhanced inference...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].numpy().flatten()
            
            resnet_features = backbone(images)
            reconstructed = ae(images)
            
            # 1. Structural Residual Map (AE-based)
            res_ae = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
            # 2. Template Residual Map (Golden Sample-based)
            res_temp = torch.abs(images - template).mean(dim=1, keepdim=True)
            
            # Combine signals
            residual_map = 0.5 * res_ae + 0.5 * res_temp
            
            # Predictor with Context
            pred_map = predictor(residual_map, resnet_features)
            
            # SSIM signal for extra structural consistency
            ssim_val_map = 1.0 - ssim(images, reconstructed, size_average=False)
            
            # Fusion
            combined_map = pred_map * ssim_val_map
            
            # 2. Hậu xử lý DIP (Otsu, Morphology, CCA)
            combined_np = combined_map[0, 0].cpu().numpy()
            processed_mask = post_process(combined_np) 
            
            soft_refined = combined_np * (processed_mask > 0)
            
            all_preds.extend(soft_refined.flatten())
            all_masks.extend(masks)
            
            label = batch['label'].item()
            if (label == 1 and i < 50) or i < 5:
                fig, axes = plt.subplots(1, 6, figsize=(30, 5))
                axes[0].imshow(images[0].cpu().permute(1, 2, 0))
                axes[0].set_title(f"Original ({label})")
                axes[1].imshow(template[0].cpu().permute(1, 2, 0))
                axes[1].set_title("Golden Template")
                axes[2].imshow(reconstructed[0].cpu().permute(1, 2, 0))
                axes[2].set_title("Reconstructed")
                axes[3].imshow(batch['mask'][0, 0], cmap='gray')
                axes[3].set_title("GT Mask")
                axes[4].imshow(combined_np, cmap='hot')
                axes[4].set_title("Raw Score")
                axes[5].imshow(processed_mask, cmap='gray')
                axes[5].set_title("DIP Refined")
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"results/{category}_template_{i}.png")
                plt.close()

    all_masks = (np.array(all_masks) > 0.5).astype(np.int32)
    all_preds = np.array(all_preds)
    
    auc = roc_auc_score(all_masks, all_preds)
    precision, recall, thresholds = precision_recall_curve(all_masks, all_preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    f1_max = np.max(f1_scores)
    
    print(f"\n--- Industrial Template-Aware Metrics for {category} ---")
    print(f"Pixel-level AUC: {auc:.4f}")
    print(f"Pixel-level F1-max: {f1_max:.4f}")

if __name__ == "__main__":
    evaluate("bottle")
