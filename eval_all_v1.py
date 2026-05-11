import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

from src.utils import Config, ssim, post_process
Config.IMG_SIZE = 128 # Force IMG_SIZE to 128 for old checkpoints
Config.PATCH_SIZE = 8 # Old checkpoints used 1024 out_dim for RL, which implies PATCH_SIZE=8 (128/8 = 16, 16^2=256, wait: 256/8=32 -> 1024)

from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN, ResNetBackbone

categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'wood']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for category in categories:
    try:
        dataset = MVTecDataset(category, split='test')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        ae = Autoencoder().to(device)
        predictor = PredictorFCN().to(device)
        backbone = ResNetBackbone().to(device).eval()
        
        ae_path = f"checkpoints/ae_{category}.pth"
        pred_path = f"checkpoints/pred_{category}.pth"
        temp_path = f"checkpoints/template_{category}.pth"
        
        if not os.path.exists(ae_path):
            print(f"[{category}] Skipping, {ae_path} not found.")
            continue
            
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        predictor.load_state_dict(torch.load(pred_path, map_location=device))
        
        temp_data = torch.load(temp_path, map_location=device)
        if isinstance(temp_data, torch.Tensor):
            template = temp_data.to(device)
        else:
            print(f"[{category}] Warning: template is not a tensor")
            template = None
        
        ae.eval()
        predictor.eval()
        
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images = batch['image'].to(device)
                
                # Masks need to be resized if they are loaded at 128x128
                masks = batch['mask'].numpy().flatten()
                
                resnet_features = backbone(images)
                reconstructed = ae(images)
                
                res_ae = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
                if template is not None:
                    res_temp = torch.abs(images - template).mean(dim=1, keepdim=True)
                    residual_map = 0.5 * res_ae + 0.5 * res_temp
                else:
                    residual_map = res_ae
                
                pred_map = predictor(residual_map, resnet_features)
                ssim_val_map = 1.0 - ssim(images, reconstructed, size_average=False)
                combined_map = pred_map * ssim_val_map
                combined_np = combined_map[0, 0].cpu().numpy()
                
                processed_mask = post_process(combined_np) 
                soft_refined = combined_np * (processed_mask > 0)
                
                all_preds.extend(soft_refined.flatten())
                all_masks.extend(masks)

        all_masks = (np.array(all_masks) > 0.5).astype(np.int32)
        all_preds = np.array(all_preds)
        
        auc = roc_auc_score(all_masks, all_preds)
        precision, recall, thresholds = precision_recall_curve(all_masks, all_preds)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
        f1_max = np.max(f1_scores)
        
        print(f"[{category}] Pixel-AUC: {auc:.4f} | F1-Max: {f1_max:.4f}")
        results.append((auc, f1_max))
    except Exception as e:
        print(f"[{category}] Error: {e}")

if results:
    avg_auc = np.mean([r[0] for r in results])
    avg_f1 = np.mean([r[1] for r in results])
    print(f"\nOVERALL -> Average Pixel-AUC: {avg_auc:.4f} | Average F1-Max: {avg_f1:.4f}")
else:
    print("No results.")