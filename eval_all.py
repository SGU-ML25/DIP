import os
import torch
import numpy as np
from src.evaluateV2 import PDIPRLEvaluator

categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'wood']

results = []

for cat in categories:
    try:
        evaluator = PDIPRLEvaluator(category=cat)
        
        from src.dataset import MVTecDataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.metrics import roc_auc_score, precision_recall_curve
        
        dataset = MVTecDataset(cat, split='test')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_preds, all_masks = [], []
        
        for i, batch in enumerate(dataloader):
            image = batch['image'].to(evaluator.device)
            gt_mask = batch['mask'].numpy().flatten()
            h_map = np.zeros((1, evaluator.config.IMG_SIZE // evaluator.config.PATCH_SIZE, 
                              evaluator.config.IMG_SIZE // evaluator.config.PATCH_SIZE), dtype=np.float32)
            anomaly_map, tau = evaluator.compute_anomaly_map(image, h_map)
            obj_mask = evaluator.get_object_mask(image)
            anomaly_map *= obj_mask
            if anomaly_map.max() > 0:
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            from src.utils import post_process
            binary_mask = post_process(anomaly_map, min_area=50)
            final_score = anomaly_map * (binary_mask > 0)
            all_preds.extend(final_score.flatten())
            all_masks.extend(gt_mask)
            
        all_masks = (np.array(all_masks) > 0.5).astype(np.int32)
        all_preds = np.array(all_preds)
        pixel_auc = roc_auc_score(all_masks, all_preds)
        precision, recall, _ = precision_recall_curve(all_masks, all_preds)
        f1_max = np.max(2 * recall * precision / (recall + precision + 1e-8))
        print(f"[{cat}] Pixel-AUC: {pixel_auc:.4f} | F1-Max: {f1_max:.4f}")
        results.append((pixel_auc, f1_max))
    except Exception as e:
        print(f"[{cat}] Error: {e}")

if results:
    avg_auc = np.mean([r[0] for r in results])
    avg_f1 = np.mean([r[1] for r in results])
    print(f"\nOVERALL -> Average Pixel-AUC: {avg_auc:.4f} | Average F1-Max: {avg_f1:.4f}")
else:
    print("No results.")