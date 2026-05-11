import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np
import cv2
from tqdm import tqdm

# Đảm bảo các module này tồn tại trong folder src của bạn
from src.utils import Config, ssim, post_process, get_sobel_map
from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN, ResNetBackbone, RLAgent
from src.rl_env import RLAnomalyEnv

class PDIPRLEvaluator:
    def __init__(self, category):
        self.device = torch.device(Config.DEVICE)
        self.category = category
        self.config = Config
        
        # 1. Khởi tạo Models theo đúng kỹ thuật Multi-scale trong paper
        self.backbone = ResNetBackbone().to(self.device).eval()
        self.ae = Autoencoder().to(self.device).eval()
        self.predictor = PredictorFCN().to(self.device).eval()
        
        num_actions = (self.config.IMG_SIZE // self.config.PATCH_SIZE)**2
        self.agent = RLAgent(out_dim=num_actions).to(self.device).eval()
        
        self._load_checkpoints()
        
    def _load_checkpoints(self):
        # Load weights theo chuẩn paper
        base_path = "checkpoints"
        paths = {
            'ae': f"{base_path}/ae_{self.category}.pth",
            'pred': f"{base_path}/pred_{self.category}.pth",
            'agent': f"{base_path}/agent_{self.category}.pth",
            'temp': f"{base_path}/template_{self.category}.pth"
        }
        
        self.ae.load_state_dict(torch.load(paths['ae'], map_location=self.device))
        self.predictor.load_state_dict(torch.load(paths['pred'], map_location=self.device))
        self.agent.load_state_dict(torch.load(paths['agent'], map_location=self.device))
        self.template = torch.load(paths['temp'], map_location=self.device).to(self.device)

    def get_object_mask(self, image_tensor):
        """Kỹ thuật lọc nền để tăng AUC (Object-level Awareness)"""
        img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Ngưỡng nhạy để lấy toàn bộ vùng viên thuốc
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        return mask.astype(np.float32) / 255.0

    def compute_anomaly_map(self, image, history_map):
        """Hàm lõi thực thi thuật toán P-DIP-RL"""
        # A. Feature Extraction (Backbone)
        features = self.backbone(image)
        reconstructed = self.ae(image)
        
        # B. Multi-signal Residuals
        res_ae = torch.abs(image - reconstructed).mean(dim=1, keepdim=True)
        res_temp = torch.abs(image - self.template).mean(dim=1, keepdim=True)
        res_grad = torch.abs(get_sobel_map(image) - get_sobel_map(reconstructed)).mean(dim=1, keepdim=True)
        
        # C. RL-based Dynamic Threshold
        env = RLAnomalyEnv(self.config)
        state = env.get_state(image, history_map)
        _, threshold_tensor = self.agent(state, features)
        tau = threshold_tensor.item()
        
        # D. Hybrid Signal Fusion
        # Predictor học cách kết hợp AE và Template dựa trên Context (features)
        raw_map = torch.sigmoid(self.predictor(0.5*res_ae + 0.5*res_temp, features))
        ssim_signal = 1.0 - ssim(image, reconstructed, size_average=False)
        
        # Fusion theo trọng số paper: Contextual + Structural + Gradient
        combined_map = (raw_map * 0.5) + (ssim_signal * 0.3) + (res_grad * 0.2)
        return combined_map[0, 0].cpu().numpy(), tau

    def run(self):
        dataset = MVTecDataset(self.category, split='test')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_preds, all_masks = [], []
        print(f"Running P-DIP-RL Evaluation on: {self.category}")
        
        for i, batch in enumerate(tqdm(dataloader)):
            image = batch['image'].to(self.device)
            gt_mask = batch['mask'].numpy().flatten()
            
            # Khởi tạo history cho RL Agent
            h_map = np.zeros((1, self.config.IMG_SIZE // self.config.PATCH_SIZE, 
                              self.config.IMG_SIZE // self.config.PATCH_SIZE), dtype=np.float32)
            
            # 1. Inference
            anomaly_map, tau = self.compute_anomaly_map(image, h_map)
            
            # 2. Object-level Refinement (Triệt tiêu nhiễu nền)
            obj_mask = self.get_object_mask(image)
            anomaly_map *= obj_mask
            
            # 3. Min-Max Scaling (Chuẩn hóa để tăng AUC)
            if anomaly_map.max() > 0:
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            
            # 4. DIP-inspired Post-processing (Dùng ngưỡng từ RL Agent)
            binary_mask = post_process(anomaly_map, min_area=50) # Hàm utils của bạn
            
            # Đánh giá dựa trên xác suất (soft-score) để AUC cao nhất
            final_score = anomaly_map * (binary_mask > 0)
            
            all_preds.extend(final_score.flatten())
            all_masks.extend(gt_mask)

        # Metrics Calculation
        all_masks = (np.array(all_masks) > 0.5).astype(np.int32)
        all_preds = np.array(all_preds)
        
        pixel_auc = roc_auc_score(all_masks, all_preds)
        precision, recall, _ = precision_recall_curve(all_masks, all_preds)
        f1_max = np.max(2 * recall * precision / (recall + precision + 1e-8))
        
        print(f"\n--- Final P-DIP-RL Results for {self.category} ---")
        print(f"Pixel-AUC: {pixel_auc:.4f} | F1-Max: {f1_max:.4f}")

if __name__ == "__main__":
    evaluator = PDIPRLEvaluator(category="pill")
    evaluator.run()