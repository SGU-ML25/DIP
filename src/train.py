import os
# from tomlkit import datetime
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# If running as script, handle imports
from src.utils import Config, get_history_map, ssim, generate_synthetic_anomalies, get_sobel_map, calculate_category_template
from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN, RLAgent, ResNetBackbone
from src.rl_env import RLAnomalyEnv
import torch.nn as nn

def train(category):
    device = torch.device(Config.DEVICE)
    print(f"Starting optimized training for {category} on {device}...")
    
    dataset = MVTecDataset(category, split='train')
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Calculate Template (Golden Sample)
    template = calculate_category_template(dataloader, device)
    
    ae = Autoencoder().to(device)
    predictor = PredictorFCN().to(device)
    num_actions = (Config.IMG_SIZE // Config.PATCH_SIZE)**2
    agent = RLAgent(out_dim=num_actions).to(device)
    backbone = ResNetBackbone().to(device).eval()
    
    opt_ae = optim.Adam(ae.parameters(), lr=Config.LR_AE)
    opt_pred = optim.Adam(predictor.parameters(), lr=Config.LR_PRED)
    opt_agent = optim.Adam(agent.parameters(), lr=Config.LR_RL)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    criterion_bce = nn.BCEWithLogitsLoss()
    
    env = RLAnomalyEnv(Config)
    history_maps = get_history_map(len(dataset), Config.IMG_SIZE, Config.PATCH_SIZE)
    
    for epoch in range(Config.EPOCHS):
        beta = Config.BETA_START - (Config.BETA_START - Config.BETA_END) * (epoch / Config.EPOCHS)
        total_reward = 0
        total_loss_ae = 0
        total_loss_pred = 0
        
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            idxs = batch['idx'].numpy()
            batch_history = history_maps[idxs]
            
            with torch.no_grad():
                resnet_feats = backbone(images)
            
            # --- 1. Autoencoder Training ---
            opt_ae.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                reconstructed = ae(images)
                orig_sobel = get_sobel_map(images)
                recon_sobel = get_sobel_map(reconstructed)
                
                loss_pixel = torch.mean((images - reconstructed)**2)
                loss_ssim = 1.0 - ssim(images, reconstructed)
                loss_struct = torch.mean((orig_sobel - recon_sobel)**2)
                
                loss_ae = 0.3 * loss_pixel + 0.3 * loss_ssim + 0.4 * loss_struct
            
            scaler.scale(loss_ae).backward()
            scaler.step(opt_ae)
            
            # --- 2. Predictor Training ---
            aug_images, anomaly_masks = generate_synthetic_anomalies(images)
            
            opt_pred.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    aug_reconstructed = ae(aug_images)
                    aug_resnet_feats = backbone(aug_images)
                    
                    # AE Residual
                    res_ae = torch.abs(aug_images - aug_reconstructed).mean(dim=1, keepdim=True)
                    # Template Residual
                    res_temp = torch.abs(aug_images - template).mean(dim=1, keepdim=True)
                    
                    residual_map = 0.5 * res_ae + 0.5 * res_temp
            
            with torch.amp.autocast('cuda'):
                pred_logits = predictor(residual_map, aug_resnet_feats)
                loss_pred = criterion_bce(pred_logits, anomaly_masks)
            
            scaler.scale(loss_pred).backward()
            scaler.step(opt_pred)
            
            # --- 3. RL Agent Training ---
            opt_agent.zero_grad(set_to_none=True)
            state = env.get_state(images, batch_history)
            
            with torch.amp.autocast('cuda'):
                probs, threshold = agent(state, resnet_feats)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                with torch.no_grad():
                    real_res_ae = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
                    real_res_temp = torch.abs(images - template).mean(dim=1, keepdim=True)
                    real_res = 0.5 * real_res_ae + 0.5 * real_res_temp
                    
                    real_pred_logits = predictor(real_res, resnet_feats)
                    real_pred = torch.sigmoid(real_pred_logits) 
                    reward = env.calculate_reward(images, real_pred, batch_history, action, beta)
                
                loss_agent = -(log_prob * reward).mean()
            
            scaler.scale(loss_agent).backward()
            scaler.step(opt_agent)
            scaler.update()
            
            history_maps[idxs] = batch_history
            
            total_reward += reward.mean().item()
            total_loss_ae += loss_ae.item()
            total_loss_pred += loss_pred.item()

        avg_reward = total_reward / len(dataloader)
        avg_ae = total_loss_ae / len(dataloader)
        avg_pred = total_loss_pred / len(dataloader)
        
        print(f"Epoch {epoch}/{Config.EPOCHS} | Beta: {beta:.2f} | Reward: {avg_reward:.4f} | AE Loss: {avg_ae:.6f} | Pred Loss: {avg_pred:.6f}")
        torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(ae.state_dict(), f"checkpoints/ae_{category}_{time}.pth")
    torch.save(predictor.state_dict(), f"checkpoints/pred_{category}_{time}.pth")
    torch.save(agent.state_dict(), f"checkpoints/agent_{category}_{time}.pth")
    torch.save(template, f"checkpoints/template_{category}_{time}.pth") 
    print("Training completed and models saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DIP model on a specific category.")
    parser.add_argument("--category", type=str, default="bottle", help="Category to train on.")
    args = parser.parse_args()
    
    train(args.category)
