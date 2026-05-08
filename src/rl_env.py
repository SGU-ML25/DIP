import torch
import torch.nn.functional as F
import numpy as np

class RLAnomalyEnv:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.PATCH_SIZE
        self.img_size = config.IMG_SIZE
        self.num_patches_side = self.img_size // self.patch_size

    def get_state(self, image, history_map):
        B, _, H, W = image.shape
        
        # 1-3. RGB (3 channels)
        rgb = image
        
        # 4. Context (1 channel): Grayscale
        context = torch.mean(image, dim=1, keepdim=True)
        
        # 5. History (1 channel)
        history = torch.from_numpy(history_map).to(image.device).unsqueeze(1)
        history = F.interpolate(history, size=(H, W), mode='nearest')
        
        # 6. Position (1 channel): Distance from center or coordinate grid
        # Create a coordinate grid [-1, 1]
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        pos = torch.sqrt(grid_y**2 + grid_x**2).unsqueeze(0).unsqueeze(0).to(image.device)
        pos = pos.repeat(B, 1, 1, 1)
        
        return torch.cat([rgb, context, history, pos], dim=1)

    def calculate_reward(self, image, predictor_map, history_map, action, beta):
        B = image.shape[0]
        rewards = []
        
        for i in range(B):
            row = action[i] // self.num_patches_side
            col = action[i] % self.num_patches_side
            y = row * self.patch_size
            x = col * self.patch_size
            
            patch_img = image[i, :, y:y+self.patch_size, x:x+self.patch_size]
            patch_pred = predictor_map[i, 0, y:y+self.patch_size, x:x+self.patch_size]
            
            # R_clone: Variance as proxy for detail
            r_clone = torch.var(patch_img).item()
            
            # R_cover: Penalize repeat sampling
            h_val = history_map[i, row, col]
            r_cover = 1.0 / (1.0 + h_val)
            
            # R_pred: Predictor signal
            r_pred = torch.mean(patch_pred).item()
            
            reward = beta * (r_clone + r_cover) + (1 - beta) * r_pred
            rewards.append(reward)
            
            # Update history map (this should probably be done in the training loop, but let's track it)
            history_map[i, row, col] += 1.0
            
        return torch.tensor(rewards, device=image.device)
