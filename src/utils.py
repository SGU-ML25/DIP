import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class Config:
    DATA_ROOT = "./data/"
    CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    IMG_SIZE = 128
    PATCH_SIZE = 4
    BATCH_SIZE = 4
    LR_AE = 1e-4
    LR_PRED = 2e-4
    LR_RL = 1e-5
    BETA_START = 1.0
    BETA_END = 0.1
    EPOCHS = 50     # Giảm từ 100 xuống 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 # Tận dụng đa nhân CPU để load ảnh

def get_history_map(num_images, img_size, patch_size):
    num_patches_side = img_size // patch_size
    return np.zeros((num_images, num_patches_side, num_patches_side), dtype=np.float32)

def ssim(img1, img2, window_size=11, size_average=True):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map

def get_sobel_map(images):
    B, C, H, W = images.shape
    gray = images.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(images.device)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
    return magnitude

def calculate_category_template(dataloader, device):
    """Tính ảnh trung bình (Golden Sample) của các ảnh bình thường"""
    print("Calculating category template...")
    template = torch.zeros((1, 3, Config.IMG_SIZE, Config.IMG_SIZE)).to(device)
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            template += torch.sum(images, dim=0, keepdim=True)
            count += images.shape[0]
    return template / count

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (res[0]*frequency, res[1]*frequency))
        frequency *= 2
        amplitude *= persistence
    return noise

def generate_synthetic_anomalies(images):
    B, C, H, W = images.shape
    device = images.device
    augmented_images = images.clone()
    anomaly_masks = torch.zeros((B, 1, H, W), device=device)
    
    for i in range(B):
        # 1. Perlin Noise based anomalies (natural cracks/stains)
        if np.random.rand() > 0.3:
            res_val = np.random.choice([2, 4])
            res = (res_val, res_val)
            noise = generate_fractal_noise_2d((H, W), res, octaves=2)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            
            # Create a mask by thresholding noise to get "cracks"
            noise_mask = (noise > 0.7).astype(np.float32)
            noise_mask = torch.from_numpy(noise_mask).to(device)
            
            # Apply color variation to the noise mask
            color_shift = torch.tensor([np.random.uniform(-0.2, 0.2) for _ in range(3)], device=device).view(3, 1, 1)
            augmented_images[i] = torch.clamp(augmented_images[i] * (1 - noise_mask) + (augmented_images[i] + color_shift) * noise_mask, 0, 1)
            anomaly_masks[i, 0] = torch.maximum(anomaly_masks[i, 0], noise_mask)

        # 2. Structural anomalies (patch swap)
        if np.random.rand() > 0.5:
            ph, pw = np.random.randint(10, 40), np.random.randint(10, 40)
            sy, sx = np.random.randint(0, H - ph), np.random.randint(0, W - pw)
            dy, dx = np.random.randint(0, H - ph), np.random.randint(0, W - pw)
            patch = images[i, :, sy:sy+ph, sx:sx+pw]
            patch = patch.flip(dims=[1, 2])
            augmented_images[i, :, dy:dy+ph, dx:dx+pw] = patch
            anomaly_masks[i, 0, dy:dy+ph, dx:dx+pw] = 1.0
            
    return augmented_images, anomaly_masks

def post_process(anomaly_map, threshold=None, min_area=50):
    am_norm = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
    
    if threshold is None:
        _, thresh = cv2.threshold(am_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # threshold provided by RL agent (0-1) scaled to 0-255
        t_val = int(threshold * 255)
        _, thresh = cv2.threshold(am_norm, t_val, 255, cv2.THRESH_BINARY)
        
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
    clean_mask = np.zeros_like(closing)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask
