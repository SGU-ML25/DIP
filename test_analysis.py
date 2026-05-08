import torch
from src.utils import Config, generate_synthetic_anomalies, ssim
from src.dataset import MVTecDataset
from src.models import Autoencoder, PredictorFCN
from src.evaluate import post_process
import cv2
import numpy as np

DEVICE = Config.DEVICE
CATEGORY = "bottle"

def test_roi():
    print("Testing ROI Masking...")
    dummy_img = torch.rand(1, 3, 256, 256).to(DEVICE)
    img_np = dummy_img[0].cpu().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    print("ROI Masking successful.")

def test_synthetic():
    print("Testing Synthetic Anomalies...")
    dummy_imgs = torch.rand(4, 3, 256, 256).to(DEVICE)
    aug_images, masks = generate_synthetic_anomalies(dummy_imgs)
    print("Synthetic Anomalies generation successful.")

if __name__ == "__main__":
    test_roi()
    test_synthetic()
