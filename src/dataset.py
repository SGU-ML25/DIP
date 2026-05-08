import os
import glob
from PIL import Image
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.utils import Config

class MVTecDataset(Dataset):
    def __init__(self, category, split='train', img_size=Config.IMG_SIZE, use_dip=True):
        self.category = category
        self.split = split
        self.img_size = img_size
        self.use_dip = use_dip
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        category_path = os.path.join(Config.DATA_ROOT, category)
        
        if split == 'train':
            train_path = os.path.join(category_path, 'train', 'good')
            self.image_paths = sorted(glob.glob(os.path.join(train_path, "*.png")))
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)
        else:
            test_path = os.path.join(category_path, 'test')
            defect_types = os.listdir(test_path)
            for d in defect_types:
                d_path = os.path.join(test_path, d)
                imgs = sorted(glob.glob(os.path.join(d_path, "*.png")))
                self.image_paths.extend(imgs)
                if d == 'good':
                    self.labels.extend([0] * len(imgs))
                    self.mask_paths.extend([None] * len(imgs))
                else:
                    self.labels.extend([1] * len(imgs))
                    gt_dir = os.path.join(category_path, 'ground_truth', d)
                    for img_p in imgs:
                        img_name = os.path.basename(img_p)
                        mask_name = img_name.replace(".png", "_mask.png")
                        mask_p = os.path.join(gt_dir, mask_name)
                        if os.path.exists(mask_p):
                            self.mask_paths.append(mask_p)
                        else:
                            self.mask_paths.append(None)

    def apply_dip(self, image):
        # Convert PIL to CV2
        img_cv = np.array(image)
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Apply to L channel in Lab or just Gray
        lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_dip = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # 2. Bilateral Filter (Noise reduction while preserving edges)
        img_dip = cv2.bilateralFilter(img_dip, 9, 75, 75)
        
        return Image.fromarray(img_dip)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.use_dip:
            image = self.apply_dip(image)
            
        image = self.transform(image)
        
        mask_path = self.mask_paths[idx]
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros((1, self.img_size, self.img_size))
            
        return {
            'image': image,
            'mask': mask,
            'label': self.labels[idx],
            'idx': idx
        }
