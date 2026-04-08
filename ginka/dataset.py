import json
import random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class GinkaMaskGITDataset(Dataset):
    def __init__(
        self, data_path: str, sigma_rand=0.1, blur_min=3, blur_max=6, 
        noise_prob=0.2, drop_prob=0.2
    ):
        self.data = load_data(data_path)
        self.sigma_rand = sigma_rand
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.noise_prob = noise_prob
        self.drop_prob = drop_prob
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        target_np = np.array(item['map'])
        heatmap = np.array(item['heatmap'], dtype=np.float32)
        
        # 数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            target_np = np.rot90(target_np, k)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.rot90(heatmap[i], k)

        if np.random.rand() > 0.5:
            target_np = np.fliplr(target_np)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.fliplr(heatmap[i])

        if np.random.rand() > 0.5:
            target_np = np.flipud(target_np)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.flipud(heatmap[i])
                
        
        
        target = torch.LongTensor(target_np.copy()) # [H, W]
        cond = torch.FloatTensor(item['val']) # [cond_dim]
        
        if random.random() < 0.5:
            size = random.randint(self.blur_min, self.blur_max)
            if size % 2 == 0:
                size = size + 1 if random.random() < 0.5 else size - 1
            heatmap = cv2.GaussianBlur(heatmap, (size, size), 0)
        else:
            sizeX = random.randint(self.blur_min, self.blur_max)
            sizeY = random.randint(self.blur_min, self.blur_max)
            if sizeX % 2 == 0:
                sizeX = sizeX + 1 if random.random() < 0.5 else sizeX - 1
            if sizeY % 2 == 0:
                sizeY = sizeY + 1 if random.random() < 0.5 else sizeY - 1
            heatmap = cv2.GaussianBlur(heatmap, (sizeX, sizeY), 0)
            
        heatmap = torch.FloatTensor(heatmap) # [heatmap_channel, H, W]
        
        for i in range(0, heatmap.shape[0]):
            if np.random.rand() < self.noise_prob:
                sigma = random.random() * self.sigma_rand
                heatmap[i] = heatmap * sigma + torch.rand_like(heatmap[i]) * (1 - sigma)
            elif np.random.rand() < self.drop_prob:
                heatmap[i] = torch.zeros_like(heatmap[i])
        
        if random.random() < 0.5:
            sigma = random.random() * self.sigma_rand
            rand = torch.rand_like(heatmap)
            heatmap = heatmap * (1 - sigma) + rand * sigma

        return {
            "cond": cond,
            "target_map": target,
            "heatmap": heatmap
        }
        
class GinkaHeatmapDataset(Dataset):
    def __init__(self, data_path: str, min_mask=0, max_mask=0.8, blur_min=3, blur_max=6):
        self.data = load_data(data_path)
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.min_mask = min_mask
        self.max_mask = max_mask
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        heatmap = np.array(item['heatmap'], dtype=np.float32)
        
        # 数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.rot90(heatmap[i], k)

        if np.random.rand() > 0.5:
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.fliplr(heatmap[i])

        if np.random.rand() > 0.5:
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.flipud(heatmap[i])
                
        target = heatmap.copy()
        
        if random.random() < 0.5:
            size = random.randint(self.blur_min, self.blur_max)
            if size % 2 == 0:
                size = size + 1 if random.random() < 0.5 else size - 1
            target = cv2.GaussianBlur(target, (size, size), 0)
        else:
            sizeX = random.randint(self.blur_min, self.blur_max)
            sizeY = random.randint(self.blur_min, self.blur_max)
            if sizeX % 2 == 0:
                sizeX = sizeX + 1 if random.random() < 0.5 else sizeX - 1
            if sizeY % 2 == 0:
                sizeY = sizeY + 1 if random.random() < 0.5 else sizeY - 1
            target = cv2.GaussianBlur(target, (sizeX, sizeY), 0)
            
        target = torch.FloatTensor(target) # [heatmap_channel, H, W]
        cond = torch.FloatTensor(heatmap) # [heatmap_channel, H, W]
        C, H, W = target.shape
        
        for i in range(C):
            total = H * W
            ratio = np.random.random() * (self.max_mask - self.min_mask) + self.min_mask
            num = int(total * ratio)

            idx = np.random.choice(total, num, replace=False)

            mask = np.zeros(total, dtype=bool)
            mask[idx] = True
            mask = mask.reshape(H, W)
            cond[i, mask] = 0

        return {
            "target_heatmap": heatmap,
            "cond_heatmap": cond
        }