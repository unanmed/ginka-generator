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
    def __init__(self, data_path: str, sigma_rand=0.1, blur_min=3, blur_max=6):
        self.data = load_data(data_path)
        self.sigma_rand = sigma_rand
        self.blur_min = blur_min
        self.blur_max = blur_max
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = torch.LongTensor(item['map']) # [H, W]
        cond = torch.FloatTensor(item['val']) # [cond_dim]
        heatmap = np.array(item['heatmap'], dtype=np.float32)
        
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
        
        if random.random() < 0.5:
            sigma = random.random() * self.sigma_rand
            rand = torch.randn_like(heatmap) * sigma
            heatmap = heatmap + rand

        return {
            "cond": cond,
            "target_map": target,
            "heatmap": heatmap
        }
        