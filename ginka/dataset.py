import json
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from typing import List

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list
        
class GinkaMaskGITDataset(Dataset):
    def __init__(self, data_path: str, device):
        self.data = load_data(data_path)  # 自定义数据加载函数
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = torch.LongTensor(item['map']) # [H, W]
        cond = torch.FloatTensor(item['val']) # [cond_dim]
        heatmap = torch.FloatTensor(item['heatmap']) # [heatmap_channel, H, W]

        return {
            "cond": cond,
            "target_map": target,
            "heatmap": heatmap
        }
        