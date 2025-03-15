import json
import torch
from torch.utils.data import Dataset

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class MinamoDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = load_data(data_path)  # 自定义数据加载函数
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.LongTensor(item['map1']),
            torch.LongTensor(item['map2']),
            torch.FloatTensor([item['visionSimilarity']]),
            torch.FloatTensor([item['topoSimilarity']])
        )
