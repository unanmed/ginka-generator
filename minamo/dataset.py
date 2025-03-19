import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from shared.graph import convert_soft_map_to_graph

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
        
        map1_probs = F.one_hot(torch.LongTensor(item['map1']), num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        map2_probs = F.one_hot(torch.LongTensor(item['map2']), num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        
        graph1 = convert_soft_map_to_graph(map1_probs)
        graph2 = convert_soft_map_to_graph(map2_probs)
        
        return (
            map1_probs,
            map2_probs,
            torch.FloatTensor([item['visionSimilarity']]),
            torch.FloatTensor([item['topoSimilarity']]),
            graph1,
            graph2
        )
