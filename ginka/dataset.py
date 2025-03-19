import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from minamo.model.model import MinamoModel
from shared.graph import convert_soft_map_to_graph

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class GinkaDataset(Dataset):
    def __init__(self, data_path: str, device, minamo: MinamoModel):
        self.data = load_data(data_path)  # 自定义数据加载函数
        self.max_size = 32
        self.minamo = minamo
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = F.one_hot(torch.LongTensor(item['map']), num_classes=32).permute(2, 0, 1).float().to(self.device)  # [32, H, W]
        graph = convert_soft_map_to_graph(target).to(self.device)
        vision_feat, topo_feat = self.minamo(target.unsqueeze(0), graph)
        
        return {
            "target_vision_feat": vision_feat,
            "target_topo_feat": topo_feat,
            "target": target
        }
        