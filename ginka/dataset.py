import json
import torch
from torch.utils.data import Dataset
from minamo.model.model import MinamoModel
from shared.graph import convert_map_to_graph

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class GinkaDataset(Dataset):
    def __init__(self, data_path: str, minamo: MinamoModel):
        self.data = load_data(data_path)  # 自定义数据加载函数
        self.max_size = 32
        self.minamo = minamo

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = torch.tensor(item["map"])
        graph = convert_map_to_graph(target)
        vision_feat, topo_feat = self.minamo(target, graph)
        feat_vec = torch.cat([vision_feat, topo_feat])
        
        return {
            "feat_vec": feat_vec,
            "target": target
        }
        