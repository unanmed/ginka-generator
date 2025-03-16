import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

def convert_map_to_graph(map):
    rows = len(map)
    cols = len(map[0])
    node_indices = {}
    valid_nodes = []
    node_counter = 0

    for r in range(rows):
        for c in range(cols):
            if map[r][c] != 1:  # 排除墙体
                node_indices[(r, c)] = node_counter
                valid_nodes.append((r, c, map[r][c]))  # (行, 列, 地形类型)
                node_counter += 1

    edge_list = []
    for (r, c, _) in valid_nodes:
        node = node_indices[(r, c)]
        if c + 1 < cols and (r, c + 1) in node_indices:
            edge_list.append((node, node_indices[(r, c + 1)]))
        if r + 1 < rows and (r + 1, c) in node_indices:
            edge_list.append((node, node_indices[(r + 1, c)]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    node_features = torch.tensor([node_type for (_, _, node_type) in valid_nodes], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)

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
            torch.FloatTensor([item['topoSimilarity']]),
            convert_map_to_graph(item['map1']),
            convert_map_to_graph(item['map2'])
        )
