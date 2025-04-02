import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from minamo.model.model import MinamoModel
from shared.graph import differentiable_convert_to_data
from shared.utils import random_smooth_onehot

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

def load_minamo_gan_data(data: list):
    res = list()
    for one in data:
        res.append((one['map1'], one['map2'], one['visionSimilarity'], one['topoSimilarity'], True))
    return res

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
        
        target = F.one_hot(torch.LongTensor(item['map']), num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        target_smooth = random_smooth_onehot(target)
        graph = differentiable_convert_to_data(target_smooth).to(self.device)
        target = target.to(self.device)
        vision_feat, topo_feat = self.minamo(target.unsqueeze(0), graph)
        
        return {
            "target_vision_feat": vision_feat,
            "target_topo_feat": topo_feat,
            "target": target,
        }
        
class MinamoGANDataset(Dataset):
    def __init__(self, refer_data_path):
        self.refer = load_minamo_gan_data(load_data(refer_data_path))
        self.data = list()
        self.data.extend(random.sample(self.refer, 1000))
        
    def set_data(self, data: list):
        self.data.clear()
        self.data.extend(data)
        k = min(len(data) / 4, len(self.refer))
        self.data.extend(random.sample(self.refer, int(k)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        map1, map2, vis_sim, topo_sim, review = item
        map1 = torch.LongTensor(map1)
        map2 = torch.LongTensor(map2)
        # 检查是否有 review 标签，没有的话说明是概率分布，不需要任何转换
        if review:
            map1 = F.one_hot(map1, num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        map2 = F.one_hot(map2, num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        
        min_main = random.uniform(0.75, 0.9)
        max_main = random.uniform(0.9, 1)
        epsilon = random.uniform(0, 0.25)
        
        if review:
            map1 = random_smooth_onehot(map1, min_main, max_main, epsilon)
        map2 = random_smooth_onehot(map2, min_main, max_main, epsilon)
        
        graph1 = differentiable_convert_to_data(map1)
        graph2 = differentiable_convert_to_data(map2)
        
        return (
            map1,
            map2,
            torch.FloatTensor([vis_sim]),
            torch.FloatTensor([topo_sim]),
            graph1,
            graph2
        )