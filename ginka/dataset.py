import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from typing import List
from shared.utils import random_smooth_onehot

STAGE1_MASK = [0, 1, 10, 11]
STAGE1_REMOVE = [2, 3, 4, 5, 6, 7, 8, 9, 12]
STAGE2_MASK = [6, 7, 8, 9]
STAGE2_REMOVE = [2, 3, 4, 5, 12]
STAGE3_MASK = [2, 3, 4, 5, 12]
STAGE3_REMOVE = []

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

def apply_curriculum_mask(
    maps: torch.Tensor,                    # [B, C, H, W]
    mask_classes: List[int],               # 要遮挡的类别索引
    remove_classes: List[int],             # 要移除的类别索引
    mask_ratio: float                      # 遮挡比例 0~1
) -> torch.Tensor:
    C, H, W = maps.shape
    device = maps.device
    masked_maps = maps.clone()

    # Step 1: 移除不需要的类别（全设为 0 类）
    if remove_classes:
        remove_mask = masked_maps[remove_classes, :, :].sum(dim=0, keepdim=True) > 0
        masked_maps[:, :, :][remove_mask.expand(C, -1, -1)] = 0
        masked_maps[0][remove_mask[0, :, :]] = 1  # 设置为“空地”
        
    removed_maps = masked_maps.clone()

    # Step 2: 对指定类别随机遮挡
    for cls in mask_classes:
        cls_mask = masked_maps[:, cls] > 0  # 目标类别的像素布尔掩码 [H, W]
        indices = cls_mask.nonzero(as_tuple=False)  # 所有该类像素坐标
        num_mask = int(len(indices) * mask_ratio)
        if num_mask > 0:
            selected = indices[torch.randperm(len(indices))[:num_mask]]
            masked_maps[cls, selected[:, 0], selected[:, 1]] = 0
            masked_maps[0, selected[:, 0], selected[:, 1]] = 1  # 置为“空地”

    return removed_maps, masked_maps
        
class GinkaWGANDataset(Dataset):
    def __init__(self, data_path: str, device):
        self.data = load_data(data_path)  # 自定义数据加载函数
        self.device = device
        self.train_stage = 1
        self.mask_ratio1 = 0.1
        self.mask_ratio2 = 0.1
        self.mask_ratio3 = 0.1
        self.random_ratio = 0.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = F.one_hot(torch.LongTensor(item['map']), num_classes=32).permute(2, 0, 1).float()  # [32, H, W]

        if self.train_stage == 1:
            removed1, masked1 = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, self.mask_ratio1)
            removed2, masked2 = apply_curriculum_mask(target, STAGE2_MASK, STAGE2_REMOVE, self.mask_ratio2)
            removed3, masked3 = apply_curriculum_mask(target, STAGE3_MASK, STAGE3_REMOVE, self.mask_ratio3)
        elif self.train_stage == 2:
            removed1, masked1 = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, random.uniform(0.1, 0.9))
            removed2, masked2 = apply_curriculum_mask(target, STAGE2_MASK, STAGE2_REMOVE, random.uniform(0.1, 0.9))
            removed3, masked3 = apply_curriculum_mask(target, STAGE3_MASK, STAGE3_REMOVE, random.uniform(0.1, 0.9))
        
        if self.random_ratio > 0:
            removed1 = random_smooth_onehot(removed1, min_main=1 - self.random_ratio, max_main=1.0, epsilon=self.random_ratio)
            removed2 = random_smooth_onehot(removed2, min_main=1 - self.random_ratio, max_main=1.0, epsilon=self.random_ratio)
            removed3 = random_smooth_onehot(removed3, min_main=1 - self.random_ratio, max_main=1.0, epsilon=self.random_ratio)

        return removed1, masked1, removed2, masked2, removed3, masked3
        