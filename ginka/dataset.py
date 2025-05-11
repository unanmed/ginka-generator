import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from typing import List

STAGE1_MASK = [0, 1, 2, 29, 30]
STAGE1_REMOVE = list(range(3, 29))
STAGE2_MASK = [3, 4, 5, 6, 26, 27, 28]
STAGE2_REMOVE = list(range(7, 26))
STAGE3_MASK = list(range(7, 26))
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

def apply_curriculum_remove(
    maps: torch.Tensor,
    remove_classes: List[int],             # 要移除的类别索引
):
    C, H, W = maps.shape
    device = maps.device
    removed_maps = maps.clone()

    remove_mask = removed_maps[remove_classes, :, :].sum(dim=0, keepdim=True) > 0
    removed_maps[:, :, :][remove_mask.expand(C, -1, -1)] = 0
    removed_maps[0][remove_mask[0, :, :]] = 1  # 设置为“空地”

    return removed_maps.to(device)

def apply_curriculum_mask(
    maps: torch.Tensor,                    # [C, H, W]
    mask_classes: List[int],               # 要遮挡的类别索引
    remove_classes: List[int],             # 要移除的类别索引
    mask_ratio: float                      # 遮挡比例 0~1
) -> torch.Tensor:
    C, H, W = maps.shape
    masked_maps = maps.clone()

    # Step 1: 移除不需要的类别（全设为 0 类）
    if remove_classes:
        remove_mask = masked_maps[remove_classes, :, :].sum(dim=0, keepdim=True) > 0
        masked_maps[:, :, :][remove_mask.expand(C, -1, -1)] = 0
        masked_maps[0][remove_mask[0, :, :]] = 1  # 设置为“空地”
        
    removed_maps = masked_maps.clone()

    # Step 2: 对指定类别随机遮挡
    for cls in mask_classes:
        cls_mask = masked_maps[cls] > 0  # 目标类别的像素布尔掩码 [H, W]
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

    def __len__(self):
        return len(self.data)
    
    def handle_stage1(self, target, tag_cond, val_cond):
        # 课程学习第一阶段，蒙版填充
        removed1, masked1 = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, self.mask_ratio1)
        removed2, masked2 = apply_curriculum_mask(target, STAGE2_MASK, STAGE2_REMOVE, self.mask_ratio2)
        removed3, masked3 = apply_curriculum_mask(target, STAGE3_MASK, STAGE3_REMOVE, self.mask_ratio3)
        
        return {
            "real1": removed1,
            "masked1": masked1,
            "real2": removed2,
            "masked2": masked2,
            "real3": removed3,
            "masked3": masked3,
            "tag_cond": tag_cond,
            "val_cond": val_cond
        }
    
    def handle_stage2(self, target, tag_cond, val_cond):
        # 课程学习第二阶段，完全随机蒙版
        removed1, masked1 = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, random.uniform(0.1, 0.9))
        # 后面两个阶段由于会保留一些类别，所以完全随机遮挡即可
        removed2, masked2 = apply_curriculum_mask(target, STAGE2_MASK, STAGE2_REMOVE, random.uniform(0.1, 1))
        removed3, masked3 = apply_curriculum_mask(target, STAGE3_MASK, STAGE3_REMOVE, random.uniform(0.1, 1))
            
        return {
            "real1": removed1,
            "masked1": masked1,
            "real2": removed2,
            "masked2": masked2,
            "real3": removed3,
            "masked3": masked3,
            "tag_cond": tag_cond,
            "val_cond": val_cond
        }
    
    def handle_stage3(self, target, tag_cond, val_cond):
        # 第三阶段，联合生成，输入随机蒙版
        removed1, masked1 = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, random.uniform(0.1, 0.9))
        removed2 = apply_curriculum_remove(target, STAGE2_REMOVE)
        removed3 = apply_curriculum_remove(target, STAGE3_REMOVE)
        
        return {
            "real1": removed1,
            "masked1": masked1,
            "real2": removed2,
            "masked2": torch.zeros_like(target),
            "real3": removed3,
            "masked3": torch.zeros_like(target),
            "tag_cond": tag_cond,
            "val_cond": val_cond
        }

    def handle_stage4(self, target, tag_cond, val_cond):
        # 第四阶段，完全随机输入
        removed1 = apply_curriculum_remove(target, STAGE1_REMOVE)
        removed2 = apply_curriculum_remove(target, STAGE2_REMOVE)
        removed3 = apply_curriculum_remove(target, STAGE3_REMOVE)
        _, masked = apply_curriculum_mask(target, STAGE1_MASK, STAGE1_REMOVE, 0.5)
        rand = torch.rand(32, 32, 32, device=target.device)
        
        return {
            "real1": removed1,
            "masked1": rand,
            "real2": removed2,
            "masked2": masked,
            "real3": removed3,
            "masked3": torch.zeros_like(target),
            "tag_cond": tag_cond,
            "val_cond": val_cond
        }

    def __getitem__(self, idx):
        item = self.data[idx]
        
        target = F.one_hot(torch.LongTensor(item['map']), num_classes=32).permute(2, 0, 1).float()  # [32, H, W]
        C, H, W = target.shape
        tag_cond = torch.FloatTensor(item['tag'])
        val_cond = torch.FloatTensor(item['val'])
        val_cond[9] = val_cond[9] / H / W
        val_cond[10] = val_cond[10] / H / W

        if self.train_stage == 1:
            return self.handle_stage1(target, tag_cond, val_cond)
            
        elif self.train_stage == 2:
            return self.handle_stage2(target, tag_cond, val_cond)
        
        elif self.train_stage == 3:
            return self.handle_stage3(target, tag_cond, val_cond)
        
        elif self.train_stage == 4:
            return self.handle_stage4(target, tag_cond, val_cond)

        raise RuntimeError(f"Invalid train stage: {self.train_stage}")
        