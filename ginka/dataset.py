import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    data_list = []
    for value in data["data"].values():
        data_list.append(value)

    return data_list

def compute_symmetry(target_np: np.ndarray) -> tuple:
    """从 numpy 地图矩阵中直接计算三种对称性，O(H*W)"""
    sym_h = bool(np.all(target_np == target_np[:, ::-1]))
    sym_v = bool(np.all(target_np == target_np[::-1, :]))
    sym_c = bool(np.all(target_np == target_np[::-1, ::-1]))
    return int(sym_h), int(sym_v), int(sym_c)

class GinkaSeperatedDataset(Dataset):
    FLOOR = 0
    WALL = 1
    DOOR = 2
    RESOURCE = 3
    MONSTER = 4
    ENTRANCE = 5
    MASK_ID = 6

    def __init__(
        self,
        data_path: str,
        subset_weights: tuple = (0.5, 0.3, 0.2)
    ):
        self.data = load_data(data_path)
        total = sum(subset_weights)
        self.subset_cumw = [sum(subset_weights[:i+1]) / total for i in range(len(subset_weights))]

        n = len(self.data)
        rs = sorted(item['roomCount'] for item in self.data)
        bs = sorted(item['highDegBranchCount'] for item in self.data)
        th1_r, th2_r = rs[n // 3], rs[2 * n // 3]
        th1_b, th2_b = bs[n // 3], bs[2 * n // 3]
        if th1_r == th2_r: th2_r = th1_r + 1
        if th1_b == th2_b: th2_b = th1_b + 1
        self.room_th = (th1_r, th2_r)
        self.branch_th = (th1_b, th2_b)

        for item in self.data:
            item['roomCountLevel'] = self.to_level(item['roomCount'], self.room_th)
            item['branchLevel'] = self.to_level(item['highDegBranchCount'], self.branch_th)

    def to_level(self, v, th):
        return 0 if v < th[0] else (1 if v < th[1] else 2)

    def __len__(self):
        return len(self.data)

    def degrade_tile(self, m: np.ndarray, tiles: list) -> np.ndarray:
        # 将指定 tile ID 替换为 floor(0)，原地修改
        for t in tiles:
            m[m == t] = self.FLOOR
        return m

    def std_mask(self) -> np.ndarray:
        # Beta(2,2) 采样掩码比例，50% 随机掩码 / 50% 分块掩码，返回 bool[13, 13]
        ratio = float(np.random.beta(2, 2)) * 0.95 + 0.05
        if random.random() < 0.5:
            idx = np.random.choice(169, int(169 * ratio), replace=False)
            mask = np.zeros(169, dtype=bool)
            mask[idx] = True
            return mask.reshape(13, 13)
        target = int(169 * ratio)
        mask = np.zeros((13, 13), dtype=bool)
        while mask.sum() < target:
            bh = np.random.randint(2, 7)
            bw = np.random.randint(2, 7)
            x = np.random.randint(0, 14 - bh)
            y = np.random.randint(0, 14 - bw)
            mask[x:x + bh, y:y + bw] = True
        return mask
            
    def create_degreaded(self, raw: np.ndarray):
        # 阶段一：生成墙壁和入口
        target1 = raw.copy()
        self.degrade_tile(target1, [self.DOOR, self.RESOURCE, self.MONSTER])
        inp1 = target1.copy()
        
        # 阶段二：生成怪物、门，同时也允许生成入口以适配结构
        target2 = raw.copy()
        self.degrade_tile(target2, [self.RESOURCE, self.WALL])
        inp2 = raw.copy()
        self.degrade_tile(inp2, [self.RESOURCE])
        
        # 阶段三：生成资源
        target3 = raw.copy()
        self.degrade_tile(target3, [self.WALL, self.DOOR, self.MONSTER, self.ENTRANCE])
        inp3 = raw.copy()
        
        return target1, inp1, target2, inp2, target3, inp3

    def apply_subset1(self, raw: np.ndarray):
        # 子集 1：std_mask 随机掩码
        
        target1, inp1, target2, inp2, target3, inp3 = self.create_degreaded(raw)
        
        enc1 = target1.copy()
        enc2 = inp2.copy()
        enc3 = raw.copy()

        # stage1：对整图 std_mask
        inp1[self.std_mask()] = self.MASK_ID

        # stage2：对 floor+功能元素区域 std_mask
        need_mask = np.isin(inp2, [self.FLOOR, self.DOOR, self.MONSTER, self.ENTRANCE])
        inp2[need_mask & self.std_mask()] = self.MASK_ID

        # stage3：对 floor+resource 区域 std_mask
        need_mask = np.isin(inp3, [self.FLOOR, self.RESOURCE])
        inp3[need_mask & self.std_mask()] = self.MASK_ID

        return inp1, target1, enc1, inp2, target2, enc2, inp3, target3, enc3

    def apply_subset2(self, raw: np.ndarray):
        # 子集 2：掩码所有内容，墙壁随机掩码，不掩码入口
        target1, inp1, target2, inp2, target3, inp3 = self.create_degreaded(raw)

        enc1 = target1.copy()
        enc2 = inp2.copy()
        enc3 = raw.copy()

        need_mask = np.isin(inp2, [self.FLOOR, self.WALL])
        inp1[need_mask & self.std_mask()] = self.MASK_ID
        need_mask = np.isin(inp2, [self.FLOOR, self.DOOR, self.MONSTER, self.ENTRANCE])
        inp2[need_mask] = self.MASK_ID
        need_mask = np.isin(inp3, [self.FLOOR, self.RESOURCE])
        inp3[need_mask] = self.MASK_ID

        return inp1, target1, enc1, inp2, target2, enc2, inp3, target3, enc3

    def apply_subset3(self, raw: np.ndarray):
        # 子集 3：在 2 的基础上掩码入口
        out = self.apply_subset2(raw)
        out[0][out[0] == self.ENTRANCE] = self.MASK_ID
        return out

    def __getitem__(self, idx):
        item = self.data[idx]
        map_np = np.array(item['map'], dtype=np.int64)

        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            map_np = np.rot90(map_np, k).copy()
        if np.random.rand() > 0.5:
            map_np = np.fliplr(map_np).copy()
        if np.random.rand() > 0.5:
            map_np = np.flipud(map_np).copy()

        r = random.random()
        if r < self.subset_cumw[0]:
            out = self.apply_subset1(map_np)
        elif r < self.subset_cumw[1]:
            out = self.apply_subset2(map_np)
        else:
            out = self.apply_subset3(map_np)

        sym_h, sym_v, sym_c = compute_symmetry(map_np)
        cond_sym = sym_h * 4 + sym_v * 2 + sym_c
        cond_room = item['roomCountLevel']
        cond_branch = item['branchLevel']
        cond_outer = item['outerWall']
        struct_inject = torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer])

        return {
            "input_stage1": torch.LongTensor(out[0]),
            "target_stage1": torch.LongTensor(out[1]),
            "encoder_stage1": torch.LongTensor(out[2]),
            "input_stage2": torch.LongTensor(out[3]),
            "target_stage2": torch.LongTensor(out[4]),
            "encoder_stage2": torch.LongTensor(out[5]),
            "input_stage3": torch.LongTensor(out[6]),
            "target_stage3": torch.LongTensor(out[7]),
            "encoder_stage3": torch.LongTensor(out[8]),
            "struct_inject": struct_inject
        }
