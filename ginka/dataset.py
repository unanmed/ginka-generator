import json
import random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    data_list = []
    for value in data["data"].values():
        data_list.append(value)
        
    return data_list

class GinkaMaskGITDataset(Dataset):
    def __init__(
        self, data_path: str, sigma_rand=0.1, blur_min=3, blur_max=6, 
        noise_prob=0.2, drop_prob=0.2, noise_sigma=0.1
    ):
        self.data = load_data(data_path)
        self.sigma_rand = sigma_rand
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.noise_prob = noise_prob
        self.drop_prob = drop_prob
        self.noise_sigma = noise_sigma
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        target_np = np.array(item['map'])
        heatmap = np.array(item['heatmap'], dtype=np.float32)
        
        # 数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            target_np = np.rot90(target_np, k)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.rot90(heatmap[i], k)

        if np.random.rand() > 0.5:
            target_np = np.fliplr(target_np)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.fliplr(heatmap[i])

        if np.random.rand() > 0.5:
            target_np = np.flipud(target_np)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.flipud(heatmap[i])
        
        target = torch.LongTensor(target_np.copy()) # [H, W]
        cond = torch.FloatTensor(item['val']) # [cond_dim]
        
        if random.random() < 0.5:
            size = random.randint(self.blur_min, self.blur_max)
            if size % 2 == 0:
                size = size + 1 if random.random() < 0.5 else size - 1
            heatmap = cv2.GaussianBlur(heatmap, (size, size), 0)
        else:
            sizeX = random.randint(self.blur_min, self.blur_max)
            sizeY = random.randint(self.blur_min, self.blur_max)
            if sizeX % 2 == 0:
                sizeX = sizeX + 1 if random.random() < 0.5 else sizeX - 1
            if sizeY % 2 == 0:
                sizeY = sizeY + 1 if random.random() < 0.5 else sizeY - 1
            heatmap = cv2.GaussianBlur(heatmap, (sizeX, sizeY), 0)
            
        heatmap = torch.FloatTensor(heatmap) # [heatmap_channel, H, W]
        
        for i in range(0, heatmap.shape[0]):
            if np.random.rand() < self.noise_prob:
                sigma = random.random() * self.noise_sigma
                heatmap[i] = heatmap[i] * sigma + torch.rand_like(heatmap[i]) * (1 - sigma)
            elif np.random.rand() < self.drop_prob:
                heatmap[i] = torch.zeros_like(heatmap[i])
        
        if random.random() < 0.5:
            sigma = random.random() * self.sigma_rand
            rand = torch.rand_like(heatmap)
            heatmap = heatmap * (1 - sigma) + rand * sigma

        return {
            "cond": cond,
            "target_map": target,
            "heatmap": heatmap
        }
        
class GinkaHeatmapDataset(Dataset):
    def __init__(self, data_path: str, min_mask=0, max_mask=0.8, blur_min=3, blur_max=6):
        self.data = load_data(data_path)
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.min_mask = min_mask
        self.max_mask = max_mask
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        heatmap = np.array(item['heatmap'], dtype=np.float32)
        
        # 数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.rot90(heatmap[i], k)

        if np.random.rand() > 0.5:
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.fliplr(heatmap[i])

        if np.random.rand() > 0.5:
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.flipud(heatmap[i])
                
        target = heatmap.copy()
        
        if random.random() < 0.5:
            size = random.randint(self.blur_min, self.blur_max)
            if size % 2 == 0:
                size = size + 1 if random.random() < 0.5 else size - 1
            target = cv2.GaussianBlur(target, (size, size), 0)
        else:
            sizeX = random.randint(self.blur_min, self.blur_max)
            sizeY = random.randint(self.blur_min, self.blur_max)
            if sizeX % 2 == 0:
                sizeX = sizeX + 1 if random.random() < 0.5 else sizeX - 1
            if sizeY % 2 == 0:
                sizeY = sizeY + 1 if random.random() < 0.5 else sizeY - 1
            target = cv2.GaussianBlur(target, (sizeX, sizeY), 0)
            
        target = torch.FloatTensor(target) # [heatmap_channel, H, W]
        cond = torch.FloatTensor(heatmap) # [heatmap_channel, H, W]
        C, H, W = target.shape
        
        for i in range(C):
            total = H * W
            ratio = np.random.random() * (self.max_mask - self.min_mask) + self.min_mask
            num = int(total * ratio)

            idx = np.random.choice(total, num, replace=False)

            mask = np.zeros(total, dtype=bool)
            mask[idx] = True
            mask = mask.reshape(H, W)
            cond[i, mask] = 0

        return {
            "target_heatmap": heatmap,
            "cond_heatmap": cond
        }


class GinkaJointDataset(Dataset):
    def __init__(self, data_path: str, min_mask=0, max_mask=0.8, blur_min=3, blur_max=6):
        self.data = load_data(data_path)
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.min_mask = min_mask
        self.max_mask = max_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_map = np.array(item['map'])
        heatmap = np.array(item['heatmap'], dtype=np.float32)

        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            target_map = np.rot90(target_map, k)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.rot90(heatmap[i], k)

        if np.random.rand() > 0.5:
            target_map = np.fliplr(target_map)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.fliplr(heatmap[i])

        if np.random.rand() > 0.5:
            target_map = np.flipud(target_map)
            for i in range(0, heatmap.shape[0]):
                heatmap[i] = np.flipud(heatmap[i])

        target_heatmap = heatmap.copy()

        if random.random() < 0.5:
            size = random.randint(self.blur_min, self.blur_max)
            if size % 2 == 0:
                size = size + 1 if random.random() < 0.5 else size - 1
            target_heatmap = cv2.GaussianBlur(target_heatmap, (size, size), 0)
        else:
            sizeX = random.randint(self.blur_min, self.blur_max)
            sizeY = random.randint(self.blur_min, self.blur_max)
            if sizeX % 2 == 0:
                sizeX = sizeX + 1 if random.random() < 0.5 else sizeX - 1
            if sizeY % 2 == 0:
                sizeY = sizeY + 1 if random.random() < 0.5 else sizeY - 1
            target_heatmap = cv2.GaussianBlur(target_heatmap, (sizeX, sizeY), 0)

        target_map = torch.LongTensor(target_map.copy())
        target_heatmap = torch.FloatTensor(target_heatmap)
        cond_heatmap = torch.FloatTensor(heatmap.copy())
        channels, height, width = cond_heatmap.shape

        for i in range(channels):
            total = height * width
            ratio = np.random.random() * (self.max_mask - self.min_mask) + self.min_mask
            num = int(total * ratio)

            masked_indices = np.random.choice(total, num, replace=False)

            mask = np.zeros(total, dtype=bool)
            mask[masked_indices] = True
            mask = mask.reshape(height, width)
            cond_heatmap[i, mask] = 0

        return {
            "target_map": target_map,
            "target_heatmap": target_heatmap,
            "cond_heatmap": cond_heatmap
        }


def _compute_symmetry(target_np: np.ndarray) -> tuple:
    """从 numpy 地图矩阵中直接计算三种对称性，O(H*W)"""
    sym_h = bool(np.all(target_np == target_np[:, ::-1]))
    sym_v = bool(np.all(target_np == target_np[::-1, :]))
    sym_c = bool(np.all(target_np == target_np[::-1, ::-1]))
    return int(sym_h), int(sym_v), int(sym_c)


class GinkaVQDataset(Dataset):
    """
    用于 VQ-VAE + MaskGIT 联合训练的多子集数据集。

    每次 __getitem__ 按权重随机选取以下四种子集之一：
      A (standard):     标准 MaskGIT 随机掩码，随机遮盖部分 tile
      B (wall-only):    仅保留 wall(1) + floor(0)，其余全部替换为 MASK(15)
      C (wall-random):  在 B 基础上，再随机 mask 部分 wall tile
      D (wall+entry):   仅保留 wall(1) + floor(0) + entrance(10)，其余全部替换为 MASK(15)

    返回 dict:
      raw_map:    LongTensor [H*W]  完整原始地图（供 VQ-VAE 编码）
      masked_map: LongTensor [H*W]  MaskGIT 输入（被 mask 的位置 = 15）
      target_map: LongTensor [H*W]  CE loss ground truth（等同 raw_map）
      subset:     str               子集标识，供调试/统计用
    """

    FLOOR    = 0
    WALL     = 1
    ENTRANCE = 10
    MASK_ID  = 15

    def __init__(
        self,
        data_path: str,
        subset_weights: tuple = (0.5, 0.2, 0.2, 0.1),
        wall_mask_ratio: float = 0.3,
        room_thresholds: tuple = None,
        branch_thresholds: tuple = None,
    ):
        """
        Args:
            data_path:          JSON 数据文件路径
            subset_weights:     子集 (A, B, C, D) 的采样权重，自动归一化
            wall_mask_ratio:    Subset C 中额外随机 mask 的 wall tile 比例上限
                                （每次从 [0, wall_mask_ratio] 均匀采样实际比例）
            room_thresholds:    (th1, th2) 房间数量等频分箱阈值；为 None 时自动从当前数据计算（训练集）
            branch_thresholds:  (th1, th2) 分支数量等频分箱阈值；为 None 时自动从当前数据计算（训练集）
        """
        self.data = load_data(data_path)
        self.wall_mask_ratio = wall_mask_ratio

        # 累积权重，用于快速随机子集选择
        total_w = sum(subset_weights)
        normalized = [x / total_w for x in subset_weights]
        self.subset_cumw = [sum(normalized[:i + 1]) for i in range(len(normalized))]

        # ── 两趟扫描：计算等频分箱阈值 ──────────────────────────────
        room_counts   = [item['roomCount']           for item in self.data]
        branch_counts = [item['highDegBranchCount']  for item in self.data]

        if room_thresholds is None:
            n  = len(room_counts)
            rs = sorted(room_counts)
            bs = sorted(branch_counts)
            th1_r, th2_r = rs[n // 3], rs[2 * n // 3]
            th1_b, th2_b = bs[n // 3], bs[2 * n // 3]
            # 防止 Medium 等级为空
            if th1_r == th2_r:
                th2_r = th1_r + 1
            if th1_b == th2_b:
                th2_b = th1_b + 1
            self.room_th   = (th1_r, th2_r)
            self.branch_th = (th1_b, th2_b)
        else:
            self.room_th   = room_thresholds
            self.branch_th = branch_thresholds

        def to_level(v: int, th: tuple) -> int:
            return 0 if v < th[0] else (1 if v < th[1] else 2)

        # 回填等级字段
        for item in self.data:
            item['roomCountLevel'] = to_level(item['roomCount'],           self.room_th)
            item['branchLevel']    = to_level(item['highDegBranchCount'],  self.branch_th)

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    # 内联随机掩码生成（避免 scipy 的 NumPy 版本兼容问题）
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_mask_ratio(min_r=0.05, max_r=1.0) -> float:
        """用 Beta(2,2) 分布采样掩码比例，集中在中间值。"""
        r = np.random.beta(2, 2)
        return min_r + (max_r - min_r) * r

    @staticmethod
    def _random_mask(h: int, w: int) -> np.ndarray:
        """纯随机掩码，返回 [H*W] bool。"""
        ratio = GinkaVQDataset._sample_mask_ratio()
        total = h * w
        idx = np.random.choice(total, int(total * ratio), replace=False)
        mask = np.zeros(total, dtype=bool)
        mask[idx] = True
        return mask

    @staticmethod
    def _block_mask(h: int, w: int) -> np.ndarray:
        """矩形分块随机掩码，返回 [H*W] bool。"""
        ratio = GinkaVQDataset._sample_mask_ratio()
        max_block = max(2, min(h, w) // 2)
        target = int(h * w * ratio)
        mask = np.zeros((h, w), dtype=bool)
        while mask.sum() < target:
            bh = np.random.randint(2, max_block + 1)
            bw = np.random.randint(2, max_block + 1)
            x  = np.random.randint(0, max(1, h - bh + 1))
            y  = np.random.randint(0, max(1, w - bw + 1))
            mask[x:x + bh, y:y + bw] = True
        return mask.reshape(-1)

    def _std_mask(self, h: int, w: int) -> np.ndarray:
        """标准 MaskGIT 掩码：随机选择纯随机或分块策略。"""
        if random.random() < 0.5:
            return self._random_mask(h, w)
        else:
            return self._block_mask(h, w)

    # ------------------------------------------------------------------

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """随机旋转 / 翻转数据增强，返回新 array。"""
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k).copy()
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() > 0.5:
            arr = np.flipud(arr).copy()
        return arr

    def _choose_subset(self) -> str:
        r = random.random()
        if r < self.subset_cumw[0]:
            return 'A'
        elif r < self.subset_cumw[1]:
            return 'B'
        elif r < self.subset_cumw[2]:
            return 'C'
        else:
            return 'D'

    def _apply_subset(self, raw: np.ndarray, subset: str) -> np.ndarray:
        """
        根据子集类型生成 masked_map。

        Args:
            raw:    [H, W] int64 完整原始地图
            subset: 'A' | 'B' | 'C' | 'D'

        Returns:
            [H*W] int64，被遮盖位置值为 MASK_ID(15)
        """
        H, W = raw.shape

        if subset == 'A':
            # 标准随机 mask：纯随机或分块策略
            mask = self._std_mask(H, W)              # [H*W] bool
            flat = raw.reshape(-1).copy()
            flat[mask] = self.MASK_ID
            return flat

        elif subset == 'B':
            # 仅保留 wall(1)，floor(0) 和其他非墙内容全部 mask
            flat = raw.reshape(-1).copy()
            keep = (flat == self.WALL)
            flat[~keep] = self.MASK_ID
            return flat

        elif subset == 'C':
            # Subset B + 随机 mask 部分 wall
            flat = raw.reshape(-1).copy()
            keep = (flat == self.WALL)
            flat[~keep] = self.MASK_ID

            wall_idx = np.where(flat == self.WALL)[0]
            if len(wall_idx) > 0:
                ratio = random.random() * self.wall_mask_ratio
                n = max(1, int(len(wall_idx) * ratio))
                chosen = np.random.choice(wall_idx, n, replace=False)
                flat[chosen] = self.MASK_ID
            return flat

        else:  # D
            # 仅保留 wall(1) 和 entrance(10)，floor(0) 和其他非墙内容全部 mask
            flat = raw.reshape(-1).copy()
            keep = (flat == self.WALL) | (flat == self.ENTRANCE)
            flat[~keep] = self.MASK_ID

            # 随机 mask 部分 wall（模拟真实场景，与子集 C 一致）
            wall_idx = np.where(flat == self.WALL)[0]
            if len(wall_idx) > 0:
                ratio = random.random() * self.wall_mask_ratio
                n = max(1, int(len(wall_idx) * ratio))
                chosen = np.random.choice(wall_idx, n, replace=False)
                flat[chosen] = self.MASK_ID
            return flat

    def __getitem__(self, idx):
        item = self.data[idx]

        raw_np = self._augment(np.array(item['map'], dtype=np.int64))  # [H, W]
        subset = self._choose_subset()
        masked_np = self._apply_subset(raw_np, subset)                 # [H*W]
        raw_flat  = raw_np.reshape(-1)                                 # [H*W]

        # 对称性：在增强后重新计算
        sym_h, sym_v, sym_c = _compute_symmetry(raw_np)
        cond_sym = sym_h * 4 + sym_v * 2 + sym_c  # [0, 7]

        # 其余结构标签：增强不改变拓扑结构，直接读取
        cond_room   = item['roomCountLevel']   # 0/1/2
        cond_branch = item['branchLevel']      # 0/1/2
        cond_outer  = item['outerWall']        # 0/1

        struct_cond = torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer])

        raw_t = torch.LongTensor(raw_flat)
        return {
            "raw_map":     raw_t,                              # VQ-VAE 编码器输入
            "slice1":      make_slice(raw_t, {0, 1}),          # 通道 1：floor+wall
            "slice2":      make_slice(raw_t, {0, 1, 2, 9, 10}),# 通道 2：floor+wall+门+怪+入口
            "slice3":      raw_t.clone(),                      # 通道 3：完整地图
            "masked_map":  torch.LongTensor(masked_np),        # MaskGIT 输入
            "target_map":  torch.LongTensor(raw_flat.copy()),  # CE loss ground truth
            "subset":      subset,                             # 调试/统计用
            "struct_cond": struct_cond,                        # [4]，供模型 Embedding 查表
        }


# ---------------------------------------------------------------------------
# make_slice：按保留集合切割地图，其余位置替换为 floor(0)
# ---------------------------------------------------------------------------

def make_slice(map_flat: torch.Tensor, keep_set: set) -> torch.Tensor:
    """
    从完整地图中只保留 keep_set 中的 tile 类型，其余位置替换为 floor(0)。

    Args:
        map_flat: LongTensor [H*W]  完整地图 tile ID 序列
        keep_set: set of int        需要保留的 tile 类型集合

    Returns:
        LongTensor [H*W]  切片后的地图（非保留 tile 位置值为 0）
    """
    out = map_flat.clone()
    mask = torch.zeros_like(out, dtype=torch.bool)
    for t in keep_set:
        mask |= (out == t)
    out[~mask] = 0
    return out


# ---------------------------------------------------------------------------
# GinkaSplitDataset：三通道分拆预训练专用数据集
# ---------------------------------------------------------------------------

class GinkaSplitDataset(Dataset):
    """
    三通道分拆预训练（方案 B）专用数据集。

    每个样本只提供完整地图及其三路切片，不做 MaskGIT 掩码处理。
    切片按累积式设计：
      slice1 = floor(0) + wall(1)
      slice2 = floor(0) + wall(1) + door(2) + mob(9) + entrance(10)
      slice3 = 完整地图（所有 tile）

    返回 dict:
      raw_map: LongTensor [H*W]  完整原始地图
      slice1:  LongTensor [H*W]  通道 1 切片（floor+wall）
      slice2:  LongTensor [H*W]  通道 2 切片（floor+wall+门+怪+入口）
      slice3:  LongTensor [H*W]  通道 3 切片（完整地图）
    """

    def __init__(self, data_path: str):
        self.data = load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        arr = np.array(item['map'], dtype=np.int64)  # [H, W]

        # 随机旋转 / 翻转数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k).copy()
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() > 0.5:
            arr = np.flipud(arr).copy()

        raw = torch.LongTensor(arr.reshape(-1))   # [H*W]
        return {
            "raw_map": raw,
            "slice1":  make_slice(raw, {0, 1}),
            "slice2":  make_slice(raw, {0, 1, 2, 9, 10}),
            "slice3":  raw.clone(),
        }


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ginka-dataset.json')
    ds = GinkaVQDataset(data_path)
    print(f"数据集大小: {len(ds)}")

    subset_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for i in range(200):
        sample = ds[i % len(ds)]
        subset_count[sample['subset']] += 1

    raw    = sample['raw_map']
    masked = sample['masked_map']
    target = sample['target_map']
    print(f"raw_map    shape={raw.shape}, dtype={raw.dtype}")
    print(f"masked_map shape={masked.shape}, dtype={masked.dtype}")
    print(f"target_map shape={target.shape}, dtype={target.dtype}")
    print(f"被 mask 的位置数: {(masked == 15).sum().item()} / {masked.numel()}")
    print(f"\n200 次采样子集分布: {subset_count}")
