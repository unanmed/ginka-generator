import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset

def _compute_map_labels(map_2d) -> dict:
    """
    从 2D 地图列表（或 numpy 数组）推算结构标签。
    当 JSON 数据缺少 roomCount / highDegBranchCount / outerWall 字段时调用。
    """
    arr = np.array(map_2d, dtype=np.int64)  # [H, W]
    H, W = arr.shape
    WALL, ENTRY = 1, 5

    # outerWall：最外圈中 wall+entry 占比 > 90%
    border = np.concatenate([arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]])
    total_b = border.size
    outer_wall = int(total_b > 0 and np.sum((border == WALL) | (border == ENTRY)) / total_b > 0.9)

    # roomCount：BFS 统计 floor(0)+resource(3) 连通区域，
    #            需满足：总格子 >= 4，外接矩形宽 >= 2 且高 >= 2
    FLOOR_SET = (0, 3)
    visited = np.zeros((H, W), dtype=bool)
    room_count = 0
    for sy in range(H):
        for sx in range(W):
            if arr[sy, sx] not in FLOOR_SET or visited[sy, sx]:
                continue
            queue = [(sy, sx)]
            visited[sy, sx] = True
            tiles_y, tiles_x = [sy], [sx]
            head = 0
            while head < len(queue):
                y, x = queue[head]; head += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and arr[ny, nx] in FLOOR_SET:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                        tiles_y.append(ny); tiles_x.append(nx)
            if (len(tiles_y) >= 4
                    and max(tiles_y) - min(tiles_y) >= 1
                    and max(tiles_x) - min(tiles_x) >= 1):
                room_count += 1

    # highDegBranchCount：非 wall 格子中，4 邻域非 wall 邻居 >= 3 的数量
    non_wall = (arr != WALL).astype(np.int32)
    padded = np.pad(non_wall, 1, mode='constant', constant_values=0)
    nbr_sum = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
               padded[1:-1, :-2] + padded[1:-1, 2:])
    high_deg = int(np.sum((non_wall == 1) & (nbr_sum >= 3)))

    return {'outerWall': outer_wall, 'roomCount': room_count, 'highDegBranchCount': high_deg}


def load_data(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    data_list = []
    for value in data["data"].values():
        # 兼容旧版数据集（缺少结构标签字段）
        if 'roomCount' not in value:
            labels = _compute_map_labels(value['map'])
            value.update(labels)
        # symmetry 字段由 __getitem__ 在增强后重新计算，此处不需要从 JSON 读取
        data_list.append(value)

    return data_list

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
      B (wall-only):    仅保留 wall(1) + floor(0)，其余全部替换为 MASK(6)
      C (wall-random):  在 B 基础上，再随机 mask 部分 wall tile
      D (wall+entry):   仅保留 wall(1) + floor(0) + entrance(5)，其余全部替换为 MASK(6)

    返回 dict:
      raw_map:    LongTensor [H*W]  完整原始地图（供 VQ-VAE 编码）
      masked_map: LongTensor [H*W]  MaskGIT 输入（被 mask 的位置 = 6）
      target_map: LongTensor [H*W]  CE loss ground truth（等同 raw_map）
      subset:     str               子集标识，供调试/统计用
    """

    FLOOR    = 0
    WALL     = 1
    ENTRANCE = 5
    MASK_ID  = 6

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
            [H*W] int64，被遮盖位置値为 MASK_ID(6)
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
            # 仅保留 wall(1) 和 entrance(5)，floor(0) 和其他非墙内容全部 mask
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
            "slice2":      make_slice(raw_t, {0, 1, 2, 4, 5}), # 通道 2：floor+wall+门+怪+入口
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
      slice2 = floor(0) + wall(1) + door(2) + mob(4) + entrance(5)
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
            "slice2":  make_slice(raw, {0, 1, 2, 4, 5}),
            "slice3":  raw.clone(),
        }


# ---------------------------------------------------------------------------
# GinkaStageDataset：三阶段级联训练专用数据集
# ---------------------------------------------------------------------------

class GinkaStageDataset(Dataset):
    """
    三阶段级联生成训练专用 Dataset。

    每个阶段只预测特定类别的 tile，后续阶段以前序阶段输出作为上下文。
    训练时统一使用 GT 作为前序上下文（teacher forcing），避免误差级联。

    阶段划分：
      stage=1  结构骨架：预测 floor(0) + wall(1)
      stage=2  功能元素：预测 door(2) + monster(4) + entrance(5)，以 floor/wall 为上下文
      stage=3  资源放置：预测 resource(3)，以完整骨架为上下文

    返回 dict:
      raw_map:      LongTensor [H*W]  完整原始地图（供 VQ-VAE 编码）
      vq_slice:     LongTensor [H*W]  当前阶段 VQ 编码器的输入切片
      stage_input:  LongTensor [H*W]  MaskGIT 输入（含上下文 + MASK 位置）
      target_map:   LongTensor [H*W]  CE loss ground truth
      loss_mask:    BoolTensor [H*W]  只对 True 位置计算损失
      subset:       str               子集标识 A/B/C/D
      struct_cond:  LongTensor [4]    [sym, room, branch, outer]
    """

    FLOOR    = 0
    WALL     = 1
    DOOR     = 2
    RESOURCE = 3
    MONSTER  = 4
    ENTRANCE = 5
    MASK_ID  = 6

    STAGE1_TARGETS = frozenset({0, 1})
    STAGE2_TARGETS = frozenset({2, 4, 5})
    STAGE3_TARGETS = frozenset({3})

    # VQ 切片集合：各阶段编码器只"看"与自身相关的 tile
    _VQ_KEEP = {
        1: frozenset({0, 1}),
        2: frozenset({0, 1, 2, 4, 5}),
        3: None,  # 完整地图
    }

    def __init__(
        self,
        data_path: str,
        stage: int,
        subset_weights: tuple = (0.5, 0.2, 0.2, 0.1),
        wall_mask_ratio: float = 0.3,
        room_thresholds: tuple = None,
        branch_thresholds: tuple = None,
    ):
        """
        Args:
            data_path:         JSON 数据文件路径
            stage:             生成阶段 1/2/3
            subset_weights:    子集 (A, B, C, D) 的采样权重，自动归一化
            wall_mask_ratio:   Subset C 中额外随机 mask 的 wall 比例上限
            room_thresholds:   等频分箱阈值（None 时自动计算）
            branch_thresholds: 等频分箱阈值（None 时自动计算）
        """
        assert stage in (1, 2, 3), f"stage 必须是 1/2/3，收到 {stage}"
        self.stage = stage
        self.data = load_data(data_path)
        self.wall_mask_ratio = wall_mask_ratio

        total_w = sum(subset_weights)
        normalized = [x / total_w for x in subset_weights]
        self.subset_cumw = [sum(normalized[:i + 1]) for i in range(len(normalized))]

        room_counts   = [item['roomCount']          for item in self.data]
        branch_counts = [item['highDegBranchCount'] for item in self.data]

        if room_thresholds is None:
            n  = len(room_counts)
            rs = sorted(room_counts)
            bs = sorted(branch_counts)
            th1_r, th2_r = rs[n // 3], rs[2 * n // 3]
            th1_b, th2_b = bs[n // 3], bs[2 * n // 3]
            if th1_r == th2_r: th2_r = th1_r + 1
            if th1_b == th2_b: th2_b = th1_b + 1
            self.room_th   = (th1_r, th2_r)
            self.branch_th = (th1_b, th2_b)
        else:
            self.room_th   = room_thresholds
            self.branch_th = branch_thresholds

        def to_level(v, th):
            return 0 if v < th[0] else (1 if v < th[1] else 2)

        for item in self.data:
            item['roomCountLevel'] = to_level(item['roomCount'],          self.room_th)
            item['branchLevel']    = to_level(item['highDegBranchCount'], self.branch_th)

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    # 掩码辅助（与 GinkaVQDataset 相同逻辑）
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_mask_ratio(min_r=0.05, max_r=1.0) -> float:
        r = np.random.beta(2, 2)
        return min_r + (max_r - min_r) * r

    @staticmethod
    def _random_mask(h: int, w: int) -> np.ndarray:
        ratio = GinkaStageDataset._sample_mask_ratio()
        total = h * w
        idx   = np.random.choice(total, int(total * ratio), replace=False)
        mask  = np.zeros(total, dtype=bool)
        mask[idx] = True
        return mask

    @staticmethod
    def _block_mask(h: int, w: int) -> np.ndarray:
        ratio     = GinkaStageDataset._sample_mask_ratio()
        max_block = max(2, min(h, w) // 2)
        target    = int(h * w * ratio)
        mask      = np.zeros((h, w), dtype=bool)
        while mask.sum() < target:
            bh = np.random.randint(2, max_block + 1)
            bw = np.random.randint(2, max_block + 1)
            x  = np.random.randint(0, max(1, h - bh + 1))
            y  = np.random.randint(0, max(1, w - bw + 1))
            mask[x:x + bh, y:y + bw] = True
        return mask.reshape(-1)

    def _std_mask(self, h: int, w: int) -> np.ndarray:
        return self._random_mask(h, w) if random.random() < 0.5 else self._block_mask(h, w)

    # ------------------------------------------------------------------
    # 子集选择
    # ------------------------------------------------------------------
    def _choose_subset(self) -> str:
        r = random.random()
        if r < self.subset_cumw[0]: return 'A'
        if r < self.subset_cumw[1]: return 'B'
        if r < self.subset_cumw[2]: return 'C'
        return 'D'

    # ------------------------------------------------------------------
    # 阶段一：结构骨架（floor + wall）
    # ------------------------------------------------------------------
    def _make_stage1(self, raw_flat: np.ndarray, subset: str):
        """
        阶段一：预测 floor/wall，所有非 floor/wall tile 在目标中重映射为 floor。
        子集决定向模型提供多少 wall 作为上下文条件。
        """
        H = W = 13

        # 目标：非 floor/wall → floor
        target = raw_flat.copy()
        target[~np.isin(target, [self.FLOOR, self.WALL])] = self.FLOOR

        inp = target.copy()

        if subset == 'A':
            # 标准随机 mask：随机遮盖部分 floor/wall
            mask = self._std_mask(H, W)
            inp[mask] = self.MASK_ID

        elif subset == 'B':
            # 保留全部 wall，MASK floor
            inp[inp == self.FLOOR] = self.MASK_ID

        elif subset == 'C':
            # 随机保留部分 wall，MASK 其余（含全部 floor）
            inp[inp == self.FLOOR] = self.MASK_ID
            wall_idx = np.where(inp == self.WALL)[0]
            if len(wall_idx) > 0:
                ratio   = random.random() * self.wall_mask_ratio
                n       = max(1, int(len(wall_idx) * ratio))
                chosen  = np.random.choice(wall_idx, n, replace=False)
                inp[chosen] = self.MASK_ID

        else:  # D：与 B 相同（阶段一无 entrance 维度）
            inp[inp == self.FLOOR] = self.MASK_ID

        loss_mask = (inp == self.MASK_ID)
        return inp, target, loss_mask

    # ------------------------------------------------------------------
    # 阶段二：功能元素（door + monster + entrance）
    # ------------------------------------------------------------------
    def _make_stage2(self, raw_flat: np.ndarray, subset: str):
        """
        阶段二：以 floor/wall 为上下文，预测 door/monster/entrance。
        resource 在输入与目标中均视为 floor（阶段二不负责资源）。
        子集决定 wall 上下文的完整程度与 door/monster/entrance 的掩码方式。
        """
        # 目标：resource → floor
        target = raw_flat.copy()
        target[target == self.RESOURCE] = self.FLOOR

        # 基础输入：resource → floor，功能元素先保留，再按子集处理
        inp = raw_flat.copy()
        inp[inp == self.RESOURCE] = self.FLOOR

        if subset == 'A':
            # 随机遮盖部分 door/monster/entrance（部分上下文补全）
            func_idx = np.where(np.isin(inp, [self.DOOR, self.MONSTER, self.ENTRANCE]))[0]
            if len(func_idx) > 0:
                ratio  = random.random() * 0.8 + 0.2   # 20%~100%
                n      = max(1, int(len(func_idx) * ratio))
                chosen = np.random.choice(func_idx, n, replace=False)
                inp[chosen] = self.MASK_ID
        else:
            # B/C/D：全部 door/monster/entrance → MASK
            inp[np.isin(inp, [self.DOOR, self.MONSTER, self.ENTRANCE])] = self.MASK_ID

            if subset == 'C':
                # 额外随机 mask 部分 wall（降低 wall 上下文质量）
                wall_idx = np.where(inp == self.WALL)[0]
                if len(wall_idx) > 0:
                    ratio  = random.random() * self.wall_mask_ratio
                    n      = max(1, int(len(wall_idx) * ratio))
                    chosen = np.random.choice(wall_idx, n, replace=False)
                    inp[chosen] = self.MASK_ID

        # loss_mask：阶段二只对 door/monster/entrance 原始位置计算损失，
        # 不对被额外 mask 的 wall 位置计算（它们在 target 中已知为 wall）
        loss_mask = np.isin(raw_flat, [self.DOOR, self.MONSTER, self.ENTRANCE])
        return inp, target, loss_mask

    # ------------------------------------------------------------------
    # 阶段三：资源放置（resource）
    # ------------------------------------------------------------------
    def _make_stage3(self, raw_flat: np.ndarray, subset: str):
        """
        阶段三：以完整骨架为上下文，预测 resource 位置。
        所有 resource 位置在输入中替换为 MASK。
        子集 A 随机保留部分 resource 作为上下文（部分补全训练），
        其余子集始终 MASK 全部 resource。
        """
        target = raw_flat.copy()
        inp    = raw_flat.copy()

        if subset == 'A':
            # 随机遮盖部分 resource（部分上下文补全）
            res_idx = np.where(inp == self.RESOURCE)[0]
            if len(res_idx) > 0:
                ratio  = random.random() * 0.8 + 0.2   # 20%~100%
                n      = max(1, int(len(res_idx) * ratio))
                chosen = np.random.choice(res_idx, n, replace=False)
                inp[chosen] = self.MASK_ID
            else:
                pass  # 无 resource 时无需处理
        else:
            # B/C/D：全部 resource → MASK
            inp[inp == self.RESOURCE] = self.MASK_ID

        loss_mask = (inp == self.MASK_ID)
        return inp, target, loss_mask

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def _augment(self, arr: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            k   = np.random.randint(1, 4)
            arr = np.rot90(arr, k).copy()
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() > 0.5:
            arr = np.flipud(arr).copy()
        return arr

    def __getitem__(self, idx):
        item = self.data[idx]

        raw_np  = self._augment(np.array(item['map'], dtype=np.int64))  # [H, W]
        raw_flat = raw_np.reshape(-1)                                    # [H*W]
        subset   = self._choose_subset()

        if self.stage == 1:
            stage_input_np, target_np, loss_mask_np = self._make_stage1(raw_flat, subset)
        elif self.stage == 2:
            stage_input_np, target_np, loss_mask_np = self._make_stage2(raw_flat, subset)
        else:
            stage_input_np, target_np, loss_mask_np = self._make_stage3(raw_flat, subset)

        # 若 loss_mask 全为 False（如地图中无 resource 时的 stage3），
        # 退回为全图损失，避免 NaN
        if not loss_mask_np.any():
            loss_mask_np = np.ones_like(loss_mask_np)

        # VQ 切片：当前阶段编码器的输入（仅保留相关 tile）
        raw_t = torch.LongTensor(raw_flat)
        vq_keep = self._VQ_KEEP[self.stage]
        if vq_keep is None:
            vq_slice = raw_t.clone()
        else:
            vq_slice = make_slice(raw_t, vq_keep)

        # 结构标签
        sym_h, sym_v, sym_c = _compute_symmetry(raw_np)
        cond_sym    = sym_h * 4 + sym_v * 2 + sym_c
        cond_room   = item['roomCountLevel']
        cond_branch = item['branchLevel']
        cond_outer  = item['outerWall']
        struct_cond = torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer])

        return {
            "raw_map":     raw_t,
            "vq_slice":    vq_slice,
            "stage_input": torch.LongTensor(stage_input_np),
            "target_map":  torch.LongTensor(target_np),
            "loss_mask":   torch.BoolTensor(loss_mask_np),
            "subset":      subset,
            "struct_cond": struct_cond,
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
    print(f"被 mask 的位置数: {(masked == 6).sum().item()} / {masked.numel()}")
    print(f"\n200 次采样子集分布: {subset_count}")
