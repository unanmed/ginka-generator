from collections import deque
import numpy as np
import torch

DIST_MAX_BUCKET = 12 # 距离分桶上限
DIST_VOCAB = DIST_MAX_BUCKET + 1 # 0-12 共 13 个桶

def compute_distance_field(map_matrix: np.ndarray) -> np.ndarray:
    # map_matrix: [H, W] 整数矩阵
    # 返回: [H * W] 分桶后的整数距离，值域 [0, DIST_MAX_BUCKET]
    wall_mask = (map_matrix == 1)
    h, w = map_matrix.shape
    dist = np.full((h, w), h + w, dtype=np.int32)
    dist[wall_mask] = 0

    queue = deque()
    for i in range(h):
        for j in range(w):
            if wall_mask[i, j]:
                queue.append((i, j))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        i, j = queue.popleft()
        d = dist[i, j]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and dist[ni, nj] > d + 1:
                dist[ni, nj] = d + 1
                queue.append((ni, nj))

    dist = np.clip(dist, 0, DIST_MAX_BUCKET)
    return dist.flatten().astype(np.int64)

def compute_distance_field_tensor(map_tensor: torch.Tensor) -> torch.Tensor:
    # map_tensor: [B, H, W] 或 [1, H, W]
    batch = []
    for i in range(map_tensor.shape[0]):
        d = compute_distance_field(map_tensor[i].cpu().numpy())
        batch.append(torch.from_numpy(d))
    return torch.stack(batch, dim=0).to(map_tensor.device)
