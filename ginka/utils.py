import torch
import torch.nn.functional as F
import numpy as np

def print_memory(device, tag=""):
    if torch.cuda.is_available():
        print(f"{tag} | 当前显存: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    else:
        print("当前设备不支持 cuda.")
        
def nms_sampling(noise: np.ndarray, k: int, radius=2):
    # noise: [H, W]
    noise = noise.copy()
    points = []

    for _ in range(k):
        idx = np.argmax(noise)
        x, y = np.unravel_index(idx, noise.shape)

        points.append((x, y))

        # 抑制周围
        x0 = max(0, x - radius)
        x1 = min(noise.shape[0], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(noise.shape[1], y + radius + 1)

        noise[x0:x1, y0:y1] = -np.inf

    result = np.zeros_like(noise)
    for x, y in points:
        result[y, x] = 1
        
    return result


def masked_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    tile_set: set,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    通道专属掩码 Focal Loss：仅在 tile_set 中指定的 tile 位置计算损失。

    Args:
        logits:   [B, H*W, num_classes]  解码头输出（未经 softmax）
        target:   [B, H*W]               完整地图 ground truth（整数 tile ID）
        tile_set: set of int             本通道专属 tile 集合，其余位置损失权重为 0
        gamma:    Focal Loss 聚焦参数
        eps:      数值稳定的分母偏置

    Returns:
        scalar tensor  通道专属掩码 Focal Loss
    """
    B, S, C = logits.shape

    # 构造掩码：仅在专属 tile 位置为 True
    mask = torch.zeros(B, S, dtype=torch.bool, device=logits.device)
    for t in tile_set:
        mask |= (target == t)

    if not mask.any():
        return logits.sum() * 0.0  # 保留计算图，返回零梯度

    # Focal Loss（reduction='none'）
    ce = F.cross_entropy(
        logits.view(-1, C),
        target.view(-1),
        reduction='none',
    ).view(B, S)                          # [B, S]

    pt = torch.exp(-ce.detach())          # 正确类预测概率，stop-gradient
    fl = (1.0 - pt) ** gamma * ce

    return (fl * mask).sum() / (mask.sum() + eps)
