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
) -> torch.Tensor:
    """
    通道专属 Focal Loss：tile_set 内的位置以真实 tile ID 为目标，
    tile_set 外的位置以 0（空地）为目标，全部位置均参与损失计算。

    这样模型不仅要学会"这里是什么 tile"，还要学会"这里不应该是本通道的 tile"，
    避免解码器在所有位置都输出专属类别来规避损失。

    Args:
        logits:   [B, H*W, num_classes]  解码头输出（未经 softmax）
        target:   [B, H*W]               完整地图 ground truth（整数 tile ID）
        tile_set: set of int             本通道专属 tile 集合
        gamma:    Focal Loss 聚焦参数

    Returns:
        scalar tensor  通道专属 Focal Loss（均值）
    """
    B, S, C = logits.shape

    # 非专属 tile 位置目标替换为 0（空地），专属 tile 位置保持原始标签
    in_set = torch.zeros(B, S, dtype=torch.bool, device=logits.device)
    for t in tile_set:
        in_set |= (target == t)

    corrected = target.clone()
    corrected[~in_set] = 0

    # Focal Loss，全部位置参与计算
    ce = F.cross_entropy(
        logits.view(-1, C),
        corrected.view(-1),
        reduction='none',
    ).view(B, S)                          # [B, S]

    pt = torch.exp(-ce.detach())          # 正确类预测概率，stop-gradient
    fl = (1.0 - pt) ** gamma * ce

    return fl.mean()
