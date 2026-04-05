import torch
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
    