import time
import torch
import torch.nn as nn
from ..utils import print_memory

class GinkaMaskGITCond(nn.Module):
    def __init__(self,  heatmap_channel=4, output_dim=256):
        super().__init__()
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(heatmap_channel, output_dim // 4, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(output_dim // 4),
            nn.GELU(),
            
            nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )
    
    def forward(self, heatmap):
        # heatmap: [B, C, H, W]
        heatmap = self.heatmap_conv(heatmap)
        return heatmap
    
if __name__ == "__main__":
    device = torch.device("cpu")
    
    heatmap = torch.rand(1, 4, 13, 13).to(device)
    
    # 初始化模型
    model = GinkaMaskGITCond().to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    cond, heatmap = model(heatmap)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: cond={cond.shape}, heatmap={heatmap.shape}")
    print(f"Cond FC parameters: {sum(p.numel() for p in model.cond_fc.parameters())}")
    print(f"Heatmap Conv parameters: {sum(p.numel() for p in model.heatmap_conv.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
