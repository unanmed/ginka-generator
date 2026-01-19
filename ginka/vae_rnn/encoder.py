import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import print_memory

class VAEEncoder(nn.Module):
    def __init__(self, tile_classes=32, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(tile_classes, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    
    def forward(self, x):
        h = self.conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 13, 13]).to(device)
    input = F.one_hot(input, 32).permute(0, 3, 1, 2).float()
    
    # 初始化模型
    model = VAEEncoder().to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    mu, logvar = model(input)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: mu={mu.shape}, logvar={logvar.shape}")
    print(f"CNN parameters: {sum(p.numel() for p in model.conv.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
