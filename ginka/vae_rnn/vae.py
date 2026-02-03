import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VAEEncoder
from .decoder import VAEDecoder
from ..utils import print_memory

class GinkaVAE(nn.Module):
    def __init__(self, device, tile_classes=32, latent_dim=32):
        super().__init__()
        self.encoder = VAEEncoder(device, tile_classes, latent_dim)
        self.decoder = VAEDecoder(device, map_vec_dim=latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, target_map: torch.Tensor, use_self_probility=0):
        z = self.encoder(target_map)
        logits = self.decoder(z, target_map, use_self_probility)
        return logits, z
    
if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 13, 13]).to(device)
    
    # 初始化模型
    model = GinkaVAE(device).to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    logits, z = model(input)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: logits= {logits.shape}, z={z.shape}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
