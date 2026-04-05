import time
import torch
import torch.nn as nn
from ..utils import print_memory
from .cond import GinkaMaskGITCond
from .maskGIT import MaskGIT

class GinkaMaskGIT(nn.Module):
    def __init__(
        self, num_classes=16, heatmap_channel=4, d_model=256, 
        dim_ff=512, nhead=8, num_layers=4, map_size=13*13
    ):
        super().__init__()
        
        self.tile_embedding = nn.Embedding(num_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, map_size, d_model))
        
        self.cond_encoder = GinkaMaskGITCond(heatmap_channel=heatmap_channel, output_dim=d_model)
        
        self.transformer = MaskGIT(d_model=d_model, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers)
        
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, map: torch.Tensor, heatmap: torch.Tensor):
        # map: [B, H * W]
        # heatmap: [B, C, H, W]
        # output: [B, H * W, num_classes]
        heatmap = self.cond_encoder(heatmap)
        # cond: [B, d_model]
        # heatmap: [B, d_model, H, W]
        
        B, C, H, W = heatmap.shape
        
        heatmap = heatmap.view(B, C, H * W).permute(0, 2, 1)
        x = self.tile_embedding(map) + heatmap
        x = x + self.pos_embedding
        x = self.transformer(x)
        
        logits = self.output_fc(x)
        
        return logits
        
if __name__ == "__main__":
    device = torch.device("cpu")
    
    map = torch.randint(0, 16, [1, 169]).to(device)
    heatmap = torch.rand(1, 4, 13, 13).to(device)
    
    # 初始化模型
    model = GinkaMaskGIT().to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    output = model(map, heatmap)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: output={output.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Condition Encoder parameters: {sum(p.numel() for p in model.cond_encoder.parameters())}")
    print(f"MaskGIT parameters: {sum(p.numel() for p in model.transformer.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output_fc.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
