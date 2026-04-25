import time
import torch
import torch.nn as nn
from .cond import HeatmapCond
from ..maskGIT.maskGIT import Transformer
from ..utils import print_memory

class GinkaHeatmapModel(nn.Module):
    def __init__(
        self, T=100, embed_dim=128, heatmap_dim=8, d_model=128, dim_ff=512, nhead=8,
        num_layers=4, map_size=13*13
    ):
        super().__init__()
        self.heatmap_dim = heatmap_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, map_size, d_model))
        self.cond = HeatmapCond(T, embed_dim=embed_dim, heatmap_dim=heatmap_dim, output_dim=d_model)
        self.input = HeatmapCond(T, embed_dim=embed_dim, heatmap_dim=heatmap_dim, output_dim=d_model)
        self.transformer = Transformer(d_model=d_model, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(0.3),
            nn.GELU(),
            
            nn.Linear(d_model // 2, heatmap_dim)
        )
        
    def forward(self, input: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        # input: [B, heatmap_dim, H, W] 噪声
        # cond: [B, heatmap_dim, H, W] 点图
        # t: [B]
        input = self.input(input, t) # [B, d_model, H, W]
        cond = self.cond(cond, t) # [B, d_model, H, W]
        B, C, H, W = input.shape
        scale = torch.sigmoid(cond) # [B, d_model, H, W]
        hidden = input * (1 + scale) + cond # [B, d_model, H, W]
        hidden = hidden.view(B, C, H * W).permute(0, 2, 1) # [B, H * W, d_model]
        hidden = hidden + self.pos_embedding # [B, H * W, d_model]
        hidden = self.transformer(hidden) # [B, H * W, d_model]
        cond_tokens = cond.view(B, C, H * W).permute(0, 2, 1) # [B, H * W, d_model]
        attn, _ = self.cross_attn(hidden, cond_tokens, cond_tokens) # [B, H * W, d_model]
        hidden = hidden + attn # [B, H * W, d_model]
        output = self.output_fc(hidden) # [B, H * W, heatmap_dim]
        return output.view(B, H, W, self.heatmap_dim).permute(0, 3, 1, 2) # [B, heatmap_dim, H, W]
        
if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randn(1, 9, 13, 13).to(device)
    cond = torch.randint(0, 1, [1, 9, 13, 13]).to(device)
    t = torch.randint(0, 100, [1]).to(device)
    
    # 初始化模型
    model = GinkaHeatmapModel(heatmap_dim=9).to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    output = model(input, cond.float(), t)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: output={output.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.cond.parameters())}")
    print(f"Condition Encoder parameters: {sum(p.numel() for p in model.input.parameters())}")
    print(f"MaskGIT parameters: {sum(p.numel() for p in model.transformer.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output_fc.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

