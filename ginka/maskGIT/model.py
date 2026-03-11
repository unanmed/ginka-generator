import time
import torch
import torch.nn as nn
from ..utils import print_memory
from .cond import GinkaMaskGITCond

class GinkaMaskGIT(nn.Module):
    def __init__(
        self, num_classes=16, cond_dim=16, heatmap_channel=4, d_model=256, 
        dim_ff=512, nhead=8, num_layers=4, map_size=13*13
    ):
        super().__init__()
        
        self.tile_embedding = nn.Embedding(num_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, map_size + 1, d_model))
        
        self.cond_encoder = GinkaMaskGITCond(cond_dim=cond_dim, heatmap_channel=heatmap_channel, output_dim=d_model)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True),
            num_layers=num_layers
        )
        
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, map: torch.Tensor, cond: torch.Tensor, heatmap: torch.Tensor):
        # map: [B, H * W]
        # cond: [B, cond_dim]
        # heatmap: [B, C, H, W]
        # output: [B, H * W, num_classes]
        cond, heatmap = self.cond_encoder(cond, heatmap)
        # cond: [B, d_model]
        # heatmap: [B, d_model, H, W]
        
        B, C, H, W = heatmap.shape
        
        heatmap = heatmap.view(B, C, H * W).permute(0, 2, 1)
        x = self.tile_embedding(map) + heatmap
        x = torch.cat([cond.unsqueeze(1), x], dim=1) + self.pos_embedding
        
        m = self.encoder(x)
        out = self.decoder(x, m)
        
        logits = self.output_fc(out)
        
        return logits[:, :-1, :]
        
if __name__ == "__main__":
    device = torch.device("cpu")
    
    map = torch.randint(0, 16, [1, 169]).to(device)
    cond = torch.rand(1, 16).to(device)
    heatmap = torch.rand(1, 4, 13, 13).to(device)
    
    # 初始化模型
    model = GinkaMaskGIT().to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    output = model(map, cond, heatmap)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: output={output.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Condition Encoder parameters: {sum(p.numel() for p in model.cond_encoder.parameters())}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output_fc.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
