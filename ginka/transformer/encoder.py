import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import print_memory

class GinkaTransformerEncoder(nn.Module):
    def __init__(self, dim_ff=256, nhead=4, num_layers=4):
        super().__init__()
        self.dim_ff = dim_ff
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_ff, dim_feedforward=dim_ff, nhead=nhead, batch_first=True, activation=F.gelu),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_ff, dim_feedforward=dim_ff, nhead=nhead, batch_first=True, activation=F.gelu),
            num_layers=max(num_layers // 2, 1)
        )
        
    def forward(self, x: torch.Tensor):
        # x: [B, H * W, S]
        B, L, S = x.shape
        first_token = torch.randn(B, 1, self.dim_ff).to(x.device)
        x = self.encoder(x)
        x = self.decoder(first_token, x)
        return x.squeeze(1)

class GinkaTransformerBottleneck(nn.Module):
    def __init__(self, dim_ff=256, hidden_dim=512, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ff, hidden_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        # x: [B, dim_ff]
        hidden = self.fc(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class GinkaTransformerVAEEncoder(nn.Module):
    def __init__(
        self, num_classes=32, latent_dim=32, bottleneck_dim=512, dim_ff=256, 
        nhead=4, num_layers=4, map_size=13*13
    ):
        super().__init__()
        self.map_size = map_size
        self.embedding = nn.Embedding(num_classes, dim_ff)
        self.pos_embedding = nn.Embedding(map_size, dim_ff)
        self.encoder = GinkaTransformerEncoder(dim_ff=dim_ff, nhead=nhead, num_layers=num_layers)
        self.bottleneck = GinkaTransformerBottleneck(
            dim_ff=dim_ff, hidden_dim=bottleneck_dim, latent_dim=latent_dim
        )
        
    def forward(self, x: torch.Tensor):
        # x: [B, map_size]
        pos = self.pos_embedding(torch.arange(self.map_size, dtype=torch.long).to(x.device))
        x = self.embedding(x) + pos
        x = self.encoder(x)
        mu, logvar = self.bottleneck(x)
        return mu, logvar
    
if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 169]).to(device)
    
    # 初始化模型
    model = GinkaTransformerVAEEncoder().to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    mu, logvar = model(input)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: mu={mu.shape}, logvar={logvar.shape}")
    print(f"Embedding parameters: {sum(p.numel() for p in model.embedding.parameters())}")
    print(f"Position Embedding parameters: {sum(p.numel() for p in model.pos_embedding.parameters())}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"bottleneck parameters: {sum(p.numel() for p in model.bottleneck.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

