import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import print_memory

class EncoderEmbedding(nn.Module):
    def __init__(self, tile_classes=32, width=13, height=13, hidden_dim=128, output_dim=256):
        super().__init__()
        self.tile_embedding = nn.Sequential(
            nn.Embedding(tile_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.col_embedding = nn.Sequential(
            nn.Embedding(width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.row_embedding = nn.Sequential(
            nn.Embedding(height, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.fusion = nn.Linear(hidden_dim * 3, output_dim)
        
    def forward(self, tile, x, y):
        tile_embed = self.tile_embedding(tile)
        col_embed = self.col_embedding(x)
        row_embed = self.row_embedding(y)
        embed = torch.cat([tile_embed, col_embed, row_embed], dim=2)
        fused = self.fusion(embed)
        return fused

class EncoderGRU(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        
        # GRU
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feat: torch.Tensor, hidden: torch.Tensor):
        """
        feat: [B, input_dim]
        hidden: [B, hidden_dim]
        """
        hidden = self.drop(self.gru(feat, hidden))
        logits = self.fc(hidden)
        return logits, hidden
    
class EncoderFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, dim_feedforward=d_model, nhead=2, batch_first=True
            ),
            num_layers=2
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU()
        )
    
    def forward(self, logits):
        x = self.norm(self.transformer(logits))
        h_mean = torch.mean(x, dim=1)
        h_max = torch.max(x, dim=1).values
        h = torch.cat([h_mean, h_max], dim=1)
        return self.fc(h)

class VAEEncoder(nn.Module):
    def __init__(self, device, tile_classes=32, latent_dim=32, width=13, height=13):
        super().__init__()
        self.device = device
        
        self.rnn_hidden = 512
        self.logits_dim = 256
        
        self.embedding = EncoderEmbedding(tile_classes, width, height, 128, 256)
        self.rnn = EncoderGRU(256, self.rnn_hidden, self.logits_dim)
        self.fusion = EncoderFusion(256)
        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim)
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        self.col_list = []
        self.row_list = []
        for y in range(0, height):
            for x in range(0, width):
                self.col_list.append(x)
                self.row_list.append(y)
    
    def forward(self, x: torch.Tensor):
        B, H, W = x.shape
        
        map = torch.flatten(x, start_dim=1)
        hidden = torch.zeros(B, self.rnn_hidden).to(self.device)
        output = torch.zeros(B, H * W, self.logits_dim).to(self.device)
        
        col_list = torch.IntTensor(self.col_list).to(self.device).expand(B, -1)
        row_list = torch.IntTensor(self.row_list).to(self.device).expand(B, -1)
        embed = self.embedding(map, col_list, row_list)
        
        for idx in range(0, len(self.col_list)):
            logits, h = self.rnn(embed[:, idx], hidden)
            hidden = h
            output[:, idx] = logits

        h = self.fusion(output)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 13, 13]).to(device)
    
    # 初始化模型
    model = VAEEncoder(device).to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    mu, logvar = model(input)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: mu={mu.shape}, logvar={logvar.shape}")
    print(f"Embedding parameters: {sum(p.numel() for p in model.embedding.parameters())}")
    print(f"RNN parameters: {sum(p.numel() for p in model.rnn.parameters())}")
    print(f"Fusion parameters: {sum(p.numel() for p in model.fusion.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
