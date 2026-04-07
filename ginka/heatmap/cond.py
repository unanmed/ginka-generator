import torch
import torch.nn as nn

class HeatmapCond(nn.Module):
    def __init__(self, T=100, embed_dim=128, heatmap_dim=8, output_dim=128):
        super().__init__()
        self.time_embedding = nn.Embedding(T, embed_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(heatmap_dim, output_dim // 4, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(output_dim // 4),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim // 4, output_dim // 2, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(output_dim // 2, output_dim, 3, padding=1, padding_mode='replicate')
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, output_dim // 4),
            nn.Dropout(0.3),
            nn.LayerNorm(output_dim // 4),
            nn.GELU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim, output_dim // 2),
            nn.Dropout(0.3),
            nn.LayerNorm(output_dim // 2),
            nn.GELU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, heatmap: torch.Tensor, t: torch.Tensor):
        # heatmap: [B, C, H, W]
        # t: [B]
        t_embed = self.time_embedding(t)
        x = self.conv1(heatmap) + self.fc1(t_embed).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2)
        x = self.conv2(x) + self.fc2(t_embed).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2)
        x = self.conv3(x) + self.fc3(t_embed).unsqueeze(1).unsqueeze(1).permute(0, 3, 1, 2)
        return x
    