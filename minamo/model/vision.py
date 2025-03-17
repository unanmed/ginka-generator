import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.channel_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.spatial(x) + self.channel(x) + self.channel_max(x)
        return x * attn

class MinamoVisionModel(nn.Module):
    def __init__(self, tile_types=32, embedding_dim=16, conv_channels=64, out_dim=128):
        super().__init__()
        # 嵌入层处理不同图块类型
        self.embedding = nn.Embedding(tile_types, embedding_dim)
        
        # 卷积部分
        self.vision_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 3, padding=1),
            nn.BatchNorm2d(conv_channels),
            DualAttention(conv_channels, reduction=12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_channels, conv_channels*2, 3, padding=1),
            nn.BatchNorm2d(conv_channels*2),
            DualAttention(conv_channels*2, reduction=12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_channels*2, conv_channels*4, 3, padding=1),
            nn.BatchNorm2d(conv_channels*4),
            DualAttention(conv_channels*4, reduction=12),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 输出为向量
        self.vision_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conv_channels*4, out_dim)
        )
        
    def forward(self, map):
        x = self.embedding(map)
        # print(map.shape, x.shape)
        x = x.permute(0, 3, 1, 2)

        x = self.vision_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        
        vision_vec = self.vision_head(x)
        
        return F.normalize(vision_vec, p=2, dim=-1)  # 归一化
