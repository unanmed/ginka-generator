import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        # 通道注意力
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.spatial(x) + x * self.channel(x)

class MinamoVisionModel(nn.Module):
    def __init__(self, tile_types=32, embedding_dim=16, conv_channels=16):
        super().__init__()
        # 嵌入层处理不同图块类型
        self.embedding = nn.Embedding(tile_types, embedding_dim)
        
        # 卷积部分
        self.vision_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 3, padding=1),
            DualAttention(conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            
            nn.Conv2d(conv_channels, conv_channels*2, 3, padding=1),
            DualAttention(conv_channels*2),
            nn.BatchNorm2d(conv_channels*2),
            nn.ReLU(),
            
            nn.Conv2d(conv_channels*2, conv_channels*4, 3, padding=1),
            DualAttention(conv_channels*4),
            nn.BatchNorm2d(conv_channels*4),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 预测头
        self.vision_head = nn.Sequential(
            nn.Linear(conv_channels*4, conv_channels*2),
            nn.Dropout(0.4),
            nn.Linear(conv_channels*2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, map1, map2):
        e1 = self.embedding(map1).permute(0, 3, 1, 2)
        e2 = self.embedding(map2).permute(0, 3, 1, 2)
        
        v1 = self.vision_conv(e1)
        v2 = self.vision_conv(e2)
        
        v1 = v1.view(v1.size(0), -1)  # 展平
        v2 = v2.view(v2.size(0), -1)  # 展平
        
        vision_sim = self.vision_head(torch.abs(v1 - v2))
        
        return vision_sim