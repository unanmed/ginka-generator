import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.attention import CBAM

class MinamoVisionModel(nn.Module):
    def __init__(self, tile_types=32, embedding_dim=32, conv_channels=64, out_dim=128):
        super().__init__()
        # 嵌入层处理不同图块类型
        self.embedding = nn.Embedding(tile_types, embedding_dim)
        
        # 卷积部分
        self.vision_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 3, padding=1),
            nn.BatchNorm2d(conv_channels),
            CBAM(conv_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_channels, conv_channels*2, 3, padding=1),
            nn.BatchNorm2d(conv_channels*2),
            CBAM(conv_channels*2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_channels*2, conv_channels*4, 3, padding=1),
            nn.BatchNorm2d(conv_channels*4),
            CBAM(conv_channels*4),
            nn.GELU(),
            
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 输出为向量
        self.vision_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(conv_channels*4, out_dim)
        )
        
    def forward(self, map):
        x = self.embedding(map)
        x = x.permute(0, 3, 1, 2)

        x = self.vision_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        
        vision_vec = self.vision_head(x)
        
        return F.normalize(vision_vec, p=2, dim=-1)  # 归一化
