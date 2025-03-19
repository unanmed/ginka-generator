import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.attention import CBAM

class MinamoVisionModel(nn.Module):
    def __init__(self, tile_types=32, conv_ch=32, out_dim=128):
        super().__init__()
        # 输入 softmax 概率值
        self.input_conv = nn.Conv2d(tile_types, conv_ch, 3, padding=1)
        
        # 卷积部分
        self.vision_conv = nn.Sequential(
            nn.Conv2d(conv_ch, conv_ch*2, 3, padding=1),
            nn.BatchNorm2d(conv_ch*2),
            CBAM(conv_ch*2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_ch*2, conv_ch*4, 3, padding=1),
            nn.BatchNorm2d(conv_ch*4),
            CBAM(conv_ch*4),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(conv_ch*4, conv_ch*8, 3, padding=1),
            nn.BatchNorm2d(conv_ch*8),
            CBAM(conv_ch*8),
            nn.GELU(),
            
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 输出为向量
        self.vision_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(conv_ch*8, out_dim)
        )
        
    def forward(self, map):
        x = self.input_conv(map)
        x = self.vision_conv(x)
        x = x.view(x.size(0), -1)  # 展平
        
        vision_vec = self.vision_head(x)
        
        return F.normalize(vision_vec, p=2, dim=-1)  # 归一化
