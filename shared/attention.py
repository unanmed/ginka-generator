import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ELU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        c_att = self.channel_att(x)
        x = x * c_att
        return x

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super().__init__()
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        s_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * s_att

class CBAM(nn.Module):
    """通道与空间注意力结合"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        self.channel_att = ChannelAttention(channels, reduction)
        # 空间注意力
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        c_att = self.channel_att(x)
        x = x * c_att
        
        # 空间注意力
        s_att = self.spatial_att(x)
        return x * s_att

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y