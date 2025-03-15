import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class HybridUpsample(nn.Module):
    """自适应尺寸的混合上采样"""
    def __init__(self, in_ch, out_ch, skip_ch=None):
        super().__init__()
        # 子像素卷积上采样
        self.subpixel = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2)  # 2倍上采样
        )
        
        # 跳跃连接处理
        self.skip_conv = nn.Conv2d(skip_ch, out_ch, 1) if skip_ch else None
        self.adaptive_pool = nn.AdaptiveAvgPool2d(None)

    def forward(self, x, skip=None):
        x = self.subpixel(x)  # [B, out_ch, 2H, 2W]
        
        if skip is not None and self.skip_conv:
            # 自动对齐尺寸
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='nearest')
            
            # 融合特征
            x = x + self.skip_conv(skip)
        
        return x

class DiscreteAwareUpsample(nn.Module):
    """离散感知的智能上采样模块"""
    def __init__(self, in_ch, out_ch, base_size=16):
        super().__init__()
        self.base_size = base_size
        self.scale_factors = [2, 4, 8]  # 支持放大倍数
        
        # 可变形卷积增强几何感知
        self.deform_conv = ops.DeformConv2d(in_ch, in_ch, kernel_size=3, padding=1)
        
        # 多尺度特征融合
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch//4, 1),
                nn.Upsample(scale_factor=s, mode='nearest')
            ) for s in self.scale_factors
        ])
        
        # 门控上采样机制
        self.gate_conv = nn.Conv2d(in_ch*2, len(self.scale_factors)+1, 3, padding=1)
        
        # 离散化输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, 3, padding=1),
            nn.PixelShuffle(2),  # 亚像素卷积
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
    def forward(self, x, target_size):
        # 几何特征提取
        deform_feat = self.deform_conv(x)
        
        # 生成多尺度特征
        scale_features = [f(deform_feat) for f in self.multi_scale]
        
        # 动态门控选择
        gate_map = F.softmax(self.gate_conv(torch.cat([x, deform_feat], dim=1)), dim=1)
        
        # 加权融合多尺度特征
        combined = sum(g * F.interpolate(f, size=target_size, mode='nearest') 
                     for g, f in zip(gate_map.unbind(1), scale_features+[x]))
        
        # 离散化上采样
        out = self.final_conv(combined)
        
        # 结构化约束（保持通道独立性）
        return out.argmax(dim=1).unsqueeze(1).float()  # 伪梯度保留
