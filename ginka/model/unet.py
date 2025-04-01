import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class ConditionFusionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习融合系数
        
    def forward(self, x, cond_feat):
        return x + self.alpha * cond_feat  # 残差融合
    
class FusionConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.fusion = ConditionFusionBlock()
        
    def forward(self, x, feat):
        x = self.conv(x)
        x = self.fusion(x, feat)
        return x

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        self.fusion = ConditionFusionBlock()

    def forward(self, x, feat):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fusion(x, feat)
        return x
    
class GinkaUpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class GinkaDecoder(nn.Module):
    """解码器（上采样）部分"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = GinkaUpSample(in_ch, in_ch // 2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.fusion = ConditionFusionBlock()
        
    def forward(self, x, skip, feat):
        dec = self.upsample(x)
        x = torch.cat([dec, skip], dim=1)
        x = self.conv(x)
        x = self.fusion(x, feat)
        return x

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, out_ch=32, feat_dim=1024):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.in_conv = FusionConvBlock(in_ch, base_ch)
        self.down1 = GinkaEncoder(base_ch, base_ch*2)
        self.down2 = GinkaEncoder(base_ch*2, base_ch*4)
        self.down3 = GinkaEncoder(base_ch*4, base_ch*8)

        self.bottleneck = GinkaEncoder(base_ch*8, base_ch*16)

        self.up1 = GinkaDecoder(base_ch*16, base_ch*8)
        self.up2 = GinkaDecoder(base_ch*8, base_ch*4)
        self.up3 = GinkaDecoder(base_ch*4, base_ch*2)
        self.up4 = GinkaDecoder(base_ch*2, base_ch)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
        )
        
    def forward(self, x, feat, cond):
        x1 = self.in_conv(x, feat[0])
        x2 = self.down1(x1, feat[1])
        x3 = self.down2(x2, feat[2])
        x4 = self.down3(x3, feat[3])
        x5 = self.bottleneck(x4, feat[4])
        
        x = self.up1(x5, x4, feat[3])
        x = self.up2(x, x3, feat[2])
        x = self.up3(x, x2, feat[1])
        x = self.up4(x, x1, feat[0])
        
        return self.final(x)
