import torch
import torch.nn as nn
import torch.nn.functional as F

class GinkaAdaIN(nn.Module):
    def __init__(self, num_features, condition_dim):
        """
        自适应实例归一化 (AdaIN)
        参数:
            num_features: 归一化的通道数
            condition_dim: 条件输入的特征维度
        """
        super(GinkaAdaIN, self).__init__()
        self.fc = nn.Linear(condition_dim, num_features * 2)  # γ 和 β

    def forward(self, x, condition):
        """
        x: [B, C, H, W] - 输入特征图
        condition: [B, condition_dim] - 需要注入的条件向量
        """
        gamma, beta = self.fc(condition).chunk(2, dim=1)  # 分割为 γ 和 β
        gamma = gamma.view(x.shape[0], x.shape[1], 1, 1)  # 调整形状
        beta = beta.view(x.shape[0], x.shape[1], 1, 1)
        
        x = F.instance_norm(x)  # 标准化
        return gamma * x + beta  # 进行变换

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
    
class AdaINConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, feat_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.adain = GinkaAdaIN(out_ch, feat_dim)
        
    def forward(self, x, feat):
        x = self.conv(x)
        x = self.adain(x, feat)
        return x

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_ch, out_ch, feat_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        self.adain = GinkaAdaIN(out_ch, feat_dim)

    def forward(self, x, feat):
        x = self.conv(x)
        x = self.pool(x)
        x = self.adain(x, feat)
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
    def __init__(self, in_ch, out_ch, feat_dim):
        super().__init__()
        self.upsample = GinkaUpSample(in_ch, in_ch // 2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.adain = GinkaAdaIN(out_ch, feat_dim)
        
    def forward(self, x, skip, feat):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.adain(x, feat)
        return x

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, out_ch=32, feat_dim=1024):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.in_conv = AdaINConvBlock(in_ch, base_ch, feat_dim)
        self.down1 = GinkaEncoder(base_ch, base_ch*2, feat_dim)
        self.down2 = GinkaEncoder(base_ch*2, base_ch*4, feat_dim)
        self.down3 = GinkaEncoder(base_ch*4, base_ch*8, feat_dim)

        self.bottleneck = GinkaEncoder(base_ch*8, base_ch*16, feat_dim)

        self.up1 = GinkaDecoder(base_ch*16, base_ch*8, feat_dim)
        self.up2 = GinkaDecoder(base_ch*8, base_ch*4, feat_dim)
        self.up3 = GinkaDecoder(base_ch*4, base_ch*2, feat_dim)
        self.up4 = GinkaDecoder(base_ch*2, base_ch, feat_dim)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
        )
        
    def forward(self, x, feat):
        x1 = self.in_conv(x, feat)
        x2 = self.down1(x1, feat)
        x3 = self.down2(x2, feat)
        x4 = self.down3(x3, feat)
        x5 = self.bottleneck(x4, feat)
        
        x = self.up1(x5, x4, feat)
        x = self.up2(x, x3, feat)
        x = self.up3(x, x2, feat)
        x = self.up4(x, x1, feat)
        
        return self.final(x)
