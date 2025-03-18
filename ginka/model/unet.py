import torch
import torch.nn as nn

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_res = self.conv(x)  # 卷积提取特征
        x_down = self.pool(x_res)  # 进行池化
        return x_down, x_res  # 返回池化后的特征和跳跃连接特征
    
class GinkaDecoder(nn.Module):
    """解码器（上采样）部分"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 上采样（双线性插值 + 卷积）
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 跳跃连接融合
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # 跳跃连接融合
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class GinkaBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=32):
        """Ginka Model UNet 部分
        """
        super().__init__()

        self.down1 = GinkaEncoder(in_ch, in_ch*2)
        self.down2 = GinkaEncoder(in_ch*2, in_ch*4)
        
        self.bottleneck = GinkaBottleneck(in_ch*4, in_ch*4)
        
        self.up1 = GinkaDecoder(in_ch*4, in_ch*2)
        self.up2 = GinkaDecoder(in_ch*2, in_ch)
        
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1)
        )
        
    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        
        x = self.bottleneck(x)
        
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        
        return self.final(x)
