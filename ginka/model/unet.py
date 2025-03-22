import torch
import torch.nn as nn
from shared.attention import CBAM

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # CBAM(out_channels),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_res = self.conv(x)  
        x_down = self.pool(x_res)  
        return x_down, x_res  
    
class GinkaDecoder(nn.Module):
    """解码器（上采样）部分"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # CBAM(out_channels),
            nn.GELU()
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  
        x = self.conv(x)
        return x
    
class GinkaBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=32, out_ch=32):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.down1 = GinkaEncoder(in_ch, in_ch*2)
        self.down2 = GinkaEncoder(in_ch*2, in_ch*4)
        self.down3 = GinkaEncoder(in_ch*4, in_ch*8)
        self.down4 = GinkaEncoder(in_ch*8, in_ch*16)

        self.bottleneck = GinkaBottleneck(in_ch*16, in_ch*16)

        self.up1 = GinkaDecoder(in_ch*16, in_ch*8)
        self.up2 = GinkaDecoder(in_ch*8, in_ch*4)
        self.up3 = GinkaDecoder(in_ch*4, in_ch*2)
        self.up4 = GinkaDecoder(in_ch*2, in_ch)

        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            # nn.Softmax(dim=1)  # 适用于分类任务
        )
        
    def forward(self, x):
        x_down1, skip1 = self.down1(x)
        x_down2, skip2 = self.down2(x_down1)
        x_down3, skip3 = self.down3(x_down2)
        x_down4, skip4 = self.down4(x_down3)

        x = self.bottleneck(x_down4)

        x = self.up1(x, skip4)  # 用 down2 的 skip
        x = self.up2(x, skip3)  # 用 down2 的 skip
        x = self.up3(x, skip2)  # 用 down1 的 skip
        x = self.up4(x, skip1)  # 用 down1 的 skip
        
        return self.final(x)
