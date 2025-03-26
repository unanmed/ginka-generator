import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.attention import CBAM, SEBlock

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_channels, out_channels, attention=False, block='CBAM'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
        )
        # 注意力
        if attention:
            if block == 'CBAM':
                self.conv.append(CBAM(out_channels))
            elif block == 'SEBlock':
                self.conv.append(SEBlock(out_channels))
        self.conv.append(nn.GELU())
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x_res = self.conv(x)
        x_down = self.down(x_res)
        return x_down, x_res
    
class GinkaDecoder(nn.Module):
    """解码器（上采样）部分"""
    def __init__(self, in_channels, out_channels, attention=False, block='CBAM'):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
        )
        # 注意力
        if attention:
            if block == 'CBAM':
                self.conv.append(CBAM(out_channels))
            elif block == 'SEBlock':
                self.conv.append(SEBlock(out_channels))
        self.conv.append(nn.GELU())
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  
        x = self.conv(x)
        return x
    
class GinkaBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
        )
        if attention:
            self.conv.append(SEBlock(out_channels))
        self.conv.append(nn.GELU())

    def forward(self, x):
        return self.conv(x)

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=32):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.down1 = GinkaEncoder(in_ch, in_ch*2, attention=True)
        self.down2 = GinkaEncoder(in_ch*2, in_ch*4, attention=True)
        self.down3 = GinkaEncoder(in_ch*4, in_ch*8, attention=True, block='SEBlock')
        self.down4 = GinkaEncoder(in_ch*8, in_ch*16, attention=True, block='SEBlock')

        self.bottleneck = GinkaBottleneck(in_ch*16, in_ch*16, attention=True)

        self.up1 = GinkaDecoder(in_ch*16, in_ch*8, attention=True, block='SEBlock')
        self.up2 = GinkaDecoder(in_ch*8, in_ch*4, attention=True, block='SEBlock')
        self.up3 = GinkaDecoder(in_ch*4, in_ch*2, attention=True)
        self.up4 = GinkaDecoder(in_ch*2, in_ch, attention=True)

        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
        )
        
    def forward(self, x):
        x_down1, skip1 = self.down1(x)
        x_down2, skip2 = self.down2(x_down1)
        x_down3, skip3 = self.down3(x_down2)
        x_down4, skip4 = self.down4(x_down3)

        x = self.bottleneck(x_down4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        return self.final(x)
