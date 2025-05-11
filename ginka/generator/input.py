import torch
import torch.nn as nn
from ..common.common import ConvFusionModule
from ..common.cond import ConditionInjector
from .unet import GinkaEncoderPath, GinkaDecoderPath

class RandomInputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GinkaEncoderPath(32, 32)
        self.dec = GinkaDecoderPath(32)
        self.out_conv = nn.Sequential(
            nn.AdaptiveMaxPool2d((15, 15)),
            nn.Conv2d(32, 64, 3, padding=0),
            nn.InstanceNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 32, 1),
        )
        
    def forward(self, x, cond):
        x1, x2, x3, x4 = self.enc(x, cond)
        x = self.dec(x1, x2, x3, x4, cond)
        x = self.out_conv(x)
        return x
    
class InputUpsample(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvFusionModule(in_ch, hidden_ch, hidden_ch, 13, 13),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13x13 → 26x26
            ConvFusionModule(hidden_ch, hidden_ch, hidden_ch, 26, 26),
            
            nn.Upsample(size=(32, 32), mode='nearest'),  # 26x26 → 32x32
            ConvFusionModule(hidden_ch, hidden_ch, out_ch, 32, 32),
        )

    def forward(self, x): # [B, C, 13, 13]
        x = self.net(x)   # [B, C, 32, 32]
        return x

class GinkaInput(nn.Module):
    def __init__(self, in_ch=32, out_ch=64, in_size=(13, 13), out_size=(32, 32)):
        super().__init__()
        self.out_size = out_size
        self.upsample = InputUpsample(in_ch, in_ch*2, out_ch)
        self.enc = ConvFusionModule(out_ch, out_ch*2, out_ch, out_size[0], out_size[1])
        self.inject1 = ConditionInjector(256, out_ch)
        self.inject2 = ConditionInjector(256, out_ch)
        
    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.inject1(x, cond)
        x = self.enc(x)
        x = self.inject2(x, cond)
        return x
