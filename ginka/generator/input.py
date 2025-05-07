import torch
import torch.nn as nn
from ..common.common import ConvFusionModule
from ..common.cond import ConditionInjector

class RandomInputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ConvFusionModule(32, 256, 256, 32, 32)
        self.out_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(128),
            nn.ELU(),
            
            nn.AdaptiveMaxPool2d((13, 13)),
            nn.Conv2d(128, 32, 1),
        )
        self.inject = ConditionInjector(256, 256)
        
    def forward(self, x, cond):
        x = self.enc(x)
        x = self.inject(x, cond)
        x = self.out_conv(x)
        return x
    
class InputUpsample(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ELU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),  # 13x13 → 26x26
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ELU(),
            
            nn.Upsample(size=(32, 32), mode='nearest'),  # 26x26 → 32x32
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1),
            nn.ELU(),
        )

    def forward(self, x): # [B, C, 13, 13]
        x = self.net(x)   # [B, C, 32, 32]
        return x

class GinkaInput(nn.Module):
    def __init__(self, in_ch=32, out_ch=64, in_size=(13, 13), out_size=(32, 32)):
        super().__init__()
        self.out_size = out_size
        self.enc1 = ConvFusionModule(in_ch, in_ch*4, in_ch, in_size[0], in_size[1])
        self.upsample = InputUpsample(in_ch, in_ch*2, out_ch)
        self.enc2 = ConvFusionModule(out_ch, out_ch*4, out_ch, out_size[0], out_size[1])
        self.inject1 = ConditionInjector(256, in_ch)
        self.inject2 = ConditionInjector(256, out_ch)
        
    def forward(self, x, cond):
        x = self.enc1(x)
        x = self.inject1(x, cond)
        x = self.upsample(x)
        x = self.enc2(x)
        x = self.inject2(x, cond)
        return x
