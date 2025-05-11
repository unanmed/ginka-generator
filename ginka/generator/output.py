import torch
import torch.nn as nn
from ..common.common import ConvFusionModule
from ..common.cond import ConditionInjector

class StageHead(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=(13, 13)):
        super().__init__()
        self.dec = ConvFusionModule(in_ch, in_ch*2, in_ch, 32, 32)
        self.pool = nn.Sequential(
            ConvFusionModule(in_ch, in_ch*2, in_ch*2, 32, 32),
            ConvFusionModule(in_ch*2, in_ch*2, in_ch, 32, 32),
            
            nn.AdaptiveMaxPool2d(out_size),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.inject = ConditionInjector(256, in_ch)
        
    def forward(self, x, cond):
        x = self.dec(x)
        x = self.inject(x, cond)
        x = self.pool(x)
        return x

class GinkaOutput(nn.Module):
    def __init__(self, in_ch=64, out_ch=32, out_size=(13, 13)):
        super().__init__()
        self.head1 = StageHead(in_ch, out_ch, out_size)
        self.head2 = StageHead(in_ch, out_ch, out_size)
        self.head3 = StageHead(in_ch, out_ch, out_size)
        
    def forward(self, x, stage, cond):
        if stage == 1:
            x = self.head1(x, cond)
        elif stage == 2:
            x = self.head2(x, cond)
        elif stage == 3:
            x = self.head3(x, cond)
        else:
            raise RuntimeError("Unknown generate stage.")
        return x
