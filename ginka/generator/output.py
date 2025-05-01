import torch
import torch.nn as nn
from ..common.common import GCNBlock, DoubleConvBlock
from ..common.cond import ConditionInjector

class StageHead(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=(13, 13)):
        super().__init__()
        self.cnn_head = DoubleConvBlock([in_ch, in_ch*2, in_ch])
        self.gcn_head = GCNBlock(in_ch, in_ch*2, in_ch, 32, 32)
        self.fusion = DoubleConvBlock([in_ch*2, in_ch*4, in_ch])
        self.pool = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(in_ch*2),
            nn.ELU(),
            
            nn.Conv2d(in_ch*2, in_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(in_ch),
            nn.ELU(),
            
            nn.AdaptiveMaxPool2d(out_size),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.inject = ConditionInjector(256, in_ch)
        
    def forward(self, x, cond):
        x_cnn = self.cnn_head(x)
        x_gcn = self.gcn_head(x)
        x = torch.cat([x_cnn, x_gcn], dim=1)
        x = self.fusion(x)
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
