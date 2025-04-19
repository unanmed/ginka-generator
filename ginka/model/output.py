import torch
import torch.nn as nn
from .common import GCNBlock, DoubleConvBlock

class StageHead(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=(13, 13)):
        super().__init__()
        self.cnn_head = DoubleConvBlock([in_ch, in_ch*2, in_ch])
        self.gcn_head = GCNBlock(in_ch, in_ch*2, in_ch, 32, 32)
        self.fusion = DoubleConvBlock([in_ch*2, in_ch*4, in_ch])
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(out_size),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        
    def forward(self, x):
        x_cnn = self.cnn_head(x)
        x_gcn = self.gcn_head(x)
        x = torch.cat([x_cnn, x_gcn], dim=1)
        x = self.fusion(x)
        x = self.pool(x)
        return x

class GinkaOutput(nn.Module):
    def __init__(self, in_ch=64, out_ch=32, out_size=(13, 13)):
        super().__init__()
        self.head1 = StageHead(in_ch, out_ch, out_size)
        self.head2 = StageHead(in_ch, out_ch, out_size)
        self.head3 = StageHead(in_ch, out_ch, out_size)
        
    def forward(self, x, stage):
        if stage == 1:
            x = self.head1(x)
        elif stage == 2:
            x = self.head2(x)
        elif stage == 3:
            x = self.head3(x)
        else:
            raise RuntimeError("Unknown generate stage.")
        return x
