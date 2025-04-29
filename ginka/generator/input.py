import torch
import torch.nn as nn
from ..common.common import GCNBlock, DoubleConvBlock
from ..common.cond import ConditionInjector

class RandomInputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = DoubleConvBlock([32, 64, 128])
        self.gcn = GCNBlock(32, 128, 128, 32, 32)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(256),
            nn.ELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(128),
            nn.ELU(),
            
            nn.AdaptiveMaxPool2d((13, 13)),
            nn.Conv2d(128, 32, 1),
        )
        self.inject = ConditionInjector(256, 256)
        
    def forward(self, x, cond):
        x_cnn = self.conv(x)
        x_gcn = self.gcn(x)
        x = torch.cat([x_cnn, x_gcn], dim=1)
        x = self.fusion(x)
        x = self.inject(x, cond)
        x = self.out_conv(x)
        return x

class GinkaInput(nn.Module):
    def __init__(self, in_ch=32, out_ch=64, in_size=(13, 13), out_size=(32, 32)):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Sequential(
            nn.Linear(in_size[0] * in_size[1], out_size[0] * out_size[1]),
            nn.LayerNorm(out_size[0] * out_size[1]),
            nn.ELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.ELU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = self.fc(x)
        x = x.view(B, C, self.out_size[0], self.out_size[1])
        x = self.conv(x)
        return x
