import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class GinkaInput(nn.Module):
    def __init__(self, feat_dim=1024, out_ch=64):
        super().__init__()
        fc_dim = out_ch * 8 * 4 * 4
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            ResidualUpsampleBlock(out_ch*8, out_ch*8),
            ResidualUpsampleBlock(out_ch*8, out_ch*4),
            ResidualUpsampleBlock(out_ch*4, out_ch)
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.out_ch*8, 4, 4)
        x = self.upsample(x)
        return x
