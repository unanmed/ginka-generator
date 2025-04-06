import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class MinamoVisionModel(nn.Module):
    def __init__(self, in_ch=32, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch*2, 3, stride=2)), # 6*6
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(in_ch*2, in_ch*4, 3)), #4*4
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(in_ch*4, in_ch*8, 3)), # 2*2
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_ch*8*2*2, out_dim))
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
