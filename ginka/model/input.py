import torch
import torch.nn as nn

class RandomInputHead(nn.Module):
    def __init__(self, in_size=(32, 32), out_size=(32, 32)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(32),
            nn.ELU(),
            
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(64),
            nn.ELU(),
            
            nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(128),
            nn.ELU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1),
        )
        
    def forward(self, x):
        x = self.conv(x)
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
