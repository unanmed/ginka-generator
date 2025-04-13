import torch
import torch.nn as nn

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
