import torch
import torch.nn as nn

class GinkaOutput(nn.Module):
    def __init__(self, out_ch=32, base_ch=64, out_size=(13, 13)):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.AdaptiveMaxPool2d(out_size),
            nn.Conv2d(base_ch, out_ch, 1)
        )
        
    def forward(self, x):
        return self.conv_down(x)
