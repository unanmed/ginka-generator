import torch
import torch.nn as nn

class MapDownSample(nn.Module):
    def __init__(self, in_ch=32, out_ch=32):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0)
        )
        
    def forward(self, x):
        x = self.down(x)
        return x
