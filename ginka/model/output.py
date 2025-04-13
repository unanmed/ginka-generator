import torch
import torch.nn as nn

class StageHead(nn.Module):
    def __init__(self, in_ch, out_ch, out_size=(13, 13)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(in_ch),
            nn.ELU(),
            
            nn.Conv2d(in_ch, in_ch, 1),
            nn.InstanceNorm2d(in_ch),
            nn.ELU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(out_size),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        
    def forward(self, x):
        x = self.head(x)
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
