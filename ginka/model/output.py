import torch
import torch.nn as nn

class GinkaOutput(nn.Module):
    def __init__(self, num_classes=32, out_size=(13, 13)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(out_size)
        
    def forward(self, x):
        return self.pool(x)
