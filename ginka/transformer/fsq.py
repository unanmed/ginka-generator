import torch
import torch.nn as nn

class FSQ(nn.Module):
    def __init__(self, levels=7):
        super().__init__()

        self.levels = levels
        self.scale = (levels - 1) / 2

    def forward(self, z):

        # 限制范围
        z = torch.tanh(z)

        # 量化
        z_q = torch.round(z * self.scale) / self.scale

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q
