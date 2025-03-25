import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MinamoVisionModel(nn.Module):
    def __init__(self, tile_types=32, out_dim=512):
        super().__init__()
        self.resnet = resnet18(num_classes=out_dim)
        self.resnet.conv1 = nn.Conv2d(tile_types, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x):
        vision_vec = self.resnet(x)
        return F.normalize(vision_vec, p=2, dim=-1)  # 归一化
