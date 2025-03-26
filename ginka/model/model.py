import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .input import GinkaInput
from .output import GinkaOutput

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=1024, base_ch=64, num_classes=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.input = GinkaInput(feat_dim, base_ch)
        self.unet = GinkaUNet(base_ch, num_classes)
        self.output = GinkaOutput(num_classes, (13, 13))
        print(f"Input parameters: {sum(p.numel() for p in self.input.parameters())}")
        print(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters())}")
        print(f"Output parameters: {sum(p.numel() for p in self.output.parameters())}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        
    def forward(self, x):
        """
        Args:
            feat: 参考地图的特征向量
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        x = self.input(x)
        x = self.unet(x)
        x = self.output(x)
        return x, F.softmax(x, dim=1)
    