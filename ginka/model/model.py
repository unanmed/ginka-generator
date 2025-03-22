import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .sample import MapDownSample

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=256, base_ch=32, num_classes=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 32 * 32 * base_ch)
        )
        self.unet = GinkaUNet(base_ch, num_classes)
        self.down_sample = MapDownSample(num_classes, num_classes)
        
    def forward(self, feat):
        """
        Args:
            feat: 参考地图的特征向量
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        x = self.fc(feat)
        x = x.view(-1, self.base_ch, 32, 32)
        x = self.unet(x)
        x = self.down_sample(x)
        return F.softmax(x, dim=1)
    