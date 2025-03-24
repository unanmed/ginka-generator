import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .sample import MapDownSample

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=1024, base_ch=64, num_classes=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.base_ch = base_ch
        fc_dim = base_ch * 8 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=4, stride=2, padding=1),  # Upsample 2x
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=4, stride=2, padding=1),   # Upsample 2x
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=4, stride=2, padding=1),   # Upsample 2x
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        self.unet = GinkaUNet(base_ch, num_classes)
        self.down_sample = MapDownSample(num_classes, num_classes)
        self.pool = nn.AdaptiveMaxPool2d((13, 13))
        
    def forward(self, feat):
        """
        Args:
            feat: 参考地图的特征向量
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        x = self.fc(feat)
        x = x.view(-1, self.base_ch*8, 4, 4)
        x = self.deconv_layers(x)
        x = self.unet(x)
        x = F.interpolate(x, (13, 13), mode='bilinear')
        return x, F.softmax(x, dim=1)
    