import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet

class GumbelSoftmax(nn.Module):
    def __init__(self, tau=1.0, hard=True):
        super().__init__()
        self.tau = tau  # 温度参数
        self.hard = hard  # 是否生成硬性one-hot
    
    def forward(self, logits):
        # logits形状: [BS, C, H, W]
        y = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard)
        
        # 转换为类索引的连续表示
        class_indices = torch.arange(y.size(1), device=y.device).view(1, -1, 1, 1)
        return (y * class_indices).sum(dim=1)  # 形状[BS, H, W]

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=256, base_ch=64, num_classes=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 32 * 32 * base_ch)
        )
        self.unet = GinkaUNet(base_ch, num_classes)
        self.softmax = GumbelSoftmax()
        
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
        x = F.interpolate(x, (13, 13), mode='bilinear', align_corners=False)
        return x, self.softmax(x)
    