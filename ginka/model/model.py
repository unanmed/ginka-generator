import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .output import GinkaOutput

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=1024, base_ch=64, num_classes=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.unet = GinkaUNet(1, base_ch, num_classes, feat_dim)
        self.output = GinkaOutput(num_classes, (13, 13))
        
    def forward(self, x, feat):
        """
        Args:
            feat: 参考地图的特征向量
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        x = self.unet(x, feat)
        x = self.output(x)
        return x, F.softmax(x, dim=1)
    
# 检查显存占用
if __name__ == "__main__":
    x = torch.randn((1, 1, 32, 32)).cuda()
    feat = torch.randn((1, 1024)).cuda()
    
    # 初始化模型
    model = GinkaModel().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output, output_softmax = model(x, feat)
    
    print_memory("前向传播后")
    
    print(f"输入形状: x={x.shape}, feat={feat.shape}")
    print(f"输出形状: output={output.shape}, softmax={output_softmax.shape}")
    print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    