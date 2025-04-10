import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .output import GinkaOutput
from .input import GinkaInput

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class GinkaModel(nn.Module):
    def __init__(self, feat_dim=1024, base_ch=64, out_ch=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.unet = GinkaUNet(base_ch, base_ch, feat_dim)
        self.output = GinkaOutput(base_ch, out_ch, (13, 13))
        
    def forward(self, x):
        """
        Args:
            x: 参考地图的特征向量
        Returns:
            logits: 输出logits [BS, num_classes, H, W]
        """
        x = self.unet(x)
        x = self.output(x)
        return F.softmax(x, dim=1)
    
# 检查显存占用
if __name__ == "__main__":
    feat = torch.randn((1, 1024)).cuda()
    
    # 初始化模型
    model = GinkaModel().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output = model(feat)
    
    print_memory("前向传播后")
    
    print(f"输入形状: feat={feat.shape}")
    print(f"输出形状: output={output.shape}")
    # print(f"Input parameters: {sum(p.numel() for p in model.input.parameters())}")
    # print(f"Feature Encoder parameters: {sum(p.numel() for p in model.feat_enc.parameters())}")
    print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    