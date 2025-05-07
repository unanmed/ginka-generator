import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import GinkaUNet
from .output import GinkaOutput
from .input import GinkaInput, RandomInputHead
from ..common.cond import ConditionEncoder

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class GinkaModel(nn.Module):
    def __init__(self, base_ch=64, out_ch=32):
        """Ginka Model 模型定义部分
        """
        super().__init__()
        self.head = RandomInputHead()
        self.cond = ConditionEncoder(64, 16, 256, 256)
        self.input = GinkaInput(32, 64, (13, 13), (32, 32))
        self.unet = GinkaUNet(64, base_ch, base_ch)
        self.output = GinkaOutput(base_ch, out_ch, (13, 13))
        
    def forward(self, x, stage, tag_cond, val_cond, random=False):
        B, D = tag_cond.shape
        stage_tensor = torch.Tensor([stage]).expand(B, 1).to(x.device)
        cond = self.cond(tag_cond, val_cond, stage_tensor)
        if random:
            x_in = F.softmax(self.head(x, cond), dim=1)
        else:
            x_in = x
        x = self.input(x_in, cond)
        x = self.unet(x, cond)
        x = self.output(x, stage, cond)
        return x, x_in
    
# 检查显存占用
if __name__ == "__main__":
    input = torch.rand(1, 32, 32, 32).cuda()
    tag = torch.rand(1, 64).cuda()
    val = torch.rand(1, 16).cuda()
    
    # 初始化模型
    model = GinkaModel().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output, _ = model(input, 1, tag, val, True)
    
    print_memory("前向传播后")
    
    print(f"输入形状: feat={input.shape}")
    print(f"输出形状: output={output.shape}")
    print(f"Random parameters: {sum(p.numel() for p in model.head.parameters())}")
    print(f"Cond parameters: {sum(p.numel() for p in model.cond.parameters())}")
    print(f"Input parameters: {sum(p.numel() for p in model.input.parameters())}")
    print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Output parameters: {sum(p.numel() for p in model.output.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    