import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.attention import ChannelAttention
from ..common.common import GCNBlock, TransformerGCNBlock, DoubleConvBlock, ConvFusionModule
from ..common.cond import ConditionInjector

class GinkaTransformerEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, token_size, ff_dim, num_heads=8, num_layers=6):
        super().__init__()
        in_dim = in_dim // token_size
        hidden_dim = hidden_dim // token_size
        out_dim = out_dim // token_size
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, token_size, hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=ff_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        # 输入 [B, L, in_dim]
        # 输出 [B, L, out_dim]
        x = self.embedding(x) # [B, L, hidden_dim]
        x = x + self.pos_embedding # [B, L, hidden_dim]
        x = self.transformer(x) # [B, L, hidden_dim]
        x = self.fc(x) # [B, L, out_dim]
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attn=True):
        super().__init__()
        self.conv = DoubleConvBlock([in_ch, out_ch, out_ch])
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
        #     nn.InstanceNorm2d(out_ch),
        #     nn.ELU(),
        #     nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
        #     nn.InstanceNorm2d(out_ch),
        # )
        # if attn:
        #     self.conv.append(ChannelAttention(out_ch))
        # self.conv.append(nn.ELU())
        
    def forward(self, x):
        return self.conv(x)
    
class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate')
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    
class GinkaUNetInput(nn.Module):
    def __init__(self, in_ch, out_ch, w, h):
        super().__init__()
        self.conv = ConvFusionModule(in_ch, out_ch, out_ch, w, h)
        self.inject = ConditionInjector(256, out_ch)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.inject(x, cond)
        return x
    
class GinkaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, w, h):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvFusionModule(in_ch, out_ch, out_ch, w, h)
        self.inject = ConditionInjector(256, out_ch)

    def forward(self, x, cond):
        x = self.pool(x)
        x = self.conv(x)
        x = self.inject(x, cond)
        return x
    
class GinkaUpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.conv(x)
    
class GinkaDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, w, h):
        super().__init__()
        self.upsample = GinkaUpSample(in_ch, in_ch // 2)
        self.fusion = nn.Conv2d(in_ch, in_ch, 1)
        self.conv = ConvFusionModule(in_ch, out_ch, out_ch, w, h)
        self.inject = ConditionInjector(256, out_ch)
        
    def forward(self, x, feat, cond):
        x = self.upsample(x)
        x = torch.cat([x, feat], dim=1)
        x = self.fusion(x)
        x = self.conv(x)
        x = self.inject(x, cond)
        return x
    
class GinkaBottleneck(nn.Module):
    def __init__(self, module_ch, w, h):
        super().__init__()
        # self.transformer = GinkaTransformerEncoder(
        #     in_dim=module_ch*w*h, hidden_dim=module_ch*w*h, out_dim=module_ch*w*h,
        #     token_size=16, ff_dim=1024, num_layers=4
        # )
        # self.gcn = TransformerGCNBlock(module_ch, module_ch*2, module_ch, 4, 4)
        # self.fusion = nn.Conv2d(module_ch*3, module_ch, 1)
        self.conv = ConvFusionModule(module_ch, module_ch, module_ch, w, h)
        self.inject = ConditionInjector(256, module_ch)
        
    def forward(self, x, cond):
        # x1 = x.view(B, 512, 16).permute(0, 2, 1) # [B, 16, in_ch]
        # x1 = self.transformer(x1)
        # x1 = x1.permute(0, 2, 1).view(B, 512, 4, 4) # [B, out_ch, 4, 4]
        x = self.conv(x)
        x = self.inject(x, cond)
        return x
    
class GinkaEncoderPath(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.down1 = GinkaUNetInput(in_ch, base_ch, 32, 32)
        self.down2 = GinkaEncoder(base_ch, base_ch*2, 16, 16)
        self.down3 = GinkaEncoder(base_ch*2, base_ch*4, 8, 8)
        self.down4 = GinkaEncoder(base_ch*4, base_ch*8, 4, 4)
        
    def forward(self, x, cond):
        x1 = self.down1(x, cond) # [B, 64, 32, 32]
        x2 = self.down2(x1, cond) # [B, 128, 16, 16]
        x3 = self.down3(x2, cond) # [B, 256, 8, 8]
        x4 = self.down4(x3, cond) # [B, 512, 4, 4]
        
        return x1, x2, x3, x4
    
class GinkaDecoderPath(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.up1 = GinkaDecoder(base_ch*8, base_ch*4, 8, 8)
        self.up2 = GinkaDecoder(base_ch*4, base_ch*2, 16, 16)
        self.up3 = GinkaDecoder(base_ch*2, base_ch, 32, 32)
        
    def forward(self, x1, x2, x3, x4, cond):
        x = self.up1(x4, x3, cond) # [B, 256, 8, 8]
        x = self.up2(x, x2, cond) # [B, 128, 16, 16]
        x = self.up3(x, x1, cond) # [B, 64, 32, 32]
        return x

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=32, base_ch=32, out_ch=32):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.enc = GinkaEncoderPath(in_ch, base_ch)
        self.bottleneck = GinkaBottleneck(base_ch*8, 4, 4)
        self.dec = GinkaDecoderPath(base_ch)

        self.final = ConvFusionModule(base_ch, base_ch, out_ch, 32, 32)
        
    def forward(self, x, cond):
        x1, x2, x3, x4 = self.enc(x, cond)
        x4 = self.bottleneck(x4, cond) # [B, 512, 4, 4]
        x = self.dec(x1, x2, x3, x4, cond)

        x = self.final(x) # [B, 32, 32, 32]
        
        return x
