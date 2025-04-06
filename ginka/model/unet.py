import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.constant import FEAT_DIM

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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        return self.conv(x)

class GinkaEncoder(nn.Module):
    """编码器（下采样）部分"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class GinkaUpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class GinkaDecoder(nn.Module):
    """解码器（上采样）部分"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = GinkaUpSample(in_ch, in_ch // 2)
        self.conv = ConvBlock(in_ch, out_ch)
        
    def forward(self, x, feat):
        x = self.upsample(x)
        x = torch.cat([x, feat], dim=1)
        x = self.conv(x)
        return x

class GinkaUNet(nn.Module):
    def __init__(self, base_ch=64, out_ch=32, feat_dim=1024):
        """Ginka Model UNet 部分
        """
        super().__init__()
        self.input = GinkaTransformerEncoder(
            in_dim=feat_dim, hidden_dim=feat_dim*2, out_dim=2*32*32, # 自动除以 token_size
            token_size=4, ff_dim=feat_dim*2, num_layers=4
        )
        self.down1 = ConvBlock(2, base_ch)
        self.down2 = GinkaEncoder(base_ch, base_ch*2)
        self.down3 = GinkaEncoder(base_ch*2, base_ch*4)
        self.down4 = GinkaEncoder(base_ch*4, base_ch*8)
        self.bottleneck = GinkaTransformerEncoder(
            in_dim=base_ch*8*4*4, hidden_dim=base_ch*8*4*4, out_dim=base_ch*8*4*4,
            token_size=16, ff_dim=1024, num_layers=4
        )
        
        self.up1 = GinkaDecoder(base_ch*8, base_ch*4)
        self.up2 = GinkaDecoder(base_ch*4, base_ch*2)
        self.up3 = GinkaDecoder(base_ch*2, base_ch)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        B, D = x.shape # [B, 1024]
        x = x.view(B, 4, D // 4) # [B, 4, 256]
        x = self.input(x) # [B, 4, 512]
        x = x.view(B, 2, 32, 32) # [B, 2, 32, 32]
        x1 = self.down1(x) # [B, 64, 32, 32]
        x2 = self.down2(x1) # [B, 128, 16, 16]
        x3 = self.down3(x2) # [B, 256, 8, 8]
        x4 = self.down4(x3) # [B, 512, 4, 4]
        x4 = x4.view(B, 512, 16).permute(0, 2, 1) # [B, 16, 512]
        x4 = self.bottleneck(x4) # [B, 16, 512]
        x4 = x4.permute(0, 2, 1).view(B, 512, 4, 4) # [B, 512, 4, 4]
        
        # 上采样
        x = self.up1(x4, x3) # [B, 256, 8, 8]
        x = self.up2(x, x2) # [B, 128, 16, 16]
        x = self.up3(x, x1) # [B, 64, 32, 32]
        
        return self.final(x) # [B, 32, 32, 32]
