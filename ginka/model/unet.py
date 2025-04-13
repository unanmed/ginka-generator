import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from shared.attention import ChannelAttention

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
    def __init__(self, in_ch, out_ch, atte=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
        )
        if atte:
            self.conv.append(ChannelAttention(out_ch))
        self.conv.append(nn.ELU())
        
    def forward(self, x):
        return self.conv(x)
    
class GCNBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w, h):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, out_ch)
        self.norm1 = nn.LayerNorm(hidden_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.single_edge_index, _ = grid(h, w)  # [2, E] for a single map

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to [B * H * W, C]
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Construct batched edge index
        device = x.device
        edge_index = self._batch_edge_index(B, self.single_edge_index.to(device), H * W)

        # Batch vector for PyG (not strictly needed for GCNConv, but useful if you switch to GAT/Pooling)
        # batch = torch.arange(B, device=device).repeat_interleave(H * W)

        # GCN forward
        x = self.conv1(x, edge_index)
        x = F.elu(self.norm1(x))
        x = self.conv2(x, edge_index)
        x = F.elu(self.norm2(x))

        # Reshape back to [B, C, H, W]
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        return x

    def _batch_edge_index(self, B, edge_index, num_nodes_per_batch):
        # 批次偏移 edge_index
        edge_index = edge_index.clone()  # [2, E]
        batch_edge_index = []
        for i in range(B):
            offset = i * num_nodes_per_batch
            batch_edge_index.append(edge_index + offset)
        return torch.cat(batch_edge_index, dim=1)
    
class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU()
        )
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

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
    
class GinkaGCNFusedEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, w, h):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.gcn = GCNBlock(out_ch, out_ch*2, out_ch, w, h)
        self.pool = nn.MaxPool2d(2)
        self.fusion = FusionModule(out_ch*2, out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x2 = self.gcn(x)
        x = self.fusion(x, x2)
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
    
class GinkaGCNFusedDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, w, h):
        super().__init__()
        self.upsample = GinkaUpSample(in_ch, in_ch // 2)
        self.conv = ConvBlock(in_ch, out_ch)
        self.gcn = GCNBlock(out_ch, out_ch*2, out_ch, w, h)
        self.fusion = FusionModule(out_ch*2, out_ch)
        
    def forward(self, x, feat):
        x = self.upsample(x)
        x = torch.cat([x, feat], dim=1)
        x = self.conv(x)
        x2 = self.gcn(x)
        x = self.fusion(x, x2)
        return x
    
class GinkaBottleneck(nn.Module):
    def __init__(self, module_ch, w, h):
        super().__init__()
        self.transformer = GinkaTransformerEncoder(
            in_dim=module_ch*w*h, hidden_dim=module_ch*w*h, out_dim=module_ch*w*h,
            token_size=16, ff_dim=1024, num_layers=4
        )
        self.gcn = GCNBlock(module_ch, module_ch*2, module_ch, 4, 4)
        self.fusion = FusionModule(module_ch*2, module_ch)
        
    def forward(self, x):
        B = x.size(0)
        
        x1 = x.view(B, 512, 16).permute(0, 2, 1) # [B, 16, in_ch]
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1).view(B, 512, 4, 4) # [B, out_ch, 4, 4]
        x2 = self.gcn(x)
        
        x = self.fusion(x1, x2)
        
        return x

class GinkaUNet(nn.Module):
    def __init__(self, in_ch=32, base_ch=64, out_ch=32):
        """Ginka Model UNet 部分
        """
        super().__init__()
        # self.input = GinkaTransformerEncoder(
        #     in_dim=feat_dim, hidden_dim=feat_dim*2, out_dim=2*32*32, # 自动除以 token_size
        #     token_size=4, ff_dim=feat_dim*2, num_layers=4
        # )
        self.down1 = ConvBlock(in_ch, base_ch)
        self.down2 = GinkaGCNFusedEncoder(base_ch, base_ch*2, 16, 16)
        self.down3 = GinkaGCNFusedEncoder(base_ch*2, base_ch*4, 8, 8)
        self.down4 = GinkaEncoder(base_ch*4, base_ch*8)
        self.bottleneck = GinkaBottleneck(base_ch*8, 4, 4)
        
        self.up1 = GinkaGCNFusedDecoder(base_ch*8, base_ch*4, 8, 8)
        self.up2 = GinkaGCNFusedDecoder(base_ch*4, base_ch*2, 16, 16)
        self.up3 = GinkaGCNFusedDecoder(base_ch*2, base_ch, 32, 32)

        self.final = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        x1 = self.down1(x) # [B, 64, 32, 32]
        x2 = self.down2(x1) # [B, 128, 16, 16]
        x3 = self.down3(x2) # [B, 256, 8, 8]
        x4 = self.down4(x3) # [B, 512, 4, 4]
        x4 = self.bottleneck(x4) # [B, 512, 4, 4]
        
        # 上采样
        x = self.up1(x4, x3) # [B, 256, 8, 8]
        x = self.up2(x, x2) # [B, 128, 16, 16]
        x = self.up3(x, x1) # [B, 64, 32, 32]
        x = self.final(x) # [B, 32, 32, 32]
        
        return x
