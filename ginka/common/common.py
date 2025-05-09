import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.utils import grid

def batch_edge_index(B, edge_index, num_nodes_per_batch):
    # 批次偏移 edge_index
    edge_index = edge_index.clone()  # [2, E]
    batch_edge_index = []
    for i in range(B):
        offset = i * num_nodes_per_batch
        batch_edge_index.append(edge_index + offset)
    return torch.cat(batch_edge_index, dim=1)

class DoubleConvBlock(nn.Module):
    def __init__(self, feats: tuple[int, int, int]):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(feats[0], feats[1], 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(feats[1]),
            nn.ELU(),
            
            nn.Conv2d(feats[1], feats[2], 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(feats[2]),
            nn.ELU(),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        return x

class GCNBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w, h):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, hidden_ch)
        self.conv3 = GCNConv(hidden_ch, out_ch)
        self.norm1 = nn.LayerNorm(hidden_ch)
        self.norm2 = nn.LayerNorm(hidden_ch)
        self.norm3 = nn.LayerNorm(out_ch)
        self.single_edge_index, _ = grid(h, w)  # [2, E] for a single map

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to [B * H * W, C]
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Construct batched edge index
        device = x.device
        edge_index = batch_edge_index(B, self.single_edge_index.to(device), H * W)

        # Batch vector for PyG (not strictly needed for GCNConv, but useful if you switch to GAT/Pooling)
        # batch = torch.arange(B, device=device).repeat_interleave(H * W)

        # GCN forward
        x = self.conv1(x, edge_index)
        x = F.elu(self.norm1(x))
        x = self.conv2(x, edge_index)
        x = F.elu(self.norm2(x))
        x = self.conv3(x, edge_index)
        x = F.elu(self.norm3(x))

        # Reshape back to [B, C, H, W]
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        return x
    
class TransformerGCNBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w, h):
        super().__init__()
        self.conv1 = TransformerConv(in_ch, hidden_ch // 8, heads=8, concat=True)
        self.conv2 = TransformerConv(hidden_ch, out_ch, heads=1)
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
        edge_index = batch_edge_index(B, self.single_edge_index.to(device), H * W)

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
    
class ConvFusionModule(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w: int, h: int):
        super().__init__()
        self.cnn = DoubleConvBlock([in_ch, hidden_ch, in_ch])
        self.gcn = GCNBlock(in_ch, hidden_ch, in_ch, w, h)
        self.fusion = DoubleConvBlock([in_ch*2, hidden_ch*2, out_ch])
        
    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.gcn(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion(x)
        return x
    
class DoubleFCModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU()
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
        