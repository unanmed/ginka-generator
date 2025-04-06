import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GATConv, global_max_pool, GCNConv, global_mean_pool
from torch_geometric.data import Data

class MinamoTopoModel(nn.Module):
    def __init__(
        self, tile_types=32, emb_dim=128, hidden_dim=256, out_dim=512, feat_dim=512
    ):
        super().__init__()
        # 传入 softmax 概率值，直接映射
        self.input_proj = nn.Sequential(
            nn.Linear(tile_types, emb_dim),
            nn.LeakyReLU(0.2)
        )
        # 图卷积层
        self.conv1 = GATConv(emb_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim*8, hidden_dim, heads=8)
        self.conv3 = GATConv(hidden_dim*8, out_dim, heads=1)
        
        # self.norm1 = nn.LayerNorm(hidden_dim*8)
        # self.norm2 = nn.LayerNorm(hidden_dim*8)
        # self.norm3 = nn.LayerNorm(out_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(out_dim, feat_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, graph: Data):
        x = self.input_proj(graph.x)
        
        x = self.conv1(x, graph.edge_index)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x, graph.edge_index)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv3(x, graph.edge_index)
        x = F.leaky_relu(x, 0.2)
        
        # 池化
        x = global_mean_pool(x, graph.batch)
        
        topo_vec = self.fc(x)
        
        return topo_vec
    