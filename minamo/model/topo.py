import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, AttentionalAggregation, global_max_pool
from torch_geometric.data import Data

class MinamoTopoModel(nn.Module):
    def __init__(
        self, tile_types=32, emb_dim=128, hidden_dim=128, out_dim=512, mlp_dim=512
    ):
        super().__init__()
        # 传入 softmax 概率值，直接映射
        self.input_proj = nn.Linear(tile_types, emb_dim)
        # 图卷积层
        self.conv1 = GATConv(emb_dim, hidden_dim*2, heads=8, dropout=0.2)
        self.conv2 = GATConv(hidden_dim*16, hidden_dim*2, heads=8)
        self.conv3 = GATConv(hidden_dim*16, hidden_dim*2, heads=8)
        self.conv4 = GATConv(hidden_dim*16, out_dim, heads=1)
        
        # 正则化
        self.norm1 = nn.LayerNorm(hidden_dim*16)
        self.norm2 = nn.LayerNorm(hidden_dim*16)
        self.norm3 = nn.LayerNorm(hidden_dim*16)
        self.norm4 = nn.LayerNorm(out_dim)
        
        self.drop = nn.Dropout(0.3)
        
        # 增强MLP
        self.fc = nn.Sequential(
            nn.Linear(out_dim, mlp_dim),
        )
        
    def forward(self, graph: Data):
        x = self.input_proj(graph.x)
        
        x = self.conv1(x, graph.edge_index)
        x = F.relu(self.norm1(x))
        
        x = self.conv2(x, graph.edge_index)
        x = F.relu(self.norm2(x))
        
        x = self.conv3(x, graph.edge_index)
        x = F.relu(self.norm3(x))
        
        x = self.conv4(x, graph.edge_index)
        x = F.relu(self.norm4(x))
        
        # 池化
        x = self.drop(x)
        x = global_max_pool(x, graph.batch)
        
        topo_vec = self.fc(x)
        
        # 归一化
        return F.normalize(topo_vec, p=2, dim=-1)
    