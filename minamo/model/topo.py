import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TopKPooling, GATConv
from torch_geometric.data import Data

class MinamoTopoModel(nn.Module):
    def __init__(
        self, tile_types=32, emb_dim=64, hidden_dim=64, out_dim=512, mlp_dim=128
    ):
        super().__init__()
        # 传入 softmax 概率值，直接映射
        self.input_proj = torch.nn.Linear(tile_types, emb_dim)
        # 图卷积层
        self.conv1 = GATConv(emb_dim, hidden_dim*2, heads=8, dropout=0.2)
        self.conv2 = GATConv(hidden_dim*16, hidden_dim*4, heads=4)
        self.conv3 = GATConv(hidden_dim*16, out_dim, concat=False)
        
        # 正则化
        self.norm1 = nn.LayerNorm(hidden_dim*16)
        self.norm2 = nn.LayerNorm(hidden_dim*16)
        self.norm3 = nn.LayerNorm(out_dim)
        
        # 池化层
        self.pool = TopKPooling(out_dim, ratio=0.8)  # 保留80%关键节点
        self.drop = nn.Dropout(0.3)
        
        # 增强MLP
        self.fc = nn.Sequential(
            nn.Linear(out_dim, mlp_dim),
        )
        
    def forward(self, graph: Data):
        x = self.input_proj(graph.x)
        # identity = x
        
        x = self.conv1(x, graph.edge_index)
        x = F.elu(self.norm1(x))
        
        x = self.conv2(x, graph.edge_index)
        x = F.elu(self.norm2(x))
        
        x = self.conv3(x, graph.edge_index)
        x = F.elu(self.norm3(x))
        
        # 分层池化
        x = self.drop(x)
        # x, _, _, batch, _, _ = self.pool(x, graph.edge_index, batch=graph.batch)
        x = global_mean_pool(x, graph.batch)
        
        topo_vec = self.fc(x)
        
        # 增强MLP
        return F.normalize(topo_vec, p=2, dim=-1)
    