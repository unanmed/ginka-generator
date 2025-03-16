import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class MinamoTopoModel(nn.Module):
    def __init__(
        self, tile_types=32, emb_dim=16, hidden_dim=32, out_dim=16, mlp_dim=8
    ):
        super().__init__()
        # 嵌入层
        self.embedding = torch.nn.Embedding(tile_types, emb_dim)
        # 图卷积层
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.fc = torch.nn.Linear(out_dim, mlp_dim)  # 降维全连接层
        
    def forward(self, graph: Data):
        x = self.embedding(graph.x)
        x = self.conv1(x, graph.edge_index)
        x = F.relu(x)
        x = self.conv2(x, graph.edge_index)
        x = global_mean_pool(x, graph.batch)

        # 全连接层降维
        x = self.fc(x)
        return x  # (batch_size, mlp_dim)
    