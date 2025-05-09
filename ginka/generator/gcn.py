import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.utils import grid
from ..common.cond import ConditionInjector

# 考虑使用 GCN 作为生成器主路径，暂时先留着

class GCNBlock(nn.Module):
    def __init__(self, feats: tuple[int, int, int]):
        super().__init__()
        self.conv1 = GCNConv(feats[0], feats[1])
        self.conv2 = GCNConv(feats[1], feats[2])
        
        self.norm1 = nn.LayerNorm(feats[1])
        self.norm2 = nn.LayerNorm(feats[2])
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(self.norm1(x))
        
        x = self.conv2(x, edge_index)
        x = F.elu(self.norm2(x))
        return x

class GinkaGCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
class GinkaGCNDecoder(nn.Module):
    def __init__(self):
        super().__init__()

class GinkaGCNModel(nn.Module):
    def __init__(self):
        super().__init__()