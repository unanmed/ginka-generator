import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class MinamoSimilarityVision(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, 3, padding=1),
            nn.InstanceNorm2d(in_ch * 2),
            nn.ReLU(),
            
            nn.Conv2d(in_ch * 2, in_ch * 4, 3, padding=1),
            nn.InstanceNorm2d(in_ch * 4),
            nn.ReLU(),
            
            nn.Conv2d(in_ch * 4, in_ch * 8, 3),
            nn.InstanceNorm2d(in_ch * 8),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 8, out_ch),
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class MinamoSimilarityTopo(nn.Module):
    def __init__(self, in_ch, hidden_dim, out_ch):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim*2)
        self.conv2 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.conv3 = GCNConv(hidden_dim*4, hidden_dim*8)
        
        self.norm1 = nn.LayerNorm(hidden_dim*2)
        self.norm2 = nn.LayerNorm(hidden_dim*4)
        self.norm3 = nn.LayerNorm(hidden_dim*8)
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim*8, out_ch)
        )
        
    def forward(self, graph: Data):
        x = self.input_fc(graph.x)
        
        x = self.conv1(x, graph.edge_index)
        x = F.relu(self.norm1(x))
        
        x = self.conv2(x, graph.edge_index)
        x = F.relu(self.norm2(x))
        
        x = self.conv3(x, graph.edge_index)
        x = F.relu(self.norm3(x))
        
        x = global_mean_pool(x, graph.batch)
        x = self.output_fc(x)
        
        return x

class MinamoSimilarityModel(nn.Module):
    def __init__(self, tile_type=32):
        super().__init__()
        self.vision = MinamoSimilarityVision(tile_type, 512)
        self.topo = MinamoSimilarityTopo(tile_type, 64, 512)
        
    def forward(self, x, graph):
        vis_feat = self.vision(x)
        topo_feat = self.topo(graph)
        return vis_feat, topo_feat
        