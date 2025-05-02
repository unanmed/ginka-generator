import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import global_max_pool, GCNConv
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT
from shared.graph import batch_convert_soft_map_to_graph
from .vision import MinamoVisionModel
from .topo import MinamoTopoModel
from ..common.cond import ConditionEncoder

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class CNNHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3)),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveMaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_ch*2*2, 1))
        )
        self.proj = spectral_norm(nn.Linear(256, in_ch*2*2))

    def forward(self, x, cond):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        cond = self.proj(cond)
        proj = torch.sum(x * cond, dim=1, keepdim=True)
        x = self.fc(x) + proj
        return x
    
class GCNHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, in_dim)
        self.proj = spectral_norm(nn.Linear(256, in_dim))
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, 1))
        )
        
    def forward(self, x, graph, cond):
        x = self.gcn(x, graph.edge_index)
        x = F.leaky_relu(x, 0.2)
        x = global_max_pool(x, graph.batch)
        cond = self.proj(cond)
        proj = torch.sum(x * cond, dim=1, keepdim=True)
        x = self.fc(x) + proj
        return x

class MinamoScoreHead(nn.Module):
    def __init__(self, vision_dim, topo_dim):
        super().__init__()
        self.vision_head = CNNHead(vision_dim)
        self.topo_head = GCNHead(topo_dim)
        
    def forward(self, vis, topo, graph, cond):
        vis_score = self.vision_head(vis, cond)
        topo_score = self.topo_head(topo, graph, cond)
        return vis_score, topo_score
    
class MinamoModel(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        self.topo_model = MinamoTopoModel(tile_types)
        self.vision_model = MinamoVisionModel(tile_types)
        self.cond = ConditionEncoder(64, 16, 256, 256)
        # 输出层
        self.head1 = MinamoScoreHead(512, 512)
        self.head2 = MinamoScoreHead(512, 512)
        self.head3 = MinamoScoreHead(512, 512)

    def forward(self, map, graph, stage, tag_cond, val_cond):
        B, D = tag_cond.shape
        stage_tensor = torch.Tensor([stage]).expand(B, 1).to(map.device)
        vision = self.vision_model(map)
        topo = self.topo_model(graph)
        cond = self.cond(tag_cond, val_cond, stage_tensor)
        if stage == 1:
            vision_score, topo_score = self.head1(vision, topo, graph, cond)
        elif stage == 2:
            vision_score, topo_score = self.head2(vision, topo, graph, cond)
        elif stage == 3:
            vision_score, topo_score = self.head3(vision, topo, graph, cond)
        else:
            raise RuntimeError("Unknown critic stage.")
        score = VISION_WEIGHT * vision_score + TOPO_WEIGHT * topo_score
        return score, vision_score, topo_score

# 检查显存占用
if __name__ == "__main__":
    input = torch.randn((1, 32, 13, 13)).cuda()
    tag = torch.rand(1, 64).cuda()
    val = torch.rand(1, 16).cuda()
    
    # 初始化模型
    model = MinamoModel().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output, _, _ = model(input, batch_convert_soft_map_to_graph(input), 1, tag, val)
    
    print_memory("前向传播后")
    
    print(f"输入形状: feat={input.shape}")
    print(f"输出形状: output={output.shape}")
    print(f"Cond parameters: {sum(p.numel() for p in model.cond.parameters())}")
    print(f"Topo parameters: {sum(p.numel() for p in model.topo_model.parameters())}")
    print(f"Vision parameters: {sum(p.numel() for p in model.vision_model.parameters())}")
    print(f"Head parameters: {sum(p.numel() for p in model.head1.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
