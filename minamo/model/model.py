import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import global_max_pool, GCNConv, global_mean_pool
from .vision import MinamoVisionModel
from .topo import MinamoTopoModel
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT
from shared.graph import batch_convert_soft_map_to_graph

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

class MinamoModel(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        # 视觉相似度部分
        self.vision_model = MinamoVisionModel(tile_types)
        # 拓扑相似度部分
        self.topo_model = MinamoTopoModel(tile_types)

    def forward(self, map, graph):
        vision_feat = self.vision_model(map)
        topo_feat = self.topo_model(graph)
        
        return vision_feat, topo_feat

class CNNHead(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3)),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveMaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_ch*2*2, out_dim))
        )

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = self.fc(x)
        return x
    
class GCNHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, in_dim)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, out_dim))
        )
        
    def forward(self, x, graph):
        x = self.gcn(x, graph.edge_index)
        x = F.leaky_relu(x, 0.2)
        x = global_max_pool(x, graph.batch)
        x = self.fc(x)
        return x

class MinamoScoreHead(nn.Module):
    def __init__(self, vision_dim, topo_dim, out_dim):
        super().__init__()
        self.vision_head = CNNHead(vision_dim, out_dim)
        self.topo_head = GCNHead(topo_dim, out_dim)
        
    def forward(self, vis, topo, graph):
        vis_score = self.vision_head(vis)
        topo_score = self.topo_head(topo, graph)
        return vis_score, topo_score
    
class MinamoScoreModule(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        self.topo_model = MinamoTopoModel(tile_types)
        self.vision_model = MinamoVisionModel(tile_types)
        # 输出层
        self.head1 = MinamoScoreHead(512, 512, 1)
        self.head2 = MinamoScoreHead(512, 512, 1)
        self.head3 = MinamoScoreHead(512, 512, 1)

    def forward(self, map, graph, stage):
        vision = self.vision_model(map)
        topo = self.topo_model(graph)
        if stage == 1:
            vision_score, topo_score = self.head1(vision, topo, graph)
        elif stage == 2:
            vision_score, topo_score = self.head2(vision, topo, graph)
        elif stage == 3:
            vision_score, topo_score = self.head3(vision, topo, graph)
        else:
            raise RuntimeError("Unknown critic stage.")
        score = VISION_WEIGHT * vision_score + TOPO_WEIGHT * topo_score
        return score, vision_score, topo_score

# 检查显存占用
if __name__ == "__main__":
    input = torch.randn((1, 32, 13, 13)).cuda()
    
    # 初始化模型
    model = MinamoScoreModule().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output, _, _ = model(input, batch_convert_soft_map_to_graph(input), 1)
    
    print_memory("前向传播后")
    
    print(f"输入形状: feat={input.shape}")
    print(f"输出形状: output={output.shape}")
    print(f"Topo parameters: {sum(p.numel() for p in model.topo_model.parameters())}")
    print(f"Vision parameters: {sum(p.numel() for p in model.vision_model.parameters())}")
    print(f"Head parameters: {sum(p.numel() for p in model.head1.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
