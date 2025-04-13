import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .vision import MinamoVisionModel
from .topo import MinamoTopoModel
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT

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
    
class MinamoScoreHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.vision_fc = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, out_dim)),
        )
        self.topo_fc = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, out_dim))
        )
        
    def forward(self, vis_feat, topo_feat):
        vis_score = self.vision_fc(vis_feat)
        topo_score = self.topo_fc(topo_feat)
        return vis_score, topo_score
    
class MinamoScoreModule(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        self.topo_model = MinamoTopoModel(tile_types)
        self.vision_model = MinamoVisionModel(tile_types)
        # 输出层
        self.head1 = MinamoScoreHead(512, 1)
        self.head2 = MinamoScoreHead(512, 1)
        self.head3 = MinamoScoreHead(512, 1)

    def forward(self, map, graph, stage):
        vision_feat = self.vision_model(map)
        topo_feat = self.topo_model(graph)
        if stage == 1:
            vision_score, topo_score = self.head1(vision_feat, topo_feat)
        elif stage == 2:
            vision_score, topo_score = self.head2(vision_feat, topo_feat)
        elif stage == 3:
            vision_score, topo_score = self.head3(vision_feat, topo_feat)
        else:
            raise RuntimeError("Unknown critic stage.")
        score = VISION_WEIGHT * vision_score + TOPO_WEIGHT * topo_score
        return score, vision_score, topo_score
