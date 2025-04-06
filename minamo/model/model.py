import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
class MinamoVisionScore(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        # 视觉相似度部分
        self.vision_model = MinamoVisionModel(tile_types)
        # 输出层
        self.vision_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1)
        )

    def forward(self, map):
        vision_feat = self.vision_model(map)
        vis_score = self.vision_fc(vision_feat)
        return vis_score
    
class MinamoTopoScore(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        # 拓扑相似度部分
        self.topo_model = MinamoTopoModel(tile_types)
        # 输出层
        self.topo_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1)
        )

    def forward(self, graph):
        topo_feat = self.topo_model(graph)
        topo_score = self.topo_fc(topo_feat)
        return topo_score
    
class MinamoScoreModule(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        self.topo_model = MinamoTopoModel(tile_types)
        self.vision_model = MinamoVisionModel(tile_types)
        # 输出层
        self.topo_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1)
        )
        self.vision_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1)
        )

    def forward(self, map, graph):
        topo_feat = self.topo_model(graph)
        topo_score = self.topo_fc(topo_feat)
        vision_feat = self.vision_model(map)
        vision_score = self.vision_fc(vision_feat)
        score = VISION_WEIGHT * vision_score + TOPO_WEIGHT * topo_score
        return score, vision_score, topo_score
