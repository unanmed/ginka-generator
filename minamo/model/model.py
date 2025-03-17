import torch.nn as nn
import torch.nn.functional as F
from .vision import MinamoVisionModel
from .topo import MinamoTopoModel

class MinamoModel(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        # 视觉相似度部分
        self.vision_model = MinamoVisionModel(tile_types)
        # 拓扑相似度部分
        self.topo_model = MinamoTopoModel(tile_types)

    def forward(self, map1, map2, graph1, graph2):
        vision_feat1 = self.vision_model(map1)
        vision_feat2 = self.vision_model(map2)
        
        topo_feat1 = self.topo_model(graph1)
        topo_feat2 = self.topo_model(graph2)
        
        vision_sim = F.cosine_similarity(vision_feat1, vision_feat2, -1).unsqueeze(-1)
        topo_sim = F.cosine_similarity(topo_feat1, topo_feat2, -1).unsqueeze(-1)
        
        return vision_sim, topo_sim
