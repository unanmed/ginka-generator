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

    def forward(self, map, graph):
        vision_feat = self.vision_model(map)
        topo_feat = self.topo_model(graph)
        
        return vision_feat, topo_feat
