import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import global_max_pool, GCNConv, TransformerConv
from torch_geometric.utils import grid
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT
from .vision import MinamoVisionModel
from .topo import MinamoTopoModel

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

def batch_edge_index(B, edge_index, num_nodes_per_batch):
    # 批次偏移 edge_index
    edge_index = edge_index.clone()  # [2, E]
    batch_edge_index = []
    for i in range(B):
        offset = i * num_nodes_per_batch
        batch_edge_index.append(edge_index + offset)
    return torch.cat(batch_edge_index, dim=1)

class DoubleConvBlock(nn.Module):
    def __init__(self, feats: tuple[int, int, int]):
        super().__init__()
        self.cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(feats[0], feats[1], 3, padding=1, padding_mode='replicate')),
            nn.GELU(),
            
            spectral_norm(nn.Conv2d(feats[1], feats[2], 3, padding=1, padding_mode='replicate')),
            nn.GELU(),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        return x

class TransformerGCNBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w, h):
        super().__init__()
        self.conv1 = TransformerConv(in_ch, hidden_ch // 8, heads=8, concat=True)
        self.conv2 = TransformerConv(hidden_ch, out_ch, heads=1)
        self.single_edge_index, _ = grid(h, w)  # [2, E] for a single map

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        device = x.device
        edge_index = batch_edge_index(B, self.single_edge_index.to(device), H * W)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        return x
    
class ConvFusionModule(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, w: int, h: int):
        super().__init__()
        self.cnn = DoubleConvBlock([in_ch, hidden_ch, in_ch])
        self.gcn = TransformerGCNBlock(in_ch, hidden_ch, in_ch, w, h)
        self.fusion = DoubleConvBlock([in_ch*2, hidden_ch, out_ch])
        
    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.gcn(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fusion(x)
        return x
    
class DoubleFCModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.GELU(),
            
            spectral_norm(nn.Linear(hidden_dim, out_dim)),
            nn.GELU()
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class ConditionEncoder(nn.Module):
    def __init__(self, tag_dim, val_dim, hidden_dim, out_dim):
        super().__init__()
        self.tag_embed = DoubleFCModule(tag_dim, hidden_dim, hidden_dim)
        self.val_embed = DoubleFCModule(val_dim, hidden_dim, hidden_dim)
        self.stage_embed = DoubleFCModule(1, hidden_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=4
        )
        self.fusion = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            
            spectral_norm(nn.Linear(hidden_dim, out_dim))
        )
        
    def forward(self, tag, val, stage):
        tag = self.tag_embed(tag)
        val = self.val_embed(val)
        stage = self.stage_embed(stage)
        feat = torch.stack([tag, val, stage], dim=1)
        feat = self.encoder(feat)
        feat = torch.mean(feat, dim=1)
        feat = self.fusion(feat)
        return feat
    
class ConditionInjector(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.gamma_layer = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim, out_dim))
        )
        self.beta_layer = nn.Sequential(
            spectral_norm(nn.Linear(cond_dim, out_dim))
        )

    def forward(self, x, cond):
        gamma = self.gamma_layer(cond).unsqueeze(2).unsqueeze(3)
        beta = self.beta_layer(cond).unsqueeze(2).unsqueeze(3)
        return x * gamma + beta

class CNNHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_ch, 3)),
            nn.GELU(),
            
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
        x = F.gelu(x)
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
    
class MinamoHead2(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.conv = ConvFusionModule(in_ch, hidden_ch, hidden_ch, 13, 13)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.proj = spectral_norm(nn.Linear(256, hidden_ch))
        self.fc = spectral_norm(nn.Linear(hidden_ch, 1))
        
    def forward(self, x, cond):
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(3).squeeze(2)
        cond = self.proj(cond)
        proj = torch.sum(x * cond, dim=1, keepdim=True)
        x = self.fc(x) + proj
        return x
    
class MinamoModel2(nn.Module):
    def __init__(self, tile_types=32):
        super().__init__()
        self.cond = ConditionEncoder(64, 16, 256, 256)
        
        self.conv1 = ConvFusionModule(tile_types, 256, 256, 13, 13)
        self.conv2 = ConvFusionModule(256, 512, 256, 13, 13)
        self.conv3 = ConvFusionModule(256, 512, 256, 13, 13)
        
        self.head0 = MinamoHead2(256, 256) # 随机头的判别头
        self.head1 = MinamoHead2(256, 256)
        self.head2 = MinamoHead2(256, 256)
        self.head3 = MinamoHead2(256, 256)
        
        # self.inject1 = ConditionInjector(256, 256)
        # self.inject2 = ConditionInjector(256, 256)
        self.inject3 = ConditionInjector(256, 256)
        
    def forward(self, x, stage, tag_cond, val_cond):
        B, D = tag_cond.shape
        stage_tensor = torch.Tensor([stage]).expand(B, 1).to(x.device)
        cond = self.cond(tag_cond, val_cond, stage_tensor)
        x = self.conv1(x)
        # x = self.inject1(x, cond)
        x = self.conv2(x)
        # x = self.inject2(x, cond)
        x = self.conv3(x)
        x = self.inject3(x, cond)
        
        if stage == 0:
            score = self.head0(x, cond)
        elif stage == 1:
            score = self.head1(x, cond)
        elif stage == 2:
            score = self.head2(x, cond)
        elif stage == 3:
            score = self.head3(x, cond)
        else:
            raise RuntimeError("Unknown critic stage.")
        
        return score

# 检查显存占用
if __name__ == "__main__":
    input = torch.randn((1, 32, 13, 13)).cuda()
    tag = torch.rand(1, 64).cuda()
    val = torch.rand(1, 16).cuda()
    
    # 初始化模型
    model = MinamoModel2().cuda()
    
    print_memory("初始化后")
    
    # 前向传播
    output = model(input, 1, tag, val)
    
    print_memory("前向传播后")
    
    print(f"输入形状: feat={input.shape}")
    print(f"输出形状: output={output.shape}")
    # print(f"Vision parameters: {sum(p.numel() for p in model.vision_model.parameters())}")
    # print(f"Topo parameters: {sum(p.numel() for p in model.topo_model.parameters())}")
    print(f"Cond parameters: {sum(p.numel() for p in model.cond.parameters())}")
    print(f"Head parameters: {sum(p.numel() for p in model.head1.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
