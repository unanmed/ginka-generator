import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        # 通道注意力
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.spatial(x) + x * self.channel(x)
    
class DirectionalAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.direction_convs = nn.ModuleDict({
            dir: nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, 
                           padding_mode='replicate')
            for dir in ['h', 'v', 'd1', 'd2']
        })
        
    def forward(self, x):
        B, C, H, W = x.shape
        # 各方向特征
        h_att = self.direction_convs['h'](x.mean(1, keepdim=True))
        v_att = self.direction_convs['v'](x.mean(1, keepdim=True))
        d1_att = self.direction_convs['d1'](x.mean(1, keepdim=True))
        d2_att = self.direction_convs['d2'](x.mean(1, keepdim=True))
        
        # 动态融合
        combined = torch.stack([h_att, v_att, d1_att, d2_att], dim=1)  # [B,4,1,H,W]
        att_weights = F.softmax(combined.mean([3,4]), dim=1)  # [B,4]
        return x * (combined * att_weights.unsqueeze(-1).unsqueeze(-1)).sum(1)

class MinamoModel(nn.Module):
    def __init__(self, tile_types=32, embedding_dim=64, conv_channels=256):
        super().__init__()
        # 嵌入层处理不同图块类型
        self.embedding = nn.Embedding(tile_types, embedding_dim)
        
        self.vision_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 3, padding=1),
            DualAttention(conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels*2, 3, padding=1),
            DualAttention(conv_channels*2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 拓扑特征分支
        self.topo_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 5, padding=2),  # 更大卷积核捕捉结构
            nn.MaxPool2d(2),
            # GraphConvLayer(128, 256),  # 图卷积层
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 多任务预测头
        self.vision_head = nn.Sequential(
            nn.Linear(conv_channels*2, 1),
            nn.Sigmoid()
        )
        
        self.topo_head = nn.Sequential(
            nn.Linear(conv_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, map1, map2):
        e1 = self.embedding(map1).permute(0, 3, 1, 2)
        e2 = self.embedding(map2).permute(0, 3, 1, 2)
        
        v1 = self.vision_conv(e1).squeeze()
        v2 = self.vision_conv(e2).squeeze()
        
        t1 = self.topo_conv(e1).squeeze()
        t2 = self.topo_conv(e2).squeeze()
        
        # 多任务输出
        vision_sim = self.vision_head(torch.abs(v1 - v2))
        topo_sim = self.topo_head(torch.abs(t1 - t2))
        
        return vision_sim, topo_sim
