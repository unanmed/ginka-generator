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
    def __init__(self, num_tile_types, embedding_dim=64, conv_channels=256):
        super().__init__()
        # 嵌入层处理不同图块类型
        self.embedding = nn.Embedding(num_tile_types, embedding_dim)
        
        # 共享特征提取的卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, conv_channels, 3, padding=1),
            DualAttention(conv_channels),
            DirectionalAttention(),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels),
            
            nn.Conv2d(conv_channels, conv_channels*2, 3, padding=1),
            DualAttention(conv_channels*2),
            DirectionalAttention(),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels*2),
            
            nn.Conv2d(conv_channels*2, conv_channels*4, 3, padding=1),
            DualAttention(conv_channels*4),
            DirectionalAttention(),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels*4),
        )
        
        # 自适应池化处理任意尺寸
        self.pool = nn.ModuleDict({
            'avg': nn.AdaptiveAvgPool2d((1,1)),
            'max': nn.AdaptiveMaxPool2d((1,1))
        })
        
        # 多任务预测头
        head_dim = conv_channels * 4 * 2 * 4  # 2个池化，四个交互项
        self.vision_head = nn.Sequential(
            nn.Linear(head_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.topo_head = nn.Sequential(
            nn.Linear(head_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, map1, map2):
        # 增强特征提取
        def process_map(x):
            x = self.embedding(x).permute(0,3,1,2)
            x = self.conv_layers(x)
            return torch.cat([
                self.pool['avg'](x),
                self.pool['max'](x)
            ], dim=1).flatten(1)
        
        f1 = process_map(map1)
        f2 = process_map(map2)
        
        # 特征融合
        combined = torch.cat([f1, f2, f1-f2, f1*f2], dim=1)  # [B, 256]
        
        # 多任务输出
        vision_sim = self.vision_head(combined)
        topo_sim = self.topo_head(combined)
        
        return vision_sim, topo_sim
