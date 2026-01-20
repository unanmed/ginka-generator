import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import print_memory
    
class GinkaMapPatch(nn.Module):
    def __init__(self, tile_classes=32, width=13, height=13):
        super().__init__()
        
        # 地图局部卷积，用于捕获局部结构信息
        
        self.width = width
        self.height = height
        self.tile_classes = 32
        
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(tile_classes + 1, 64, 3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, 3),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 3 * 3, 256)
        
    def forward(self, map: torch.Tensor, x: int, y: int):
        """
        map: [B, H, W]
        """
        B, H, W = map.shape
        mask = torch.zeros([B, 5, 5]).to(map.device)
        result = torch.zeros([B, 5, 5], dtype=torch.long).to(map.device)
        left = x - 2  if x >= 2             else 0
        right = x + 3 if x < self.width - 2 else self.width
        top = y - 4   if y >= 4             else 0
        bottom = y + 1
        
        res_left = left - (x - 2)
        res_right = right - (x + 3) + 5
        res_top = top - (y - 4)
        res_bottom = 5
        
        result[:, res_top:res_bottom, res_left:res_right] = map[:, top:bottom, left:right]
        # 没画到的地方要置为 0
        result[:, 4, 2] = 0
        result[:, 4, 3] = 0
        result[:, 4, 4] = 0
        mask[:, res_top:res_bottom, res_left:res_right] = 1
        mask[:, 4, 2] = 0
        mask[:, 4, 3] = 0
        mask[:, 4, 4] = 0
        masked_result = torch.zeros([B, self.tile_classes + 1, 5, 5]).to(map.device)
        masked_result[:, 0:32] = F.one_hot(result, num_classes=32).permute(0, 3, 2, 1).float()
        masked_result[:, 32] = mask
        
        feat = self.patch_cnn(masked_result)
        feat = self.fc(feat)
        return feat
    
class GinkaTileEmbedding(nn.Module):
    def __init__(self, tile_classes=32, embed_dim=256):
        super().__init__()
        
        # 图块编码，上一次画的图块
        
        self.embedding = nn.Embedding(tile_classes, embed_dim)
        
    def forward(self, tile: torch.Tensor):
        return self.embedding(tile)
    
class GinkaPosEmbedding(nn.Module):
    def __init__(self, width=13, height=13, embed_dim=256):
        super().__init__()
        
        # 位置编码
        
        self.width = width
        self.height = height
        
        self.row_embedding = nn.Embedding(height, embed_dim)
        self.col_embedding = nn.Embedding(width, embed_dim)
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        row = self.row_embedding(y)
        col = self.col_embedding(x)
        embed = torch.cat([row, col], dim=2)
        fused = self.fusion(embed)
        
        return fused
    
class GinkaInputFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        # 使用 Transformer 进行信息整合
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=d_model, batch_first=True,
                dropout=0.2
            ),
            num_layers=3
        )
        
    def forward(
        self, tile_embed: torch.Tensor, cond_vec: torch.Tensor, 
        pos_embed: torch.Tensor, patch_vec: torch.Tensor
    ):
        """
        tile_embed: [B, 256]
        cond_vec: [B, 256]
        pos_embed: [B, 256]
        patch_vec: [B, 256]
        """
        vec = torch.stack([tile_embed, cond_vec, pos_embed, patch_vec], dim=1)
        feat = self.transformer(vec)
        return feat[:, 0]

class GinkaRNN(nn.Module):
    def __init__(self, tile_classes=32, input_dim=256, hidden_dim=512):
        super().__init__()
        
        # GRU
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, tile_classes)
        )

    def forward(self, feat_fusion: torch.Tensor, hidden: torch.Tensor):
        """
        feat_fusion: [B, input_dim]
        hidden: [B, hidden_dim]
        """
        hidden = self.drop(self.gru(feat_fusion, hidden))
        logits = self.fc(hidden)
        return logits, hidden
    
class VAEDecoder(nn.Module):
    def __init__(self, device: torch.device, start_tile=31, map_vec_dim=32, width=13, height=13):
        super().__init__()
        
        self.device = device
        self.width = width
        self.height = height
        self.start_tile = start_tile
        
        self.rnn_hidden = 512
        self.tile_classes = 32
        
        # 模型结构
        self.map_vec_fc = nn.Sequential(
            nn.Linear(map_vec_dim, 256)
        )
        self.tile_embedding = GinkaTileEmbedding(tile_classes=self.tile_classes)
        self.pos_embedding = GinkaPosEmbedding()
        self.map_patch = GinkaMapPatch(tile_classes=self.tile_classes)
        self.feat_fusion = GinkaInputFusion()
        self.rnn = GinkaRNN(tile_classes=self.tile_classes, hidden_dim=self.rnn_hidden)
        
        self.col_list = []
        self.row_list = []
        for y in range(0, height):
            for x in range(0, width):
                self.col_list.append(x)
                self.row_list.append(y)
        
    def forward(self, map_vec: torch.Tensor, target_map: torch.Tensor, use_self_probility=0):
        """
        map_vec: [B, vec_dim]
        target_map: [B, H, W]
        use_self: 是否使用自己生成的上一步结果执行下一步
        """
        B, C = map_vec.shape
        
        # 张量声明
        now_tile = torch.LongTensor([self.start_tile]).to(self.device).expand(B)
        
        map = torch.zeros([B, self.height, self.width], dtype=torch.int32).to(self.device)
        output_logits = torch.zeros([B, self.height, self.width, self.tile_classes]).to(self.device)
        hidden: torch.Tensor = torch.zeros(B, self.rnn_hidden).to(self.device)
        
        col_list = torch.IntTensor(self.col_list).to(self.device).expand(B, -1)
        row_list = torch.IntTensor(self.row_list).to(self.device).expand(B, -1)
        pos_embed = self.pos_embedding(col_list, row_list)
        
        map_vec = self.map_vec_fc(map_vec)
        
        for y in range(0, self.height):
            for x in range(0, self.width):
                idx = y * self.width + x
                # 图块编码、地图局部编码
                tile_embed = self.tile_embedding(now_tile)
                use_self = random.random() < use_self_probility
                map_patch = self.map_patch(map if use_self else target_map, x, y)
                # 编码特征融合
                feat = self.feat_fusion(tile_embed, map_vec, pos_embed[:, idx], map_patch)
                # RNN 输出
                logits, h = self.rnn(feat, hidden)
                # 处理输出
                output_logits[:, y, x] = logits[:]
                hidden = h
                tile_id = torch.argmax(logits, dim=1).detach()
                map[:, y, x] = tile_id[:]
                now_tile = tile_id if use_self else target_map[:, y, x].detach()
                
        return output_logits.permute(0, 3, 1, 2)

if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 13, 13]).to(device)
    map_vec = torch.rand(1, 32).to(device)
    
    # 初始化模型
    model = VAEDecoder("cpu").to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    fake_logits = model(map_vec, input, 0)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: fake_logits={fake_logits.shape}")
    print(f"Map Vector FC parameters: {sum(p.numel() for p in model.map_vec_fc.parameters())}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Position Embedding parameters: {sum(p.numel() for p in model.pos_embedding.parameters())}")
    print(f"Map Patch parameters: {sum(p.numel() for p in model.map_patch.parameters())}")
    print(f"Feature Fusion parameters: {sum(p.numel() for p in model.feat_fusion.parameters())}")
    print(f"RNN parameters: {sum(p.numel() for p in model.rnn.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
