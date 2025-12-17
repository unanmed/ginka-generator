import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNConditionEncoder(nn.Module):
    def __init__(self, val_dim=16, output_dim=256, width=13, height=13):
        super().__init__()
        
        # 条件编码
        self.val_fc = nn.Sequential(
            nn.Linear(val_dim, output_dim * 4),
            nn.LayerNorm(output_dim * 4),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim)
        )
        
    def forward(self, val_cond: torch.Tensor):
        val_hidden = self.val_fc(val_cond)
        return self.fusion(val_hidden)
    
class GinkaMapPatch(nn.Module):
    def __init__(self, tile_classes=32, width=13, height=13):
        super().__init__()
        
        # 地图局部卷积，用于捕获局部结构信息
        
        self.width = width
        self.height = height
        self.tile_classes = 32
        
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(tile_classes + 1, 256, 3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            nn.Conv2d(256, 512, 3),
            nn.Dropout(0.2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            
            nn.Flatten()
        )
        self.fc = nn.Linear(512 * 3 * 3, 256)
        
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
        
        self.row_embedding = nn.Embedding(width, embed_dim)
        self.col_embedding = nn.Embedding(height, embed_dim)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        row = self.row_embedding(x).squeeze(1)
        col = self.col_embedding(y).squeeze(1)
        
        return row, col
    
class GinkaInputFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        # 使用 Transformer 进行信息整合
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=2, dim_feedforward=d_model*2, batch_first=True,
                dropout=0.2
            ),
            num_layers=4
        )
        
    def forward(
        self, tile_embed: torch.Tensor, cond_vec: torch.Tensor, 
        row_embed: torch.Tensor, col_embed: torch.Tensor, patch_vec: torch.Tensor
    ):
        """
        tile_embed: [B, 256]
        cond_vec: [B, 256]
        row_embed: [B, 256]
        col_embed: [B, 256]
        patch_vec: [B, 256]
        """
        vec = torch.stack([tile_embed, cond_vec, row_embed, col_embed, patch_vec], dim=1)
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
    
class GinkaRNNModel(nn.Module):
    def __init__(self, device: torch.device, start_tile=31, width=13, height=13):
        super().__init__()
        
        self.device = device
        self.width = width
        self.height = height
        self.start_tile = start_tile
        
        self.rnn_hidden = 512
        self.tile_classes = 32
        
        # 模型结构
        self.cond = RNNConditionEncoder()
        self.tile_embedding = GinkaTileEmbedding(tile_classes=self.tile_classes)
        self.pos_embedding = GinkaPosEmbedding()
        self.map_patch = GinkaMapPatch(tile_classes=self.tile_classes)
        self.feat_fusion = GinkaInputFusion()
        self.rnn = GinkaRNN(tile_classes=self.tile_classes, hidden_dim=self.rnn_hidden)
        
    def forward(self, val_cond: torch.Tensor, target_map: torch.Tensor, use_self=False):
        """
        val_cond: [B, val_dim]
        target_map: [B, H, W]
        use_self: 是否使用自己生成的上一步结果执行下一步
        """
        B, C = val_cond.shape
        
        # 张量声明
        now_tile = torch.LongTensor([self.start_tile]).to(self.device).expand(B)
        
        map = torch.zeros([B, self.height, self.width], dtype=torch.int32).to(self.device)
        output_logits = torch.zeros([B, self.height, self.width, self.tile_classes]).to(self.device)
        hidden: torch.Tensor = torch.zeros(B, self.rnn_hidden).to(self.device)
        
        # 条件编码，全局，所以只用一次
        cond = self.cond(val_cond)
        
        for y in range(0, self.height):
            for x in range(0, self.width):
                x_tensor = torch.LongTensor([x]).to(self.device).expand(B, -1)
                y_tensor = torch.LongTensor([y]).to(self.device).expand(B, -1)
                # 位置编码、图块编码、地图局部编码
                tile_embed = self.tile_embedding(now_tile)
                row_embed, col_embed = self.pos_embedding(x_tensor, y_tensor)
                map_patch = self.map_patch(map if use_self else target_map, x, y)
                # 编码特征融合
                feat = self.feat_fusion(tile_embed, cond, row_embed, col_embed, map_patch)
                # RNN 输出
                logits, h = self.rnn(feat, hidden)
                # 处理输出
                output_logits[:, y, x] = logits[:]
                hidden = h
                tile_id = torch.argmax(logits, dim=1).detach()
                map[:, y, x] = tile_id[:]
                now_tile = tile_id if use_self else target_map[:, y, x].detach()
                
        return output_logits.permute(0, 3, 1, 2), map

def print_memory(device, tag=""):
    if torch.cuda.is_available():
        print(f"{tag} | 当前显存: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    else:
        print("当前设备不支持 cuda.")

if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randint(0, 32, [1, 13, 13]).to(device)
    cond = torch.rand(1, 16).to(device)
    
    # 初始化模型
    model = GinkaRNNModel("cpu").to(device)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    fake_logits, fake_map = model(cond, input, False)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: fake_logits={fake_logits.shape}, fake_map={fake_map.shape}")
    print(f"Condition Encoder parameters: {sum(p.numel() for p in model.cond.parameters())}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Position Embedding parameters: {sum(p.numel() for p in model.pos_embedding.parameters())}")
    print(f"Map Patch parameters: {sum(p.numel() for p in model.map_patch.parameters())}")
    print(f"Feature Fusion parameters: {sum(p.numel() for p in model.feat_fusion.parameters())}")
    print(f"RNN parameters: {sum(p.numel() for p in model.rnn.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
