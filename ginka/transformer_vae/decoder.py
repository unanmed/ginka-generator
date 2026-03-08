import time
import torch
import torch.nn as nn
from ..utils import print_memory

class GinkaTransformerDecoder(nn.Module):
    def __init__(self, num_classes=32, dim_ff=256, nhead=4, num_layers=4, map_size=13*13):
        super().__init__()
        self.autoregressive = False
        self.dim_ff = dim_ff
        self.map_size = map_size
        self.embedding = nn.Embedding(num_classes, dim_ff)
        self.pos_embedding = nn.Embedding(map_size, dim_ff)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_ff, dim_feedforward=dim_ff, nhead=nhead, batch_first=True),
            num_layers=max(num_layers // 2, 1)
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim_ff, dim_feedforward=dim_ff, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_ff, num_classes)
        )
        
    def forward(self, z: torch.Tensor, target_map: torch.Tensor):
        # z: [B, dim_ff]
        # target_map: [B, H * W]
        # training output: [B, H * W, dim_ff]
        # evaling output: [B, H * W]
        B, L = target_map.shape
        
        memory = self.encoder(z.unsqueeze(1)) # [B, 1, dim_ff]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool)).to(z.device) # [B, H * W, H * W]
        
        # when training, use teacher forcing
        if not self.autoregressive:
            map = self.embedding(target_map)
            pos_embed = self.pos_embedding(torch.arange(L, dtype=torch.long).to(z.device))
            map = map + pos_embed # [B, H * W, dim_ff]
            decoded = self.decoder(map, memory, tgt_mask=mask) # [B, H * W, dim_ff]
            output = self.fc(decoded)
            return output
        
        # when evaling, use autoregressive generation
        else:
            output = torch.zeros([B, L], dtype=torch.int).to(z.device)
            for idx in range(0, self.map_size):
                embed = self.embedding(output)
                pos_embed = self.pos_embedding(torch.IntTensor([idx]).repeat(B, 1).to(z.device))
                map = embed + pos_embed # [B, H * W, dim_ff]
                decoded = self.decoder(map, memory, tgt_mask=mask)
                decoded = self.fc(decoded) # [B, H * W, dim_ff]
                output[:, idx] = torch.argmax(decoded[:, idx, :], dim=1)
            
            return output
        
class GinkaTransformerVAEDecoder(nn.Module):
    def __init__(
        self, latent_dim=32, num_classes=32, dim_ff=256, nhead=4, num_layers=4,
        map_size=13*13
    ):
        super().__init__()
        self.map_size = map_size
        self.input = nn.Sequential(
            nn.Linear(latent_dim, dim_ff),
            nn.Dropout(0.3),
            nn.LayerNorm(dim_ff),
            nn.ReLU(),
            
            nn.Linear(dim_ff, dim_ff)
        )
        self.decoder = GinkaTransformerDecoder(
            num_classes=num_classes, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers, map_size=map_size
        )
        
    def forward(self, z: torch.Tensor, map: torch.Tensor):
        hidden = self.input(z)
        output = self.decoder(hidden, map)
        return output[:, 0:self.map_size]

if __name__ == "__main__":
    device = torch.device("cpu")
    
    input = torch.randn(1, 32).to(device)
    map = torch.randint(0, 32, [1, 169]).to(device)
    
    # 初始化模型
    model = GinkaTransformerVAEDecoder().to(device)
    model.eval()
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    output = model(input, map)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输出形状: output={output.shape}")
    print(f"Input Embedding parameters: {sum(p.numel() for p in model.input.parameters())}")
    print(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
