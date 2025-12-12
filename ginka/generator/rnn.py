import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class GinkaRNN(nn.Module):
    def __init__(self, tile_classes=32, cond_dim=256, input_dim=256, hidden_dim=1024, num_layers=2):
        super().__init__()
        
        # 输入部分
        self.embedding = nn.Embedding(tile_classes, input_dim)
        self.input_fc = nn.Linear(input_dim, input_dim)
        
        self.gru = nn.GRU(input_dim + cond_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tile_classes)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [B, T]
        cond: [B, cond_dim]
        """
        B, T = x.shape
        tile_emb = self.input_fc(self.embedding(x)) # [B, T, input_dim]
        cond_expand = cond.unsqueeze(1).expand(B, T, cond.shape[-1]) # [B, T, cond_dim]

        # 拼接 tile + cond
        step_input = torch.cat([tile_emb, cond_expand], dim=-1)

        out, _ = self.gru(step_input)
        logits = self.fc(out)
        return logits

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated('cuda:1') / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated('cuda:1') / 1024**2:.2f} MB")

if __name__ == "__main__":
    input = torch.argmax(torch.rand(1, 32, 13 * 13).cuda(1), dim=1)
    cond = torch.rand(1, 256).cuda(1)
    
    # 初始化模型
    model = GinkaRNN().cuda(1)
    
    print_memory("初始化后")
    
    # 前向传播
    start = time.perf_counter()
    fake = model(input, cond)
    end = time.perf_counter()
    
    print_memory("前向传播后")
    
    print(f"推理耗时: {end - start}")
    print(f"输入形状: feat={input.shape}")
    print(f"输出形状: output={fake.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
