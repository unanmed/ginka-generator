import torch
import torch.nn as nn

class AdaLN(nn.Module):
    # 自适应 LayerNorm：条件向量 c 动态预测 LayerNorm 的 gamma 和 beta
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, d_model * 2)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]  c: [B, d_cond]
        gamma, beta = self.proj(c).chunk(2, dim=-1) # 各 [B, d_model]
        return (1 + gamma.unsqueeze(1)) * self.norm(x) + beta.unsqueeze(1)

class CondTransformerLayer(nn.Module):
    # 带 AdaLN 条件注入的 Transformer Encoder 层
    # 结构：AdaLN → Self-Attn → 残差；AdaLN → FFN → 残差（Pre-norm）
    def __init__(self, d_model: int, nhead: int, dim_ff: int, d_cond: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.adaln1 = AdaLN(d_model, d_cond)
        self.adaln2 = AdaLN(d_model, d_cond)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]  c: [B, d_cond]
        residual = x
        normed = self.adaln1(x, c)
        x, _ = self.self_attn(normed, normed, normed)
        x = residual + x
        residual = x
        x = self.ffn(self.adaln2(x, c))
        x = residual + x
        return x

class Transformer(nn.Module):
    # 纯 encoder Transformer，每层使用 AdaLN 注入全局条件向量 c
    def __init__(
        self, d_model: int = 256, dim_ff: int = 512,
        nhead: int = 8, num_layers: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CondTransformerLayer(d_model=d_model, nhead=nhead, dim_ff=dim_ff, d_cond=d_model)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]  c: [B, d_model]  全局条件向量
        for layer in self.layers:
            x = layer(x, c)
        return x
