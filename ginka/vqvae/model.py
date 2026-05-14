import time
import torch
import torch.nn as nn
from .quantize import VectorQuantizer
from typing import Tuple
from ..utils import print_memory

class _DecodeLayer(nn.Module):
    """单个解码层：Pre-LN Cross-Attention + Pre-LN FFN。"""

    def __init__(self, d_z: int, nhead: int, dim_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_z)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_z,
            num_heads=nhead,
            batch_first=True,
            dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(d_z)
        self.ffn = nn.Sequential(
            nn.Linear(d_z, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_z),
        )

    def forward(self, x: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), z_q, z_q)[0]   # Pre-LN cross-attn
        x = x + self.ffn(self.norm2(x))                        # Pre-LN FFN
        return x


class VQDecodeHead(nn.Module):
    """
    VQ-VAE 预训练用解码头（堆叠 Cross-Attention + FFN，Pre-LN 风格）。

    将 z_q [B, L, d_z] 通过多层 Cross-Attention 解码为地图 logits [B, H*W, num_classes]。
    预训练结束后此模块被丢弃，不影响联合训练路径。

    架构（每层）：
        Pre-LN Cross-Attention（query=可学习位置查询, key/value=z_q）
        Pre-LN FFN
    × num_layers 层 → LayerNorm → 线性分类头
    """

    def __init__(
        self,
        num_classes: int,
        d_z: int,
        map_size: int,
        nhead: int = 8,
        dim_ff: int = 512,
        num_layers: int = 4,
    ):
        """
        Args:
            num_classes: tile 类别数
            d_z:         z 向量维度（须与 GinkaVQVAE 的 d_z 一致）
            map_size:    地图 token 总数（H * W）
            nhead:       Cross-Attention 的注意力头数（d_z 须能被 nhead 整除）
            dim_ff:      FFN 隐层维度
            num_layers:  解码层数（建议与编码器 num_layers 相同）
        """
        super().__init__()

        # 每个格子一个可学习位置查询
        self.pos_queries = nn.Parameter(torch.randn(1, map_size, d_z) * 0.02)

        # 条件地图嵌入：将切片地图 tile ID 映射到 d_z 空间，叠加到位置查询
        self.cond_embedding = nn.Embedding(num_classes, d_z)

        # 堆叠多层解码块
        self.layers = nn.ModuleList([
            _DecodeLayer(d_z, nhead, dim_ff) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_z)
        self.classifier = nn.Linear(d_z, num_classes)

    def forward(self, z_q: torch.Tensor, cond_map: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z_q:      [B, L, d_z]    量化后的 z 向量
            cond_map: [B, map_size]  条件切片地图（整数 tile ID）；
                                     为 None 时退化为纯位置查询（与旧行为一致）

        Returns:
            logits: [B, map_size, num_classes]
        """
        B = z_q.shape[0]
        x = self.pos_queries.expand(B, -1, -1)   # [B, map_size, d_z]
        if cond_map is not None:
            x = x + self.cond_embedding(cond_map) # 叠加切片上下文
        for layer in self.layers:
            x = layer(x, z_q)
        x = self.norm_out(x)
        return self.classifier(x)                # [B, map_size, num_classes]


class GinkaVQVAE(nn.Module):
    def __init__(
        self, num_classes: int = 16, L: int = 2, K: int = 16, d_z: int = 64, d_model: int = 128,
        nhead: int = 4, num_layers: int = 2, dim_ff: int = 256, map_h: int = 13, map_w: int = 13
    ):
        super().__init__()
        self.L = L
        self.K = K
        self.map_h = map_h
        self.map_w = map_w

        # Tile 嵌入
        self.tile_embedding = nn.Embedding(num_classes, d_model)

        # 二维因式分解位置编码：行嵌入 + 列嵌入，共享行列语义
        self.row_embedding = nn.Parameter(torch.randn(1, map_h, d_model) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(1, map_w, d_model) * 0.02)

        # L 个可学习 summary token，拼接到序列头部
        self.summary_tokens = nn.Parameter(torch.randn(1, L, d_model) * 0.02)

        # Pre-LN Transformer Encoder（训练更稳定）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True,
            activation='gelu', norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 将 Transformer 输出投影到 codebook 维度 d_z
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_z),
            nn.LayerNorm(d_z),
        )

    def forward(self, map: torch.Tensor) -> torch.Tensor:
        # map: [B, H * W]
        B, _ = map.shape

        row_idx = torch.arange(self.map_h, device=map.device).repeat_interleave(self.map_w)
        col_idx = torch.arange(self.map_w, device=map.device).repeat(self.map_h)
        pos = self.row_embedding[0, row_idx] + self.col_embedding[0, col_idx] # [H*W, d_model]

        x = self.tile_embedding(map) + pos # [B, H * W, d_model]
        summary = self.summary_tokens.expand(B, -1, -1) # [B, L, d_model]
        x = torch.cat([summary, x], dim=1) # [B, L+H*W, d_model]
        
        x = self.transformer(x)
        
        z_e = self.proj(x[:, :self.L])
        
        return z_e

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    map_input = torch.randint(0, 7, (4, 13 * 13)).to(device)  # [B=4, 169]

    model = GinkaVQVAE(
        num_classes=7, L=2, K=16, d_z=64, d_model=128,
        nhead=4, num_layers=2, dim_ff=256, map_h=13, map_w=13
    ).to(device)

    print_memory(device, "初始化后")

    start = time.perf_counter()
    z_e = model(map_input)
    end = time.perf_counter()

    print_memory(device, "前向传播后")

    print(f"推理耗时: {end - start:.4f}s")
    print(f"输出形状: z_e={z_e.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Transformer parameters: {sum(p.numel() for p in model.transformer.parameters())}")
    print(f"Projection parameters: {sum(p.numel() for p in model.proj.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
