import torch
import torch.nn as nn
from .quantize import VectorQuantizer
from typing import Tuple


class GinkaVQVAE(nn.Module):
    """
    VQ-VAE 风格地图编码器。

    将一张完整的地图（[B, H*W] 整数 tile ID 序列）编码为 L 个离散码字，
    输出 z [B, L, d_z] 作为 MaskGIT 模型的生成条件。

    架构：
        tile embedding + 位置编码
        → L 个可学习 summary token（拼接到序列头部）
        → Transformer Encoder（Pre-LN，自注意力）
        → 取前 L 个输出
        → 线性投影到 d_z
        → VectorQuantizer（直通估计 + 熵最大化正则）

    设计约束：
        - 参数量目标 < 1M
        - 不含解码器，z 的语义由 MaskGIT 端的交叉熵损失间接约束
        - z 定位为风格/多样性控制信号，而非结构重建指导
    """

    def __init__(
        self,
        num_classes: int = 16,
        L: int = 2,
        K: int = 16,
        d_z: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        map_size: int = 13 * 13,
        beta: float = 0.25,
        gamma: float = 0.1,
        vq_temp: float = 1.0,
    ):
        """
        Args:
            num_classes: tile 类别数（含 MASK token）
            L:           码字序列长度，即 z 的序列维度
            K:           codebook 大小（码字总数）
            d_z:         码字嵌入维度
            d_model:     Transformer 内部维度
            nhead:       注意力头数
            num_layers:  Transformer 层数
            dim_ff:      前馈网络隐层维度
            map_size:    地图 token 总数（H * W）
            beta:        承诺损失权重
            gamma:       熵正则损失权重
            vq_temp:     VQ 软分配 softmax 温度
        """
        super().__init__()
        self.L = L
        self.K = K
        self.d_z = d_z
        self.beta = beta
        self.gamma = gamma

        # Tile 嵌入
        self.tile_embedding = nn.Embedding(num_classes, d_model)

        # 地图位置编码（仅覆盖 map_size 个位置，不含 summary token）
        self.pos_embedding = nn.Parameter(torch.randn(1, map_size, d_model) * 0.02)

        # L 个可学习 summary token，拼接到序列头部
        self.summary_tokens = nn.Parameter(torch.randn(1, L, d_model) * 0.02)

        # Pre-LN Transformer Encoder（训练更稳定）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation='gelu',
            norm_first=True,      # Pre-LN
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 将 Transformer 输出投影到 codebook 维度 d_z
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_z),
            nn.LayerNorm(d_z),
        )

        # 向量量化层
        self.vq = VectorQuantizer(K=K, d_z=d_z, temp=vq_temp)

    def encode(self, map: torch.Tensor) -> torch.Tensor:
        """
        将地图编码为量化前的连续向量序列。

        Args:
            map: [B, H*W]  整数 tile ID

        Returns:
            z_e: [B, L, d_z]  量化前的编码向量
        """
        B = map.shape[0]

        x = self.tile_embedding(map)                        # [B, H*W, d_model]
        x = x + self.pos_embedding                          # [B, H*W, d_model]

        summary = self.summary_tokens.expand(B, -1, -1)    # [B, L, d_model]
        x = torch.cat([summary, x], dim=1)                 # [B, L+H*W, d_model]

        x = self.transformer(x)                            # [B, L+H*W, d_model]

        z_e = self.proj(x[:, :self.L])                     # [B, L, d_z]
        return z_e

    def forward(self, map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播：编码 → 量化 → 计算损失。

        Args:
            map: [B, H*W]  整数 tile ID（训练时传入完整真实地图）

        Returns:
            z_q:   [B, L, d_z]  量化后的 z（含直通梯度），供 MaskGIT 使用
            indices: [B, L]     每个位置对应的码字索引
            vq_loss: scalar     VQ 总损失 = beta * commit_loss + gamma * entropy_loss
        """
        z_e = self.encode(map)
        z_q, indices, commit_loss, entropy_loss = self.vq(z_e)

        vq_loss = self.beta * commit_loss + self.gamma * entropy_loss
        return z_q, indices, vq_loss, commit_loss, entropy_loss

    def sample(self, B: int, device: torch.device) -> torch.Tensor:
        """
        推理阶段：从 codebook 中随机均匀采样 L 个码字。

        Args:
            B:      batch size
            device: 目标设备

        Returns:
            z: [B, L, d_z]
        """
        indices = torch.randint(0, self.K, (B, self.L), device=device)
        z = self.vq.codebook(indices)   # [B, L, d_z]
        return z


if __name__ == "__main__":
    device = torch.device("cpu")

    model = GinkaVQVAE(
        num_classes=16,
        L=2,
        K=16,
        d_z=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_ff=256,
        map_size=13 * 13,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}  ({total_params / 1e6:.3f}M)")

    # 分模块参数统计
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    # 前向传播测试
    map_input = torch.randint(0, 15, (4, 13 * 13)).to(device)  # [B=4, 169]

    z_q, indices, vq_loss = model(map_input)

    print(f"\nz_q shape:    {z_q.shape}")      # [4, 2, 64]
    print(f"indices shape:{indices.shape}")    # [4, 2]
    print(f"vq_loss:      {vq_loss.item():.4f}")

    # 推理采样测试
    z_sample = model.sample(B=4, device=device)
    print(f"sample shape: {z_sample.shape}")   # [4, 2, 64]
