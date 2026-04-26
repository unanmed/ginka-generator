import time
import torch
import torch.nn as nn
from ..utils import print_memory
from .maskGIT import Transformer


class GinkaMaskGIT(nn.Module):
    """
    改造后的 MaskGIT 地图生成模型。

    以掩码地图序列和 VQ-VAE 输出的离散隐变量 z 为输入，
    通过 Transformer encoder-decoder 结构预测被遮盖位置的 tile 类别。

    z 通过 cross-attention 注入到 Transformer decoder，
    作为风格/多样性控制信号，而非结构重建指导。
    """

    def __init__(
        self,
        num_classes: int = 16,
        d_model: int = 192,
        d_z: int = 64,
        dim_ff: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        map_size: int = 13 * 13,
        z_dropout: float = 0.1,
    ):
        """
        Args:
            num_classes: tile 类别数（含 MASK token=15）
            d_model:     Transformer 内部维度
            d_z:         VQ-VAE 码字嵌入维度，需与 GinkaVQVAE.d_z 一致
            dim_ff:      前馈网络隐层维度
            nhead:       注意力头数
            num_layers:  Transformer 层数
            map_size:    地图 token 总数（H * W）
            z_dropout:   训练时随机替换 z 为随机码字的概率（提升鲁棒性）
        """
        super().__init__()
        self.z_dropout = z_dropout

        # Tile 嵌入 + 位置编码
        self.tile_embedding = nn.Embedding(num_classes, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, map_size, d_model) * 0.02)

        # z 投影：将 VQ 码字从 d_z 维映射到 d_model 维，供 cross-attention 使用
        self.z_proj = nn.Sequential(
            nn.Linear(d_z, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer：encoder 做 map token 自注意力，decoder 做与 z 的 cross-attention
        self.transformer = Transformer(
            d_model=d_model, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers
        )

        self.output_fc = nn.Linear(d_model, num_classes)

    def forward(self, map: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map: [B, H*W]      掩码后的地图 token 序列（MASK token = 15）
            z:   [B, L, d_z]   VQ-VAE 量化后的离散隐变量

        Returns:
            logits: [B, H*W, num_classes]
        """
        # z dropout：训练时以一定概率将 z 替换为随机均匀噪声，
        # 模拟推理时随机采样 z 的分布，避免模型过拟合于精确的 z 语义
        if self.training and self.z_dropout > 0:
            mask = torch.rand(z.shape[0], 1, 1, device=z.device) < self.z_dropout
            rand_z = torch.randn_like(z)
            z = torch.where(mask, rand_z, z)

        # 投影 z 到 d_model 维度
        z_mem = self.z_proj(z)          # [B, L, d_model]

        # tile embedding + 位置编码
        x = self.tile_embedding(map)    # [B, H*W, d_model]
        x = x + self.pos_embedding      # [B, H*W, d_model]

        # Transformer：encoder 做 map 自注意力，decoder cross-attend z
        x = self.transformer(x, memory=z_mem)  # [B, H*W, d_model]

        logits = self.output_fc(x)      # [B, H*W, num_classes]
        return logits


if __name__ == "__main__":
    device = torch.device("cpu")

    model = GinkaMaskGIT(
        num_classes=16,
        d_model=192,
        d_z=64,
        dim_ff=512,
        nhead=8,
        num_layers=4,
        map_size=13 * 13,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}  ({total_params / 1e6:.3f}M)")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,}")

    map_input = torch.randint(0, 16, (4, 13 * 13)).to(device)   # [B=4, 169]
    z_input = torch.randn(4, 2, 64).to(device)                   # [B=4, L=2, d_z=64]

    model.train()
    logits = model(map_input, z_input)
    print(f"\nlogits shape: {logits.shape}")  # [4, 169, 16]

    print_memory(device, "前向传播后")
