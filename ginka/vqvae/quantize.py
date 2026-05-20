import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        K: int,
        d_z: int,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.K = K
        self.d_z = d_z
        self.decay = decay
        self.epsilon = epsilon

        self.codebook = nn.Embedding(K, d_z)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)
        self.codebook.weight.requires_grad_(False)

        # EMA 统计量：码字访问次数与对应编码向量和。
        self.register_buffer("ema_cluster_size", torch.ones(K))
        self.register_buffer(
            "ema_weight",
            self.codebook.weight.detach().clone()
        )

    def codebook_stats(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_indices = indices.reshape(-1)
        one_hot = F.one_hot(flat_indices, num_classes=self.K).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(
            -(avg_probs * torch.log(avg_probs.clamp_min(1e-10))).sum()
        )
        usage_rate = (avg_probs > 0).float().mean()
        usage_count = one_hot.sum(dim=0)
        return perplexity, usage_rate, usage_count

    def ema_update(self, z_flat: torch.Tensor, flat_indices: torch.Tensor):
        one_hot = F.one_hot(flat_indices, num_classes=self.K).type_as(z_flat)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.transpose(0, 1) @ z_flat

        self.ema_cluster_size.mul_(self.decay).add_(
            cluster_size,
            alpha=1.0 - self.decay
        )
        self.ema_weight.mul_(self.decay).add_(
            embed_sum,
            alpha=1.0 - self.decay
        )

        total_count = self.ema_cluster_size.sum()
        normalized_cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (total_count + self.K * self.epsilon) * total_count
        )
        normalized_weight = self.ema_weight / normalized_cluster_size.unsqueeze(1)
        self.codebook.weight.data.copy_(normalized_weight)

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, L, d_z]
        B, L, d_z = z_e.shape

        z_flat = z_e.reshape(B * L, d_z) # [B * L, d_z]

        codebook_w = self.codebook.weight  # [K, d_z]

        # 计算 L2 距离：||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e · e_k
        # distances: [B*L, K]
        ze_square = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        ek_square = torch.sum(codebook_w ** 2, dim=1)
        mul = z_flat @ codebook_w.t()
        distances = ze_square + ek_square - 2 * mul

        # Hard assignment：取最近码字索引
        flat_indices = distances.argmin(dim=1) # [B*L]

        # 量化向量
        z_q_flat = self.codebook(flat_indices) # [B*L, d_z]
        z_q = z_q_flat.reshape(B, L, d_z)

        # 直通估计：前向传 z_q，反向传 z_e 的梯度
        z_q_st = z_e + (z_q - z_e).detach()

        # 承诺损失：拉近编码向量与其对应的码字（仅更新编码器）
        commit_loss = F.mse_loss(z_e, z_q.detach())

        # 训练时使用 EMA 更新码本；验证与推理阶段保持码本固定。
        if self.training and z_e.requires_grad:
            self.ema_update(z_flat.detach(), flat_indices.detach())

        indices = flat_indices.reshape(B, L)
        perplexity, usage_rate, usage_count = self.codebook_stats(indices)
        return z_q_st, indices, commit_loss, perplexity, usage_count

    def sample(self, B: int, L: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.K, (B, L), device=device)
        return self.codebook(indices)
