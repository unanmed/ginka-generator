import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VectorQuantizer(nn.Module):
    def __init__(self, K: int, d_z: int):
        """
        Args:
            K:    codebook 大小（码字数量）
            d_z:  码字嵌入维度
            temp: 软分配 softmax 温度，越小越接近 hard assignment
        """
        super().__init__()
        self.K = K
        self.d_z = d_z

        self.codebook = nn.Embedding(K, d_z)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)

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

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, L, d_z]
        """
        Args:
            z_e: [B, L, d_z]  编码器输出的连续向量序列

        Returns:
            z_q_st:       [B, L, d_z]  量化后向量（直通梯度）
            indices:      [B, L]       每个位置对应的码字索引
            commit_loss:  scalar       承诺损失 ||z_e - sg(z_q)||^2
        """
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
        indices = distances.argmin(dim=1) # [B*L]

        # 量化向量
        z_q_flat = self.codebook(indices) # [B*L, d_z]
        z_q = z_q_flat.reshape(B, L, d_z)

        # 直通估计：前向传 z_q，反向传 z_e 的梯度
        z_q_st = z_e + (z_q - z_e).detach()

        # 承诺损失：拉近编码向量与其对应的码字（仅更新编码器）
        commit_loss = F.mse_loss(z_e, z_q.detach())

        indices = indices.reshape(B, L)
        perplexity, usage_rate, usage_count = self.codebook_stats(indices)
        return z_q_st, indices, commit_loss, perplexity, usage_count

    def sample(self, B: int, L: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.K, (B, L), device=device)
        return self.codebook(indices)
