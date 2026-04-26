import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    向量量化层（Vector Quantization）。

    将连续的编码向量序列映射到离散的 codebook 码字索引，
    并通过直通估计（Straight-Through Estimator）保持梯度流。

    均匀分布正则化采用软分配熵最大化方案：
      通过对距离做 softmax 得到软分配概率，计算平均码字使用率的熵，
      最小化负熵以鼓励所有码字被均等使用。
    """

    def __init__(self, K: int, d_z: int, temp: float = 1.0):
        """
        Args:
            K:    codebook 大小（码字数量）
            d_z:  码字嵌入维度
            temp: 软分配 softmax 温度，越小越接近 hard assignment
        """
        super().__init__()
        self.K = K
        self.d_z = d_z
        self.temp = temp

        self.codebook = nn.Embedding(K, d_z)
        nn.init.uniform_(self.codebook.weight, -1.0 / K, 1.0 / K)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, L, d_z]  编码器输出的连续向量序列

        Returns:
            z_q_st:       [B, L, d_z]  量化后向量（直通梯度）
            indices:      [B, L]       每个位置对应的码字索引
            commit_loss:  scalar       承诺损失 ||z_e - sg(z_q)||^2
            entropy_loss: scalar       负熵损失（最小化 = 最大化码字使用均匀度）
        """
        B, L, d_z = z_e.shape

        # 展平到 [B*L, d_z]
        z_flat = z_e.reshape(B * L, d_z)

        codebook_w = self.codebook.weight  # [K, d_z]

        # 计算 L2 距离：||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e · e_k
        # distances: [B*L, K]
        distances = (
            (z_flat ** 2).sum(dim=1, keepdim=True)   # [B*L, 1]
            + (codebook_w ** 2).sum(dim=1)            # [K]
            - 2.0 * z_flat @ codebook_w.t()           # [B*L, K]
        )

        # Hard assignment：取最近码字索引
        indices = distances.argmin(dim=1)             # [B*L]

        # 量化向量
        z_q_flat = self.codebook(indices)             # [B*L, d_z]
        z_q = z_q_flat.reshape(B, L, d_z)

        # 直通估计：前向传 z_q，反向传 z_e 的梯度
        z_q_st = z_e + (z_q - z_e).detach()

        # 承诺损失：拉近编码向量与其对应的码字（仅更新编码器）
        commit_loss = F.mse_loss(z_e, z_q.detach())

        # 熵最大化正则：通过软分配计算平均码字使用率，最小化负熵
        # soft_assign: [B*L, K]，对距离做 softmax（距离越小，概率越大）
        soft_assign = F.softmax(-distances / self.temp, dim=1)
        avg_assign = soft_assign.mean(dim=0)          # [K]，平均码字使用率
        # entropy_loss = -H(p) = sum(p * log(p))，最小化即最大化熵
        entropy_loss = (avg_assign * torch.log(avg_assign + 1e-10)).sum()

        indices = indices.reshape(B, L)
        return z_q_st, indices, commit_loss, entropy_loss
