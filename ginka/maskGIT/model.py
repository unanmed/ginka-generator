import time
import torch
import torch.nn as nn
from ..utils import print_memory
from .maskGIT import Transformer

# 结构标签词表大小
SYM_VOCAB = 8 # symmetryH/V/C 三位组合 0-7
ROOM_VOCAB = 3 # roomCountLevel 0-2
BRANCH_VOCAB = 3 # branchLevel 0-2
OUTER_VOCAB = 2 # outerWall 0-1

# 密度标签词表大小（Low/Medium/High 三档）
DOOR_DENSITY_VOCAB = 3
MONSTER_DENSITY_VOCAB = 3
RESOURCE_DENSITY_VOCAB = 3

class GinkaMaskGIT(nn.Module):
    def __init__(
        self, num_classes: int = 16, d_model: int = 192, dim_ff: int = 512,
        nhead: int = 8, num_layers: int = 4, map_h: int = 13, map_w: int = 13,
        d_z: int = 64, z_seq_len: int = 6
    ):
        super().__init__()
        self.map_h = map_h
        self.map_w = map_w

        # Tile 嵌入 + 二维因式分解位置编码
        self.tile_embedding = nn.Embedding(num_classes, d_model)
        self.row_embedding = nn.Parameter(torch.randn(1, map_h, d_model) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(1, map_w, d_model) * 0.02)

        # 结构标签嵌入：各自独立嵌入到 d_z 维度，作为独立 token
        self.sym_embed = nn.Embedding(SYM_VOCAB, d_z)
        self.room_embed = nn.Embedding(ROOM_VOCAB, d_z)
        self.branch_embed = nn.Embedding(BRANCH_VOCAB, d_z)
        self.outer_embed = nn.Embedding(OUTER_VOCAB, d_z)

        # 密度标签嵌入：各自独立嵌入到 d_z 维度，作为独立 token
        self.door_density_embed = nn.Embedding(DOOR_DENSITY_VOCAB, d_z)
        self.monster_density_embed = nn.Embedding(MONSTER_DENSITY_VOCAB, d_z)
        self.resource_density_embed = nn.Embedding(RESOURCE_DENSITY_VOCAB, d_z)

        # z 投影：逐 token 线性变换，保持序列结构
        self.z_proj = nn.Linear(d_z, d_z)

        # 条件融合投影：将 (z_seq_len + 4 + 3) 个 d_z 维 token 拼接后降维到 d_model
        # 拼接顺序：z_seq_len 个 z token + 4 个结构 token + 3 个密度 token
        self.cond_proj = nn.Linear((z_seq_len + 7) * d_z, d_model)

        # 纯 encoder Transformer，条件向量 c 通过 AdaLN 注入每一层
        self.transformer = Transformer(
            d_model=d_model, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers
        )

        self.output_fc = nn.Linear(d_model, num_classes)

    def forward(
        self,
        map: torch.Tensor,
        z: torch.Tensor,
        struct: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        # map: [B, H * W]
        # z: [B, z_seq_len, d_z]
        # struct: [B, 4]
        # density: [B, 3] — [door_level, monster_level, resource_level]

        # 结构标签：各自嵌入为独立 token，stack 成序列 [B, 4, d_z]
        e_struct = torch.stack([
            self.sym_embed(struct[:, 0]),
            self.room_embed(struct[:, 1]),
            self.branch_embed(struct[:, 2]),
            self.outer_embed(struct[:, 3])
        ], dim=1)

        # 密度标签：各自嵌入为独立 token，stack 成序列 [B, 3, d_z]
        e_density = torch.stack([
            self.door_density_embed(density[:, 0]),
            self.monster_density_embed(density[:, 1]),
            self.resource_density_embed(density[:, 2])
        ], dim=1)

        # z：逐 token 投影，保留序列结构 [B, z_seq_len, d_z]
        z_proj = self.z_proj(z)

        # 拼接所有条件 token → [B, z_seq_len+7, d_z]，展平后投影到 d_model
        cond_seq = torch.cat([z_proj, e_struct, e_density], dim=1)
        c = self.cond_proj(cond_seq.reshape(cond_seq.size(0), -1)) # [B, d_model]

        # tile embedding + 位置编码
        row_idx = torch.arange(self.map_h, device=map.device).repeat_interleave(self.map_w)
        col_idx = torch.arange(self.map_w, device=map.device).repeat(self.map_h)
        pos = self.row_embedding[0, row_idx] + self.col_embedding[0, col_idx] # [H*W, d_model]
        x = self.tile_embedding(map) + pos # [B, H * W, d_model]

        # Transformer：纯 encoder，每层通过 AdaLN 接收全局条件向量 c
        x = self.transformer(x, c) # [B, H * W, d_model]

        logits = self.output_fc(x) # [B, H * W, num_classes]
        return logits

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    map_input = torch.randint(0, 7, (4, 13 * 13)).to(device) # [4, 169]
    z_input = torch.randn(4, 6, 64).to(device) # [4, L*3, 64]
    struct_input = torch.tensor([
        [3, 1, 0, 1],
        [0, 2, 1, 0],
        [5, 1, 2, 1],
        [1, 0, 1, 0],
    ], dtype=torch.long).to(device) # [4, 4]
    density_input = torch.tensor([
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0],
        [0, 0, 1],
    ], dtype=torch.long).to(device) # [4, 3]

    model = GinkaMaskGIT(
        num_classes=7,
        d_model=256,
        d_z=64,
        dim_ff=1024,
        nhead=4,
        num_layers=6,
        map_h=13,
        map_w=13,
        z_seq_len=6
    ).to(device)

    print_memory(device, "初始化后")

    start = time.perf_counter()
    logits = model(map_input, z_input, struct_input, density_input)
    end = time.perf_counter()

    print_memory(device, "前向传播后")

    print(f"推理耗时: {end - start:.4f}s")
    print(f"输出形状: logits={logits.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Struct Projection parameters: {sum(p.numel() for p in model.struct_proj.parameters())}")
    print(f"Density Projection parameters: {sum(p.numel() for p in model.density_proj.parameters())}")
    print(f"Z Projection parameters: {sum(p.numel() for p in model.z_proj.parameters())}")
    print(f"Transformer parameters: {sum(p.numel() for p in model.transformer.parameters())}")
    print(f"Output FC parameters: {sum(p.numel() for p in model.output_fc.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
