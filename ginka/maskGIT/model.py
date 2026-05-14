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

class GinkaMaskGIT(nn.Module):
    def __init__(
        self, num_classes: int = 16, d_model: int = 192, dim_ff: int = 512,
        nhead: int = 8, num_layers: int = 4, map_h: int = 13, map_w: int = 13, d_z: int = 64
    ):
        super().__init__()
        self.map_h = map_h
        self.map_w = map_w

        # Tile 嵌入 + 二维因式分解位置编码
        self.tile_embedding = nn.Embedding(num_classes, d_model)
        self.row_embedding = nn.Parameter(torch.randn(1, map_h, d_model) * 0.02)
        self.col_embedding = nn.Parameter(torch.randn(1, map_w, d_model) * 0.02)

        # z 投影：将 VQ 码字从 d_z 维映射到 d_model 维，供 cross-attention 使用
        self.z_proj = nn.Sequential(
            nn.Linear(d_z, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 结构标签嵌入（编码到 d_z 维度）
        # 注意：结构标签与 VQ 码字语义不同，使用独立投影层避免混用
        self.sym_embed = nn.Embedding(SYM_VOCAB, d_z)
        self.room_embed = nn.Embedding(ROOM_VOCAB, d_z)
        self.branch_embed = nn.Embedding(BRANCH_VOCAB, d_z)
        self.outer_embed = nn.Embedding(OUTER_VOCAB, d_z)

        self.struct_proj  = nn.Sequential(
            nn.Linear(d_z, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Transformer：encoder 做 map token 自注意力，decoder 做与 z 的 cross-attention
        self.transformer = Transformer(
            d_model=d_model, dim_ff=dim_ff, nhead=nhead, num_layers=num_layers
        )

        self.output_fc = nn.Linear(d_model, num_classes)

    def forward(
        self,
        map: torch.Tensor,
        z: torch.Tensor,
        struct: torch.Tensor
    ) -> torch.Tensor:
        # map: [B, H * W]
        # z: [B, L * 3, d_z]
        # struch: [B, 4]

        sym_idx = struct[:, 0]
        room_idx = struct[:, 1]
        branch_idx = struct[:, 2]
        outer_idx = struct[:, 3]

        # 嵌入结构标签到 d_z 维度，拼接到 z 序列末尾
        e_sym = self.sym_embed(sym_idx).unsqueeze(1) # [B, 1, d_z]
        e_room = self.room_embed(room_idx).unsqueeze(1) # [B, 1, d_z]
        e_branch = self.branch_embed(branch_idx).unsqueeze(1) # [B, 1, d_z]
        e_outer = self.outer_embed(outer_idx).unsqueeze(1) # [B, 1, d_z]

        struct_seq = torch.cat([e_sym, e_room, e_branch, e_outer], dim=1) # [B, 4, d_z]

        # VQ 码字与结构标签语义不同，使用各自独立的投影层后再拼接
        z_mem_vq = self.z_proj(z) # [B, L, d_model]
        z_mem_struct = self.struct_proj(struct_seq) # [B, 4, d_model]
        z_mem = torch.cat([z_mem_vq, z_mem_struct], dim=1) # [B, L * 3 + 4, d_model]

        # tile embedding + 位置编码
        row_idx = torch.arange(self.map_h, device=map.device).repeat_interleave(self.map_w)
        col_idx = torch.arange(self.map_w, device=map.device).repeat(self.map_h)
        pos = self.row_embedding[0, row_idx] + self.col_embedding[0, col_idx] # [H*W, d_model]
        x = self.tile_embedding(map) + pos # [B, H * W, d_model]

        # Transformer：encoder 做 map 自注意力，decoder cross-attend z+struct
        x = self.transformer(x, memory=z_mem) # [B, H * W, d_model]

        logits = self.output_fc(x) # [B, H * W, num_classes]
        return logits

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    map_input = torch.randint(0, 7, (4, 13 * 13)).to(device) # [4, 169]
    z_input = torch.randn(4, 2, 64).to(device) # [4, 2, 64]
    struct_input = torch.tensor([
        [3, 1, 0, 1],
        [0, 2, 1, 0],
        [5, 1, 2, 1],
        [1, 0, 1, 0],
    ], dtype=torch.long).to(device) # [4, 4]

    model = GinkaMaskGIT(
        num_classes=7,
        d_model=192,
        d_z=64,
        dim_ff=2048,
        nhead=8,
        num_layers=6,
        map_h=13,
        map_w=13
    ).to(device)

    print_memory(device, "初始化后")

    start = time.perf_counter()
    logits = model(map_input, z_input, struct_input)
    end = time.perf_counter()

    print_memory(device, "前向传播后")

    print(f"推理耗时: {end - start:.4f}s")
    print(f"输出形状: logits={logits.shape}")
    print(f"Tile Embedding parameters: {sum(p.numel() for p in model.tile_embedding.parameters())}")
    print(f"Z Projection parameters: {sum(p.numel() for p in model.z_proj.parameters())}")
    print(f"Struct Projection parameters: {sum(p.numel() for p in model.struct_proj.parameters())}")
    print(f"Transformer parameters: {sum(p.numel() for p in model.transformer.parameters())}")
    print(f"Output FC parameters: {sum(p.numel() for p in model.output_fc.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
