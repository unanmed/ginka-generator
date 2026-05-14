# Stage1 二维空间感知改进设计文档

## 问题诊断

### 核心现象

第一阶段（floor/wall 骨架生成）质量显著劣于第二、三阶段。生成结果常见表现为：

- 墙壁分布破碎，缺乏连通性；
- 房间边界不完整，出现孤立墙块；
- 走廊未能形成闭合通道；
- 生成结果的空间拓扑结构与训练集分布偏差较大。

第二、三阶段（门/怪物/资源放置）问题相对较少，因为这些阶段的任务是"在已有结构上填充稀疏元素"，对空间结构的整体一致性要求较低。

### 根本原因：一维位置编码与二维网格的结构失配

#### 现有架构

`GinkaMaskGIT` 当前使用以下位置编码：

```python
self.pos_embedding = nn.Parameter(torch.randn(1, map_size, d_model) * 0.02)
```

这是一个纯一维可学习位置嵌入，将 13×13=169 个格子视为一条线性序列。`GinkaVQVAE` 中同样如此。

`maskGIT.py` 中的 `Transformer` 使用标准 `nn.TransformerEncoder` + `nn.TransformerDecoder`，注意力机制中没有任何二维空间偏置。

#### 一维展平带来的结构失配

将 13×13 地图按行展平后，位置关系如下：

```
位置(0,12) → token 12
位置(1, 0) → token 13
```

这两个 token 在一维序列中相邻（距离 1），但在二维地图上相距 12 列（横跨整行）。而真正的二维邻居关系：

```
位置(0, 0) 和 位置(1, 0) 的二维距离 = 1 格
对应 token 0 和 token 13，一维距离 = 13
```

一维位置嵌入告诉模型"token 0 和 token 13 相距较远"，但实际上它们是相邻的竖向邻格。注意力机制的相对偏置完全依赖位置嵌入的初始化，无法从一维嵌入中自动推断二维邻接关系。

#### 为何 Stage1 特别敏感

Stage1 负责生成 floor/wall 骨架，这是整个地图中**空间结构约束最强**的层次：

- 墙壁需要形成封闭或半封闭的房间边界；
- 走廊需要是连通的、宽度一致的通道；
- 整体拓扑（房间数、对称性、外围走廊）需要全局一致。

上述约束全部是**二维局部连通性约束**：一个墙壁格子是否合理，取决于它的上下左右四个邻格，而非它前后若干 token。一维位置编码使模型必须从数据中隐式学习这种行列边界，代价高昂且泛化差。

相比之下，Stage2/3 的任务（在走廊或房间内散布门/怪物/资源）对全局结构的空间一致性要求较低，位置编码的精确性影响较小。

---

## 改进方案

### 方案 A：二维因式分解位置嵌入（推荐首选）

#### 思路

将当前单一的一维位置嵌入替换为行嵌入与列嵌入的加和：

```
pos_embed[i, j] = row_embed[i] + col_embed[j]
```

这样，同一行的所有格子共享相同的行嵌入，同一列的所有格子共享相同的列嵌入。模型可以直接从嵌入中感知行列身份，而无需从一维序号中隐式推断。

#### 具体实现

```python
# 替换原有的 pos_embedding
self.row_embedding = nn.Parameter(torch.randn(1, MAP_H, d_model) * 0.02)
self.col_embedding = nn.Parameter(torch.randn(1, MAP_W, d_model) * 0.02)
```

前向传播中：

```python
# map: [B, H*W]
row_idx = torch.arange(MAP_H, device=map.device).repeat_interleave(MAP_W)
col_idx = torch.arange(MAP_W, device=map.device).repeat(MAP_H)
pos = self.row_embedding[0, row_idx] + self.col_embedding[0, col_idx]
x = self.tile_embedding(map) + pos.unsqueeze(0)
```

也可以预计算展开后的索引并缓存。

#### 特点

- 参数量变化：从 `169 × d_model` 变为 `13 × d_model + 13 × d_model = 26 × d_model`，显著减少，且参数共享有助于泛化；
- 无需修改注意力机制，改动最小；
- 直接赋予模型行列语义，改进效果立竿见影。

---

### 方案 B：二维相对位置偏置（推荐次选，与 A 叠加）

#### 思路

在注意力计算中，对每对 query-key 的打分加入一个可学习偏置，偏置由两个 token 的**相对行列偏移量**决定：

```
score(i, j) = (q_i · k_j) / sqrt(d) + B[Δrow, Δcol]
```

其中 `Δrow = row(i) - row(j)`，`Δcol = col(i) - col(j)`。偏置表 B 的形状为 `(2H-1, 2W-1)` = `(25, 25)`，每个注意力头各一张。

这种方式的核心优势：注意力打分天然理解"相邻格子应更强相关"，模型无需从位置嵌入中隐式学习距离感。

#### 具体实现

**步骤一：预计算相对位置索引表**

对于 13×13 的地图，预计算每对 token (i, j) 的相对位置，将二维偏移量映射为一维索引，供后续从偏置表中 gather：

```python
def build_relative_position_index(H: int, W: int) -> torch.Tensor:
    coords_h = torch.arange(H)
    coords_w = torch.arange(W)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij')) # [2, H, W]
    coords_flat = coords.flatten(1) # [2, H*W]
    rel = coords_flat[:, :, None] - coords_flat[:, None, :] # [2, H*W, H*W]
    rel[0] += H - 1
    rel[1] += W - 1
    rel_index = rel[0] * (2 * W - 1) + rel[1] # [H*W, H*W]
    return rel_index
```

**步骤二：在模型中注册偏置表**

需要自定义 `SelfAttentionWithRPB`，替换 `nn.TransformerEncoderLayer` 中的注意力：

```python
class SelfAttentionWithRPB(nn.Module):
    def __init__(self, d_model: int, nhead: int, H: int, W: int):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        # 偏置表：(2H-1) * (2W-1) 个可能的相对位置，每个头各一组
        self.rel_bias_table = nn.Parameter(
            torch.zeros(nhead, (2 * H - 1) * (2 * W - 1))
        )
        rel_index = build_relative_position_index(H, W) # [H*W, H*W]
        self.register_buffer('rel_index', rel_index.flatten()) # [H*W * H*W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.d_head)
        q, k, v = qkv.unbind(2) # 各 [B, N, nhead, d_head]
        q = q.permute(0, 2, 1, 3) # [B, nhead, N, d_head]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, nhead, N, N]

        # 从偏置表中取出对应偏置，reshape 成 [nhead, N, N]
        bias = self.rel_bias_table[:, self.rel_index].reshape(self.nhead, N, N)
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)
```

**步骤三：替换 TransformerEncoderLayer 中的注意力**

由于 PyTorch 标准 `TransformerEncoderLayer` 不支持直接替换注意力实现，需要手写包含 RPB 的编码器层：

```python
class RPBEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, H: int, W: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttentionWithRPB(d_model, nhead, H, W)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

#### 参数量评估

对于 Stage1 MaskGIT（d_model=192, nhead=8, num_layers=6，H=W=13）：

- 偏置表：每层 `8 × 25 × 25 = 5000` 参数
- 共 6 层：30,000 参数
- 占总参数量的比例极低，但对 attention 的几何感知能力提升显著。

#### 特点

- 与方案 A 互补，可叠加使用；
- 理论上最接近"正确"的二维注意力感应偏置；
- 实现较复杂，需要手写注意力层；
- 参数量增加极少。

---

### 方案 C：轴向注意力（Axial Attention）

#### 思路

将每一个标准自注意力层替换为两个顺序执行的注意力：

1. **行轴注意力**：每一行内的格子互相注意，13 个 group，每组 13 个 token；
2. **列轴注意力**：每一列内的格子互相注意，13 个 group，每组 13 个 token。

两种注意力交替叠加：`Row → Col → Row → Col → ...`

```
标准自注意力：O((H*W)²) = O(169²) = 28,561
轴向注意力：  O(H * W²) + O(W * H²) = O(H*W*(H+W)) = O(169*26) ≈ 4,394
```

复杂度大幅下降，且强制模型以行/列为单位建立空间关联。

#### 具体实现

```python
class AxialAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W
        self.row_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_row = nn.LayerNorm(d_model)
        self.norm_col = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, W = self.H, self.W
        x = x.reshape(B, H, W, C)

        # 行注意力：每行的 W 个 token 互相注意
        x_row = x.reshape(B * H, W, C)
        x_row, _ = self.row_attn(x_row, x_row, x_row)
        x = x + x_row.reshape(B, H, W, C)
        x = self.norm_row(x)

        # 列注意力：每列的 H 个 token 互相注意
        x_col = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        x_col, _ = self.col_attn(x_col, x_col, x_col)
        x = x + x_col.reshape(B, W, H, C).permute(0, 2, 1, 3)
        x = self.norm_col(x)

        return x.reshape(B, N, C)


class AxialEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, H: int, W: int):
        super().__init__()
        self.axial_attn = AxialAttention(d_model, nhead, H, W)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.axial_attn(x)
        x = x + self.ffn(self.norm_ff(x))
        return x
```

#### 特点

- 最强的二维结构归纳偏置：明确区分行内关系和列内关系；
- 改造最彻底，需要替换整个 Transformer 编码器；
- 计算量低于标准全局注意力；
- 不适合 cross-attention 部分（decoder 中 map token cross-attend z 的部分保持不变）。

---

### 方案 D：双轴并行输入（Dual-Axis Input）

#### 思路

同一张地图展平两次，分别按行优先和列优先展平，两份序列并行送入各自的嵌入+编码器，最终在 token 维度相加合并：

```
地图 [H, W]
  ├─ 行优先展平 → [H*W] → Embedding + 行位置编码 → 编码器 A → [H*W, d_model]
  └─ 列优先展平 → [W*H] → Embedding + 列位置编码 → 编码器 B → [W*H, d_model]
       ↓ 重排到行优先顺序
  相加合并 → [H*W, d_model] → Decoder + z → logits
```

两个编码器可以共享权重（只有位置编码不同），以减少参数量。

#### 特点

- 无需修改注意力机制，复用现有 Transformer；
- 计算量加倍（两次编码），但可通过共享权重缓解；
- 同时为模型提供横向和纵向的序列上下文；
- 参数复用程度较高（共享编码器权重时）。

---

## 方案对比

| 方案                | 改动范围     | 额外参数量                         | 二维感知能力 | 实现复杂度 |
| ------------------- | ------------ | ---------------------------------- | ------------ | ---------- |
| A：二维因式位置嵌入 | 位置嵌入层   | 减少（共享）                       | 中等         | 低         |
| B：二维相对位置偏置 | 注意力层     | 极少（~30k）                       | 强           | 中等       |
| C：轴向注意力       | 整个编码器   | 基本不变                           | 最强         | 高         |
| D：双轴并行输入     | 输入与编码器 | 按编码器大小翻倍（共享权重则不变） | 中等         | 中等       |

---

## 推荐实施策略

### 第一步：替换位置嵌入（方案 A）

优先实施方案 A，这是改动最小、风险最低的基础改进。仅需修改 `GinkaMaskGIT` 和 `GinkaVQVAE` 中的 `pos_embedding` 初始化与使用方式。

修改范围：

- `ginka/maskGIT/model.py`：替换 `pos_embedding`，调整 `forward` 中的位置嵌入加法；
- `ginka/vqvae/model.py`：同上。

### 第二步：叠加二维相对位置偏置（方案 B）

在方案 A 的基础上，为 `GinkaMaskGIT` 的编码器部分叠加 RPB，只需新增一个自定义 `RPBEncoderLayer`，替换 `maskGIT.py` 中 `Transformer.encoder` 的层类型。

VQ-VAE 编码器的 RPB 改造优先级较低（VQ-VAE 负责全图压缩，对局部连通性感知需求低于 MaskGIT）。

### 可选第三步：轴向注意力替换（方案 C）

若前两步改进后 Stage1 质量仍不满足要求，可进一步将 `GinkaMaskGIT` 的编码器改为轴向注意力。由于 decoder（cross-attention 部分）不涉及空间 token，无需改动。

### 其他注意事项

- 上述改动仅针对 **Stage1 MaskGIT**。Stage2/3 可以同步修改，也可以保持原结构，视实际效果决定；
- 改动后需重置训练，位置嵌入的结构变化会使旧检查点不兼容（shape 不匹配）；
- 验证时重点观察生成地图中墙壁的连通性、房间闭合度和整体拓扑是否改善。
