# 条件注入方式改进：从 Cross Attention 到 FiLM / AdaLN

## 问题背景

### 当前条件注入方式

`GinkaMaskGIT` 当前使用的条件注入策略如下：

1. VQ 码字 `z`（形状 `[B, L*3, d_z]`）通过 `z_proj` 投影到 `d_model` 维度
2. 结构标签（`sym / room / branch / outer`）各自嵌入后拼接为 `[B, 4, d_model]`
3. 密度标签（`door / monster / resource`）三个嵌入相加后经 MLP 得到 `[B, 1, d_model]`
4. 上述三部分拼接为 `memory`（`[B, L*3+5, d_model]`），作为 cross-attention 的 key/value
5. Transformer decoder 以 map token 作为 query，对 `memory` 做 cross-attention

### 问题分析

Cross-attention 的本质是**查询驱动**（query-driven）的检索机制：模型只在需要时才主动去 `memory` 中寻找相关信息，且注意力权重由 query（地图 token）与 key 的相似度决定。

这一机制对**空间局部条件**（如参考图像特征、空间先验）效果良好，但对**全局标量条件**（如"资源密度为 High"）存在以下问题：

#### 1. 隐式性：条件无法强制生效

模型可以选择性地"忽视"某个 memory 条目。结构/密度条件只是 memory 序列中的几个 token，与 VQ 码字并列竞争注意力权重。当 VQ 码字已经携带了足够多的生成信息时，模型倾向于将注意力集中在 VQ 码字上，而对结构/密度 token 的注意力权重趋近于零。

实验现象印证了这一点：即使将密度标签设置为 High，模型生成的怪物/资源数量与 Low 时差异极小，说明密度条件被模型基本忽略。

#### 2. 语义不匹配：全局信号与局部查询不对齐

Cross-attention 的设计假设 key/value 携带**空间位置相关**的信息（例如编码器输出的特征图），query 在不同位置关注不同的 key。然而：

- 密度标签是一个全局标量（表示整张地图的资源密度档位），没有空间维度
- 所有地图位置（169 个 token）的 query 若都要接收该全局信号，需要所有 query 一致地高度关注同一个 key，这与 cross-attention 的设计初衷相悖

#### 3. 与 VQ 码字竞争导致梯度稀释

结构/密度条件作为 memory token，与 VQ 码字通过同一个 softmax 竞争注意力。当 VQ 码字数量远多于条件 token（当前 L\*3=6 对 5），且 VQ 码字携带了更多"有用信息"时，梯度信号倾向于强化对 VQ 的关注，条件 token 的参数得不到有效更新。

#### 4. VQ 码字 z 本身也未被充分利用

即使将结构/密度从 cross-attention 中移出，VQ 码字 `z` 本身也存在相同的问题。训练前期观察到模型倾向于输出高度相似的地图（风格单一、多样性极低），这表明模型并未有效利用随机采样的 `z`。根本原因相同：cross-attention 是 query-driven 的，模型可以在不关注 `z` 的情况下仅靠地图 token 自注意力完成预测，`z` 的梯度信号因此极为稀弱。因此，`z` 同样需要改为全局 AdaLN 注入，而非仅依赖 cross-attention。

---

## 改进方案

### 核心思路

全局条件（结构标签、密度标签）应当作用于 **每一层的特征变换**，以加法偏移或缩放仿射的形式强制施加到所有 map token 上，使模型**无法绕过**该条件。这正是 FiLM 和 AdaLN 的设计目标。

### FiLM（Feature-wise Linear Modulation）

FiLM 对特征向量做逐元素仿射变换：

$$
\text{FiLM}(x, c) = \gamma(c) \odot x + \beta(c)
$$

其中 $\gamma(c)$ 和 $\beta(c)$ 是从条件 $c$ 预测出的缩放和偏移向量（维度均为 `d_model`），$\odot$ 为逐元素乘法。

FiLM 直接修改特征分布，条件信号强制影响所有 token 的表示，而不依赖 query 主动发起的检索。

### AdaLN（Adaptive Layer Normalization）

AdaLN 将 FiLM 与 LayerNorm 结合，用条件向量预测 LayerNorm 的缩放和偏移参数，替代原有的固定参数：

$$
\text{AdaLN}(x, c) = \gamma(c) \odot \frac{x - \mu}{\sigma} + \beta(c)
$$

与标准 LayerNorm 的区别仅在于 $\gamma$ 和 $\beta$ 不是可学习的静态参数，而是由条件 $c$ 动态生成。AdaLN 在 DiT（Diffusion Transformer）和 MaskGIT 的改进版本中已有广泛验证。

**选用 AdaLN** 作为主要方案，理由：

- 在 Transformer 架构中，LayerNorm 是特征归一化的核心节点，在此处注入条件效果最稳定
- AdaLN 的参数量增加极少（仅新增 `2 * d_model` 的线性层输出）
- 与 FiLM 效果等价，但更符合 Transformer 的设计惯例

---

## 架构设计

### 条件向量的构建

将结构标签、密度标签和 VQ 码字 `z` 全部融合为**单一全局条件向量** `c`（维度 `d_model`），通过 AdaLN 在每一层强制施加到所有 map token 上。

**结构标签**（4 个离散标量）各自独立嵌入后**拼接**，再经 Linear 投影：

```
struct: [B, 4] → 各自 Embedding(d_cond) → cat → [B, 4*d_cond] → Linear → [B, d_model]
```

**密度标签**（3 个离散标量）各自独立嵌入后**拼接**，再经 Linear 投影（不使用相加，避免各档位嵌入相互抵消）：

```
density: [B, 3] → 各自 Embedding(d_cond) → cat → [B, 3*d_cond] → Linear → [B, d_model]
```

**VQ 码字 z**（序列）先做均值池化压缩为单个向量，再经 Linear 投影：

```
z: [B, L*3, d_z] → mean(dim=1) → [B, d_z] → Linear → [B, d_model]
```

三路向量相加得到最终条件向量：

```
c = struct_vec + density_vec + z_vec  # [B, d_model]
```

> 说明：`z` 改为全局注入的动机在于，训练前期模型观察到输出地图高度相似、多样性极低，表明 cross-attention 方式下模型未能有效利用随机采样的 `z`。均值池化保留了 `z` 序列的整体语义，同时将其压缩为标量条件，适合 AdaLN 注入。

### 自定义 Transformer 层

由于 PyTorch 的 `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` 不支持外部注入 AdaLN 参数，需要自行实现：

#### AdaLN 模块

```python
class AdaLN(nn.Module):
    # 自适应 LayerNorm：用条件向量 c 预测 LayerNorm 的 gamma 和 beta
    def __init__(self, d_model: int, d_cond: int):
        ...
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, d_model * 2)  # 输出 [gamma, beta]

    def forward(self, x, c):
        # x: [B, S, d_model]
        # c: [B, d_model]  全局条件向量
        gamma, beta = self.proj(c).chunk(2, dim=-1)  # 各 [B, d_model]
        return (1 + gamma.unsqueeze(1)) * self.norm(x) + beta.unsqueeze(1)
```

#### 自定义 Transformer 层

替换标准的 `TransformerEncoderLayer`，在每个 sub-layer 的 LayerNorm 处注入条件：

```python
class CondTransformerLayer(nn.Module):
    # 带 AdaLN 条件注入的 Transformer Encoder 层
    # 结构：AdaLN-Self-Attn → AdaLN-FFN
    def __init__(self, d_model, nhead, dim_ff, d_cond):
        ...
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.adaln1 = AdaLN(d_model, d_cond)  # 自注意力前的归一化
        self.adaln2 = AdaLN(d_model, d_cond)  # FFN 前的归一化
        self.ffn = nn.Sequential(Linear, GELU, Linear)

    def forward(self, x, c, key_padding_mask=None):
        # Pre-norm 结构
        residual = x
        x = self.adaln1(x, c)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        residual = x
        x = self.adaln2(x, c)
        x = self.ffn(x)
        x = residual + x
        return x
```

#### Cross-attention 层（移除）

`z` 已改为通过均值池化后加入全局条件向量 `c`，由 AdaLN 注入每一层，不再需要单独的 cross-attention 层。整个 Transformer 退化为纯 encoder（自注意力）结构，仅由 `CondTransformerLayer` 堆叠而成，无 decoder。

### 整体前向流程

```
map → tile_embed + pos_embed → x  [B, H*W, d_model]

struct: [B, 4] → 各自 Embed → cat → Linear → [B, d_model]
density: [B, 3] → 各自 Embed → cat → Linear → [B, d_model]
z: [B, L*3, d_z] → mean → Linear → [B, d_model]
c = struct_vec + density_vec + z_vec              # [B, d_model]

for each layer:
    x = CondTransformerLayer(x, c)               # AdaLN 自注意力，纯 encoder 结构

logits = output_fc(x)  [B, H*W, num_classes]
```

---

## 参数量对比

以 `d_model=256, nhead=4, dim_ff=1024, num_layers=6` 为基准估算：

| 模块                | 当前方案                 | 新方案（AdaLN）                              |
| ------------------- | ------------------------ | -------------------------------------------- |
| 条件嵌入层          | 小（各 Embedding + MLP） | 小（相似，略有增加）                         |
| 每层 AdaLN 额外参数 | 0                        | `2 * d_model * d_model = 131K` × 6 层 ≈ 786K |
| cross-attention 层  | 6 层完整 decoder         | 0（移除，z 改为 AdaLN 全局注入）             |
| 总参数量变化        | 基准                     | +约 5~10%（可接受）                          |

---

## 实现文件规划

| 文件                       | 改动内容                                                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `ginka/maskGIT/maskGIT.py` | 重写 `Transformer` 为自定义纯 encoder 架构，新增 `AdaLN`、`CondTransformerLayer`；移除 `ZCrossAttentionLayer`                     |
| `ginka/maskGIT/model.py`   | 更新 `GinkaMaskGIT`：struct/density/z 三路融合为条件向量 `c`，密度标签改为拼接，z 改为均值池化后注入；移除旧 cross-attention 路径 |
| `ginka/train_seperated.py` | 无需修改（接口不变，`forward` 签名保持）                                                                                          |

---

## 预期效果

- 密度标签、结构标签、VQ 码字 `z` 三路均通过 AdaLN 在每一层强制影响特征分布，模型无法绕过任何一路条件
- 密度标签改为拼接（而非相加），避免不同档位嵌入线性叠加时相互抵消，使各密度维度保持独立的表示空间
- `z` 通过均值池化压缩为全局向量后注入，保留 codebook 多样性的同时消除对 cross-attention 的依赖，预期解决训练前期输出地图高度相似的问题
- 架构简化为纯 encoder，去掉 encoder-decoder 分离结构，降低实现复杂度和计算量
