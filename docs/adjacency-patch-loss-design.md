# 邻接损失与 Patch 损失设计文档

## 1. 问题分析

当前三阶段 MaskGIT 训练仅使用逐格 Cross Entropy（CE）损失，每个格子的损失完全独立计算。然而地图生成任务中不同格子之间存在强烈的空间相关性：

- **邻接关系**：两格是否同为可通行区域（地板）是地图连通性的基础，孤立的地板块意味着该区域无法到达
- **局部一致性**：一个格子的类别高度依赖其周围格子的布局，例如门必须出现在墙与地板的交界处、怪物通常成簇分布

CE 损失将每个格子的预测视为独立的多分类问题，无法显式建模这些空间依赖，导致生成的地图在连通性和局部结构上表现不佳。

本方案引入两种互补的空间损失函数：

| 损失类型 | 作用范围 | 核心机制 | 主要受益阶段 |
| -------- | -------- | -------- | ------------ |
| 邻接损失 | 紧邻四邻域 | BCE 约束相邻两格同时为空地的概率 | 阶段一（墙壁骨架） |
| Patch 损失 | K×K 邻域 | 高斯核加权 CE 平滑，使损失梯度在空间上扩散 | 全部三个阶段 |

## 2. 邻接损失

### 2.1 设计思路

对于地图中的每一条相邻边（共 312 条：13×12 水平 + 12×13 垂直），计算两端格子均为空地的联合概率，并与真实标注做 BCE（二元交叉熵）对比。

$$
P(\text{both floor})_{(u,v)} = P(\text{floor} \mid u) \cdot P(\text{floor} \mid v)
$$

其中 $P(\text{floor} \mid u) = \text{softmax}(\text{logits}_u)[\text{class}=0]$，即第 0 类（地板）的 softmax 概率。

真实标注为二元值：若目标地图中两格均为地板（ID=0），则为 1；否则为 0。

**直觉解释**：
- 若两格都是地板，BCE 推动二者的 floor 概率同时提高，强化连通区域的一致性
- 若一格是墙、一格是地板，BCE 推动墙格的 floor 概率降低，与 CE 损失形成梯度叠加
- 若两格都是墙，BCE 推动二者的 floor 概率同时降低，与 CE 一致

邻接损失的核心价值不在于引入新信息，而在于**改变损失曲面的几何形状**——通过将逐格独立损失改为逐边联合损失，使相邻格子的梯度互相耦合。当一个格子的预测出错时，其四个邻居的梯度也会受到影响，从而加速模型学习空间一致性。

### 2.2 数学定义

设地图尺寸为 $H \times W = 13 \times 13$，logits 形状为 $[B, S, C]$，目标形状为 $[B, S]$（$S = 169$，$C = 8$）。

对水平边 $(i,j) \leftrightarrow (i,j+1)$（共 $H \times (W-1) = 156$ 条）：

$$
\mathcal{L}_{\text{adj}}^h = \frac{1}{B \cdot H \cdot (W-1)} \sum_b \sum_{i=0}^{H-1} \sum_{j=0}^{W-2} \text{BCE}\left(
    p_{b,i,j}^f \cdot p_{b,i,j+1}^f,\;
    \mathbb{I}[t_{b,i,j} = 0 \land t_{b,i,j+1} = 0]
\right)
$$

对垂直边 $(i,j) \leftrightarrow (i+1,j)$（共 $(H-1) \times W = 156$ 条）：

$$
\mathcal{L}_{\text{adj}}^v = \frac{1}{B \cdot (H-1) \cdot W} \sum_b \sum_{i=0}^{H-2} \sum_{j=0}^{W-1} \text{BCE}\left(
    p_{b,i,j}^f \cdot p_{b,i+1,j}^f,\;
    \mathbb{I}[t_{b,i,j} = 0 \land t_{b,i+1,j} = 0]
\right)
$$

总邻接损失为二者均值：

$$
\mathcal{L}_{\text{adj}} = \frac{1}{2} (\mathcal{L}_{\text{adj}}^h + \mathcal{L}_{\text{adj}}^v)
$$

### 2.3 实现

```python
import torch
import torch.nn.functional as F

def adjacency_loss(logits, target):
    # logits: [B, S, C] — MaskGIT 解码器输出
    # target: [B, S] — 目标类别 ID，不含 MASK 标记
    B, S, C = logits.shape
    H = W = 13
    probs = F.softmax(logits, dim=-1) # [B, S, C]
    p_floor = probs[:, :, 0].view(B, H, W) # [B, H, W] — 地板概率
    t = target.view(B, H, W) # [B, H, W]
    t_floor = (t == 0).float() # 地板标注为 1，其余为 0

    # 水平边：左格 × 右格
    joint_h = p_floor[:, :, :-1] * p_floor[:, :, 1:] # [B, H, W-1]
    target_h = t_floor[:, :, :-1] * t_floor[:, :, 1:] # [B, H, W-1]

    # 垂直边：上格 × 下格
    joint_v = p_floor[:, :-1, :] * p_floor[:, 1:, :] # [B, H-1, W]
    target_v = t_floor[:, :-1, :] * t_floor[:, 1:, :] # [B, H-1, W]

    loss_h = F.binary_cross_entropy(joint_h, target_h, reduction='mean')
    loss_v = F.binary_cross_entropy(joint_v, target_v, reduction='mean')
    return (loss_h + loss_v) / 2.0
```

**边界处理**：只计算完整存在的边（水平 13×12=156 条，垂直 12×13=156 条），共计 312 条边。边界格子的某些方向没有邻居，自然不产生该方向的边。

**阶段适用性**：邻接损失直接对地板-地板连通性建模，在三个阶段中的适用程度不同：

| 阶段 | target 地板占比 | 邻接损失有效性 | 说明 |
| ---- | --------------- | -------------- | ---- |
| 一   | ~50%            | 高             | 地板/墙壁二分类，地板连通性决定地图骨架质量 |
| 二   | ~70%            | 中             | 墙壁已降级为地板，地板占比更高，但仍有门与怪物放置的地板连通约束 |
| 三   | ~90%            | 低             | 仅保留地板与资源，大部分区域为地板，BCE 信号趋弱 |

**建议**：邻接损失默认仅作用于阶段一。若想对所有阶段生效，可为二、三阶段设置更小的权重 `LAMBDA_ADJ_STAGE2`、`LAMBDA_ADJ_STAGE3`。

### 2.4 超参数

| 参数 | 建议值 | 说明 |
| ---- | ------ | ---- |
| `LAMBDA_ADJ` | 0.1 | 邻接损失总权重，作用于阶段一 |
| `LAMBDA_ADJ_STAGE2` | 0.05 | 阶段二邻接损失权重（可选，默认与 LAMBDA_ADJ 相同） |
| `LAMBDA_ADJ_STAGE3` | 0.02 | 阶段三邻接损失权重（可选） |

### 2.5 扩展方向

当前方案仅建模地板-地板成对关系。可扩展为多类别邻接矩阵：

- 对 7 个有效类别两两组合得到 7×7 = 49 种邻接关系
- 计算联合概率 `P(class_a | u) * P(class_b | v)` 得到 49 维向量
- 与目标地图的 one-hot 邻接关系做 CE

多类邻接能显式建模"门必须出现在墙与地板交界""怪物应位于地板上"等约束，但计算量较大（49 × 312 = 15,288 对/样本），建议作为地板块邻接验证有效后的进阶方案。

## 3. Patch 损失

### 3.1 设计思路

逐格 CE 损失的梯度仅作用于单格。Patch 损失通过高斯核对每格的 CE 损失进行空间卷积，使一个格子的损失值等于其 K×K 邻域内所有格子 CE 的加权平均。这带来的效果是：

- 一个格子的预测错误会通过高斯核"扩散"到其邻居，邻居的梯度也会向纠错方向移动
- 等效于增强了局部区域的梯度信号，使模型更难忽视局部上下文
- 相比邻接损失关心特定类别关系（地板-地板），Patch 损失是**类别无关**的空间平滑，对所有 tile 类型均有效

**与邻接损失的区别**：

| 维度 | 邻接损失 | Patch 损失 |
| ---- | -------- | ---------- |
| 作用范围 | 仅紧邻四邻域（距离 1） | K×K 邻域（距离 2，可调） |
| 类别感知 | 是（地板-地板特定） | 否（CE 已包含类别） |
| 权重机制 | 等权（乘积） | 高斯衰减权重 |
| 梯度耦合 | 乘性（P_A × P_B） | 加性（加权 CE 求和） |

### 3.2 数学定义

设 CE 损失在每格的定义为（不做 reduction）：

$$
\text{ce}_{b,i,j} = -\log \frac{\exp(\text{logits}_{b,i,j}[t_{b,i,j}])}{\sum_{c=0}^{C-1} \exp(\text{logits}_{b,i,j}[c])}
$$

高斯核 $G \in \mathbb{R}^{K \times K}$，中心 $(k_c, k_c) = (\lfloor K/2 \rfloor, \lfloor K/2 \rfloor)$：

$$
G_{u,v} = \frac{1}{Z} \exp\left(-\frac{(u - k_c)^2 + (v - k_c)^2}{2\sigma^2}\right), \quad Z = \sum_{u,v} G_{u,v}
$$

Patch 损失定义为逐格平滑后的 CE 的均值：

$$
\mathcal{L}_{\text{patch}} = \frac{1}{B \cdot H \cdot W} \sum_b \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1}
    G_{u,v} \cdot \text{ce}_{b,\; i+u-k_c,\; j+v-k_c}
$$

其中越界的 $\text{ce}$ 用边缘复制（replicate padding）填充。

### 3.3 实现

```python
import torch
import torch.nn.functional as F

def gaussian_kernel(kernel_size, sigma, device):
    # 生成归一化二维高斯卷积核 [1, 1, K, K]
    k = kernel_size
    center = (k - 1) / 2.0
    xs = torch.arange(k, dtype=torch.float32, device=device) - center
    gx = torch.exp(-xs ** 2 / (2.0 * sigma ** 2))
    gy = torch.exp(-xs ** 2 / (2.0 * sigma ** 2))
    g2d = gx[:, None] * gy[None, :] # [K, K]
    g2d = g2d / g2d.sum() # 归一化
    return g2d.view(1, 1, k, k)

def patch_loss(logits, target, kernel_size=5, sigma=1.2):
    # logits: [B, S, C]
    # target: [B, S]
    B, S, C = logits.shape
    H = W = 13

    # 逐格 CE（不做 reduction）
    ce = F.cross_entropy(
        logits.reshape(-1, C), target.reshape(-1), reduction='none'
    ).view(B, H, W) # [B, H, W]

    # 高斯核
    kernel = gaussian_kernel(kernel_size, sigma, logits.device) # [1, 1, K, K]

    # replicate 填充后用 unfold 提取邻域
    pad = kernel_size // 2
    ce_padded = F.pad(ce.view(B, 1, H, W), (pad, pad, pad, pad), mode='replicate')
    # patches: [B, K*K, H*W]
    patches = F.unfold(ce_padded, kernel_size=(kernel_size, kernel_size))
    patches = patches.view(B, kernel_size * kernel_size, H, W) # [B, K*K, H, W]

    # 加权求和
    k_flat = kernel.view(1, kernel_size * kernel_size, 1, 1) # [1, K*K, 1, 1]
    smoothed = (patches * k_flat).sum(dim=1) # [B, H, W]

    return smoothed.mean()
```

**设计要点**：

- `replicate` 填充：边界格子的邻域超出地图范围时，用最近的边缘值填充，避免引入零值的虚假信号
- 核归一化：高斯核归一化到和为 1，确保 Patch 损失的数值量级与原始 CE 大致相当，便于权重调参
- `F.unfold` 实现：比手动循环或 `conv2d` 更高效，在 CUDA 上有针对性的算子优化

### 3.4 超参数

| 参数 | 建议值 | 说明 |
| ---- | ------ | ---- |
| `PATCH_KERNEL_SIZE` | 5 | 卷积核尺寸，5×5 覆盖约 15% 的地图面积 |
| `PATCH_SIGMA` | 1.2 | 高斯核标准差，控制权重衰减速度。σ 越大平滑越强 |
| `LAMBDA_PATCH` | 0.1 | Patch 损失总权重 |
| `LAMBDA_PATCH_STAGE1` | 0.1 | 阶段一 Patch 权重（可选，默认同 LAMBDA_PATCH） |
| `LAMBDA_PATCH_STAGE2` | 0.1 | 阶段二 Patch 权重 |
| `LAMBDA_PATCH_STAGE3` | 0.1 | 阶段三 Patch 权重 |

**Kernel Size 选择**：13×13 地图上，5×5 是合理的起点（覆盖每个方向 ±2 格）。若效果不明显可尝试 7×7，但需注意边界效应增强。

**Sigma 选择**：σ = 1.2 在 5×5 内权重分布为：

| 距离 | 0 | 1 | 2 |
| ---- | --- | --- | --- |
| 权重 | ~0.273 | ~0.164 | ~0.053 |

中心权重约为边缘的 5 倍，保证自身 CE 仍占主导，同时给予邻居适度的梯度贡献。

### 3.5 扩展方向

**可学习卷积核**：将固定高斯核替换为可训练参数 `nn.Parameter`，在训练过程中自适应学习最优的空间权重分布。实现简单，只需将 `gaussian_kernel` 改为可学习参数即可。

**多尺度 Patch**：同时使用 3×3、5×5、7×7 三种核，各自的平滑结果取加权和，使模型在不同空间尺度上均受到约束。

**类感知核**：不同类别使用不同的平滑权重。例如墙壁类不应与怪物类互相平滑，可以按类别加权。实现时需在 CE 中保留类别维度，对每类分别卷积。

## 4. 总损失组合

最终训练损失为原有各项与新两项的加权和：

$$
\begin{aligned}
\mathcal{L}_{\text{total}} = &
\sum_{s=1}^{3} w_{\text{ce}}^s \cdot \mathcal{L}_{\text{ce}}^s \\
+ & \sum_{s=1}^{3} \lambda_{\text{adj}}^s \cdot \mathcal{L}_{\text{adj}}^s \\
+ & \sum_{s=1}^{3} \lambda_{\text{patch}}^s \cdot \mathcal{L}_{\text{patch}}^s \\
+ & \beta \cdot \mathcal{L}_{\text{commit}} + \beta_{\text{dist}} \cdot \mathcal{L}_{\text{commit\_dist}}
\end{aligned}
$$

其中 $w_{\text{ce}}^s$ 即现有的 `STAGE1_CE_WEIGHT` / `STAGE2_CE_WEIGHT` / `STAGE3_CE_WEIGHT`（默认均为 1.0）。

**推荐权重配置**：

| 损失项 | 阶段一 | 阶段二 | 阶段三 |
| ------ | ------ | ------ | ------ |
| CE 损失 | 1.0 | 1.0 | 1.0 |
| 邻接损失 | 0.1 | 0.05 | 0.02 |
| Patch 损失 | 0.1 | 0.1 | 0.1 |

阶段一地板/墙壁二分类是连通性建模的核心，邻接损失权重最高。阶段二、三地板占比渐增，邻接损失的判别力减弱，权重递减。Patch 损失作为类别无关的平滑约束，三个阶段等权重即可。

### 与现有超参数的位置关系

新增超参数应追加在 `train_seperated.py` 现有超参数块（第 42-118 行）之后：

```python
# 邻接损失权重（三阶段）
LAMBDA_ADJ1 = 0.1
LAMBDA_ADJ2 = 0.05
LAMBDA_ADJ3 = 0.02

# Patch 损失权重（三阶段）及核参数
LAMBDA_PATCH1 = 0.1
LAMBDA_PATCH2 = 0.1
LAMBDA_PATCH3 = 0.1
PATCH_KERNEL_SIZE = 5
PATCH_SIGMA = 1.2
```

## 5. 训练集成

### 5.1 代码变更范围

| 文件 | 变更类型 | 说明 |
| ---- | -------- | ---- |
| `ginka/train_seperated.py` | 修改 | 新增两个损失函数定义、超参数、训练循环中的损失累加 |

损失函数直接定义在 `train_seperated.py` 中，与现有的 `cross_entropy_loss`（第 201 行）并列放置，保持代码组织的一致性。

### 5.2 训练循环修改

在现有损失计算（第 969-977 行）的基础上，追加邻接损失与 Patch 损失：

```python
# 三阶段 Cross Entropy（不变）
loss1 = cross_entropy_loss(logits1, target1)
loss2 = cross_entropy_loss(logits2, target2)
loss3 = cross_entropy_loss(logits3, target3)

# 新增：邻接损失
adj1 = adjacency_loss(logits1, target1) if LAMBDA_ADJ1 > 0 else 0.0
adj2 = adjacency_loss(logits2, target2) if LAMBDA_ADJ2 > 0 else 0.0
adj3 = adjacency_loss(logits3, target3) if LAMBDA_ADJ3 > 0 else 0.0

# 新增：Patch 损失
patch1 = patch_loss(
    logits1, target1, PATCH_KERNEL_SIZE, PATCH_SIGMA
) if LAMBDA_PATCH1 > 0 else 0.0
patch2 = patch_loss(
    logits2, target2, PATCH_KERNEL_SIZE, PATCH_SIGMA
) if LAMBDA_PATCH2 > 0 else 0.0
patch3 = patch_loss(
    logits3, target3, PATCH_KERNEL_SIZE, PATCH_SIGMA
) if LAMBDA_PATCH3 > 0 else 0.0

# 加权汇总
loss1_weighted = STAGE1_CE_WEIGHT * loss1
loss2_weighted = STAGE2_CE_WEIGHT * loss2
loss3_weighted = STAGE3_CE_WEIGHT * loss3
adj_weighted = (
    LAMBDA_ADJ1 * adj1 + LAMBDA_ADJ2 * adj2 + LAMBDA_ADJ3 * adj3
)
patch_weighted = (
    LAMBDA_PATCH1 * patch1 + LAMBDA_PATCH2 * patch2 + LAMBDA_PATCH3 * patch3
)
commit_weighted = VQ_BETA * commit_loss + VQ_BETA_DIST * commit_loss_dist

loss = (
    loss1_weighted + loss2_weighted + loss3_weighted
    + adj_weighted + patch_weighted + commit_weighted
)
```

对应地在日志输出区域追加邻接损失和 Patch 损失的值，便于监控。

### 5.3 验证循环同步修改

`validate` 函数（第 720-780 行附近）中同样需要追加邻接损失和 Patch 损失的计算，保持训练/验证损失口径一致。

### 5.4 性能影响

| 损失函数 | 额外计算量 | 额外显存 |
| -------- | ---------- | -------- |
| adjacency_loss | softmax + 4 次逐元素乘法 + 2 次 BCE | 极小（仅 B×H×W 级中间张量） |
| patch_loss | 逐格 CE + unfold + 加权求和 | 中等（unfold 产生 B×K²×H×W 中间张量） |

对于 B=64、K=5、H=W=13，unfold 中间张量大小为 64×25×169 ≈ 270k float，显存开销约 1MB，可以忽略。总额外计算时间预估在 5% 以内。

## 6. 验证方案

### 6.1 定量指标

在验证集上对比以下指标（基线 vs 加邻接损失 vs 加 Patch 损失 vs 两者叠加）：

| 指标 | 含义 | 期望趋势 |
| ---- | ---- | -------- |
| 阶段一 CE 损失 | 墙壁骨架准确度 | 基本持平或微降 |
| 阶段二 CE 损失 | 门/怪物/入口准确度 | 持平或改善 |
| 阶段三 CE 损失 | 资源放置准确度 | 持平或改善 |
| 邻接损失值 | 地板连通性 | 显著下降 |
| 地板连通分量数 | 生成地图中可通行区域的连通分量个数（理想为 1） | 下降 |
| 孤立地板块比例 | 四邻域均为墙壁的地板格子占比 | 下降 |
| Codebook 使用率 | VQ 码本利用率 | 不变或微升 |

### 6.2 可视化验证

对相同随机种子、相同随机采码条件下生成的地图进行可视化对比，重点关注：

1. **墙壁骨架质量**（阶段一）：墙壁是否形成合理的房间轮廓和通道，是否出现孤立墙壁碎片
2. **连通性**（全图）：从入口出发能否到达所有可通行区域，是否存在被墙壁完全包围的孤岛
3. **局部一致性**：门是否出现在墙壁上并连接两个地板区域、怪物是否出现在地板上而非墙壁上
4. **资源分布**（阶段三）：资源是否均匀分布在可通行区域内

验证时对不同 z 采样分别生成 4-8 张图，并排对比，使用 `shared/image.py` 的 `matrix_to_image_cv` 输出 PNG 文件。

### 6.3 消融实验

| 实验编号 | CE 损失 | 邻接损失 | Patch 损失 | 目的 |
| -------- | ------- | -------- | ---------- | ---- |
| E0 | ✓ | ✗ | ✗ | 基线（当前模型） |
| E1 | ✓ | ✓ | ✗ | 验证邻接损失单独效果 |
| E2 | ✓ | ✗ | ✓ | 验证 Patch 损失单独效果 |
| E3 | ✓ | ✓ | ✓ | 验证两者叠加效果 |

### 6.4 超参数搜索建议

若初值效果不理想，按以下优先级调参：

1. `LAMBDA_ADJ1`：[0.05, 0.1, 0.2, 0.5] — 邻接损失最关键，先找到合适的强度
2. `LAMBDA_PATCH1`：[0.05, 0.1, 0.2] — Patch 损失作为补充平滑
3. `PATCH_SIGMA`：[0.8, 1.2, 1.6] — 控制平滑范围，sigma 越大平滑越强
4. `PATCH_KERNEL_SIZE`：[3, 5, 7] — 控制邻域大小，注意 7×7 在地图上的覆盖接近 30%
