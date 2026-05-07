# 三阶段级联地图生成设计文档

## 背景与问题诊断

### 当前问题：墙壁过度生成

现有单模型方案（VQ-VAE + MaskGIT 联合训练）在实践中呈现出**模型严重偏向生成墙壁**的现象，具体表现为：

- 生成地图中墙壁密度显著偏高，可通行空间不足；
- 门、怪物、入口等功能性元素稀少甚至缺失；
- 资源类 tile 几乎不出现。

### 根本原因分析

从 token 分布角度来看，13×13 = 169 个格子中，各类 tile 的分布严重不均：

| tile 类型     | 典型比例 | 训练信号 |
| ------------- | -------- | -------- |
| 墙壁（wall）  | 40%~60%  | 强、密集 |
| 空地（floor） | 20%~40%  | 强、密集 |
| 门（door）    | 1%~5%    | 极弱     |
| 怪物          | 5%~15%   | 弱       |
| 入口          | 1%~3%    | 极弱     |
| 资源          | 5%~15%   | 弱       |

单模型将以上所有类别的 **交叉熵损失**混合在一起优化。由于墙壁和空地占据绝大多数 token，模型可以通过反复预测墙壁/空地获得低训练损失，而无需真正学习如何放置稀有类别。

这是**类别不均衡问题的结构性体现**：即使引入 Focal Loss，调整 loss 权重，也难以从根本上解决三类任务学习难度不匹配的问题——因为结构骨架（floor/wall）是其他所有元素放置的前提，混合训练会导致模型困于局部最优解（"把所有位置都预测为墙壁"）。

---

## 核心思路：三阶段级联生成

将单次全类别生成拆分为**三个独立的 MaskGIT 阶段**，每阶段只负责一组语义相近、结构约束相似的 tile 类别。后续阶段以前序阶段的输出作为已知上下文。

```
┌──────────────────────────────────────────────────────────────────────┐
│  阶段一：结构骨架生成                                                  │
│                                                                      │
│  全 MASK  ──► [Stage1-MaskGIT + z₁]  ──► floor/wall 地图            │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ 已知 floor/wall 上下文
┌──────────────────────────────────────────────────────────────────────┐
│  阶段二：功能元素放置                                                  │
│                                                                      │
│  floor/wall + MASK  ──► [Stage2-MaskGIT + z₂]  ──► door/monster/入口│
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ 已知 floor/wall/door/monster/入口 上下文
┌──────────────────────────────────────────────────────────────────────┐
│  阶段三：资源放置                                                      │
│                                                                      │
│  完整上下文 + MASK  ──► [Stage3-MaskGIT + z₃]  ──► resource          │
└──────────────────────────────────────────────────────────────────────┘
```

### 各阶段职责

| 阶段   | 负责类别                         | tile 数 | 类别数（含 MASK） | 结构约束强度 |
| ------ | -------------------------------- | ------- | ----------------- | ------------ |
| 阶段一 | floor(0)、wall(1)                | 2       | 3                 | 极强         |
| 阶段二 | door(2)、monster(4)、entrance(5) | 3       | 6                 | 强           |
| 阶段三 | resource(3)                      | 1       | 3                 | 弱           |

注：各阶段模型的词表不需要只含本阶段 tile，完整词表（7 类）可以保留不变，只是**被 MASK 的位置**和**计算 Loss 的位置**会有所不同（见训练策略）。

---

## 每阶段 tile 映射与 MASK 策略

### 阶段一

**输入**：所有位置均填充 MASK token（`tile=6`）。

**目标**：预测 floor(0) 和 wall(1)；原始地图中所有非 floor/wall 的 tile 在训练目标中被**重映射为 floor(0)**，因为阶段一模型不关心功能性元素的具体种类。

```python
STAGE1_REMAP = {
    0: 0,   # floor → floor
    1: 1,   # wall  → wall
    2: 0,   # door  → floor（视作空地，让阶段二填充）
    3: 0,   # resource → floor
    4: 0,   # monster → floor
    5: 0,   # entrance → floor
    6: 6,   # MASK  → MASK（不参与损失计算）
}
```

**Loss 计算范围**：所有非 MASK 位置（即全局所有位置，因为输入均为 MASK）。

### 阶段二

**输入**：将阶段一输出的 floor/wall 地图作为固定上下文，原始地图中属于 door/monster/entrance 的位置替换为 MASK token，其余（floor/wall）位置保持不变。

```python
STAGE2_MASK_IDS = {2, 4, 5}  # 需要在阶段二中被预测的 tile ID
```

**训练时输入构造**（使用 GT 地图中的 floor/wall 作为上下文，不使用阶段一的实际输出）：

```python
def make_stage2_input(gt_map: np.ndarray) -> np.ndarray:
    """
    gt_map: [H*W] 整数数组，包含完整原始地图
    返回: stage2 输入地图，door/monster/entrance 位置替换为 MASK
    """
    inp = gt_map.copy()
    # 先将所有资源归一化为 floor（阶段二不负责资源）
    inp[np.isin(inp, [3])] = 0
    # 将 door/monster/entrance 位置 MASK 掉
    inp[np.isin(inp, [2, 4, 5])] = 6  # MASK
    return inp
```

**目标**：预测 door(2)、monster(4)、entrance(5) 位置的类别（以及它们所在位置原本是否是 floor）。

**Loss 计算范围**：仅对输入为 MASK（即原本是 door/monster/entrance 的位置）计算损失，floor/wall 位置不参与损失（它们已经确定）。

### 阶段三

**输入**：将阶段二输出的完整功能性地图（floor/wall/door/monster/entrance）作为固定上下文，原始地图中资源（tile=3）位置替换为 MASK token。

```python
def make_stage3_input(gt_map: np.ndarray) -> np.ndarray:
    """
    gt_map: [H*W] 整数数组，包含完整原始地图
    返回: stage3 输入地图，resource 位置替换为 MASK
    """
    inp = gt_map.copy()
    inp[inp == 3] = 6  # resource → MASK
    return inp
```

**Loss 计算范围**：仅对输入为 MASK（即原本是 resource 的位置）计算损失。

---

## 模型架构

### 共享基础架构

三个阶段均采用与现有方案**相同的 MaskGIT + VQ-VAE 架构**，不需要设计新的模型结构。核心差异在于：

1. **输入词表**：与现有方案一致，均使用 7 类词表（`NUM_CLASSES=7`，`MASK_TOKEN=6`）；
2. **z 来源**：每阶段使用来自对应 VQ 通道的隐变量；
3. **Loss 掩码**：只对该阶段负责的位置计算 CE Loss。

### VQ-VAE z 的阶段分配

现有 VQ-VAE 已采用三通道设计（CH1/CH2/CH3），与三个生成阶段自然对应：

| 通道 | 编码目标          | 供应阶段 | 描述                         |
| ---- | ----------------- | -------- | ---------------------------- |
| CH1  | floor/wall 骨架   | 阶段一   | z₁，控制墙壁结构多样性       |
| CH2  | door/monster/入口 | 阶段二   | z₂，控制功能元素布局多样性   |
| CH3  | resource 分布     | 阶段三   | z₃，控制资源密度与位置多样性 |

```
真实地图 ──► CH1 encoder ──► z₁  ──► Stage1 MaskGIT
         ──► CH2 encoder ──► z₂  ──► Stage2 MaskGIT
         ──► CH3 encoder ──► z₃  ──► Stage3 MaskGIT
```

**通道专属编码**：各通道编码器在编码前只"看"与自身相关的 tile，其余 tile 视作 floor(0)：

```python
def ch1_mask(gt_map):
    """只保留 floor/wall，其余置 0"""
    m = gt_map.copy()
    m[~np.isin(m, [0, 1])] = 0
    return m

def ch2_mask(gt_map):
    """只保留 door/monster/entrance，其余置 0"""
    m = gt_map.copy()
    m[~np.isin(m, [2, 4, 5])] = 0
    return m

def ch3_mask(gt_map):
    """只保留 resource，其余置 0"""
    m = gt_map.copy()
    m[m != 3] = 0
    return m
```

### Loss 计算掩码实现

训练时，每阶段额外接收一个 `loss_mask: torch.BoolTensor [B, H*W]`，指示哪些位置需要计算损失：

```python
# 阶段一：所有位置（因为输入全为 MASK）
loss_mask_s1 = torch.ones(B, MAP_SIZE, dtype=torch.bool)

# 阶段二：只有原本是 door/monster/entrance 的位置
loss_mask_s2 = torch.isin(raw_map, torch.tensor([2, 4, 5]))

# 阶段三：只有原本是 resource 的位置
loss_mask_s3 = (raw_map == 3)
```

Focal Loss 修改为只对 `loss_mask` 为 True 的位置求和后做归一化：

```python
def stage_focal_loss(logits, targets, loss_mask, gamma=2.0):
    # logits: [B, C, H*W], targets: [B, H*W], loss_mask: [B, H*W]
    per_token_loss = focal_loss(logits, targets, gamma, reduction='none')  # [B, H*W]
    masked_loss = per_token_loss[loss_mask]
    return masked_loss.mean() if masked_loss.numel() > 0 else per_token_loss.mean()
```

---

## 训练策略

### 训练方式：顺序训练（推荐）

三个阶段**依次训练**，后续阶段训练时使用 GT 地图作为前序阶段的"完美输出"（teacher forcing），而非使用前序阶段模型的实际推理结果。

```
训练阶段一:
    data:     (全MASK输入, stage1目标)
    loss:     focal loss on all positions

训练阶段二:
    data:     (floor/wall上下文 + MASK输入, stage2目标)
    loss:     focal loss only on door/monster/entrance positions

训练阶段三:
    data:     (floor/wall/door/monster/入口上下文 + MASK输入, stage3目标)
    loss:     focal loss only on resource positions
```

**使用 GT 而非前序模型输出的理由**：

- 避免误差级联（前序模型若生成错误的骨架，后续模型的训练将在错误分布上进行）；
- 各阶段训练更稳定，收敛更快；
- 阶段之间解耦，方便单独迭代和调试。

### 各阶段训练子集划分

每个阶段均沿用现有 A/B/C/D 子集划分逻辑，但 MASK 策略应用在对应阶段的目标 tile 上：

| 子集 | 阶段一                        | 阶段二                         | 阶段三                      |
| ---- | ----------------------------- | ------------------------------ | --------------------------- |
| A    | 随机遮盖部分 floor/wall       | 随机遮盖部分 door/monster/入口 | 随机遮盖部分 resource       |
| B    | 保留全部 wall，MASK floor     | 给定全部 wall，MASK 功能元素   | 给定全部骨架，MASK 部分资源 |
| C    | 随机保留部分 wall，MASK 其余  | 同 B                           | 同 B                        |
| D    | 保留 wall+entrance，MASK 其余 | 给定 wall+entrance，MASK 门/怪 | 同 B                        |

### 各阶段专属损失

每阶段的总损失：

$$\mathcal{L}^{(s)} = \mathcal{L}_{CE}^{(s)} + \beta \cdot \mathcal{L}_{commit}^{(s)} + \gamma \cdot \mathcal{L}_{uniform}^{(s)}$$

其中 $s \in \{1, 2, 3\}$ 表示阶段编号，$\mathcal{L}_{CE}^{(s)}$ 只在该阶段负责的 tile 位置上计算。

---

## 推理流程

### 完整推理管线

```
1. 随机采样 z₁, z₂, z₃（各自从对应 codebook 均匀采样 L 个 index）

2. 阶段一推理（结构骨架）
   初始状态: 全部 169 个位置 = MASK token
   迭代 MaskGIT 解码（cosine schedule，约 18 步）:
     输入: MASK地图 + z₁
     输出: 逐步填充 floor/wall，直到无 MASK 位置
   结果: floor/wall 骨架地图 M₁

3. 阶段二推理（功能元素放置）
   初始状态: 继承 M₁，在 floor 位置随机选取候选位置置为 MASK
   ─── 或 ───
   初始状态: 继承 M₁，所有 floor 位置均置为 MASK（让模型决定放置密度）
   迭代 MaskGIT 解码:
     输入: 含已知 wall 的掩码地图 + z₂
     约束: wall 位置不参与 unmask，保持不变
     输出: 逐步填充 door/monster/entrance
   结果: 含功能元素的地图 M₂

4. 阶段三推理（资源放置）
   初始状态: 继承 M₂，所有 floor 位置（未被阶段二填充的）置为 MASK
   迭代 MaskGIT 解码:
     输入: 含已知 wall/door/monster/入口的掩码地图 + z₃
     约束: 非 floor 位置保持不变
     输出: 逐步填充 resource（或保持 floor）
   结果: 完整地图 M₃
```

### 阶段二初始 MASK 策略

阶段二的初始状态有两种选择：

| 策略          | 描述                                               | 适用场景         |
| ------------- | -------------------------------------------------- | ---------------- |
| 全 floor MASK | 所有 floor 位置均置为 MASK，让模型自主决定放置密度 | 完全随机生成     |
| 候选位置 MASK | 只 MASK 用户指定或随机抽取的少数位置               | 用户指定部分位置 |

对于完全随机生成场景，**推荐使用全 floor MASK**——此时 z₂ 决定功能元素的总体风格（密集/稀疏/集中/分散），模型负责在此约束下寻找合理位置。

### 用户交互场景

| 场景                | 阶段一输入           | 阶段二输入               | 阶段三输入 |
| ------------------- | -------------------- | ------------------------ | ---------- |
| 完全随机            | 全 MASK              | 继承 M₁                  | 继承 M₂    |
| 用户手绘墙壁        | 已知 wall + MASK     | 继承 M₁（固定 wall）     | 继承 M₂    |
| 用户指定入口        | 已知 entrance + MASK | 继承 M₁（固定 entrance） | 继承 M₂    |
| 用户手绘墙+指定入口 | 已知 wall/entrance   | 继承 M₁                  | 继承 M₂    |

---

## 实现方案

### 方案一：三个独立 GinkaMaskGIT 实例（推荐）

每个阶段分别实例化一个 `GinkaMaskGIT`，共用同一个 `GinkaVQVAE`（其中已含三通道编码器）。各阶段模型独立训练、独立存储。

```python
# 模型结构
vqvae   = GinkaVQVAE(...)         # 三通道共享编码器
stage1  = GinkaMaskGIT(...)       # 结构骨架
stage2  = GinkaMaskGIT(...)       # 功能元素
stage3  = GinkaMaskGIT(...)       # 资源
```

**优点**：

- 各阶段模型完全解耦，可独立调整超参、单独重训；
- 模型大小可以针对任务难度调整（阶段一可以更大，阶段三可以更小）；
- 便于调试和增量式开发。

**缺点**：

- 三个模型分别保存，推理时需依次加载；
- 总参数量约为原方案的 3 倍（但可以通过缩小各阶段模型来对冲）。

### 方案二：单模型 + 阶段 Embedding（备选）

复用同一个 `GinkaMaskGIT`，添加阶段 embedding（类似 BERT 的 segment embedding）：

```python
self.stage_embedding = nn.Embedding(3, d_model)  # 三个阶段

def forward(self, map, z, stage: int):
    x = self.tile_embedding(map) + self.pos_embedding
    x = x + self.stage_embedding(torch.tensor(stage))  # 阶段条件
    ...
```

**优点**：参数量与原方案相同，推理时只需加载一个模型。  
**缺点**：三阶段共享所有权重，可能导致阶段间干扰；阶段一的结构任务与阶段三的稀疏资源任务表示空间差异大，单一模型难以同时擅长。

**结论**：**推荐方案一**，尤其是在当前阶段（验证多阶段框架可行性），可先分别训练三个轻量模型进行快速验证。

### 各阶段模型规模建议

| 阶段   | 任务难度   | 建议 d_model | 建议 num_layers | 参数量估算 |
| ------ | ---------- | ------------ | --------------- | ---------- |
| 阶段一 | 高（结构） | 256          | 6               | ~4M        |
| 阶段二 | 中（功能） | 192          | 4               | ~2M        |
| 阶段三 | 低（稀疏） | 128          | 3               | ~0.8M      |

---

## Dataset 修改方案

### 新增 `GinkaStageDataset`

需要扩展 `dataset.py`，增加针对三阶段的 Dataset 类，或在 `GinkaVQDataset` 中添加 `stage` 参数：

```python
class GinkaStageDataset(Dataset):
    """
    三阶段级联训练专用 Dataset。

    返回 dict:
      raw_map:      LongTensor [H*W]  完整原始地图（供 VQ-VAE 编码）
      stage_input:  LongTensor [H*W]  当前阶段 MaskGIT 输入（含上下文 + MASK）
      target_map:   LongTensor [H*W]  CE loss ground truth（等同 raw_map）
      loss_mask:    BoolTensor [H*W]  只对 True 位置计算损失
      subset:       str               子集标识
    """

    STAGE1_TARGETS = {0, 1}       # floor, wall
    STAGE2_TARGETS = {2, 4, 5}    # door, monster, entrance
    STAGE3_TARGETS = {3}          # resource
```

### 数据构造函数

```python
def make_stage1_sample(gt_map: np.ndarray, mask_id: int = 6):
    """阶段一：全 MASK 输入，目标是 floor/wall（其余归一为 floor）"""
    stage_input = np.full_like(gt_map, mask_id)
    target = gt_map.copy()
    target[~np.isin(target, [0, 1])] = 0  # 非结构 tile → floor
    loss_mask = np.ones_like(gt_map, dtype=bool)
    return stage_input, target, loss_mask

def make_stage2_sample(gt_map: np.ndarray, mask_id: int = 6):
    """阶段二：floor/wall 为上下文，door/monster/entrance 位置 MASK"""
    stage_input = gt_map.copy()
    stage_input[stage_input == 3] = 0      # 资源 → floor（阶段二不负责）
    target_ids = np.isin(stage_input, [2, 4, 5])
    stage_input[target_ids] = mask_id      # 功能元素 → MASK
    target = gt_map.copy()
    target[target == 3] = 0               # target 中资源也视为 floor
    loss_mask = (gt_map != 0) & (gt_map != 1) & (gt_map != 3)  # 只计算功能元素位置
    return stage_input, target, loss_mask

def make_stage3_sample(gt_map: np.ndarray, mask_id: int = 6):
    """阶段三：全上下文保留，只 MASK 资源位置"""
    stage_input = gt_map.copy()
    stage_input[stage_input == 3] = mask_id  # 资源 → MASK
    target = gt_map.copy()
    loss_mask = (gt_map == 3)               # 只计算资源位置
    return stage_input, target, loss_mask
```

---

## 训练脚本设计

### 新增 `ginka/train_stage.py`

```
用法示例：
    python -m ginka.train_stage --stage 1
    python -m ginka.train_stage --stage 2
    python -m ginka.train_stage --stage 3 --resume True --state result/stage3/stage3-10.pth
```

各阶段检查点分别存储到：

- `result/stage1/stage1-{epoch}.pth`
- `result/stage2/stage2-{epoch}.pth`
- `result/stage3/stage3-{epoch}.pth`

### 阶段二/三的 VQ 编码器冻结策略

阶段二训练时，CH1 编码器（已在阶段一训练中充分收敛）可选择冻结，只更新 CH2 编码器和 Stage2 MaskGIT：

```python
if args.stage == 2:
    for p in vqvae.encoder_ch1.parameters():
        p.requires_grad_(False)  # 冻结 CH1
```

类似地，阶段三训练时可冻结 CH1 和 CH2 编码器。

---

## 与现有方案的对比

| 维度           | 现有单模型方案           | 三阶段级联方案                           |
| -------------- | ------------------------ | ---------------------------------------- |
| 墙壁过度生成   | 存在，难以从根本解决     | 阶段一单独训练骨架，Loss 聚焦 floor/wall |
| 训练信号均衡性 | 墙壁主导，稀有类欠拟合   | 各阶段 Loss 只计算本阶段 tile，信号均衡  |
| 模型可调试性   | 单一模型，各类别相互干扰 | 各阶段独立，可单独分析每阶段表现         |
| 推理速度       | 1 次完整 MaskGIT 解码    | 3 次级联解码（总步数约为原来 3 倍）      |
| 误差累积       | 无                       | 存在，前序阶段错误会传播到后续阶段       |
| 用户可控性     | 较难（条件混合）         | 好（可在任意阶段注入用户约束）           |
| 参数量         | ~4M                      | 约 ~7M（可通过缩减各阶段规模控制）       |

---

## 预期收益

1. **解决墙壁过度生成**：阶段一专门针对 floor/wall 训练，类别分布从 7 类压缩到 3 类，Loss 完全聚焦，模型不再有逃避路径；
2. **功能元素召回率提升**：阶段二以已知骨架为前提生成 door/monster/entrance，训练信号不再被墙壁噪声稀释；
3. **资源分布更合理**：阶段三在完整上下文下放置资源，能感知到门/怪物位置，避免资源与关键功能元素重叠；
4. **可交互性增强**：用户可在任意阶段注入约束（固定某些 tile），天然支持层次化编辑。

---

## 风险与应对

| 风险             | 描述                                      | 应对策略                                                 |
| ---------------- | ----------------------------------------- | -------------------------------------------------------- |
| 误差累积         | 阶段一骨架不准确会导致阶段二/三布局失真   | 阶段一优先保证质量，推理时对骨架做后处理校验             |
| 推理耗时增加     | 三次 MaskGIT 解码约 3 倍耗时              | 减少 MaskGIT 迭代步数（阶段二/三任务更简单，步数可减半） |
| 阶段二稀疏性问题 | 169 格子中功能元素极少，Loss 计算覆盖率低 | 适当提高 stage 2 的 loss_mask 覆盖（周边 floor 也计入）  |
| Codebook 对齐    | 三通道 VQ-VAE 的 z 在分阶段训练时各自优化 | 联合训练阶段一+VQ-CH1，联合训练阶段二+VQ-CH2，以此类推   |

---

## 实施顺序

- [ ] 新增 `GinkaStageDataset`，实现三阶段数据构造函数
- [ ] 新增 `ginka/train_stage.py`，支持 `--stage 1/2/3` 参数
- [ ] 阶段一训练：仅 floor/wall，验证骨架生成质量（关键里程碑）
- [ ] 阶段二训练：以 GT floor/wall 为上下文，验证功能元素召回率
- [ ] 阶段三训练：以 GT 完整地图为上下文，验证资源放置合理性
- [ ] 实现三阶段级联推理脚本，接入现有可视化工具
- [ ] 对比实验：三阶段方案 vs 现有单模型方案在墙壁密度、功能元素召回率、资源分布等指标上的差异
