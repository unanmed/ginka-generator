# VQ 编码器三通道分拆预训练设计文档

## 背景与问题诊断

### 核心问题

在当前 VQ-VAE + MaskGIT 联合训练方案（以及方案 A 闭环约束、方案 D 全图预训练）中，一个根本性的类别不均衡问题始终未被解决：

地图中约 70–85% 的格子为墙壁（tile=1）和空地（tile=0），而怪物（9）、门（2）、入口（10）、宝石（4/5/6）、钥匙（3）、药水（7）、道具（8）等功能性 tile 仅占少数。当编码器以完整地图为输入、重建损失对所有位置平等计算时：

- 编码器的梯度信号被墙壁/空地主导，特征向量主要编码了空间结构，而非功能性内容；
- 解码（或生成）时，怪物/门/资源类 tile 的召回率远低于墙壁；
- 即使方案 D 进行了全图预训练，由于损失仍是全位置等权 Focal Loss，预训练阶段本身就强化了这种偏向；
- **已将损失由 CE 替换为 Focal Loss，但召回率仍无明显改善**——说明问题的根源在于梯度信号的来源比例，而非损失函数本身的形式；
- 方案 A 的闭环一致性约束在 z 本身质量不足时，仅能强化已有的偏向，难以从根本上改善。

### 改进目标

通过**按语义层次将地图拆分为三个独立通道**，采用**累积式输入 + 通道专属损失**的设计：每个通道的编码器输入包含当前通道及所有低等级 tile（提供空间上下文），而解码头的损失**仅在本通道专属 tile 的位置计算，低等级 tile 的损失权重为 0**。这样既保证了编码器能在有意义的空间结构中学习（避免功能性 tile 沦为孤立散点），又迫使解码器必须正确预测本通道 tile 才能降低损失——无法靠拟合高频的墙壁/空地来刷低损失值。

---

## Tile 类型划分

根据 `data/src/shared.ts` 中的定义，将 15 个有效 tile 类型（不含 MASK=15）按游戏语义分为三个通道：

| 通道   | 编码器输入（切片内容）                               | 解码损失计算范围   | 语义含义                             |
| ------ | ---------------------------------------------------- | ------------------ | ------------------------------------ |
| 通道 1 | floor(0) + wall(1)                                   | {1}（仅墙壁）      | 空间骨架（地形结构）                 |
| 通道 2 | floor(0) + wall(1) + door(2) + mob(9) + entrance(10) | {2, 9, 10}         | 关卡门控（交互元素，决定路径可达性） |
| 通道 3 | 完整地图（所有 tile）                                | {3, 4, 5, 6, 7, 8} | 收集资源（奖励与道具）               |

**通道划分的关键设计原则**：

- 通道 1 仅包含 floor 和 wall，编码器集中学习地图的空间骨架结构；
- 通道 2 的输入**保留墙壁与空地作为空间上下文**，在此基础上叠加 door/mob/entrance，确保编码器能感知功能性 tile 在地图中的位置关系，而非将其视为无空间依附的孤立散点；
- 通道 3 的输入为**完整地图**，编码器在包含骨架与关卡结构的完整上下文中学习资源 tile 的空间分布；
- 预训练时每个解码头的损失**仅在本通道专属 tile 的位置计算，低等级 tile（floor、wall 及前级通道的 tile）损失权重为 0**——迫使编码器必须通过正确预测本通道 tile 来降低损失，无法靠拟合高频背景来规避优化压力。

---

## 整体架构

### 预训练阶段

```
完整地图 [B, H*W]
    │
    ├──► 切片 1：floor(0)+wall(1)（这已是全部内容，无需替换）
    │       │
    │       ▼
    │   Encoder_1 → VQ_1 → z_1 [B, L_1, d_z]
    │       │
    │       ▼
    │   DecodeHead_1 → logits [B, H*W, C]
    │       │
    │       ▼
    │   Loss_1：仅在 tile∈{1} 的位置计算 Focal Loss（仅墙壁，空地权重为 0）
    │
    ├──► 切片 2：保留 floor(0)+wall(1)+door(2)+mob(9)+entry(10)，其余→floor(0)
    │       │
    │       ▼
    │   Encoder_2 → VQ_2 → z_2 [B, L_2, d_z]
    │       │
    │       ▼
    │   DecodeHead_2 → logits [B, H*W, C]
    │       │
    │       ▼
    │   Loss_2：仅在 tile∈{2,9,10} 的位置计算 Focal Loss（floor/wall 权重为 0）
    │
    └──► 切片 3：完整地图（所有 tile，无需替换）
            │
            ▼
        Encoder_3 → VQ_3 → z_3 [B, L_3, d_z]
            │
            ▼
        DecodeHead_3 → logits [B, H*W, C]
            │
            ▼
        Loss_3：仅在 tile∈{3,4,5,6,7,8} 的位置计算 Focal Loss（其余 tile 权重为 0）
```

三路编码器**相互独立预训练**，每路的预训练损失：

$$\mathcal{L}_{pretrain}^{(k)} = \mathcal{L}_{FL}^{(k)} + \beta \cdot \mathcal{L}_{commit}^{(k)} + \gamma \cdot \mathcal{L}_{uniform}^{(k)}$$

其中 $\mathcal{L}_{FL}^{(k)}$ 为通道 $k$ 的通道专属掩码 Focal Loss（见下节）。

### 联合训练阶段

```
完整地图 ──► 三路切片 ──► [Enc_1, Enc_2, Enc_3] ──► [z_1, z_2, z_3]
                                                           │
                                          z = Concat([z_1, z_2, z_3], dim=1)
                                                           │
                                                           ▼
掩码地图 + z ──► MaskGIT (Cross-Attention) ──► 预测 logits ──► Focal Loss
```

联合训练总损失（不含预训练解码头）：

$$\mathcal{L}_{joint} = \mathcal{L}_{FL}^{MaskGIT} + \sum_{k=1}^{3} \left( \beta \cdot \mathcal{L}_{commit}^{(k)} + \gamma \cdot \mathcal{L}_{uniform}^{(k)} \right)$$

---

## 通道专属掩码 Focal Loss

这是方案的核心机制。对于通道 $k$，设其专属 tile 集合为 $\mathcal{T}_k$（不含低等级 tile），则损失计算为：

$$\mathcal{L}_{FL}^{(k)} = \frac{\sum_{i=1}^{H \times W} \mathbf{1}[y_i \in \mathcal{T}_k] \cdot \text{FL}(\hat{y}_i, y_i)}{\sum_{i=1}^{H \times W} \mathbf{1}[y_i \in \mathcal{T}_k] + \epsilon}$$

其中 $y_i$ 为真实 tile 类型，$\hat{y}_i$ 为解码头输出的 logits，$\text{FL}$ 为 Focal Loss。

**实现方式**（PyTorch 伪代码）：

```python
# 通道 2 示例（输入切片已包含 floor+wall 作为上下文）
CHANNEL_2_TILES = {2, 9, 10}  # door, mob, entrance

# target: 完整地图 ground truth，[B, H*W]
# logits: DecodeHead_2 输出，[B, H*W, num_classes]

mask = torch.zeros_like(target, dtype=torch.bool)
for t in CHANNEL_2_TILES:
    mask |= (target == t)            # [B, H*W] bool，仅通道 2 专属 tile 的位置为 True

# focal_loss: reduction='none'，返回 [B * H*W]
fl = focal_loss(
    logits.view(-1, num_classes),
    target.view(-1),
)                                    # [B * H*W]
fl = fl.view(B, -1)                  # [B, H*W]

loss_ch2 = (fl * mask).sum() / (mask.sum() + 1e-6)
```

**为什么输入包含墙壁、但损失不计算墙壁**：通道 2 的切片中保留了 floor+wall，是为了给编码器提供空间结构上下文，使门/怪/入口的位置有意义（否则孤立散点难以形成有效表示）。但损失仅在 `{2, 9, 10}` 位置计算，确保梯度信号完全来自这三类 tile——编码器如果只靠拟合高频的墙壁/空地，解码头在 `{2, 9, 10}` 位置的损失无法降低，从而被迫学习功能性 tile 的空间分布。

---

## 切片构造规则

| 通道 | 切片中保留的 tile | 其余位置替换为 | 解码损失计算的位置（专属 tile 集合 $\mathcal{T}_k$）    |
| ---- | ----------------- | -------------- | ------------------------------------------------------- |
| 1    | 0, 1              | —（无需替换）  | {1}（仅墙壁；空地是墙壁的补集，能预测墙壁即能区分空地） |
| 2    | 0, 1, 2, 9, 10    | 0（floor）     | {2, 9, 10}（floor/wall 损失权重为 0）                   |
| 3    | 全部（完整地图）  | —（无需替换）  | {3, 4, 5, 6, 7, 8}（其余 tile 损失权重为 0）            |

---

## 编码器架构设计

### 三路独立编码器

三个编码器均复用现有的 `GinkaVQVAE` 类，配置略有差异：

| 参数            | Encoder_1（结构骨架） | Encoder_2（关卡门控） | Encoder_3（收集资源） |
| --------------- | --------------------- | --------------------- | --------------------- |
| `L`（码字数）   | 2                     | 2                     | 2                     |
| `K`（码本大小） | 16                    | 16                    | 16                    |
| `d_z`           | 64                    | 64                    | 64                    |
| `d_model`       | 128                   | 64                    | 64                    |
| `num_layers`    | 2                     | 2                     | 2                     |

- Encoder_1 处理高频 tile，适当加大 `d_model`；
- Encoder_2 和 Encoder_3 的功能性 tile 稀疏，信息量较小，可使用较小的 `d_model`；
- 三路 `d_z` 保持一致，以便拼接后维度齐整；
- 总参数量估算：Encoder_1 ~400K + Encoder_2 ~150K + Encoder_3 ~150K ≈ **700K**，在 1M 预算内。

> 考虑到训练集数量较少，可以考虑适当降低 K 和 L 的值，避免模型死记硬背，也可以防止训练集没有覆盖全部所有情况。1M 的参数量仅做估计，可以先尝试较大的参数量，如出现过拟合再降低。

### 联合训练时的 z 拼接

$$z = \text{Concat}([z_1, z_2, z_3], \dim=1) \in \mathbb{R}^{B \times (L_1+L_2+L_3) \times d_z}$$

以各通道 `L=2` 为例，总 memory 长度为 6，与当前 MaskGIT Cross-Attention 的 memory 规模（原来 L=2）相比略有增加，但绝对长度仍很小，不影响计算效率。

---

## 预训练流程

### 解码头复用

预训练时三路各自使用一个 `VQDecodeHead` 实例（现有类，`num_classes=16`），预训练结束后整体丢弃。解码头参数不迁移到联合训练阶段。

### 训练脚本

新增 `ginka/train_pretrain_split.py`（独立于现有的 `train_pretrain.py`）：

```python
# 伪代码结构
for epoch in ...:
    for batch in dataloader:
        raw_map = batch["raw_map"]        # [B, H*W] 完整地图

        # ─── 通道 1 ───
        slice1 = make_slice(raw_map, keep={0, 1})  # floor+wall，切片即完整输入
        z_q1, z_e1, _, vq_loss1, *_ = enc1(slice1)
        logits1 = head1(z_q1)
        fl1 = masked_focal(logits1, raw_map, tile_set={1})   # 仅对 wall 计损失
        loss1 = fl1 + vq_loss1

        # ─── 通道 2 ───
        slice2 = make_slice(raw_map, keep={0, 1, 2, 9, 10})  # 保留 wall/floor 作为上下文
        z_q2, z_e2, _, vq_loss2, *_ = enc2(slice2)
        logits2 = head2(z_q2)
        fl2 = masked_focal(logits2, raw_map, tile_set={2, 9, 10})  # 仅对专属 tile 计损失
        loss2 = fl2 + vq_loss2

        # ─── 通道 3 ───
        slice3 = raw_map  # 完整地图，无需切片
        z_q3, z_e3, _, vq_loss3, *_ = enc3(slice3)
        logits3 = head3(z_q3)
        fl3 = masked_focal(logits3, raw_map, tile_set={3, 4, 5, 6, 7, 8})  # 仅对专属 tile 计损失
        loss3 = fl3 + vq_loss3

        total = loss1 + loss2 + loss3
        total.backward()
        optimizer.step()
```

三路编码器可以**同步训练**（同一 optimizer），也可以分别独立训练——独立训练更灵活，可以对收敛速度差异大的通道单独调参。

### 预训练监控指标

| 指标                    | 说明                                                             |
| ----------------------- | ---------------------------------------------------------------- |
| 通道 1 wall 位置准确率  | Encoder_1 能否正确重建墙壁分布                                   |
| 通道 2 功能 tile 召回率 | Encoder_2 对 door/mob/entrance 各类的召回，应 > 50%（稀疏 tile） |
| 通道 3 资源 tile 召回率 | Encoder_3 对各资源类的召回                                       |
| codebook 使用熵（各路） | 各通道 codebook 是否均匀使用，避免 collapse                      |

由于通道 2/3 的 tile 在每张地图中数量极少（典型地图中 door/mob/entrance 合计约 10–20 格，资源合计约 10–15 格），召回率指标比准确率更有意义。

---

## 联合训练流程

### 三阶段训练

| 阶段                 | 模型状态                                  | 目标                                             | 建议轮数     |
| -------------------- | ----------------------------------------- | ------------------------------------------------ | ------------ |
| 阶段 0：分通道预训练 | 三路 Encoder + 三路 DecodeHead            | 各通道 Focal Loss 收敛，功能 tile 召回率达到目标 | 30–60 epoch  |
| 阶段 1：冻结热身     | 三路 Encoder 冻结 + MaskGIT 全参训练      | MaskGIT 适应三路 z 的联合分布                    | 20–40 epoch  |
| 阶段 2：完整联合训练 | 全部参数解冻，Encoder 使用较小 LR（×0.1） | 端到端联合优化                                   | 正常训练轮数 |

### 联合训练数据集

`GinkaJointDataset` 需扩展为同时提供三路切片：

```python
# 返回字典新增字段
{
    "raw_map":    ...,   # [H*W] 完整地图（VQ 编码器输入）
    "slice1":     ...,   # [H*W] 通道 1 切片
    "slice2":     ...,   # [H*W] 通道 2 切片
    "slice3":     ...,   # [H*W] 通道 3 切片
    "masked_map": ...,   # [H*W] MaskGIT 输入（掩码后地图）
    "target_map": ...,   # [H*W] MaskGIT CE ground truth
}
```

---

## 推理时的 z 采样

推理时三路编码器均独立采样，无需用户输入：

```python
# 完全随机生成
z1 = enc1.sample(B, device)   # [B, L1, d_z]
z2 = enc2.sample(B, device)   # [B, L2, d_z]
z3 = enc3.sample(B, device)   # [B, L3, d_z]
z  = torch.cat([z1, z2, z3], dim=1)  # [B, L1+L2+L3, d_z]
```

**分通道条件控制**（可选扩展）：

| 场景         | 通道 1 z 来源 | 通道 2 z 来源 | 通道 3 z 来源 |
| ------------ | ------------- | ------------- | ------------- |
| 完全随机生成 | 随机采样      | 随机采样      | 随机采样      |
| 指定墙壁布局 | 用户地图编码  | 随机采样      | 随机采样      |
| 指定关卡结构 | 随机采样      | 参考图编码    | 随机采样      |
| 风格迁移     | 参考图编码    | 参考图编码    | 随机采样      |

---

## 与现有方案的关系

| 方案   | 与本方案的关系                                                                                                                                                                            |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 方案 A | 兼容。由于通道 2/3 的编码器输入包含墙壁上下文（而非孤立散点），其 z 具备稳定的空间语义，一致性约束可同样作用于三个通道：通道 1 约束骨架结构，通道 2 约束关卡元素分布，通道 3 约束资源分布 |
| 方案 D | 本方案**替代并强化**方案 D 的预训练思路：方案 D 是全图等权 Focal Loss 预训练，本方案通过累积式输入 + 通道专属损失从根本上解决了类别不均衡问题                                             |
| 方案 C | 兼容。多阶段生成（先墙后门后资源）可以将通道 1/2/3 的 z 分别作为各阶段的生成条件                                                                                                          |

---

## 超参数建议

| 参数                       | 建议初始值 | 备注                                     |
| -------------------------- | ---------- | ---------------------------------------- |
| 各通道 `L`（码字数）       | 2          | 三路合计 6，可视效果适当扩大             |
| 各通道 `K`（码本大小）     | 16         | 通道 2/3 可减小到 8（tile 种类少）       |
| `d_z`                      | 64         | 三路保持一致，便于拼接                   |
| `β`（commit loss 权重）    | 0.25       | 同现有配置                               |
| `γ`（uniform loss 权重）   | 0.1        | 通道 2/3 码本小，可适当增大到 0.2        |
| 预训练 epoch               | 30–60      | 以功能 tile 召回率达标为准，不以轮数为限 |
| 联合训练 Encoder LR 缩放比 | 0.1        | 阶段 2 解冻后使用较小 LR 微调            |
| z dropout 概率（联合训练） | 0.1–0.2    | 三路 z 各自独立 dropout                  |

---

## 实施步骤

- [ ] 在 `ginka/dataset.py` 中实现 `make_slice(map, keep_set)` 辅助函数，生成三路切片
- [ ] 扩展 `GinkaJointDataset.__getitem__`，新增 `slice1/slice2/slice3` 字段
- [ ] 在 `ginka/vqvae/model.py` 中确认 `GinkaVQVAE` 可独立实例化三次（无全局状态）
- [ ] 实现 `masked_focal(logits, target, tile_set)` 工具函数（`ginka/utils.py`）
- [ ] 新增 `ginka/train_pretrain_split.py` 预训练脚本（支持三路同步或分路训练）
- [ ] 修改 `ginka/train_vq.py`（联合训练脚本），支持加载三路编码器权重并拼接 z
- [ ] 修改 `GinkaMaskGIT.forward()` 以接受 `[B, L1+L2+L3, d_z]` 的拼接 z（Cross-Attention memory）
- [ ] 添加联合训练监控：各通道 codebook 使用熵、功能 tile 召回率
