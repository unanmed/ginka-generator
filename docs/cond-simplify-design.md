# 条件简化与密度连续化设计文档

## 背景

当前三阶段级联生成模型的条件系统存在以下问题：

1. **结构条件中的房间数和分支数对生成指导意义有限**：这两个指标依赖数据集中预计算的离散分档，与实际生成质量的相关性较弱，且分档边界处噪声大，容易引入无效条件信号。

2. **实体密度条件（门/怪物/资源）的离散三档存在明显一对多问题**：三档划分过于粗糙，同一档内样本分布差异极大（例如 Medium 档中资源数可以从 2 到 8 不等），导致模型无法建立条件与生成结果之间的精确映射。连续值能够更精确地描述目标密度，避免档位内分布散乱导致的条件信号模糊。

## 改动总览

| 模块                       | 改动类型 | 说明                                                           |
| -------------------------- | -------- | -------------------------------------------------------------- |
| `ginka/dataset.py`         | 修改     | 删除房间/分支分档；密度改为连续归一化；输出 FloatTensor        |
| `ginka/maskGIT/model.py`   | 修改     | 删除房间/分支嵌入；密度嵌入层改为线性投影；更新 cond_proj 维度 |
| `ginka/train_seperated.py` | 修改     | 更新 random_struct/random_density；更新 annotate_labels        |

---

## 一、条件向量格式变更

### 1.1 struct_inject

**当前格式**（4 个离散整数）：

```
[cond_sym(0-7), cond_room(0-2), cond_branch(0-2), cond_outer(0-1)]
```

**新格式**（2 个离散整数，删除 room 和 branch）：

```
[cond_sym(0-7), cond_outer(0-1)]
```

`cond_sym` 的计算方式不变（水平/垂直/中心对称的三位二进制组合，0–7），`cond_outer` 不变。

### 1.2 density_inject

**当前格式**（3 个离散整数，`LongTensor`）：

```
[door_level(0-2), monster_level(0-2), resource_level(0-2)]
```

**新格式**（3 个连续浮点数，`FloatTensor`，值域 [0, 1]）：

```
[door_norm, monster_norm, resource_norm]  ∈ [0.0, 1.0]^3
```

---

## 二、密度归一化方案

### 2.1 统计量定义

在训练集初始化阶段，对原始地图统计三类图块的实际数量（非密度，直接计数）：

- `door_count = 图块ID为2的数量`
- `monster_count = 图块ID为4的数量`
- `resource_count = 图块ID为3的数量`

对每类分别求训练集内的 **最小值** 和 **最大值**：

```
door_min, door_max
monster_min, monster_max
resource_min, resource_max
```

### 2.2 归一化公式

对每个样本的 count，归一化为 [0, 1]：

$$
\text{norm}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \epsilon}
$$

其中 $\epsilon = 1\text{e-}6$，防止分母为零（当所有样本计数相同时）。

结果裁剪到 [0, 1]：`norm = clamp(norm, 0.0, 1.0)`。

### 2.3 验证集复用训练集统计量

`GinkaSeperatedDataset` 新增参数：

```python
def __init__(
    self,
    data_path: str,
    subset_weights: tuple = (0.5, 0.3, 0.2),
    density_stats: dict | None = None   # 新增：外部传入统计量
):
```

- 训练集：`density_stats=None`，自行计算并保存 `min/max` 到 `self.density_stats`
- 验证集：传入训练集的 `self.density_stats`，直接复用，保证归一化语义一致

`density_stats` 的结构：

```python
{
    "door_min": float, "door_max": float,
    "monster_min": float, "monster_max": float,
    "resource_min": float, "resource_max": float,
}
```

### 2.4 输出字段变更

`__getitem__` 中 `density_inject` 由 `LongTensor` 改为 `FloatTensor`：

```python
# 删除旧的离散分档逻辑
density_inject = torch.FloatTensor([
    self.norm_density(count_door, "door"),
    self.norm_density(count_monster, "monster"),
    self.norm_density(count_resource, "resource"),
])
```

删除以下字段（不再写入 item 也不再输出）：

- `doorDensityLevel`, `monsterDensityLevel`, `resourceDensityLevel`
- `roomCountLevel`, `branchLevel`

删除以下实例变量：

- `self.room_th`, `self.branch_th`
- `self.door_density_th`, `self.monster_density_th`, `self.resource_density_th`

---

## 三、模型结构变更（`model.py`）

### 3.1 删除房间/分支嵌入

删除：

```python
self.room_embed = nn.Embedding(ROOM_VOCAB, d_z)
self.branch_embed = nn.Embedding(BRANCH_VOCAB, d_z)
```

保留：

```python
self.sym_embed = nn.Embedding(SYM_VOCAB, d_z)   # SYM_VOCAB = 8
self.outer_embed = nn.Embedding(OUTER_VOCAB, d_z) # OUTER_VOCAB = 2
```

删除的常量：`ROOM_VOCAB`, `BRANCH_VOCAB`，保留 `SYM_VOCAB`, `OUTER_VOCAB`。

### 3.2 密度嵌入层改为线性投影

删除：

```python
self.door_density_embed = nn.Embedding(DOOR_DENSITY_VOCAB, d_z)
self.monster_density_embed = nn.Embedding(MONSTER_DENSITY_VOCAB, d_z)
self.resource_density_embed = nn.Embedding(RESOURCE_DENSITY_VOCAB, d_z)
```

删除的常量：`DOOR_DENSITY_VOCAB`, `MONSTER_DENSITY_VOCAB`, `RESOURCE_DENSITY_VOCAB`。

新增：

```python
# 连续密度投影：将 3 个归一化浮点数映射为 1 个 d_z 维 token
self.density_proj = nn.Linear(3, d_z)
```

### 3.3 cond_proj 维度更新

**当前 cond_seq 形状**：`[B, z_seq_len + 4_struct + 3_density, d_z]`，即 `[B, z_seq_len+7, d_z]`，展平后输入维度 `(z_seq_len+7) * d_z`。

**新 cond_seq 形状**：`[B, z_seq_len + 2_struct + 1_density, d_z]`，即 `[B, z_seq_len+3, d_z]`，展平后输入维度 `(z_seq_len+3) * d_z`。

```python
# 旧
self.cond_proj = nn.Linear((z_seq_len + 7) * d_z, d_model)
# 新
self.cond_proj = nn.Linear((z_seq_len + 3) * d_z, d_model)
```

### 3.4 forward 流程变更

```python
def forward(
    self,
    map: torch.Tensor,
    z: torch.Tensor,
    struct: torch.Tensor,   # [B, 2]  ← 由 [B, 4] 改为 [B, 2]
    density: torch.Tensor   # [B, 3] float ← 由 [B, 3] long 改为 float
) -> torch.Tensor:

    # 结构标签：sym + outer，各嵌入为 d_z 维 token
    e_struct = torch.stack([
        self.sym_embed(struct[:, 0]),   # [B, d_z]
        self.outer_embed(struct[:, 1]), # [B, d_z]
    ], dim=1)  # [B, 2, d_z]

    # 密度：连续值投影为单个 d_z 维 token
    e_density = self.density_proj(density).unsqueeze(1)  # [B, 1, d_z]

    # z：逐 token 投影（不变）
    z_proj = self.z_proj(z)  # [B, z_seq_len, d_z]

    # 拼接 → [B, z_seq_len+3, d_z] → 展平 → 投影到 d_model
    cond_seq = torch.cat([z_proj, e_struct, e_density], dim=1)
    c = self.cond_proj(cond_seq.reshape(cond_seq.size(0), -1))  # [B, d_model]

    # 后续不变（tile embedding + Transformer + output_fc）
```

---

## 四、训练脚本变更（`train_seperated.py`）

### 4.1 random_struct

```python
def random_struct(device: torch.device) -> torch.Tensor:
    # struct_inject 格式：[cond_sym(0-7), cond_outer(0-1)]
    cond_sym = random.randint(0, 7)   # 地图对称类型
    cond_outer = random.randint(0, 1) # 是否有外围走廊
    return torch.LongTensor([cond_sym, cond_outer]).unsqueeze(0).to(device)
```

### 4.2 random_density

```python
def random_density(device: torch.device) -> torch.Tensor:
    # density_inject 格式：[door_norm, monster_norm, resource_norm] ∈ [0, 1]
    return torch.rand(1, 3, device=device)  # 均匀分布随机采样
```

### 4.3 annotate_labels

更新标注格式，删除 room/branch，密度显示为两位小数：

```python
def annotate_labels(
    img: np.ndarray,
    struct: torch.Tensor,   # [2] long
    density: torch.Tensor   # [3] float
) -> np.ndarray:
    s = struct.tolist()
    d = density.tolist()
    line1 = f"sym:{s[0]} outer:{s[1]}"
    line2 = f"door:{d[0]:.2f} enemy:{d[1]:.2f} res:{d[2]:.2f}"
    img = img.copy()
    for text, y in [(line1, 12), (line2, 24)]:
        cv2.putText(img, text, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(img, text, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return img
```

### 4.4 训练集与验证集初始化

```python
train_dataset = GinkaSeperatedDataset(args.train, subset_weights=SUBSET_WEIGHTS)
val_dataset = GinkaSeperatedDataset(
    args.validate,
    subset_weights=SUBSET_WEIGHTS,
    density_stats=train_dataset.density_stats  # 复用训练集统计量
)
```

### 4.5 DataLoader collate_fn（FloatTensor 适配）

PyTorch 默认 collate 会自动将 FloatTensor 列表合并为 float 类型批张量，无需额外修改 DataLoader 配置。

### 4.6 验证阶段密度对照图（density_var）

`visualize_density_var` 内对比不同密度条件时，改为使用 5 个均匀分布采样点：

```python
# 旧（三档枚举）：density_levels = [0, 1, 2]
# 新（连续采样）：5 个均匀间隔值
density_values = [0.0, 0.25, 0.5, 0.75, 1.0]
for v in density_values:
    d = torch.FloatTensor([[v, v, v]]).to(device)  # 三类等密度扫描
    ...
```

---

## 五、不需要改动的部分

- `ginka/maskGIT/maskGIT.py`：AdaLN / CondTransformerLayer / Transformer 均不感知条件维度，无需修改
- `ginka/vqvae/` 目录：VQ-VAE 部分与条件系统无关
- `ginka/train_seperated.py` 中的 `maskgit_sample`、`full_generate_random_z`、`full_generate_specific_z`：接口签名不变（仍接受 struct/density 张量），内部无直接操作条件内容，无需修改
- `data/` 目录的 TypeScript 数据处理脚本：数据文件格式不变，Python 端自行计算标签

---

## 六、旧 checkpoint 兼容性

由于 `cond_proj` 输入维度和嵌入层数量均发生变化，**旧 checkpoint 不兼容**，需从头训练。

---

## 七、实施顺序

1. 修改 `ginka/dataset.py`：删除 room/branch 分档，新增密度归一化和 `density_stats` 参数
2. 修改 `ginka/maskGIT/model.py`：删除多余嵌入，新增 `density_proj`，更新 `cond_proj` 维度和 `forward`
3. 修改 `ginka/train_seperated.py`：更新 `random_struct`、`random_density`、`annotate_labels`、数据集初始化
4. 运行小规模过拟合测试（单 batch 跑 50 步）验证前向通路无误
