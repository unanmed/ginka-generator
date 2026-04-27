# 数据集标签设计文档

## 背景

在当前 VQ-VAE + MaskGIT 的联合训练方案中，VQ-VAE 的 codebook 承担着地图风格与多样性的控制职能，但缺少可用户感知的语义维度。为提升生成的可控性，计划在数据集中添加一组**可程序化标注的结构标签**，作为额外条件输入训练，从而使模型能够接受来自用户的高层语义约束（如"生成一个左右对称的地图"、"高房间数量"等）。

所有标签均可在数据集构建阶段（`info.ts` / `parseFloorInfo`）或训练前的两趟扫描中自动计算，无需人工标注。

---

## 标签一览

| 标签名           | 类型      | 取值                | 标注时机 |
| ---------------- | --------- | ------------------- | -------- |
| `symmetryH`      | `boolean` | 左右对称            | 单张地图 |
| `symmetryV`      | `boolean` | 上下对称            | 单张地图 |
| `symmetryC`      | `boolean` | 中心对称            | 单张地图 |
| `roomCountLevel` | `0\|1\|2` | Low / Medium / High | 两趟扫描 |
| `branchLevel`    | `0\|1\|2` | Low / Medium / High | 两趟扫描 |
| `outerWall`      | `boolean` | 是否外包围墙壁      | 单张地图 |

---

## 标签一：对称性

### 定义

针对经过转换后的地图（`convertedMap`）矩阵逐格比较，仅当**完全满足条件**时才标记为对应对称类型。三种对称相互独立，可同时成立。

| 对称类型 | 条件（对所有 `x ∈ [0, W)`, `y ∈ [0, H)` 成立） |
| -------- | ---------------------------------------------- |
| 左右对称 | `map[y][x] === map[y][W - 1 - x]`              |
| 上下对称 | `map[y][x] === map[H - 1 - y][x]`              |
| 中心对称 | `map[y][x] === map[H - 1 - y][W - 1 - x]`      |

### 实现要点

- 比较使用 `convertedMap`（标签化图块编号），而非原始 `originMap`，使不同塔的同类图块具有可比性。
- 中心对称与左右、上下对称在数学上不蕴含关系（非充分也非必要），需独立计算。
- 对于奇数尺寸的地图（如 13×13），中心行/列与自身比较必然成立，无需特殊处理。

### 字段扩展（`IFloorInfo`）

```typescript
/** 左右对称 */
readonly symmetryH: boolean;
/** 上下对称 */
readonly symmetryV: boolean;
/** 中心对称 */
readonly symmetryC: boolean;
```

---

## 标签二：房间数量等级

### 定义

**房间（Room）** 是地图中由"空白节点"或"资源节点"组成的区域，同时满足以下三个条件：

1. **位置条件**：该节点在拓扑图中至少与 **1 个分支节点（Branch）** 相邻；
2. **面积条件**：节点所包含的地图格子数（`tiles.size`）**≥ 4**；
3. **形状条件**：节点所有格子的**外接矩形的宽和高都大于 1**（避免把单行/单列走廊计入房间）。

> **为什么这样定义房间**
>
> 在拓扑图中，空白/资源节点天然是游戏空间的"腔体"，而分支节点（门/怪物）是进入腔体的关卡节点。只要与至少一个分支节点相邻，就说明这片空间是需要"先过关才能进入/离开"的区域——包括怪物守着宝箱这类单入口房间，同样是典型的房间结构。面积和形状约束则过滤掉通道和死胡同。

### 等级划分

等级为 **三档**：Low（0）/ Medium（1）/ High（2），通过以下两趟扫描确定：

1. **第一趟**：遍历整个训练集，计算每张地图的房间数量 `roomCount`，收集为数组；
2. **第二趟**：对数组升序排序，取 1/3 和 2/3 分位数作为阈值 `[th1, th2]`：
    - `roomCount < th1` → Low（0）
    - `th1 ≤ roomCount < th2` → Medium（1）
    - `roomCount ≥ th2` → High（2）

等级划分力求三档样本数量均等（等频分箱），而非等距分箱。

### 外接矩形计算

给定节点的所有格子坐标集合（`tiles`，存储 `y * width + x` 的平坦坐标），还原为 `(x, y)` 后：

$$
x_{\min} = \min_{t \in tiles}(t \bmod W), \quad x_{\max} = \max_{t \in tiles}(t \bmod W)
$$

$$
y_{\min} = \min_{t \in tiles}(\lfloor t / W \rfloor), \quad y_{\max} = \max_{t \in tiles}(\lfloor t / W \rfloor)
$$

外接矩形宽 = $x_{\max} - x_{\min} + 1$，高 = $y_{\max} - y_{\min} + 1$，两者均需 $> 1$。

### 字段扩展

```typescript
/** 房间数量（原始统计值，供两趟扫描使用） */
readonly roomCount: number;
/** 房间数量等级：0=Low, 1=Medium, 2=High（需两趟扫描后赋值） */
roomCountLevel: 0 | 1 | 2;
```

---

## 标签三：分支数量等级

### 定义

**高连接度分支节点**：拓扑图中 `type === Branch` 的节点，其**非墙邻居节点**总数（`neighbors.size`）**≥ 3**。由于当前拓扑图中已不含墙节点，`neighbors.size` 即等于非墙邻居数，两者在实现上等价。

这类节点是地图中的"交叉口"——一个门或怪物后方至少有三条不同路线，是地图分叉度和策略深度的指征。

等级划分方式与房间数量等级相同：先统计每张地图中高连接度分支节点的数量，再等频分箱为 Low（0）/ Medium（1）/ High（2）。

### 与房间数量的区别

| 维度     | 房间数量等级               | 分支数量等级                 |
| -------- | -------------------------- | ---------------------------- |
| 度量对象 | 空白/资源节点区域的封闭性  | 分支节点的路径分叉度         |
| 反映特征 | 地图内封闭房间的数量与密度 | 关键路口/多分支门怪的复杂度  |
| 典型高值 | 多房间迷宫风格地图         | 高度分叉、策略选择丰富的地图 |

### 字段扩展

```typescript
/** 高连接度分支节点数量（原始统计值） */
readonly highDegBranchCount: number;
/** 分支数量等级：0=Low, 1=Medium, 2=High（需两趟扫描后赋值） */
branchLevel: 0 | 1 | 2;
```

---

## 标签四：外包围墙壁

### 定义

地图**最外圈**（最外一圈格子）的格子中，**墙壁格子与入口格子**之和占外圈总格子数的比例 > **90%**，则标记为 `outerWall = true`。

$$
\text{outerWall} = \frac{|\{(x,y) \in \text{border} : \text{isWall}(x,y) \lor \text{isEntry}(x,y)\}|}{|\text{border}|} > 0.9
$$

### 最外圈定义

对于 $H \times W$ 的地图，最外圈为所有满足下列条件之一的格子：

$$
x = 0 \; \lor \; x = W - 1 \; \lor \; y = 0 \; \lor \; y = H - 1
$$

最外圈格子总数为 $2(H + W) - 4$（对于 13×13，共 48 格）。

### 为什么入口也算"通过"

入口格子在游戏中是楼梯/传送点，不属于可通行的空地，在视觉和结构上等价于边界开口，属于外圈围合结构的合理组成部分，不应被视为"破坏围合"的元素。

### 实现要点

- 使用 `convertedMap` 判断墙壁（`tile === config.wall`）；
- 使用 `originMap` + `converter.isEntry()` 或直接对 `convertedMap` 判断入口（`tile === config.entry`）判断入口；
- 两项合取计入分子。

### 字段扩展

```typescript
/** 是否外包围墙壁 */
readonly outerWall: boolean;
```

---

## 实现方案

### 单张地图可直接计算的标签

以下标签可在 `parseFloorInfo` 中直接计算，加入 `IFloorInfo`：

- `symmetryH`、`symmetryV`、`symmetryC`
- `outerWall`
- `roomCount`（原始值）
- `highDegBranchCount`（原始值）

### 需要两趟扫描的等级标签

`roomCountLevel` 和 `branchLevel` 依赖全局分位数，须在数据集构建完成后进行二次处理：

```
第一趟：构建所有楼层的 IFloorInfo，写入 roomCount / highDegBranchCount
第二趟：收集所有楼层的原始值 → 计算 1/3, 2/3 分位 → 回填等级
```

在 Python 训练侧，推荐方式：

```python
# 在 Dataset.__init__ 中完成两趟计算
counts = [item['roomCount'] for item in raw_data]
counts_sorted = sorted(counts)
th1 = counts_sorted[len(counts_sorted) // 3]
th2 = counts_sorted[2 * len(counts_sorted) // 3]

for item in raw_data:
    c = item['roomCount']
    item['roomCountLevel'] = 0 if c < th1 else (1 if c < th2 else 2)
```

同理处理 `branchLevel`。

> **注意**：分位数阈值应仅基于**训练集**统计，验证集 / 测试集使用相同的阈值映射，避免数据泄露。

---

## 训练集成

### 条件嵌入

将上述标签作为离散条件与 VQ-VAE 的 z 一同注入 MaskGIT：

```python
# 对称性：三个独立布尔值，可合并为 0~7 的整数 cond_sym
cond_sym = symmetryH * 4 + symmetryV * 2 + symmetryC * 1  # [0, 7]

# 房间等级：0 / 1 / 2
cond_room = roomCountLevel

# 分支等级：0 / 1 / 2
cond_branch = branchLevel

# 外包围墙壁：0 / 1
cond_outer = int(outerWall)
```

每个条件通过独立的 `nn.Embedding` 映射为固定维度向量，与 VQ-VAE 的 z 序列沿序列维度拼接后，一同经 Cross-Attention 注入 MaskGIT。

### 条件 Dropout

与 z dropout 类似，训练时以一定概率（如 10~20%）将部分或全部结构标签替换为"无条件"（null embedding），使模型在推理时支持条件缺省（CFG 风格）。

---

## 待细化事项

- [x] 90% 阈值是否合适？——**保持 90%**，如后续数据分布分析发现问题再调整。
- [x] 房间定义中分支节点邻居数量——**改为至少 1 个**，覆盖怪物守宝箱的单入口房间场景。
- [x] 分支等级邻居计数口径——**使用非墙邻居**（当前图结构中无墙节点，与 `neighbors.size` 等价）。
- [x] 是否新增通道数量/路径长度标签——**暂不考虑**。
- [x] 条件嵌入维度对齐——**各标签 Embedding 与 z 序列拼接后统一经 Cross-Attention 注入**。
