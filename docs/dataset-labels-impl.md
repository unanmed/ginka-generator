# 数据集标签实现指南

## 总览

本文档描述将[标签设计文档](./dataset-labels-design.md)中定义的四类结构标签落地到代码中的具体实现步骤，涉及以下文件：

| 文件                     | 变更性质                                    |
| ------------------------ | ------------------------------------------- |
| `data/src/auto/types.ts` | 扩展 `IFloorInfo` 接口                      |
| `data/src/types.ts`      | 扩展 `GinkaTrainData` 接口                  |
| `data/src/auto/info.ts`  | 新增四个 helper 函数，修改 `parseFloorInfo` |
| `data/src/auto.ts`       | 在数据集序列化时写入新字段                  |
| `ginka/dataset.py`       | 两趟扫描 + `__getitem__` 读取新标签         |

---

## 第一步：扩展 TypeScript 类型

### `data/src/auto/types.ts` — 扩展 `IFloorInfo`

在 `IFloorInfo` 接口的 `doorHeatmap` 字段之后追加以下字段：

```typescript
// ── 结构标签（新增）──────────────────────────────
/** 左右对称（基于 convertedMap 完全匹配） */
readonly symmetryH: boolean;
/** 上下对称 */
readonly symmetryV: boolean;
/** 中心对称 */
readonly symmetryC: boolean;
/** 是否外包围墙壁（最外圈墙壁+入口占比 > 90%） */
readonly outerWall: boolean;
/** 房间数量原始值（供 Python 两趟扫描使用） */
readonly roomCount: number;
/** 高连接度分支节点数量原始值（供 Python 两趟扫描使用） */
readonly highDegBranchCount: number;
```

### `data/src/types.ts` — 扩展 `GinkaTrainData`

在 `GinkaTrainData` 接口中追加序列化后写入 JSON 的字段：

```typescript
export interface GinkaTrainData {
    tag?: number[];
    val: number[];
    map: number[][];
    size: [number, number];
    heatmap?: number[][][];
    // ── 结构标签（新增）──────────────────────────────
    /** 对称性：[symmetryH, symmetryV, symmetryC]，0 或 1 */
    symmetry: [number, number, number];
    /** 是否外包围墙壁，0 或 1 */
    outerWall: number;
    /** 房间数量原始值 */
    roomCount: number;
    /** 高连接度分支节点数量原始值 */
    highDegBranchCount: number;
}
```

---

## 第二步：实现 Helper 函数（`data/src/auto/info.ts`）

以下四个函数添加到 `computeWallDensityStd` 之后、`parseFloorInfo` 之前。

### 2.1 对称性计算

```typescript
/**
 * 计算地图的三种对称性（基于 convertedMap，完全匹配才标记为 true）
 */
function computeSymmetry(map: number[][]): {
    symmetryH: boolean;
    symmetryV: boolean;
    symmetryC: boolean;
} {
    const H = map.length;
    const W = H > 0 ? map[0].length : 0;
    let symmetryH = true;
    let symmetryV = true;
    let symmetryC = true;

    outer: for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const tile = map[y][x];
            if (symmetryH && tile !== map[y][W - 1 - x]) symmetryH = false;
            if (symmetryV && tile !== map[H - 1 - y][x]) symmetryV = false;
            if (symmetryC && tile !== map[H - 1 - y][W - 1 - x])
                symmetryC = false;
            // 三种对称均已排除，提前退出
            if (!symmetryH && !symmetryV && !symmetryC) break outer;
        }
    }
    return { symmetryH, symmetryV, symmetryC };
}
```

**复杂度**：O(H × W)，最坏情况遍历全图一次，实际有短路优化。

**注意**：对于奇数尺寸地图（如 13×13），中心行/列的格子与自身比较恒为 true，不影响结果。

### 2.2 外包围墙壁检测

```typescript
/**
 * 检测地图最外圈中墙壁 + 入口的占比是否超过 90%
 * @param map convertedMap
 * @param wall 墙壁图块编号
 * @param entry 入口图块编号
 */
function computeOuterWall(
    map: number[][],
    wall: number,
    entry: number
): boolean {
    const H = map.length;
    const W = H > 0 ? map[0].length : 0;
    if (H < 2 || W < 2) return false;

    let borderCount = 0;
    let wallOrEntry = 0;

    const check = (tile: number) => {
        borderCount++;
        if (tile === wall || tile === entry) wallOrEntry++;
    };

    // 顶行 + 底行
    for (let x = 0; x < W; x++) {
        check(map[0][x]);
        check(map[H - 1][x]);
    }
    // 左列 + 右列（排除角格，已由上面计入）
    for (let y = 1; y < H - 1; y++) {
        check(map[y][0]);
        check(map[y][W - 1]);
    }

    // 13×13 地图: borderCount = 2*(13+13)-4 = 48
    return borderCount > 0 && wallOrEntry / borderCount > 0.9;
}
```

### 2.3 房间数量统计

**核心思路**：拓扑图中 Empty 节点和 Resource 节点在物理上可能彼此相邻，共同构成一个连续的游戏空间（如 2×3 的房间内放了一个宝物）。不能逐节点独立判断，而应先通过 BFS 将相邻的 Empty/Resource 节点合并为一个"候选区域"，再对整个区域判断三个条件。Branch 节点作为边界，不会被合并进区域。

```typescript
/**
 * 统计拓扑图中符合"房间"定义的连通区域数量。
 *
 * 算法：
 *   1. 以 Empty / Resource 节点为顶点，在它们之间 BFS，
 *      得到若干"候选区域"（Branch 节点作为边界，不被合并）。
 *   2. 对每个候选区域检查三个条件：
 *      a. 区域内至少一个节点有 Branch 类型邻居
 *      b. 区域内所有格子总数 >= 4
 *      c. 所有格子的外接矩形宽 > 1 且高 > 1
 *
 * @param graph  拓扑图
 * @param width  地图宽度（用于平坦坐标解码）
 */
function computeRoomCount(graph: IMapGraph, width: number): number {
    // Step 1: 收集所有 Empty 和 Resource 节点（去重）
    const allEmptyResource = new Set<MapGraphNode>();
    for (const node of graph.nodeMap.values()) {
        if (
            node.type === GraphNodeType.Empty ||
            node.type === GraphNodeType.Resource
        ) {
            allEmptyResource.add(node);
        }
    }

    // Step 2: BFS 将相邻的 Empty/Resource 节点合并为连通区域
    let roomCount = 0;
    const visited = new Set<MapGraphNode>();

    for (const startNode of allEmptyResource) {
        if (visited.has(startNode)) continue;

        // BFS：只在 Empty/Resource 节点间传播，Branch 节点阻断
        const regionNodes = new Set<MapGraphNode>();
        const queue: MapGraphNode[] = [startNode];
        visited.add(startNode);

        while (queue.length > 0) {
            const current = queue.shift()!;
            regionNodes.add(current);
            for (const nb of current.neighbors) {
                if (
                    !visited.has(nb) &&
                    (nb.type === GraphNodeType.Empty ||
                        nb.type === GraphNodeType.Resource)
                ) {
                    visited.add(nb);
                    queue.push(nb);
                }
            }
        }

        // Step 3: 对合并后的区域检查三个条件

        // 条件 a：区域内任一节点有 Branch 邻居
        let hasBranch = false;
        outer: for (const node of regionNodes) {
            for (const nb of node.neighbors) {
                if (nb.type === GraphNodeType.Branch) {
                    hasBranch = true;
                    break outer;
                }
            }
        }
        if (!hasBranch) continue;

        // 收集区域内所有格子，计算总数和外接矩形
        let totalTiles = 0;
        let minX = Infinity,
            maxX = -Infinity,
            minY = Infinity,
            maxY = -Infinity;

        for (const node of regionNodes) {
            totalTiles += node.tiles.size;
            for (const t of node.tiles) {
                const x = t % width;
                const y = (t - x) / width;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        // 条件 b：总格子数 >= 4
        if (totalTiles < 4) continue;

        // 条件 c：外接矩形宽高均 > 1（即 maxX - minX >= 1 && maxY - minY >= 1）
        if (maxX - minX < 1 || maxY - minY < 1) continue;

        roomCount++;
    }

    return roomCount;
}
```

**边界情况讨论**：

- **混合节点房间**：2×3 房间（5 个空地 + 1 个资源）→ 1 个 Empty 节点 + 1 个 Resource 节点，BFS 合并后总格子数=6，外接矩形 2×3，**计入房间**。
- **纯条形走廊**：1×4 的 Empty 节点 → 外接矩形高=1，**不计入房间**。
- **孤立资源死角**：Resource 节点的邻居全是 Entry 而无 Branch → 条件 a 不满足，**不计入房间**。
- **两个相邻的 Empty 区域被 Branch 隔开**：BFS 不会跨越 Branch，各自单独判断。

### 2.4 高连接度分支节点统计

```typescript
/**
 * 统计邻居数 >= 3 的分支节点数量
 *
 * 由于拓扑图中已不含墙节点，neighbors.size 即等于非墙邻居数。
 *
 * @param graph 拓扑图
 */
function computeHighDegBranchCount(graph: IMapGraph): number {
    let count = 0;
    const visited = new Set<MapGraphNode>();

    for (const node of graph.nodeMap.values()) {
        if (visited.has(node)) continue;
        visited.add(node);

        if (node.type === GraphNodeType.Branch && node.neighbors.size >= 3) {
            count++;
        }
    }
    return count;
}
```

---

## 第三步：在 `parseFloorInfo` 中调用

在 `parseFloorInfo` 函数内，`topo` 构建完毕、`floorInfo` 对象构造之前，调用上述四个函数：

```typescript
// ── 结构标签计算 ─────────────────────────────────
const width = map[0]?.length ?? 0;
const { symmetryH, symmetryV, symmetryC } = computeSymmetry(map);
const outerWall = computeOuterWall(
    map,
    config.classes.wall,
    config.classes.entry
);
const roomCount = computeRoomCount(topo.graph, width);
const highDegBranchCount = computeHighDegBranchCount(topo.graph);
```

然后在 `floorInfo` 字面量中追加这些字段：

```typescript
const floorInfo: IFloorInfo = {
    // ...（原有字段保持不变）
    symmetryH,
    symmetryV,
    symmetryC,
    outerWall,
    roomCount,
    highDegBranchCount
};
```

---

## 第四步：在 `data/src/auto.ts` 中序列化新字段

在构建 `GinkaTrainData` 的对象字面量中追加：

```typescript
const data: GinkaTrainData = {
    map: floor.data.map,
    size: [width, height],
    heatmap: [
        /* ...原有热力图通道... */
    ],
    val: [
        /* ...原有标量... */
    ],
    // ── 新增结构标签 ──────────────────────────────
    symmetry: [
        info.symmetryH ? 1 : 0,
        info.symmetryV ? 1 : 0,
        info.symmetryC ? 1 : 0
    ],
    outerWall: info.outerWall ? 1 : 0,
    roomCount: info.roomCount,
    highDegBranchCount: info.highDegBranchCount
};
```

布尔值存为 `0/1` 整数，便于 Python 侧直接读取，不需要类型转换。

---

## 第五步：Python 侧两趟扫描（`ginka/dataset.py`）

### 5.1 等频分箱函数

在 `dataset.py` 顶部添加通用的等频分箱 helper：

```python
def assign_level(values: list[int]) -> list[int]:
    """
    将整数列表按等频分箱映射为 0/1/2 三档等级。
    分位数阈值基于当前列表计算（训练集与验证集应共用训练集的阈值）。

    Args:
        values: 原始统计值列表，顺序与数据集一一对应

    Returns:
        与输入等长的等级列表，每项为 0 / 1 / 2
    """
    n = len(values)
    if n == 0:
        return []
    sorted_vals = sorted(values)
    th1 = sorted_vals[n // 3]
    th2 = sorted_vals[2 * n // 3]
    return [
        0 if v < th1 else (1 if v < th2 else 2)
        for v in values
    ]
```

### 5.2 修改 `GinkaVQDataset.__init__`

在加载数据之后立即进行两趟扫描，并将等级结果回填到各 item 中：

```python
class GinkaVQDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        subset_weights: tuple = (0.5, 0.2, 0.2, 0.1),
        wall_mask_min: float = 0.0,
        wall_mask_max: float = 0.5,
        # 以下两个参数仅在 train 模式下使用，验证集传入训练集的阈值即可
        room_thresholds: tuple[int, int] | None = None,
        branch_thresholds: tuple[int, int] | None = None,
    ):
        self.data = load_data(data_path)
        # ...（原有初始化逻辑）

        # ── 两趟扫描：计算等频分箱阈值 ──────────────────────────────
        room_counts   = [item['roomCount']        for item in self.data]
        branch_counts = [item['highDegBranchCount'] for item in self.data]

        if room_thresholds is None:
            # 训练集：自行计算阈值
            n = len(room_counts)
            rs = sorted(room_counts)
            bs = sorted(branch_counts)
            self.room_th   = (rs[n // 3],  rs[2 * n // 3])
            self.branch_th = (bs[n // 3],  bs[2 * n // 3])
        else:
            # 验证集/测试集：直接使用训练集的阈值，避免数据泄露
            self.room_th   = room_thresholds
            self.branch_th = branch_thresholds

        def to_level(v: int, th: tuple[int, int]) -> int:
            return 0 if v < th[0] else (1 if v < th[1] else 2)

        # 回填等级字段
        for item in self.data:
            item['roomCountLevel']  = to_level(item['roomCount'],        self.room_th)
            item['branchLevel']    = to_level(item['highDegBranchCount'], self.branch_th)
```

**调用方式**（`train_vq.py` 中）：

```python
dataset_train = GinkaVQDataset(args.train)
dataset_val   = GinkaVQDataset(
    args.validate,
    room_thresholds=dataset_train.room_th,
    branch_thresholds=dataset_train.branch_th
)
```

### 5.3 `__getitem__` 中读取结构标签

对称性标签在数据增强（旋转/翻转）后需要**重新从增强后的地图中计算**，因为 `rot90(k=1/3)` 会交换 `symmetryH` 和 `symmetryV`。其他三个标签在旋转/翻转下保持不变，可直接读取。

```python
def _compute_symmetry(target_np: np.ndarray) -> tuple[int, int, int]:
    """从 numpy 地图矩阵中直接计算三种对称性，O(H*W)"""
    H, W = target_np.shape
    sym_h = bool(np.all(target_np == target_np[:, ::-1]))
    sym_v = bool(np.all(target_np == target_np[::-1, :]))
    sym_c = bool(np.all(target_np == target_np[::-1, ::-1]))
    return int(sym_h), int(sym_v), int(sym_c)
```

在 `__getitem__` 数据增强完成后，读取所有标签：

```python
def __getitem__(self, idx):
    # ...（原有增强逻辑，target_np 已经过 rot90 / flip）

    # 对称性：在增强后重新计算
    sym_h, sym_v, sym_c = _compute_symmetry(target_np)
    cond_sym = sym_h * 4 + sym_v * 2 + sym_c  # [0, 7]

    # 其余标签：增强不改变拓扑结构，直接读取
    item    = self.data[idx]
    cond_room   = item['roomCountLevel']   # 0/1/2
    cond_branch = item['branchLevel']      # 0/1/2
    cond_outer  = item['outerWall']        # 0/1

    # 封装为 tensor
    struct_cond = torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer])

    return {
        "raw_map":    ...,
        "masked_map": ...,
        "target_map": ...,
        "subset":     ...,
        "struct_cond": struct_cond  # [4]，供模型 Embedding 查表
    }
```

---

## 第六步：模型侧条件注入（`ginka/maskGIT/model.py`）

`struct_cond` 的四个维度分别对应不同词表大小的 Embedding：

```python
# 词表大小
SYM_VOCAB    = 8   # cond_sym:    [0, 7]
ROOM_VOCAB   = 4   # cond_room:   [0, 2] + 1 个 null（dropout 用）
BRANCH_VOCAB = 4   # cond_branch: [0, 2] + 1 个 null
OUTER_VOCAB  = 3   # cond_outer:  [0, 1] + 1 个 null

# 在 GinkaMaskGIT.__init__ 中
self.sym_embed    = nn.Embedding(SYM_VOCAB,    d_z)
self.room_embed   = nn.Embedding(ROOM_VOCAB,   d_z)
self.branch_embed = nn.Embedding(BRANCH_VOCAB, d_z)
self.outer_embed  = nn.Embedding(OUTER_VOCAB,  d_z)
```

在 `forward` 中，将四个 Embedding 结果与 VQ-VAE 的 z 序列拼接，作为 Cross-Attention 的 memory：

```python
def forward(self, map_tokens, z, struct_cond, dropout_struct=False):
    # z: [B, L, d_z]
    # struct_cond: [B, 4]  — [sym, room, branch, outer]

    B = z.size(0)

    if dropout_struct or (self.training and torch.rand(1) < self.struct_dropout_prob):
        # 条件 dropout：全部替换为 null index（各词表最后一个 index）
        e_sym    = self.sym_embed(torch.full((B,), SYM_VOCAB - 1,    device=z.device))
        e_room   = self.room_embed(torch.full((B,), ROOM_VOCAB - 1,  device=z.device))
        e_branch = self.branch_embed(torch.full((B,), BRANCH_VOCAB - 1, device=z.device))
        e_outer  = self.outer_embed(torch.full((B,), OUTER_VOCAB - 1, device=z.device))
    else:
        e_sym    = self.sym_embed(struct_cond[:, 0])    # [B, d_z]
        e_room   = self.room_embed(struct_cond[:, 1])
        e_branch = self.branch_embed(struct_cond[:, 2])
        e_outer  = self.outer_embed(struct_cond[:, 3])

    # 将四个结构标签嵌入拼接为序列，与 z 合并
    # 每个 e_* 形状为 [B, d_z]，unsqueeze 后变为 [B, 1, d_z]
    struct_seq = torch.stack(
        [e_sym, e_room, e_branch, e_outer], dim=1
    )  # [B, 4, d_z]

    memory = torch.cat([z, struct_seq], dim=1)  # [B, L+4, d_z]

    # 后续 Cross-Attention 正常进行
    # query = map_token_embeddings，key/value = memory
    ...
```

**null index 规则**：各 Embedding 的最后一个 index 保留为"无条件"占位符，词表大小因此比有效类别数多 1。

---

## 边界情况与注意事项

### 关于两趟扫描的时机

两趟扫描在 `Dataset.__init__` 中完成，整个过程仅需遍历两次 Python 列表，耗时可忽略不计（数千条数据 < 1ms）。不建议延迟到 `__getitem__` 中逐条计算。

### 关于对称性在数据增强下的变化

| 增强操作          | `symmetryH` | `symmetryV` | `symmetryC` |
| ----------------- | ----------- | ----------- | ----------- |
| `fliplr`          | 不变        | 不变        | 不变        |
| `flipud`          | 不变        | 不变        | 不变        |
| `rot90(k=1 or 3)` | 与 V 交换   | 与 H 交换   | 不变        |
| `rot90(k=2)`      | 不变        | 不变        | 不变        |

因此在 `__getitem__` 中对增强后的地图**重新计算**对称性是最简洁正确的方案，无需记录增强历史。

### 关于极端分布

若训练集中某一标签的样本极度不均匀（如 90% 的地图无对称性），可以在条件 Dropout 中对不常见的条件值适当降低 dropout 概率，以确保模型充分学习该条件。初始阶段统一使用相同 dropout 概率即可，后续根据生成效果调整。

### 关于 `roomCount = 0` 和 `highDegBranchCount = 0` 的地图

这类地图在等频分箱后会进入 Low（0）等级。如果训练集中大量地图的值为 0，`th1` 可能也为 0，导致 Low 等级极少。可以在分箱前加一步检查：若 `th1 == th2`，则手动将 `th2 = th1 + 1` 以避免 Medium 等级为空。

```python
th1 = sorted_vals[n // 3]
th2 = sorted_vals[2 * n // 3]
if th1 == th2:
    th2 = th1 + 1  # 防止 Medium 等级为空
```
