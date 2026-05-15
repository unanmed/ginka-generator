# 实体密度标签设计文档

## 背景与问题

当前三阶段级联生成（stage1 骨架、stage2 功能实体、stage3 资源）在结构可行性上基本稳定，但存在明显的分布偏移：

- 怪物数量偏多
- 资源数量偏多
- 门数量在部分样本上也偏高

已尝试在采样阶段通过“随机抛弃部分新揭开位并重新掩码”的方式抑制过密生成，但效果不稳定，核心原因是该策略属于推理期启发式约束，不能从训练目标层面改变模型对全局密度的先验。

因此需要引入显式条件：将每张地图中门、怪物、资源的密度离散为三档（低/中/高），并在训练和推理时作为条件输入，让模型学习“在指定密度档位下生成”。

## 目标

- 新增 3 个可控标签：`doorDensityLevel`、`monsterDensityLevel`、`resourceDensityLevel`，取值均为 `0 | 1 | 2`。
- 标签计算与分档在 Python 端完成，保持与现有 `roomCountLevel`、`branchLevel` 一致的处理方式。
- 标签注入模型后，支持在推理时显式控制三类实体密度。
- 在不改动数据处理端（TypeScript）的前提下完成接入。

## 设计原则

- 统计口径稳定：密度分母采用固定地图面积（13x13），避免受随机掩码影响。
- 分档可迁移：使用训练集等频分箱阈值；验证/推理复用同一阈值。
- 最小侵入：优先扩展现有 Python 数据集与条件注入链路，不改变数据文件格式。
- 可回溯：训练日志与可视化中输出目标密度档位与实际密度，便于诊断。

## 标签定义

### 1. 统计对象

基于原始地图 `item['map']`（未掩码、未降级）统计三类图块数量：

- `doorCount`: 图块 ID = 2
- `resourceCount`: 图块 ID = 3
- `monsterCount`: 图块 ID = 4

### 2. 密度定义

设地图面积为 `MAP_SIZE = 13 * 13 = 169`，则：

- `doorDensity = doorCount / 169`
- `monsterDensity = monsterCount / 169`
- `resourceDensity = resourceCount / 169`

### 3. 分档定义

采用等频分箱（三档）并与现有 `to_level` 规则一致：

- 训练集上收集某一密度指标的全量样本值，升序排序
- 取 `n/3` 与 `2n/3` 位置作为阈值 `th1`、`th2`
- 分档规则：
    - `< th1` -> `0`（Low）
    - `>= th1 且 < th2` -> `1`（Medium）
    - `>= th2` -> `2`（High）

阈值退化处理（与现有实现一致）：

- 若 `th1 == th2`，将 `th2 = th1 + eps`
- 对密度值建议 `eps = 1e-6`

## Python 端处理方案

### 1. 数据集初始化阶段

在 `GinkaSeperatedDataset.__init__` 中新增一次统计流程：

- 从 `self.data` 中提取每张图的 `doorDensity`、`monsterDensity`、`resourceDensity`
- 分别计算三组阈值：
    - `self.door_density_th`
    - `self.monster_density_th`
    - `self.resource_density_th`
- 回填每个样本：
    - `item['doorDensityLevel']`
    - `item['monsterDensityLevel']`
    - `item['resourceDensityLevel']`

### 2. 样本输出阶段

在 `__getitem__` 返回字典中新增条件向量（建议独立字段，避免影响旧逻辑）：

- `density_inject = LongTensor([doorLevel, monsterLevel, resourceLevel])`

不建议直接复用旧 `struct_inject` 覆盖含义。推荐并行保留：

- `struct_inject`：结构语义（对称/房间/分支/外墙）
- `density_inject`：实体密度语义（门/怪物/资源）

## 模型接入方案

### 1. 条件输入组织

密度条件与结构条件在语义上完全不同（结构描述地图拓扑形态，密度描述实体数量先验），不复用 `struct_inject` 的处理路径。

设计：在 MaskGIT 内新增一个独立的**密度 MLP**：

- 输入：3 个独立 embedding 表（每档取值 0/1/2）输出相加后的向量
    - `emb_door_density: Embedding(3, d_embed)`
    - `emb_monster_density: Embedding(3, d_embed)`
    - `emb_resource_density: Embedding(3, d_embed)`
- 三个 embedding 相加后送入 2 层 MLP（`d_embed -> d_model -> d_model`，激活函数 GELU），输出一个 `d_model` 维向量
- 该向量作为独立条件 token 拼接到主序列头部（与 struct token 并列，不替换）

结构条件（`struct_inject`）保留原有处理方式不变。

### 2. 训练与推理接口

- 训练前向：`mgX(inpX, z_q, struct_inject, density_inject)`
- 推理采样：允许显式指定密度档位；未指定时可随机采样档位或使用数据先验分布采样

### 3. 条件 Dropout

对密度条件增加独立 dropout（例如 0.1）：

- 训练时随机置空部分密度条件，降低过拟合风险
- 推理时可在“无密度条件”与“强密度条件”两种模式间切换

## 训练与验证改造

### 1. 日志指标

在验证阶段新增统计输出：

- 按档位分组的密度 L1 误差：分别统计 door/monster/resource 三类实体在 Low/Medium/High 三档条件下，生成地图实际计数与档位中位期望值之间的 L1 距离（仅用于观察，不参与反向传播）

无需额外输出目标档位分布或实际密度均值，档位 L1 已足够直观反映控制效果。

### 2. 可视化对照

在每张验证生成图上直接标注所有条件标签，分两行显示：

- 第一行（结构标签）：`sym=N room=L/M/H branch=L/M/H outer=0/1`
- 第二行（密度标签）：`d=L/M/H m=L/M/H r=L/M/H`

其中 `sym` 取 `cond_sym` 的原始整数值（0–7），`room`/`branch`/`d`/`m`/`r` 均以 `L`/`M`/`H` 表示三档。

标注位置：图像顶部左上角，两行叠加，与现有 `fix`/`free` 标注并列（可追加到同一 `annotate` 调用后）。

额外新增一类对照图：固定同一 `z` 和结构条件，仅扫遍密度档位（Low/Medium/High 三档），分别生成地图并排排列，用于直观验证"只改密度条件，生成实体数量随档位单调变化"。该对照图在每个 checkpoint 验证时生成一次，保存到 `result/seperated/eN/density_cmp.png`。

### 3. 验收标准

至少满足以下条件后再认为方案有效：

- 同一结构条件下，密度档位从 Low -> High 时，三类实体计数总体单调上升
- 验证集上各档位的目标-实际密度 MAE 明显低于未加标签版本
- 地图可玩性不退化（入口可达、关键路径连通性不显著恶化）

## 与现有流程的兼容性

- 数据源 JSON 无需新增字段。
- 标签在 Python 读取后即时计算，不影响 `data/` 侧脚本。
- 旧 checkpoint 不兼容新增输入维度，需要从旧权重迁移或重新训练。

## 实施步骤建议

1. 在数据集类中实现三类密度统计、分档和 `density_inject` 返回。
2. 扩展 MaskGIT 条件嵌入与前向接口，打通三阶段训练调用。
3. 更新训练/验证日志与可视化标注，增加按档位评估。
4. 先做小规模过拟合与对照采样验证，再进入完整训练。

## 风险与应对

- 风险：档位边界样本噪声大，模型学习不稳定。
    - 应对：引入软标签邻域采样（可选）或在损失中增加密度一致性正则。

- 风险：实体密度受结构强约束，条件可控性受限。
    - 应对：在评估中按结构复杂度分组分析，必要时引入结构-密度联合条件建模。

- 风险：三阶段相互影响导致 stage2/stage3 条件冲突。
    - 应对：分别监控阶段内计数与最终合并计数，必要时增加阶段特异性权重。

## 后续可扩展方向

- 将三档扩展为五档，提升控制精度。
- 在密度标签之外增加“功能实体聚集度/均匀度”标签。
- 引入条件一致性判别器，进一步约束生成结果与目标档位一致。
