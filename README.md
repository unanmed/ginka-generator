# GINKA 地图生成器

GINKA Model 是一个用于生成网格状魔塔地图的模型，采用 UNet 网络，允许输入自然语言，指定地图大小。

GINKA Model 内部集成了 Minamo Model 用于判别两个地图的相似性，用于计算损失值，指导 GINKA Model 训练，避免了传统拓扑图相似度计算的不可微性质，提高模型训练性能。

## 贡献 GINKA Model 数据集

对于 HTML5 魔塔，如果你想要贡献数据集，需要对你的魔塔进行手动数据处理，流程如下：

1. 选择楼层，可以是剧情层、战斗层等，但是需要满足下述条件
2. 楼层除边缘外不应出现墙壁堆叠（例如 2\*2，边缘可以有重叠）
3. 楼层中不应该有闲置怪，不应该有连续 3 个以上的怪物，不应该有无法到达的区域，不宜有过多的入口
4. 最外面一层围上一圈墙壁（箭头楼层切换除外）
5. 将所有的墙壁换成黄墙（数字 1）
6. 将所有的血瓶换成红血瓶（数字 31），所有红宝石换成最基础的红宝石（数字 27），蓝宝石换成最基础的蓝宝石（数字 28），道具全部换为幸运金币（数字 53），剑盾可以当成红蓝宝石看待，删除除此之外的资源
7. 所有钥匙换成黄钥匙（数字 21），所有门换成黄门（数字 81）
8. 所有箭头换成样板原版箭头（数字 161 至 164），所有上下楼梯换成样板原版楼梯（数字 87 和 88）
9. 怪物分为三个强度，弱怪，中怪，强怪，弱怪换为绿头怪（数字 201），中怪换成红头怪（数字 202），强怪换成青头怪（数字 203）
10. 在 `project` 文件夹下创建 `ginka-config.json` 文件，双击进入编辑，粘贴如下模板：

```json
{
    "clip": {
        "defaults": [0, 0, 13, 13],
        "special": {
            "MT11": [3, 3, 7, 7]
        }
    },
    "data": {
        "MT1": ["MT1 楼层的第一个描述", "MT1 楼层的第二个描述"]
    }
}
```

其中，`clip` 属性表示你的每张地图的那一部分会被当成数据集，例如填写 `[0, 0, 13, 13]` 就会让坐标为 `(0, 0)`，长宽为 `(13, 13)` 的矩形内容作为数据集。`special` 属性允许你针对单独的某几层设置不同的裁剪方式，例如设置 `MT11` 为 `[3, 3, 7, 7]` 等，如果没有设置默认使用 `defaults` 的裁剪方式。最好保证每个楼层大小一致，不然我还要手动分类。

`data` 是你的每一层的楼层描述，要求每一层都要有描述，每层描述可以有多个，要求可以准确或粗略描述楼层的一部分特征（不需要是全部，不然的话文字量会很大），不超过 64 个 token（每个中文字约 0.6 token，每个英文字母约 0.3 token），可以详细可以简略，推荐每层有三个描述以上，而且描述不要有过多的语义重复。

11. 在全塔属性中的楼层列表中去除不在数据集内的楼层
12. 将 `project` 文件夹打包发给我即可

## 贡献 Minamo Model 数据集

首先需要对你的塔的地图进行处理，参考 [GINKA Model 数据处理方式](#贡献-ginka-model-数据集)的 1-9 和 11 步，同时最好每张地图至少有几个空格（没有也没有影响，只不过会导致这个地图参与训练的次数降低）

与 GINKA Model 类似，在 `project` 文件夹中添加 `minamo-config.json` 文件，打开后将如下模板粘贴进去：

```json
{
    "clip": {
        "defaults": [0, 0, 13, 13],
        "special": {
            "MT11": [3, 3, 7, 7]
        }
    }
}
```

其中 `clip` 属性与 GINKA Model 的定义相同，参考上一小节即可。

将以上内容全部设置完毕后，将 `project` 文件夹打包发送给我，添加 `ginka-config.json` 时也可以作为 GINKA Model 训练集。

## 训练

如果你想自行训练模型，首先需要安装 `reqiurements.txt` 中所需的库，然后按照以下顺序操作：

1. 准备 Minamo Model 数据集，放置在根目录下，命名为 `minamo-dataset.json`
2. 执行 `python -m minamo.train`，等待训练完毕
3. 准备 GINKA Model 数据集，放置在根目录下，命名为 `ginka-dataset.json`
4. 执行 `python -m ginka.train`，等待训练完毕
5. 目录 `result/ginka.pth` 即为训练完毕的 GINKA 模型
6. （可选）准备 Minamo Model / GINKA Model 验证集，放置在根目录下，命名为 `minamo-eval.json` / `ginka-eval.json`，训练时每 10 个 epoch 会进行一次验证推理，建议使用与训练集不同的塔数据作为验证集。

准备训练集和验证集时，可以使用命令行脚本自动处理。首先进入 `data` 文件夹，然后运行 `pnpm i` 安装所有依赖，然后执行下面的脚本：

```bash
pnpm ginka "../ginka-dataset.json" "MyTower/project" # 通过 ginka-config.json 生成 GINKA 训练集
pnpm ginka "../ginka-eval.json" "MyTower/project" # 通过 ginka-config.json 生成 GINKA 验证集
pnpm minamo "../minamo-dataset.json" "MyTower/project" # 通过 minamo-config.json 生成 Minamo 训练集
pnpm minamo "../minamo-eval.json" "MyTower/project" # 通过 minamo-config.json 生成 Minamo 验证集
```
