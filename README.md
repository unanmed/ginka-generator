# GINKA 地图生成器

GINKA Model 是一个用于生成网格状魔塔地图的模型，采用 UNet 网络。

GINKA Model 内部集成了 Minamo Model 用做判别器，与 Ginka Model 对抗训练，训练使用 Wasserstein GAN 训练方式。

## 贡献 GINKA Model 数据集

对于 HTML5 魔塔，如果你想要贡献数据集，需要对你的魔塔进行手动数据处理，流程如下：

1. 选择楼层，可以是剧情层、战斗层等，但是需要满足下述条件
2. 楼层除边缘外不应出现墙壁堆叠（例如 2\*2，边缘可以有重叠）
3. 楼层中不应该有闲置怪，不应该在直线上有无间隔连续 3 个以上的怪物，不应该有无法到达的区域，不宜有过多的入口
4. 最外面一层围上一圈墙壁（箭头楼层切换除外）
5. 将所有的墙壁换成黄墙（数字 1）
6. 将所有的血瓶换成红血瓶（数字 31），所有红宝石换成最基础的红宝石（数字 27），蓝宝石换成最基础的蓝宝石（数字 28），道具全部换为幸运金币（数字 53），剑盾可以当成红蓝宝石看待，删除除此之外的资源
7. 所有钥匙换成黄钥匙（数字 21），所有门换成黄门（数字 81）
8. 所有箭头换成样板原版箭头（数字 91 至 94），所有上下楼梯换成样板原版楼梯（数字 87 和 88）
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
    "data": {}
}
```

其中，`clip` 属性表示你的每张地图的那一部分会被当成数据集，例如填写 `[0, 0, 13, 13]` 就会让坐标为 `(0, 0)`，长宽为 `(13, 13)` 的矩形内容作为数据集。`special` 属性允许你针对单独的某几层设置不同的裁剪方式，例如设置 `MT11` 为 `[3, 3, 7, 7]` 等，如果没有设置默认使用 `defaults` 的裁剪方式。最好保证每个楼层大小一致，不然我还要手动分类。

11. 在全塔属性中的楼层列表中去除不在数据集内的楼层
12. 将 `project` 文件夹打包发给我即可
