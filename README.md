# GINKA 地图生成器

GINKA Model 是一个用于生成网格状魔塔地图的模型，采用 UNet 网络。

GINKA Model 内部集成了 Minamo Model 用做判别器，与 Ginka Model 对抗训练，训练使用 Wasserstein GAN 训练方式。

## 贡献 GINKA Model 数据集

对于 HTML5 魔塔，如果你想要贡献数据集，需要对你的魔塔进行手动数据处理，流程如下：

1. 在 `project` 文件夹下创建 `ginka-config.json` 文件，双击进入编辑，粘贴如下模板：

```json
{
    "clip": {
        "defaults": [0, 0, 13, 13],
        "special": {}
    },
    "mapping": {
        "redGem": {
            "27": 1
        },
        "blueGem": {
            "28": 1
        },
        "greenGem": {
            "29": 1
        },
        "yellowGem": {
            "30": 1
        },
        "item": {
            "47": 1,
            "49": 1,
            "50": 0,
            "51": 1,
            "52": 1,
            "53": 2
        },
        "potion": {
            "31": 100,
            "32": 200,
            "33": 400,
            "34": 800
        },
        "key": {
            "21": 0,
            "22": 1,
            "23": 2,
            "24": 2,
            "25": 2
        },
        "door": {
            "81": 0,
            "82": 1,
            "83": 2,
            "84": 2,
            "85": 3,
            "86": 2
        },
        "wall": [1, 17],
        "decoration": [],
        "floor": [87, 88],
        "arrow": [91, 92, 93, 94]
    },
    "data": {}
}
```

其中，`clip` 属性表示你的每张地图的那一部分会被当成数据集，例如填写 `[0, 0, 13, 13]` 就会让坐标为 `(0, 0)`，长宽为 `(13, 13)` 的矩形内容作为数据集。`special` 不用管。注意装饰所使用的贴图是白墙，如果白墙是墙壁的话，需要将白墙设置为墙壁。注意不要忘记保存

2. 使用 [在线工具](https://unanmed.github.io/ginka-process) 处理数据，需要给每个地图添加标签，为每个图块分配种类，有一些图块包含多种等级，需要填写正确。
3. 将 `project` 文件夹打包发给我
