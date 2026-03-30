import { readFile } from 'fs/promises';
import {
    GraphNodeType,
    IAutoLabelConfig,
    IFloorInfo,
    IMapTileConverter,
    ITowerInfo,
    TowerColor
} from './types';
import {
    doorTiles,
    enemyTiles,
    entryTiles,
    gemTiles,
    itemTiles,
    keyTiles,
    nonEmptyTiles,
    potionTiles,
    resourceTiles,
    specialDoorTiles,
    wallTiles
} from '../shared';
import { gaussainHeatmap, generateHeatmap } from './heatmap';
import { MapTopology } from './topo';

interface IRawTowerInfo {
    /** 作者 id */
    readonly authorId: string;
    /** 塔颜色 */
    readonly color: string;
    /** 评论数量 */
    readonly comment: string;
    /** 是否是比赛塔 */
    readonly competition: string;
    /** 楼层数量 */
    readonly floors: string;
    /** 塔的数字 id */
    readonly id: string;
    /** 塔的英文名 */
    readonly name: string;
    /** 塔的游玩量 */
    readonly people: string;
    /** 塔标签，每一项是对应的标签名 */
    readonly tag: string;
    /** 塔的名称 */
    readonly title: string;
    /** 测试员列表 */
    readonly topuser: string;
    /** 通关人数 */
    readonly win: string;
    /** 精美评分，第一项是评分结果，后面五项是选择每个评分的人数 */
    readonly designrate: number[];
    /** 难度评分，第一项是评分结果，后面五项是选择每个评分的人数 */
    readonly hardrate: number[];
}

/**
 * 解析出塔信息
 * @param path 塔信息文件路径
 * @returns 塔英文名到塔信息的映射
 */
export async function parseTowerInfo(
    path: string
): Promise<Map<string, ITowerInfo>> {
    const file = await readFile(path, 'utf-8');
    const data = JSON.parse(file) as IRawTowerInfo[];
    const result: ITowerInfo[] = data.map(v => {
        return {
            authorId: parseInt(v.authorId),
            color: parseInt(v.color) as TowerColor,
            comment: parseInt(v.comment),
            competition: parseInt(v.competition) === 1,
            floors: parseInt(v.floors),
            id: parseInt(v.id),
            name: v.name,
            people: parseInt(v.people),
            tag: v.tag.split('|').slice(0, -1),
            title: v.title,
            topuser: JSON.parse(v.topuser),
            win: parseInt(v.win),
            designrate: v.designrate.slice(),
            hardrate: v.hardrate.slice()
        };
    });
    const map = new Map<string, ITowerInfo>();
    result.forEach(v => {
        map.set(v.name, v);
    });
    return map;
}

function count(map: number[], set: Set<number>) {
    let count = 0;
    map.forEach(v => {
        if (set.has(v)) count++;
    });
    return count;
}

/**
 * 计算地图中墙壁密度的标准差
 * @param map 地图矩阵 number[][]
 * @param wallSet 墙壁的 tile ID 集合
 * @param kernel 卷积核大小（如 5 则是 5x5）
 */
export function computeWallDensityStd(
    map: number[][],
    wallSet: Set<number>,
    kernel: number,
    stride: number = Math.floor(kernel / 2)
): number {
    const rows = map.length;
    const cols = map[0]?.length ?? 0;

    if (rows === 0 || cols === 0) return 0;

    const densities: number[] = [];

    // 遍历 kernel window 左上角坐标
    for (let sy = 0; sy < rows; sy += stride) {
        for (let sx = 0; sx < cols; sx += stride) {
            let countWall = 0;
            let countTotal = 0;

            // 扫描 kernel 范围内的实际格子
            for (let y = sy; y < sy + kernel && y < rows; y++) {
                for (let x = sx; x < sx + kernel && x < cols; x++) {
                    countTotal++;
                    if (wallSet.has(map[y][x])) {
                        countWall++;
                    }
                }
            }

            // 避免除零
            if (countTotal > 0) {
                const density = countWall / countTotal;
                densities.push(density);
            }
        }
    }

    if (densities.length === 0) return 0;

    // 求均值
    const mean = densities.reduce((a, b) => a + b, 0) / densities.length;

    // 求方差 (总体方差)
    const variance =
        densities.reduce((a, b) => a + (b - mean) ** 2, 0) / densities.length;

    // 标准差
    return Math.sqrt(variance);
}

/**
 * 根据地图矩阵解析出地图数据
 * @param tower 地图所属塔信息
 * @param map 地图矩阵
 */
export function parseFloorInfo(
    tower: ITowerInfo,
    originMap: number[][],
    map: number[][],
    otherLayers: number[][][],
    config: IAutoLabelConfig,
    converter: IMapTileConverter,
    floorId: string
): IFloorInfo {
    const topo = new MapTopology(
        floorId,
        originMap,
        map,
        otherLayers,
        converter,
        config.classes
    );
    const flattened = map.flat();
    const area = flattened.length;

    let hasUselessBranch = false;

    // 统计拓扑图信息
    let maxEmptyArea = 0;
    let maxResourceArea = 0;
    topo.graph.areas.forEach(area => {
        area.nodes.forEach(v => {
            if (v.type === GraphNodeType.Empty) {
                let branchConnection = 0;
                v.neighbors.forEach(v => {
                    // 对节点的每个邻居遍历，如果邻居是分支节点，且直接相连的分支节点数小于 2，
                    // 说明这个连接可能会导致无用节点
                    // 至于为什么要多一次额外的邻居节点判断：
                    // |---|---|---|---|---|
                    // | W | W | D | W | W |
                    // |---|---|---|---|---|
                    // | W |   | E |   | W |
                    // |---|---|---|---|---|
                    // | W | W | D | W | W |
                    // |---|---|---|---|---|
                    if (v.type === GraphNodeType.Branch) {
                        let directBranch = 0;
                        for (const n of v.neighbors) {
                            if (n.type === GraphNodeType.Branch) {
                                directBranch++;
                            }
                        }
                        if (directBranch < 2) {
                            branchConnection++;
                        }
                    }
                });
                // 如果连接的分支数与邻居数相同，且小于等于 0，说明是门或怪物后面连接了一整片空地，是无用分支
                // 如果连接的分支数与邻居数不相同，说明可能连接了资源节点、入口节点等，这些显然不应该算入无用分支
                if (
                    branchConnection <= 1 &&
                    v.neighbors.size === branchConnection
                ) {
                    hasUselessBranch = true;
                }
                if (v.tiles.size > maxEmptyArea) {
                    maxEmptyArea = v.tiles.size;
                }
            } else if (v.type === GraphNodeType.Resource) {
                if (v.tiles.size > maxResourceArea) {
                    maxResourceArea = v.tiles.size;
                }
            }
        });
    });

    const floorInfo: IFloorInfo = {
        tower,
        topo,
        map,
        maxEmptyArea,
        maxResourceArea,
        hasUselessBranch,
        globalDensity: count(flattened, nonEmptyTiles) / area,
        wallDensity: count(flattened, wallTiles) / area,
        doorDensity: count(flattened, doorTiles) / area,
        enemyDensity: count(flattened, enemyTiles) / area,
        resourceDensity: count(flattened, resourceTiles) / area,
        gemDensity: count(flattened, gemTiles) / area,
        potionDensity: count(flattened, potionTiles) / area,
        keyDensity: count(flattened, keyTiles) / area,
        itemDensity: count(flattened, itemTiles) / area,
        entryCount: count(flattened, entryTiles),
        specialDoorCount: count(flattened, specialDoorTiles),
        wallDensityStd: computeWallDensityStd(map, wallTiles, 5),
        wallHeatmap: gaussainHeatmap(
            generateHeatmap(map, wallTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        enemyHeatmap: gaussainHeatmap(
            generateHeatmap(map, enemyTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        resourceHeatmap: gaussainHeatmap(
            generateHeatmap(map, resourceTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        potionHeatmap: gaussainHeatmap(
            generateHeatmap(map, potionTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        gemHeatmap: gaussainHeatmap(
            generateHeatmap(map, gemTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        keyHeatmap: gaussainHeatmap(
            generateHeatmap(map, keyTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        itemHeatmap: gaussainHeatmap(
            generateHeatmap(map, itemTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        entryHeatmap: gaussainHeatmap(
            generateHeatmap(map, entryTiles, config.heatmapKernel),
            config.guassainRadius
        ),
        doorHeatmap: gaussainHeatmap(
            generateHeatmap(map, doorTiles, config.heatmapKernel),
            config.guassainRadius
        )
    };

    return floorInfo;
}
