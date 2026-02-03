import { readFile } from 'fs/promises';
import { IFloorInfo, ITowerInfo, TowerColor } from './types';
import { buildTopologicalGraph } from '../topology/graph';
import {
    commonDoorTiles,
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
import { NodeType } from '../topology/interface';

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
export function parseFloorInfo(tower: ITowerInfo, map: number[][]): IFloorInfo {
    const topo = buildTopologicalGraph(map);
    const flattened = map.flat();
    const area = flattened.length;

    let hasUselessBranch = false;

    // 统计咸鱼门数量
    let fishCount = 0;
    topo.graphs.forEach(graph => {
        // 其实就是判断纯血瓶钥匙的资源节点的邻居是不是全都是门，是的话就判定为咸鱼门
        // 这么做虽然会有一定的误差，但是也大差不差了
        // 两个门对一个也判定为一个咸鱼门
        graph.areaMap.forEach(v => {
            const res = [...v.resources.entries()];
            const onlyPotion = res.every(([tile, value]) => {
                if (!potionTiles.has(tile) && !keyTiles.has(tile)) {
                    return value <= 0;
                }
                return true;
            });
            if (!onlyPotion) {
                // 包含血瓶钥匙之外的不考虑
                return;
            }

            let branchCount = 0;
            let noneBranchCount = 0;

            v.neighbor.forEach(value => {
                const node = graph.graph.get(value);
                if (!node) {
                    noneBranchCount++;
                    return;
                }

                if (node.type === NodeType.Branch) {
                    if (!commonDoorTiles.has(node.tile)) {
                        branchCount++;
                    }
                } else {
                    noneBranchCount++;
                }
            });
            if (noneBranchCount >= 0 && branchCount === 0) {
                fishCount++;
            }
        });

        graph.graph.forEach(v => {
            if (v.type === NodeType.Branch) {
                if (v.neighbor.size === 1) {
                    hasUselessBranch = true;
                }
            }
        });
    });

    const floorInfo: IFloorInfo = {
        tower,
        topo,
        map,
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
        fishCount,
        hasUselessBranch,
        wallDensityStd: computeWallDensityStd(map, wallTiles, 5)
    };

    return floorInfo;
}
