import { readFile } from 'fs/promises';
import {
    GraphNodeType,
    IAutoLabelConfig,
    IFloorInfo,
    IMapGraph,
    IMapTileConverter,
    ITowerInfo,
    MapGraphNode,
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
            if (!symmetryH && !symmetryV && !symmetryC) break outer;
        }
    }
    return { symmetryH, symmetryV, symmetryC };
}

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

    for (let x = 0; x < W; x++) {
        check(map[0][x]);
        check(map[H - 1][x]);
    }
    for (let y = 1; y < H - 1; y++) {
        check(map[y][0]);
        check(map[y][W - 1]);
    }

    return borderCount > 0 && wallOrEntry / borderCount > 0.9;
}

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
    const allEmptyResource = new Set<MapGraphNode>();
    for (const node of graph.nodeMap.values()) {
        if (
            node.type === GraphNodeType.Empty ||
            node.type === GraphNodeType.Resource
        ) {
            allEmptyResource.add(node);
        }
    }

    let roomCount = 0;
    const visited = new Set<MapGraphNode>();

    for (const startNode of allEmptyResource) {
        if (visited.has(startNode)) continue;

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

        // 条件 c：外接矩形宽高均 > 1
        if (maxX - minX < 1 || maxY - minY < 1) continue;

        roomCount++;
    }

    return roomCount;
}

/**
 * 统计邻居数 >= 3 的分支节点数量（高连接度分支节点）
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
    const width = map[0]?.length ?? 0;

    // ── 结构标签计算 ─────────────────────────────────
    const { symmetryH, symmetryV, symmetryC } = computeSymmetry(map);
    const outerWall = computeOuterWall(
        map,
        config.classes.wall,
        config.classes.entry
    );
    const roomCount = computeRoomCount(topo.graph, width);
    const highDegBranchCount = computeHighDegBranchCount(topo.graph);

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
        ),
        symmetryH,
        symmetryV,
        symmetryC,
        outerWall,
        roomCount,
        highDegBranchCount
    };

    return floorInfo;
}
