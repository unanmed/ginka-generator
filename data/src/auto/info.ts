import { readFile } from 'fs/promises';
import {
    BranchType,
    DoorKind,
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

// 格子层四方向通行检查，供无用分支主算法复用。
const branchCheckDirs: [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
];

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
 * 从拓扑节点中取出它对应的代表格子坐标。
 *
 * 当前新增的几条局部结构规则都需要把拓扑节点重新映射回格子层，
 * 例如：
 * 1. 统计格子层四方向可通行数
 * 2. 调用拓扑上的入口连通性接口
 *
 * 对于 Branch / Entry 节点，它本来就是单格节点；
 * 对于 Empty / Resource 节点，这里只需要拿其中任意一个格子作为搜索起点即可。
 */
function getNodeTile(node: MapGraphNode): number {
    return node.tiles.values().next().value as number;
}

/**
 * 判断格子层上两个相邻格子之间是否至少存在一个可通行方向。
 *
 * 这里显式回到格子层，而不是直接看拓扑图邻接关系，原因是：
 * 同一个 Empty / Resource 拓扑节点可能从多个方向贴住分支节点，
 * 但在“死胡同分支”这条快捷规则里，我们关心的是分支格子本身到底有几个可走方向。
 *
 * 当前项目会在更早的楼层过滤阶段直接剔除带 `cannotIn/cannotOut` 的地图，
 * 所以这里不再额外兼容方向限制，只要目标格子不是墙，就视为该方向可走。
 *
 * @param topo 当前楼层的拓扑信息
 * @param to 终点格子，必须是与起点四邻接的格子
 * @returns 只要目标格子不是墙，就视为这两个格子之间存在通路
 */
function hasGridPassage(topo: MapTopology, to: number): boolean {
    const width = topo.convertedMap[0]?.length ?? 0;
    const toX = to % width;
    const toY = (to - toX) / width;

    if (topo.convertedMap[toY]?.[toX] == null) {
        return false;
    }

    return topo.convertedMap[toY][toX] !== topo.config.wall;
}

/**
 * 统计某个分支格子在格子层的四方向可通行数。
 *
 * 这是无用分支主算法的第一阶段快捷判定：
 * 如果一个分支格子只有一个可通行方向，那么它在局部结构上就是典型的走廊尽头/死胡同，
 * 可以直接命中“无用分支”，不必再做后侧候选区域分析。
 *
 * @param topo 当前楼层的拓扑信息
 * @param tile 目标分支格子的平坦坐标
 * @returns 该格子四个方向中实际可走的方向数量
 */
function countGridPassableDirections(topo: MapTopology, tile: number): number {
    const width = topo.convertedMap[0]?.length ?? 0;
    const height = topo.convertedMap.length;
    const x = tile % width;
    const y = (tile - x) / width;
    let count = 0;

    for (const [dx, dy] of branchCheckDirs) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
        }

        const nextTile = ny * width + nx;
        if (hasGridPassage(topo, nextTile)) {
            count++;
        }
    }

    return count;
}

/**
 * 在“移除目标分支节点”的前提下，收集某个后侧候选区域能到达的所有拓扑节点。
 *
 * 这里的搜索允许经过其他分支节点，因为文档已经明确：
 * 我们只禁止再次穿过当前正在评估的目标分支，
 * 不禁止后侧区域继续经过其他门/怪去连接资源。
 *
 * @param startNode 后侧候选区域中的任意起点节点
 * @param ignoredNode 当前正在评估的目标分支节点，搜索时视为删除
 * @returns 移除目标分支后，从起点仍可到达的所有节点
 */
function collectReachableNodes(
    startNode: MapGraphNode,
    ignoredNode: MapGraphNode
): Set<MapGraphNode> {
    // 从后侧候选区域出发做一次搜索，并把目标分支当作已删除处理。
    const visited = new Set<MapGraphNode>([startNode]);
    const queue: MapGraphNode[] = [startNode];

    while (queue.length > 0) {
        const current = queue.shift()!;
        for (const neighbor of current.neighbors) {
            if (neighbor === ignoredNode || visited.has(neighbor)) {
                continue;
            }
            visited.add(neighbor);
            queue.push(neighbor);
        }
    }

    return visited;
}

/**
 * 判断单个分支节点是否命中“无用分支”主算法。
 *
 * 判定流程对应设计文档中的两阶段：
 * 1. 先看格子层四方向可通行数，若 <= 1 则直接按死胡同命中。
 * 2. 否则把该分支从图中临时移除，检查它周围哪些相邻区域会失去入口连通性；
 *    这些区域就是“后侧候选区域”。
 * 3. 对每个后侧候选区域做搜索，只要任意一个候选区域还能到达资源节点，
 *    就说明该分支仍然承担了资源守护作用，不应视为无用分支。
 * 4. 只有当存在后侧候选区域，且所有后侧候选区域都无法通向资源时，才返回 true。
 *
 * 这个实现刻意不处理“整块怪环只有一个出口”这类模式四，
 * 因为那属于另一类环状子图问题，不适合继续套单节点规则。
 *
 * @param topo 当前楼层的拓扑信息
 * @param branchNode 目标分支节点，必须是门或怪这类 Branch 节点
 * @returns 如果该分支满足无用分支定义，则返回 true
 */
function isUselessBranchNode(
    topo: MapTopology,
    branchNode: MapGraphNode,
    specialDoorLinkedEnemies?: Set<MapGraphNode>
): boolean {
    if (branchNode.type !== GraphNodeType.Branch) {
        return false;
    }

    if (
        branchNode.branch === BranchType.Enemy &&
        specialDoorLinkedEnemies &&
        specialDoorLinkedEnemies.has(branchNode)
    ) {
        return false;
    }

    const branchTile = getNodeTile(branchNode);
    if (countGridPassableDirections(topo, branchTile) <= 1) {
        // 格子层只有一个可通行方向时，直接按死胡同分支处理。
        return true;
    }

    // 记录已经并入后侧候选区域的节点，避免同一块区域从多个邻居重复搜索。
    const handledNodes = new Set<MapGraphNode>();
    let hasBacksideCandidate = false;

    for (const neighbor of branchNode.neighbors) {
        if (handledNodes.has(neighbor)) {
            // 多个邻居可能指向同一个后侧连通区域，已处理过就不重复搜索。
            continue;
        }

        const neighborTile = getNodeTile(neighbor);
        if (topo.connectedToAnyEntry(neighborTile, [branchNode])) {
            // 删除目标分支后，这个方向仍能回到任意入口，因此它属于前侧或旁路区域。
            continue;
        }

        // 失去入口连通性的相邻区域视为该分支的后侧候选区域。
        hasBacksideCandidate = true;

        const reachableNodes = collectReachableNodes(neighbor, branchNode);
        for (const node of reachableNodes) {
            handledNodes.add(node);
        }

        // 这里按“整块后侧候选区域”聚合判断，而不是只看相邻的单个节点。
        for (const node of reachableNodes) {
            if (node.type === GraphNodeType.Resource) {
                // 任意一个后侧候选区域还能通向资源，就不算无用分支。
                return false;
            }
        }
    }

    // 若根本不存在失去入口连通性的相邻区域，则这个分支按当前定义不属于无用分支。
    return hasBacksideCandidate;
}

/**
 * 统计同类门团 / 怪团的最大 BFS 连通块大小。
 *
 * 这里的“连续门 / 连续怪”严格按拓扑图上的分支节点邻接来定义：
 * 1. 起点必须是一个 Branch 节点
 * 2. 只沿“仍然是 Branch，且 branch 类型与起点相同”的邻接边继续 BFS
 * 3. 门和怪分别统计，绝不把混合结构合并为一个连通块
 *
 * 输出里既保留最大门团/怪团大小，也直接给出“是否超过 3”的布尔结果，
 * 这样过滤层和后续统计层都可以直接复用，不需要再次写阈值判断。
 *
 * @param branchNodes 当前楼层里所有分支节点的去重集合
 * @returns 门团/怪团的最大连通块大小，以及是否命中过大连通块
 */
function computeBranchClusterStats(branchNodes: Iterable<MapGraphNode>): {
    maxDoorClusterSize: number;
    maxEnemyClusterSize: number;
    hasLargeDoorCluster: boolean;
    hasLargeEnemyCluster: boolean;
} {
    const visited = new Set<MapGraphNode>();
    let maxDoorClusterSize = 0;
    let maxEnemyClusterSize = 0;

    for (const startNode of branchNodes) {
        if (visited.has(startNode) || startNode.type !== GraphNodeType.Branch) {
            continue;
        }

        // 每次从一个尚未访问的分支节点出发，求出它所属的同类连通块大小。
        visited.add(startNode);
        let clusterSize = 0;
        const queue: MapGraphNode[] = [startNode];

        // 只沿同类分支邻接边做 BFS，门和怪分开统计。
        while (queue.length > 0) {
            const current = queue.shift()!;
            clusterSize++;

            for (const neighbor of current.neighbors) {
                if (
                    visited.has(neighbor) ||
                    neighbor.type !== GraphNodeType.Branch ||
                    neighbor.branch !== startNode.branch
                ) {
                    continue;
                }

                visited.add(neighbor);
                queue.push(neighbor);
            }
        }

        // 门团和怪团分别维护各自的最大连通块大小。
        if (startNode.branch === BranchType.Door) {
            maxDoorClusterSize = Math.max(maxDoorClusterSize, clusterSize);
        } else {
            maxEnemyClusterSize = Math.max(maxEnemyClusterSize, clusterSize);
        }
    }

    return {
        maxDoorClusterSize,
        maxEnemyClusterSize,
        // 当前版本阈值固定为 > 3，大小恰好为 3 的团块默认保留。
        hasLargeDoorCluster: maxDoorClusterSize > 3,
        hasLargeEnemyCluster: maxEnemyClusterSize > 3
    };
}

/**
 * 统计拓扑图上“只连接到一个邻居节点”的闲置分支。
 *
 * 这条规则对应闲置节点章节中的硬规则：
 * 如果一个分支节点在拓扑图上的 `neighbors.size === 1`，说明玩家只能从同一个拓扑节点到达它，
 * 而穿过该分支后也不会暴露新的拓扑节点，因此它属于“连通但无影响”的闲置节点。
 *
 * 这里刻意使用拓扑图邻居数，而不是格子层可通行方向数，因为这条规则关注的是
 * “是否会暴露新的拓扑节点”，语义上不同于无用分支里的死胡同快捷规则。
 *
 * @param branchNodes 当前楼层里所有分支节点的去重集合
 * @returns 闲置门/怪数量，以及该楼层是否存在闲置分支
 */
function computeIdleBranchStats(
    branchNodes: Iterable<MapGraphNode>,
    specialDoorLinkedEnemies?: Set<MapGraphNode>
): {
    idleDoorBranchCount: number;
    idleEnemyBranchCount: number;
    ignoredIdleEnemyBySpecialDoorCount: number;
    hasIdleBranch: boolean;
} {
    let idleDoorBranchCount = 0;
    let idleEnemyBranchCount = 0;
    let ignoredIdleEnemyBySpecialDoorCount = 0;

    for (const node of branchNodes) {
        if (node.type !== GraphNodeType.Branch || node.neighbors.size !== 1) {
            continue;
        }

        if (node.branch === BranchType.Door) {
            idleDoorBranchCount++;
        } else if (
            specialDoorLinkedEnemies &&
            specialDoorLinkedEnemies.has(node)
        ) {
            ignoredIdleEnemyBySpecialDoorCount++;
        } else {
            idleEnemyBranchCount++;
        }
    }

    return {
        idleDoorBranchCount,
        idleEnemyBranchCount,
        ignoredIdleEnemyBySpecialDoorCount,
        hasIdleBranch: idleDoorBranchCount + idleEnemyBranchCount > 0
    };
}

interface IMergedNonBranchArea {
    readonly index: number;
    readonly nodes: Set<MapGraphNode>;
    readonly tileCount: number;
    readonly hasResource: boolean;
}

interface IRepeatedGuardCandidate {
    readonly node: MapGraphNode;
    readonly branch: BranchType;
    readonly areaA: IMergedNonBranchArea;
    readonly areaB: IMergedNonBranchArea;
}

function getNodeRepresentativeTile(node: MapGraphNode): number {
    return getNodeTile(node);
}

function buildMergedNonBranchAreas(graph: IMapGraph): {
    areas: IMergedNonBranchArea[];
    areaMap: Map<MapGraphNode, IMergedNonBranchArea>;
} {
    const visited = new Set<MapGraphNode>();
    const areas: IMergedNonBranchArea[] = [];
    const areaMap = new Map<MapGraphNode, IMergedNonBranchArea>();
    let areaIndex = 0;

    for (const startNode of graph.nodeMap.values()) {
        if (
            visited.has(startNode) ||
            (startNode.type !== GraphNodeType.Empty &&
                startNode.type !== GraphNodeType.Resource)
        ) {
            continue;
        }

        const nodes = new Set<MapGraphNode>();
        const queue: MapGraphNode[] = [startNode];
        visited.add(startNode);
        let tileCount = 0;
        let hasResource = false;

        while (queue.length > 0) {
            const current = queue.shift()!;
            nodes.add(current);
            tileCount += current.tiles.size;
            if (current.type === GraphNodeType.Resource) {
                hasResource = true;
            }

            for (const neighbor of current.neighbors) {
                if (
                    visited.has(neighbor) ||
                    (neighbor.type !== GraphNodeType.Empty &&
                        neighbor.type !== GraphNodeType.Resource)
                ) {
                    continue;
                }

                visited.add(neighbor);
                queue.push(neighbor);
            }
        }

        const area: IMergedNonBranchArea = {
            index: areaIndex++,
            nodes,
            tileCount,
            hasResource
        };
        areas.push(area);
        for (const node of nodes) {
            areaMap.set(node, area);
        }
    }

    return { areas, areaMap };
}

function findSpecialDoorLinkedEnemyNodes(
    branchNodes: Iterable<MapGraphNode>,
    areaMap: Map<MapGraphNode, IMergedNonBranchArea>
): Set<MapGraphNode> {
    const specialDoorNodes = new Set<MapGraphNode>();
    for (const node of branchNodes) {
        if (
            node.type === GraphNodeType.Branch &&
            node.branch === BranchType.Door &&
            node.doorKind === DoorKind.Special
        ) {
            specialDoorNodes.add(node);
        }
    }

    if (specialDoorNodes.size === 0) {
        return new Set();
    }

    const specialDoorLinkedAreas = new Set<IMergedNonBranchArea>();
    for (const specialDoor of specialDoorNodes) {
        for (const neighbor of specialDoor.neighbors) {
            const area = areaMap.get(neighbor);
            if (area) {
                specialDoorLinkedAreas.add(area);
            }
        }
    }

    if (specialDoorLinkedAreas.size === 0) {
        return new Set();
    }

    const linkedEnemyNodes = new Set<MapGraphNode>();
    for (const node of branchNodes) {
        if (
            node.type !== GraphNodeType.Branch ||
            node.branch !== BranchType.Enemy
        ) {
            continue;
        }
        for (const neighbor of node.neighbors) {
            const area = areaMap.get(neighbor);
            if (area && specialDoorLinkedAreas.has(area)) {
                linkedEnemyNodes.add(node);
                break;
            }
        }
    }

    return linkedEnemyNodes;
}

function buildRepeatedGuardCandidates(
    branchNodes: Iterable<MapGraphNode>,
    areaMap: Map<MapGraphNode, IMergedNonBranchArea>
): IRepeatedGuardCandidate[] {
    const candidates: IRepeatedGuardCandidate[] = [];

    for (const node of branchNodes) {
        if (node.type !== GraphNodeType.Branch) {
            continue;
        }

        const distinctAreas = new Map<number, IMergedNonBranchArea>();
        for (const neighbor of node.neighbors) {
            const area = areaMap.get(neighbor);
            if (area) {
                distinctAreas.set(area.index, area);
            }
        }

        if (distinctAreas.size !== 2) {
            continue;
        }

        const [areaA, areaB] = [...distinctAreas.values()].sort(
            (a, b) => a.index - b.index
        );

        // 保守例外：若贴着的是单格资源点，则暂不按重复守卫结构过滤。
        if (
            (areaA.tileCount === 1 && areaA.hasResource) ||
            (areaB.tileCount === 1 && areaB.hasResource)
        ) {
            continue;
        }

        candidates.push({
            node,
            branch: node.branch,
            areaA,
            areaB
        });
    }

    return candidates;
}

function areBranchTilesEightConnected(
    a: number,
    b: number,
    width: number
): boolean {
    const ax = a % width;
    const ay = (a - ax) / width;
    const bx = b % width;
    const by = (b - bx) / width;

    return Math.abs(ax - bx) <= 1 && Math.abs(ay - by) <= 1;
}

/**
 * 统计“多个同类分支重复守同一连通区域”的闲置节点模式。
 *
 * 实现口径对应文档里的保守版本：
 * 1. 先把 Empty / Resource 节点临时合并为更大的非分支连通区域。
 * 2. 若某个分支正好连接到两个不同的非分支区域，则把这两个区域和分支类型组成结构签名。
 * 3. 具有相同结构签名的同类分支，再按格子层 8 邻接做聚类。
 * 4. 当某个聚类大小 >= 2 时，视为“重复守同一连通区域”的闲置节点模式命中。
 *
 * @param graph 当前楼层的拓扑图
 * @param branchNodes 当前楼层里所有分支节点的去重集合
 * @param width 地图宽度，用于做 8 邻接聚类
 * @returns 重复守卫模式命中的门/怪数量和布尔标签
 */
function computeRepeatedGuardIdleStats(
    graph: IMapGraph,
    branchNodes: Iterable<MapGraphNode>,
    width: number
): {
    repeatedGuardDoorBranchCount: number;
    repeatedGuardEnemyBranchCount: number;
    hasRepeatedGuardIdleBranch: boolean;
} {
    const { areaMap } = buildMergedNonBranchAreas(graph);
    const candidates = buildRepeatedGuardCandidates(branchNodes, areaMap);
    const groupedCandidates = new Map<string, IRepeatedGuardCandidate[]>();

    for (const candidate of candidates) {
        const key = `${candidate.branch}:${candidate.areaA.index}:${candidate.areaB.index}`;
        const group = groupedCandidates.get(key) ?? [];
        group.push(candidate);
        groupedCandidates.set(key, group);
    }

    const repeatedGuardNodes = new Set<MapGraphNode>();

    for (const group of groupedCandidates.values()) {
        const visited = new Set<MapGraphNode>();

        for (const candidate of group) {
            if (visited.has(candidate.node)) {
                continue;
            }

            const cluster: IRepeatedGuardCandidate[] = [];
            const queue: IRepeatedGuardCandidate[] = [candidate];
            visited.add(candidate.node);

            while (queue.length > 0) {
                const current = queue.shift()!;
                cluster.push(current);
                const currentTile = getNodeRepresentativeTile(current.node);

                for (const neighbor of group) {
                    if (visited.has(neighbor.node)) {
                        continue;
                    }

                    const neighborTile = getNodeRepresentativeTile(
                        neighbor.node
                    );
                    if (
                        !areBranchTilesEightConnected(
                            currentTile,
                            neighborTile,
                            width
                        )
                    ) {
                        continue;
                    }

                    visited.add(neighbor.node);
                    queue.push(neighbor);
                }
            }

            if (cluster.length >= 2) {
                for (const member of cluster) {
                    repeatedGuardNodes.add(member.node);
                }
            }
        }
    }

    let repeatedGuardDoorBranchCount = 0;
    let repeatedGuardEnemyBranchCount = 0;
    for (const node of repeatedGuardNodes) {
        if (node.type !== GraphNodeType.Branch) {
            continue;
        }

        if (node.branch === BranchType.Door) {
            repeatedGuardDoorBranchCount++;
        } else {
            repeatedGuardEnemyBranchCount++;
        }
    }

    return {
        repeatedGuardDoorBranchCount,
        repeatedGuardEnemyBranchCount,
        hasRepeatedGuardIdleBranch:
            repeatedGuardDoorBranchCount + repeatedGuardEnemyBranchCount > 0
    };
}

/**
 * 解析单层地图的统计信息、拓扑结构标签以及局部异常标签。
 *
 * 这是楼层清洗阶段最核心的入口之一。它会在一次扫描里同时产出三类信息：
 * 1. 全局统计量，例如门/怪/资源密度、入口数量、最大空地区域等。
 * 2. 结构标签，例如对称性、房间数、高连接度分支数。
 * 3. 局部异常标签，例如无用分支、连续门团、连续怪团。
 *
 * 其中“连续门/怪”和“无用分支”是两套互相独立的规则：
 * 前者只看同类分支在拓扑图上的连通块大小，
 * 后者看删除某个分支后，后侧区域是否失去入口连通且没有资源收益。
 *
 * @param tower 当前楼层所属的塔信息
 * @param originMap 原始楼层地图，用于识别真实图块语义
 * @param map 转换后的标签地图，用于做密度统计与热力图计算
 * @param otherLayers 背景层/前景层等附加图层，用于补充不可入不可出信息
 * @param config 自动清洗配置
 * @param converter 原始图块到标签语义的转换器
 * @param floorId 当前楼层 id，用于入口识别等逻辑
 * @returns 当前楼层可供过滤与训练使用的完整解析结果
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

    // ---- 结构标签计算 ----
    const { symmetryH, symmetryV, symmetryC } = computeSymmetry(map);
    const outerWall = computeOuterWall(
        map,
        config.classes.wall,
        config.classes.entry
    );
    const roomCount = computeRoomCount(topo.graph, width);
    const highDegBranchCount = computeHighDegBranchCount(topo.graph);

    // 分支节点在拓扑图中是单格节点，先去重后再做局部结构分析。
    const branchNodes = new Set<MapGraphNode>();
    topo.graph.nodeMap.forEach(node => {
        if (node.type === GraphNodeType.Branch) {
            branchNodes.add(node);
        }
    });

    const {
        maxDoorClusterSize,
        maxEnemyClusterSize,
        hasLargeDoorCluster,
        hasLargeEnemyCluster
    } = computeBranchClusterStats(branchNodes);

    const mergedAreas = buildMergedNonBranchAreas(topo.graph);
    const specialDoorLinkedEnemies = findSpecialDoorLinkedEnemyNodes(
        branchNodes,
        mergedAreas.areaMap
    );
    const specialDoorLinkedEnemyCount = specialDoorLinkedEnemies.size;

    const {
        idleDoorBranchCount,
        idleEnemyBranchCount,
        ignoredIdleEnemyBySpecialDoorCount,
        hasIdleBranch
    } = computeIdleBranchStats(branchNodes, specialDoorLinkedEnemies);
    const {
        repeatedGuardDoorBranchCount,
        repeatedGuardEnemyBranchCount,
        hasRepeatedGuardIdleBranch
    } = computeRepeatedGuardIdleStats(topo.graph, branchNodes, width);

    let hasUselessBranch = false;
    let ignoredUselessBranchBySpecialDoorCount = 0;
    for (const node of branchNodes) {
        const useless = isUselessBranchNode(
            topo,
            node,
            specialDoorLinkedEnemies
        );
        if (useless) {
            if (
                node.branch === BranchType.Enemy &&
                specialDoorLinkedEnemies.has(node)
            ) {
                ignoredUselessBranchBySpecialDoorCount++;
            } else {
                hasUselessBranch = true;
            }
        }
    }

    // 统计拓扑图信息
    let maxEmptyArea = 0;
    let maxResourceArea = 0;
    topo.graph.areas.forEach(area => {
        area.nodes.forEach(v => {
            if (v.type === GraphNodeType.Empty) {
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

    // 把全局统计、结构标签、局部异常标签统一整理成楼层信息对象。
    const floorInfo: IFloorInfo = {
        tower,
        topo,
        map,
        maxEmptyArea,
        maxResourceArea,
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
        specialDoorLinkedEnemyCount,
        ignoredIdleEnemyBySpecialDoorCount,
        ignoredUselessBranchBySpecialDoorCount,
        maxDoorClusterSize,
        maxEnemyClusterSize,
        hasLargeDoorCluster,
        hasLargeEnemyCluster,
        idleDoorBranchCount,
        idleEnemyBranchCount,
        hasIdleBranch,
        repeatedGuardDoorBranchCount,
        repeatedGuardEnemyBranchCount,
        hasRepeatedGuardIdleBranch,
        hasUselessBranch,
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
