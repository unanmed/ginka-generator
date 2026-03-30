import {
    CannotInOut,
    GraphNodeType,
    BranchType,
    ResourceType,
    type IMapTopology,
    type IMapGraph,
    type IMapGraphArea,
    type MapGraphNode,
    type IEntryMapGraphNode,
    type IMapTileConverter,
    IMapBlockConfig
} from './types';

/** [dx, dy, 离开方向标记, 进入方向标记] */
const dirs: [number, number, CannotInOut, CannotInOut][] = [
    [-1, 0, CannotInOut.Left, CannotInOut.Right],
    [1, 0, CannotInOut.Right, CannotInOut.Left],
    [0, -1, CannotInOut.Top, CannotInOut.Bottom],
    [0, 1, CannotInOut.Bottom, CannotInOut.Top]
];

const ALL_BLOCKED =
    CannotInOut.Left | CannotInOut.Top | CannotInOut.Right | CannotInOut.Bottom;

export class MapTopology implements IMapTopology {
    readonly originMap: number[][];
    readonly otherLayersMap: number[][][];
    readonly convertedMap: number[][];
    readonly noPass: boolean[][];
    readonly cannotIn: number[][];
    readonly cannotOut: number[][];
    readonly graph: IMapGraph;

    constructor(
        readonly floorId: string,
        map: number[][],
        convertedMap: number[][],
        otherLayers: number[][][],
        converter: IMapTileConverter,
        readonly config: IMapBlockConfig
    ) {
        this.originMap = map;
        this.otherLayersMap = otherLayers;
        this.convertedMap = convertedMap;

        const height = map.length;
        const width = height > 0 ? map[0].length : 0;

        this.noPass = map.map((row, y) =>
            row.map((tile, x) => converter.getNoPass(tile, x, y))
        );

        this.cannotIn = map.map((row, y) =>
            row.map((tile, x) => {
                if (this.noPass[y][x]) return ALL_BLOCKED;
                let flags = converter.getCannotIn(tile, x, y);
                for (const layer of otherLayers) {
                    flags |= converter.getCannotIn(layer[y]?.[x] ?? 0, x, y);
                }
                return flags;
            })
        );

        this.cannotOut = map.map((row, y) =>
            row.map((tile, x) => {
                if (this.noPass[y][x]) return ALL_BLOCKED;
                let flags = converter.getCannotOut(tile, x, y);
                for (const layer of otherLayers) {
                    flags |= converter.getCannotOut(layer[y]?.[x] ?? 0, x, y);
                }
                return flags;
            })
        );

        this.graph = this.buildGraph(width, height, converter);
    }

    private buildGraph(
        width: number,
        height: number,
        converter: IMapTileConverter
    ): IMapGraph {
        const size = width * height;

        // 1. 使用 converter 对每个图块进行分类
        const tileType = new Array<GraphNodeType | null>(size).fill(null);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const tile = this.convertedMap[y][x];
                const origin = this.originMap[y][x];
                const idx = y * width + x;
                if (tile === this.config.wall) {
                    tileType[idx] = GraphNodeType.Wall;
                } else if (
                    tile === this.config.entry ||
                    converter.isEntry(origin, x, y, this.floorId)
                ) {
                    tileType[idx] = GraphNodeType.Entry;
                } else if (
                    this.config.enemies.includes(tile) ||
                    converter.isEnemy(origin)
                ) {
                    tileType[idx] = GraphNodeType.Branch;
                } else if (
                    this.config.commonDoors.includes(tile) ||
                    this.config.specialDoors.includes(tile) ||
                    converter.isDoor(origin)
                ) {
                    tileType[idx] = GraphNodeType.Branch;
                } else if (
                    this.config.potions.includes(tile) ||
                    this.config.redGems.includes(tile) ||
                    this.config.blueGems.includes(tile) ||
                    this.config.greenGems.includes(tile) ||
                    this.config.items.includes(tile) ||
                    this.config.keys.includes(tile) ||
                    converter.isResource(origin)
                ) {
                    tileType[idx] = GraphNodeType.Resource;
                } else if (
                    tile === this.config.empty ||
                    converter.isEmpty(origin)
                ) {
                    tileType[idx] = GraphNodeType.Empty;
                } else {
                    tileType[idx] = GraphNodeType.Wall;
                }
            }
        }

        // 2. 通过 BFS 将图块分组为节点
        //    空白和资源节点：相邻同类型的图块合并为一个节点
        //    分支和入口节点：每个图块独立为一个节点
        const nodeMap = new Map<number, MapGraphNode>();
        const visited = new Set<number>();
        let nodeIndex = 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                if (visited.has(idx)) continue;
                visited.add(idx);

                const type = tileType[idx]!;
                if (type === GraphNodeType.Wall) continue;
                const tiles = new Set<number>([idx]);
                const neighbors = new Set<MapGraphNode>();

                // 分支和入口节点不合并，每个图块独立为一个节点
                if (type === GraphNodeType.Entry) {
                    nodeMap.set(idx, {
                        type: GraphNodeType.Entry,
                        index: nodeIndex++,
                        tiles,
                        neighbors
                    });
                    continue;
                }

                if (type === GraphNodeType.Branch) {
                    const tile = this.originMap[y][x];
                    nodeMap.set(idx, {
                        type: GraphNodeType.Branch,
                        index: nodeIndex++,
                        tiles,
                        neighbors,
                        branch: converter.isDoor(tile)
                            ? BranchType.Door
                            : BranchType.Enemy
                    });
                    continue;
                }

                // 空白和资源节点：BFS 合并相邻同类型图块
                const queue: number[] = [idx];
                while (queue.length > 0) {
                    const ci = queue.shift()!;
                    const cx = ci % width;
                    const cy = (ci - cx) / width;

                    for (const [dx, dy, outFlag, inFlag] of dirs) {
                        const nx = cx + dx;
                        const ny = cy + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                            continue;
                        const ni = ny * width + nx;
                        if (visited.has(ni) || tileType[ni] !== type) continue;

                        // 任一方向可通行即可合并
                        const canGo =
                            !(this.cannotOut[cy][cx] & outFlag) &&
                            !(this.cannotIn[ny][nx] & inFlag);
                        const canCome =
                            !(this.cannotOut[ny][nx] & inFlag) &&
                            !(this.cannotIn[cy][cx] & outFlag);
                        if (false) continue;
                        // if (!canGo && !canCome) continue;

                        visited.add(ni);
                        tiles.add(ni);
                        queue.push(ni);
                    }
                }

                let node: MapGraphNode;
                if (type === GraphNodeType.Empty) {
                    node = {
                        type: GraphNodeType.Empty,
                        index: nodeIndex++,
                        tiles,
                        neighbors
                    };
                } else {
                    const resources = new Map<ResourceType, number>();
                    for (const t of tiles) {
                        const tx = t % width;
                        const ty = (t - tx) / width;
                        const res = converter.getResource(
                            this.originMap[ty][tx],
                            tx,
                            ty
                        );
                        for (const [k, v] of res) {
                            resources.set(k, (resources.get(k) ?? 0) + v);
                        }
                    }
                    node = {
                        type: GraphNodeType.Resource,
                        index: nodeIndex++,
                        tiles,
                        neighbors,
                        resources
                    };
                }

                for (const t of tiles) {
                    nodeMap.set(t, node);
                }
            }
        }

        // 3. 构建节点间的邻接关系
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const node = nodeMap.get(y * width + x);
                if (!node) continue;

                for (const [dx, dy, outFlag, inFlag] of dirs) {
                    const nx = x + dx;
                    const ny = y + dy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;

                    const neighbor = nodeMap.get(ny * width + nx);
                    if (!neighbor || neighbor === node) continue;

                    // 至少一个方向可通行则建立邻接关系
                    const canGo =
                        !(this.cannotOut[y][x] & outFlag) &&
                        !(this.cannotIn[ny][nx] & inFlag);
                    const canCome =
                        !(this.cannotOut[ny][nx] & inFlag) &&
                        !(this.cannotIn[y][x] & outFlag);
                    // if (canGo || canCome) {
                    if (true) {
                        node.neighbors.add(neighbor);
                        neighbor.neighbors.add(node);
                    }
                }
            }
        }

        // 4. 通过 BFS 连通分量构建区域
        const areas = new Set<IMapGraphArea>();
        const unreachableArea = new Set<IMapGraphArea>();
        const entries = new Set<IEntryMapGraphNode>();
        const visitedNodes = new Set<MapGraphNode>();

        for (const node of new Set(nodeMap.values())) {
            if (visitedNodes.has(node)) continue;
            visitedNodes.add(node);

            const areaNodes = new Set<MapGraphNode>();
            const areaEntries: IEntryMapGraphNode[] = [];
            const queue: MapGraphNode[] = [node];

            while (queue.length > 0) {
                const current = queue.shift()!;
                if (areaNodes.has(current)) continue;
                areaNodes.add(current);
                visitedNodes.add(current);

                if (current.type === GraphNodeType.Entry) {
                    areaEntries.push(current);
                }

                for (const nb of current.neighbors) {
                    if (!visitedNodes.has(nb)) {
                        visitedNodes.add(nb);
                        queue.push(nb);
                    }
                }
            }

            const area: IMapGraphArea = { nodes: areaNodes };
            areas.add(area);

            if (areaEntries.length > 0) {
                for (const e of areaEntries) {
                    entries.add(e);
                }
            } else {
                unreachableArea.add(area);
            }
        }

        // console.log(areas.size);

        return { unreachableArea, areas, entries, nodeMap };
    }

    private resolveIgnored(
        ignoredNode?: (MapGraphNode | number)[]
    ): Set<MapGraphNode> {
        const ignored = new Set<MapGraphNode>();
        if (!ignoredNode) return ignored;
        for (const item of ignoredNode) {
            if (typeof item === 'number') {
                const node = this.graph.nodeMap.get(item);
                if (node) ignored.add(node);
            } else {
                ignored.add(item);
            }
        }
        return ignored;
    }

    connectedToAnyEntry(
        pos: number,
        ignoredNode?: (MapGraphNode | number)[]
    ): boolean {
        const startNode = this.graph.nodeMap.get(pos);
        if (!startNode) return false;
        if (startNode.type === GraphNodeType.Entry) return true;

        const ignored = this.resolveIgnored(ignoredNode);
        if (ignored.has(startNode)) return false;

        const visited = new Set<MapGraphNode>([startNode]);
        const queue: MapGraphNode[] = [startNode];

        while (queue.length > 0) {
            const current = queue.shift()!;
            for (const nb of current.neighbors) {
                if (visited.has(nb) || ignored.has(nb)) continue;
                if (nb.type === GraphNodeType.Entry) return true;
                visited.add(nb);
                queue.push(nb);
            }
        }

        return false;
    }

    connectedToSpecificEntry(
        pos: number,
        entry: number | IEntryMapGraphNode,
        ignoredNode?: (MapGraphNode | number)[]
    ): boolean {
        const startNode = this.graph.nodeMap.get(pos);
        if (!startNode) return false;

        const targetNode =
            typeof entry === 'number' ? this.graph.nodeMap.get(entry) : entry;
        if (!targetNode) return false;
        if (startNode === targetNode) return true;

        const ignored = this.resolveIgnored(ignoredNode);
        if (ignored.has(startNode)) return false;

        const visited = new Set<MapGraphNode>([startNode]);
        const queue: MapGraphNode[] = [startNode];

        while (queue.length > 0) {
            const current = queue.shift()!;
            for (const nb of current.neighbors) {
                if (visited.has(nb) || ignored.has(nb)) continue;
                if (nb === targetNode) return true;
                visited.add(nb);
                queue.push(nb);
            }
        }

        return false;
    }
}
