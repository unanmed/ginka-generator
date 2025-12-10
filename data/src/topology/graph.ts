import {
    ResourceArea,
    GinkaGraph,
    BranchNode,
    GinkaTopologicalGraphs,
    ResourceNode,
    NodeType
} from './interface';

export const tileType = new Set(
    Array(13)
        .fill(0)
        .map((_, i) => i)
);
const branchType = new Set([6, 7, 8, 9]);
const entranceType = new Set([10, 11]);
const resourceType = new Set([0, 2, 3, 4, 5, 10, 11, 12, 13]);

export const directions: [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
];

function buildGraphFromEntrance(
    map: number[][],
    entrance: number,
    resourceMap: Map<number, number>,
    areaMap: ResourceArea[]
): GinkaGraph {
    const width = map[0].length;
    const height = map[1].length;

    const visitedEntrance = new Set<number>([entrance]);
    const visited = new Set<number>();
    const queue: [number, number][] = [];
    queue.push([entrance % width, Math.floor(entrance / width)]);

    const branchNodes = new Set<number>();

    // 1. BFS 检测所有分支节点
    while (queue.length > 0) {
        const item = queue.shift();
        if (!item) continue;
        const [nx, ny] = item;
        const index = ny * width + nx;
        if (visited.has(index)) continue;
        const tile = map[ny][nx];

        if (entranceType.has(tile)) {
            visitedEntrance.add(index);
        }
        if (branchType.has(tile)) {
            branchNodes.add(index);
        }
        visited.add(index);

        for (const [dx, dy] of directions) {
            const px = dx + nx;
            const py = dy + ny;
            if (px < 0 || px >= width || py < 0 || py >= height) {
                continue;
            }
            const tile = map[py][px];
            if (tile !== 1) {
                // 非墙区域可通行
                queue.push([px, py]);
            }
        }
    }

    // 2. 从分支节点构建拓扑图
    const graph = new Map<number, BranchNode | ResourceNode>();
    branchNodes.forEach(v => {
        const nx = v % width;
        const ny = Math.floor(v / width);
        if (!graph.get(v)) {
            graph.set(v, {
                type: NodeType.Branch,
                neighbor: new Set(),
                tile: map[ny][nx]
            });
        }
        const node = graph.get(v)!;
        for (const [dx, dy] of directions) {
            const px = nx + dx;
            const py = ny + dy;
            if (px < 0 || px >= width || py < 0 || py >= height) {
                continue;
            }
            const index = py * width + px;

            // 先检查临近节点是不是分支节点，是的话链接到自己
            if (branchNodes.has(index)) {
                node.neighbor.add(index);
            } else {
                // 检查是不是资源节点
                const pointer = resourceMap.get(index);
                if (pointer === void 0) continue;
                const area = areaMap[pointer];
                if (!area) continue;
                area.neighbor.add(v);
                area.members.forEach(v => {
                    node.neighbor.add(v);
                });
            }
        }
    });

    // 3. 把资源节点拆分成并排，并放入拓扑图
    areaMap.forEach(v => {
        v.members.forEach(index => {
            const nx = index % width;
            const ny = Math.floor(index / width);
            const tile = map[ny][nx];
            if (tile === 0) return;
            const node: ResourceNode = {
                type: NodeType.Resource,
                resourceType: tile,
                neighbor: v.neighbor,
                resourceArea: v
            };
            graph.set(index, node);
        });
    });

    return { graph, resourceMap, areaMap, visitedEntrance, visited };
}

function findResourceNodes(map: number[][]) {
    const width = map[0].length;
    const height = map[1].length;

    const visited = new Set<number>();
    const areas: ResourceArea[] = [];
    const resourcesMap: Map<number, number> = new Map();

    for (let ny = 0; ny < height; ny++) {
        for (let nx = 0; nx < width; nx++) {
            const tile = map[ny][nx];
            const index = ny * width + nx;
            if (visited.has(index) || !resourceType.has(tile)) {
                continue;
            }
            const queue: [number, number][] = [];
            queue.push([nx, ny]);
            const area: ResourceArea = {
                type: NodeType.Resource,
                resources: new Map([[tile, 1]]),
                members: new Set([index]),
                neighbor: new Set()
            };

            while (queue.length > 0) {
                const item = queue.shift();
                if (!item) continue;
                const [nx, ny] = item;
                const index = ny * width + nx;
                if (visited.has(index)) {
                    continue;
                }
                const tile = map[ny][nx];
                if (!resourceType.has(tile)) {
                    continue;
                }
                visited.add(index);

                const exists = area.resources.get(tile);
                if (!exists) {
                    area.resources.set(tile, 1);
                } else {
                    area.resources.set(tile, exists + 1);
                }
                area.members.add(index);
                resourcesMap.set(index, areas.length);

                for (const [dx, dy] of directions) {
                    const px = nx + dx;
                    const py = ny + dy;
                    if (px < 0 || px >= width || py < 0 || py >= height) {
                        continue;
                    }
                    queue.push([px, py]);
                }
            }

            areas.push(area);
        }
    }

    return { areaMap: areas, resourcesMap };
}

export function buildTopologicalGraph(map: number[][]): GinkaTopologicalGraphs {
    const width = map[0].length;
    const height = map[1].length;

    // 1. 找到所有入口
    const entrances = new Set<number>();
    for (let ny = 0; ny < height; ny++) {
        for (let nx = 0; nx < width; nx++) {
            const tile = map[ny][nx];
            if (entranceType.has(tile)) {
                entrances.add(ny * width + nx);
            }
        }
    }

    // 2. 找到所有的资源节点
    const { areaMap, resourcesMap } = findResourceNodes(map);

    // 3. 对每个入口计算拓扑图
    const graphs: GinkaGraph[] = [];
    const usedEntrance = new Set<number>();
    const totalVisited = new Set<number>();
    /** 入口位置到拓扑图的映射 */
    const entranceMap = new Map<number, GinkaGraph>();
    entrances.forEach(v => {
        if (usedEntrance.has(v)) {
            return;
        }
        const nx = v % width;
        const ny = Math.floor(v / width);
        const entranceGraph = buildGraphFromEntrance(
            map,
            v,
            resourcesMap,
            areaMap
        );
        const { graph, visited, visitedEntrance } = entranceGraph;
        graphs.push(entranceGraph);
        // 标记已经探索到的入口，并标记这个入口对应了哪个图
        visitedEntrance.forEach(v => {
            usedEntrance.add(v);
            entranceMap.set(v, entranceGraph);
        });
        visited.forEach(v => {
            totalVisited.add(v);
        });
    });

    // 3. 计算不可到达区域
    const unreachable = new Set<number>();
    for (let ny = 0; ny < height; ny++) {
        for (let nx = 0; nx < width; nx++) {
            const index = ny * width + nx;
            if (!totalVisited.has(index) && map[ny][nx] !== 1) {
                unreachable.add(index);
            }
        }
    }

    return { graphs, entranceMap, unreachable };
}
