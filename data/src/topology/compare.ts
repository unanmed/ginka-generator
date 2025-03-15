import { buildTopologicalGraph } from './graph';
import { GinkaTopologicalGraphs } from './interface';
import { overallSimilarity } from './similarity';

const cache = new Map<string, GinkaTopologicalGraphs>();

export function getTopologicalGraph(
    floorId: string,
    map: number[][]
): GinkaTopologicalGraphs {
    if (cache.has(floorId)) return cache.get(floorId)!;
    const graphs = buildTopologicalGraph(map);
    cache.set(floorId, graphs);
    return graphs;
}

export function compareMap(
    floorId1: string,
    floorId2: string,
    map1: number[][],
    map2: number[][]
) {
    const graph1 = getTopologicalGraph(floorId1, map1);
    const graph2 = getTopologicalGraph(floorId2, map2);

    const kernel = overallSimilarity(graph1, graph2);

    return kernel;
}
