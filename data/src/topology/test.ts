import { buildTopologicalGraph } from './graph';
import { mirrorMapX, mirrorMapY, rotateMap } from './transform';
import { overallSimilarity } from './similarity';

(() => {
    // MT3
    const map1 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 3, 1, 10, 7, 3, 1],
        [1, 6, 1, 5, 1, 7, 1],
        [1, 9, 1, 8, 1, 6, 1],
        [1, 5, 8, 0, 1, 7, 1],
        [1, 2, 1, 5, 7, 10, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ];
    // MT6
    const map2 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 5, 6, 4, 1, 10, 1],
        [1, 6, 1, 9, 1, 5, 1],
        [1, 8, 0, 6, 0, 8, 1],
        [1, 5, 1, 10, 1, 2, 1],
        [1, 9, 3, 1, 4, 9, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ];
    // MT8
    // const map2 = [
    //     [1, 1, 1, 1, 1, 1, 1],
    //     [1, 5, 8, 10, 7, 2, 1],
    //     [1, 2, 1, 5, 1, 7, 1],
    //     [1, 3, 1, 3, 6, 4, 1],
    //     [1, 6, 1, 6, 1, 8, 1],
    //     [1, 10, 7, 5, 1, 5, 1],
    //     [1, 1, 1, 1, 1, 1, 1]
    // ];
    // MT3 微调
    // const map2 = [
    //     [1, 1, 1, 1, 1, 1, 1],
    //     [1, 3, 1, 10, 7, 3, 1],
    //     [1, 6, 1, 5, 1, 7, 1],
    //     [1, 9, 1, 8, 1, 6, 1],
    //     [1, 5, 8, 0, 1, 7, 1],
    //     [1, 2, 1, 4, 7, 10, 1],
    //     [1, 1, 1, 1, 1, 1, 1]
    // ];

    // 1. 两张图与自身对比
    const graph1 = buildTopologicalGraph(map1);
    const graph2 = buildTopologicalGraph(map2);

    console.log(`map1 vs map1: ${overallSimilarity(graph1, graph1)}`);
    console.log(`map2 vs map2: ${overallSimilarity(graph2, graph2)}`);

    // 2. 两张图相互对比
    console.log(`map1 vs map2: ${overallSimilarity(graph1, graph2)}`);
    console.log(`map2 vs map1: ${overallSimilarity(graph2, graph1)}`);

    // 3. x镜像对比
    const xFlipped1 = mirrorMapX(map1);
    const xFlipped2 = mirrorMapX(map2);
    const graphX1 = buildTopologicalGraph(xFlipped1);
    const graphX2 = buildTopologicalGraph(xFlipped2);
    console.log(`map1:x vs map1: ${overallSimilarity(graphX1, graph1)}`);
    console.log(`map1:x vs map2: ${overallSimilarity(graphX1, graph2)}`);
    console.log(`map1 vs map2:x: ${overallSimilarity(graph1, graphX2)}`);
    console.log(`map2:x vs map2: ${overallSimilarity(graphX2, graph2)}`);
    console.log(`map2:x vs map1: ${overallSimilarity(graphX2, graph2)}`);
    console.log(`map2 vs map1:x: ${overallSimilarity(graph2, graphX1)}`);

    // 4. y镜像对比
    const yFlipped1 = mirrorMapY(map1);
    const yFlipped2 = mirrorMapY(map2);
    const graphY1 = buildTopologicalGraph(yFlipped1);
    const graphY2 = buildTopologicalGraph(yFlipped2);
    console.log(`map1:y vs map1: ${overallSimilarity(graphY1, graph1)}`);
    console.log(`map1:y vs map2: ${overallSimilarity(graphY1, graph2)}`);
    console.log(`map1 vs map2:y: ${overallSimilarity(graph1, graphY2)}`);
    console.log(`map2:y vs map2: ${overallSimilarity(graphY2, graph2)}`);
    console.log(`map2:y vs map1: ${overallSimilarity(graphY2, graph1)}`);
    console.log(`map2 vs map1:y: ${overallSimilarity(graph2, graphY1)}`);

    // 5. xy 镜像混合对比
    console.log(`map1:x vs map1:y: ${overallSimilarity(graphX1, graphY1)}`);
    console.log(`map1:y vs map2:x: ${overallSimilarity(graphY1, graphX2)}`);
    console.log(`map1:x vs map2:x: ${overallSimilarity(graphX1, graphX2)}`);
    console.log(`map1:x vs map2:y: ${overallSimilarity(graphX1, graphY2)}`);

    // 6. 旋转对比
    const rot901 = rotateMap(map1);
    const rot902 = rotateMap(map2);
    const graph901 = buildTopologicalGraph(rot901);
    const graph902 = buildTopologicalGraph(rot902);
    console.log(`map1:90 vs map1: ${overallSimilarity(graph1, graph901)}`);
    console.log(`map2:90 vs map2: ${overallSimilarity(graph2, graph902)}`);
})();
