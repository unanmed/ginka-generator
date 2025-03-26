import { cosineSimilarity } from 'src/utils';
import { GinkaGraph, GinkaTopologicalGraphs } from './interface';

interface WLNode {
    originalPos: number;
    originalLabel: string;
    currentLabel: string;
    neighbors: WLNode[];
}

function encodeNodeLabels(graph: GinkaGraph) {
    const nodes: WLNode[] = [];
    const nodeMap = new Map<number, WLNode>();

    graph.graph.forEach((node, pos) => {
        let label: string;

        // 编码为唯一哈希值（用字符串就行，V8 会自动帮你算哈希）
        if (node.type === 'branch') {
            label = `B:${node.tile}`;
        } else {
            label = `R:${node.resourceType}`;
        }

        const wlNode: WLNode = {
            originalPos: pos,
            originalLabel: label,
            currentLabel: label,
            neighbors: []
        };
        nodeMap.set(pos, wlNode);
        nodes.push(wlNode);
    });

    // 映射邻居节点
    nodes.forEach(node => {
        const ginkaNode = graph.graph.get(node.originalPos);
        ginkaNode?.neighbor.forEach(v => {
            const wl = nodeMap.get(v);
            if (wl) node.neighbors.push(wl);
        });
    });

    return nodes;
}

function weisfeilerLehmanIteration(
    nodes: WLNode[],
    iterations: number,
    decay: number = 0.6 // 衰减权重，减小长距离图的权重
) {
    const labelHistory: string[][] = [];

    for (let i = 0; i < iterations; i++) {
        const newLabels: string[] = [];

        // 生成新标签
        nodes.forEach(node => {
            const neighborLabels = node.neighbors
                .map(n => n.currentLabel)
                .sort();
            
            const compositeLabel = `${node.currentLabel}|${neighborLabels.join(
                ','
            )}`.slice(0, 8192);

            newLabels.push(compositeLabel);
        });

        // 更新节点标签并记录
        nodes.forEach((node, idx) => {
            node.currentLabel = newLabels[idx];
        });
        labelHistory.push([...newLabels]);
    }

    // 统计每个节点的数量
    let weight = 1;
    const numMap = new Map<string, number>();
    labelHistory.forEach(iter => {
        iter.forEach(v => {
            if (!numMap.has(v)) {
                numMap.set(v, weight);
            } else {
                numMap.set(v, numMap.get(v)! + weight);
            }
        });
        weight *= decay;
    });
    // 把每个节点的原始标签也加上，权重使用最远权重，可以认为是资源重复率
    nodes.forEach(node => {
        if (!numMap.has(node.originalLabel)) {
            numMap.set(node.originalLabel, weight);
        } else {
            numMap.set(
                node.originalLabel,
                numMap.get(node.originalLabel)! + weight
            );
        }
    });

    return numMap;
}

function vectorizeFeatures(features: Map<string, number>, vocab: string[]) {
    const vec: number[] = new Array(vocab.length).fill(0);

    features.forEach((count, label) => {
        const index = vocab.indexOf(label);
        if (index !== -1) {
            vec[index] += count;
        }
    });

    return vec;
}

function wlKernel(
    graphA: GinkaGraph,
    graphB: GinkaGraph,
    iterations = 3
): number {
    // 编码节点
    const nodesA = encodeNodeLabels(graphA);
    const nodesB = encodeNodeLabels(graphB);

    // 迭代生成标签
    const featuresA = weisfeilerLehmanIteration(nodesA, iterations);
    const featuresB = weisfeilerLehmanIteration(nodesB, iterations);

    // 构建特征向量
    const vocab = [...new Set([...featuresA.keys(), ...featuresB.keys()])];
    const vecA = vectorizeFeatures(featuresA, vocab);
    const vecB = vectorizeFeatures(featuresB, vocab);

    // 计算余弦相似度
    return cosineSimilarity(vecA, vecB);
}

export function overallSimilarity(
    a: GinkaTopologicalGraphs,
    b: GinkaTopologicalGraphs
) {
    // 使用 Weisfeiler-Lehman Kernel 方式计算拓扑图相似度
    const graphsA = a.graphs;
    const graphsB = b.graphs;

    let totalSimilarity = 0;
    const comparedGraph = new Set<GinkaGraph>();
    graphsA.forEach(ga => {
        let maxSimilarity = 0;
        let maxGraph: GinkaGraph | null = null;
        // 图之间两两比较，找到最接近的作为相似度
        for (const gb of graphsB) {
            if (comparedGraph.has(gb)) continue;
            // 计算迭代次数
            const min = Math.min(ga.graph.size, gb.graph.size);
            const iterations = Math.ceil(Math.max(1, Math.log(min)));
            const similarity = wlKernel(ga, gb, iterations);
            if (similarity > maxSimilarity && !isNaN(similarity)) {
                maxSimilarity = similarity;
                maxGraph = gb;
            }
            if (similarity === 1) break;
        }
        totalSimilarity += maxSimilarity;
        if (maxGraph) comparedGraph.add(maxGraph);
    });

    // 不可达区域惩罚
    const reduction =
        1 / (1 + Math.abs(a.unreachable.size - b.unreachable.size));
    // 取根号使结果更接近线性
    if (graphsA.length === 0) {
        return 0;
    } else {
        return Math.sqrt(totalSimilarity / graphsA.length) * reduction;
    }
}
