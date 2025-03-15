import { calculateVisualSimilarity } from './similarity';

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
    const map3 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 5, 8, 10, 7, 2, 1],
        [1, 2, 1, 5, 1, 7, 1],
        [1, 3, 1, 3, 6, 4, 1],
        [1, 6, 1, 6, 1, 8, 1],
        [1, 10, 7, 5, 1, 5, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ];
    // MT3 微调
    const map4 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 3, 1, 10, 7, 3, 1],
        [1, 6, 1, 5, 1, 7, 1],
        [1, 9, 1, 8, 1, 6, 1],
        [1, 5, 8, 0, 1, 7, 1],
        [1, 2, 1, 4, 7, 10, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ];
    // MT10
    const map5 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 5, 1, 10, 1, 5, 1],
        [1, 6, 7, 7, 7, 6, 1],
        [1, 1, 6, 5, 6, 1, 1],
        [1, 4, 5, 9, 5, 4, 1],
        [1, 3, 1, 1, 1, 3, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ];

    // 测试自我对比
    console.log(`map1 vs map1: ${calculateVisualSimilarity(map1, map1)}`);
    console.log(`map2 vs map2: ${calculateVisualSimilarity(map2, map2)}`);
    console.log(`map3 vs map3: ${calculateVisualSimilarity(map3, map3)}`);
    console.log(`map4 vs map4: ${calculateVisualSimilarity(map4, map4)}`);
    // 两两测试
    console.log(`map1 vs map2: ${calculateVisualSimilarity(map1, map2)}`);
    console.log(`map1 vs map3: ${calculateVisualSimilarity(map1, map3)}`);
    console.log(`map1 vs map4: ${calculateVisualSimilarity(map1, map4)}`);
    console.log(`map1 vs map5: ${calculateVisualSimilarity(map1, map5)}`);
    console.log(`map2 vs map3: ${calculateVisualSimilarity(map2, map3)}`);
    console.log(`map2 vs map4: ${calculateVisualSimilarity(map2, map4)}`);
    console.log(`map2 vs map5: ${calculateVisualSimilarity(map2, map5)}`);
    console.log(`map3 vs map4: ${calculateVisualSimilarity(map3, map4)}`);
    console.log(`map3 vs map5: ${calculateVisualSimilarity(map3, map5)}`);
    console.log(`map4 vs map5: ${calculateVisualSimilarity(map4, map5)}`);
    // 测试交换性
    console.log(`map2 vs map1: ${calculateVisualSimilarity(map2, map1)}`);
    console.log(`map4 vs map2: ${calculateVisualSimilarity(map4, map2)}`);
})();
