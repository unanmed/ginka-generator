interface VisualSimilarityConfig {
    // 类型重要性权重（需根据游戏设定调整）
    typeWeights: { [key: number]: number };
    // 是否启用视觉焦点增强
    enableVisualFocus: boolean;
    // 是否启用密度感知
    enableDensityAwareness: boolean;
}

const DEFAULT_CONFIG: VisualSimilarityConfig = {
    typeWeights: {
        0: 0.2, // 空地
        1: 0.3, // 墙壁
        2: 0.6, // 钥匙
        3: 0.7, // 红宝石
        4: 0.7, // 蓝宝石
        5: 0.5, // 血瓶
        6: 0.4, // 门
        7: 0.5, // 弱怪
        8: 0.6, // 中怪
        9: 0.6, // 强怪
        10: 0.4, // 楼梯
        11: 0.4, // 箭头
        12: 0.7 // 道具
    },
    enableVisualFocus: true,
    enableDensityAwareness: true
};

export function calculateVisualSimilarity(
    map1: number[][],
    map2: number[][],
    config = DEFAULT_CONFIG
): number {
    // 尺寸校验
    if (map1.length !== map2.length || map1[0]?.length !== map2[0]?.length) {
        return 0; // 或抛出异常
    }

    const rows = map1.length;
    const cols = map1[0].length;
    let totalScore = 0;
    let maxPossibleScore = 0;

    // 视觉焦点权重图
    const focusWeights = config.enableVisualFocus
        ? generateFocusWeights(rows, cols)
        : Array(rows)
              .fill(1)
              .map(() => Array(cols).fill(1));

    // 类型密度分布计算
    const densityMap = config.enableDensityAwareness
        ? calculateDensityImpact(map1, map2, config.typeWeights)
        : Array(rows)
              .fill(1)
              .map(() => Array(cols).fill(1));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const type1 = map1[i][j];
            const type2 = map2[i][j];

            // 基础类型权重
            const baseWeight = Math.max(
                config.typeWeights[type1] || 0.5,
                config.typeWeights[type2] || 0.5
            );

            // 空间权重组合
            const spatialWeight = focusWeights[i][j] * densityMap[i][j];

            // 类型匹配得分
            const typeScore = type1 === type2 ? 1 : 0;

            totalScore += typeScore * baseWeight * spatialWeight;
            maxPossibleScore += baseWeight * spatialWeight;
        }
    }

    return maxPossibleScore > 0 ? totalScore / maxPossibleScore : 0;
}

/**
 * 生成视觉焦点权重图（基于人类视觉注意力分布）
 */
function generateFocusWeights(rows: number, cols: number): number[][] {
    const weights = [];
    const centerX = cols / 2;
    const centerY = rows / 2;
    const maxDist = Math.sqrt(centerX ** 2 + centerY ** 2) * 0.7;

    for (let i = 0; i < rows; i++) {
        const rowWeights = [];
        for (let j = 0; j < cols; j++) {
            // 使用高斯分布模拟视觉焦点
            const dx = (j - centerX) / cols;
            const dy = (i - centerY) / rows;
            const distance = Math.sqrt(dx ** 2 + dy ** 2);
            const gaussian = Math.exp(-(distance ** 2) / (2 * 0.3 ** 2));
            rowWeights.push(1.0 + 0.6 * gaussian); // 中心区域最高1.6倍权重
        }
        weights.push(rowWeights);
    }
    return weights;
}

/**
 * 计算类型密度影响权重
 */
function calculateDensityImpact(
    map1: number[][],
    map2: number[][],
    typeWeights: { [key: number]: number }
): number[][] {
    const rows = map1.length;
    const cols = map1[0].length;
    const densityMap = Array(rows)
        .fill(0)
        .map(() => Array(cols).fill(0));

    // 滑动窗口分析局部密度
    const windowSize = 3;
    const halfWindow = Math.floor(windowSize / 2);

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let density = 0;
            for (let di = -halfWindow; di <= halfWindow; di++) {
                for (let dj = -halfWindow; dj <= halfWindow; dj++) {
                    const ni = i + di;
                    const nj = j + dj;
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        const weight1 = typeWeights[map1[ni][nj]] || 0.5;
                        const weight2 = typeWeights[map2[ni][nj]] || 0.5;
                        density += (weight1 + weight2) / 2;
                    }
                }
            }
            // 密度权重：高密度区域增强对比度
            densityMap[i][j] = 1.0 + 0.4 * (density / windowSize ** 2);
        }
    }
    return densityMap;
}
