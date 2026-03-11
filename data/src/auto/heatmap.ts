/**
 * 将地图转换为热力图
 * @param map 地图矩阵
 * @param tokens 计入热力图的图块
 */
export function generateHeatmap(
    map: number[][],
    tokens: Set<number>,
    kernel: number = 5
): number[][] {
    if (kernel % 2 !== 1) {
        throw new Error(`Kernal size must be odd.`);
    }
    const width = map[0].length;
    const height = map.length;
    const result: number[][] = Array.from({ length: height }, _ =>
        Array.from({ length: width }, _ => 0)
    );
    const radius = Math.floor(kernel / 2);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const left = Math.max(0, x - radius);
            const right = Math.min(width, x + radius);
            const top = Math.max(0, y - radius);
            const bottom = Math.min(height, y + radius);
            const size = (right - left) * (bottom - top);
            let num = 0;
            for (let ky = top; ky < bottom; ky++) {
                for (let kx = left; kx < right; kx++) {
                    if (tokens.has(map[ky][kx])) {
                        num++;
                    }
                }
            }
            result[y][x] = num / size;
        }
    }
    return result;
}

/**
 * 对热力图实施高斯模糊
 * @param map 热力图
 * @param sigma 标准差
 */
export function gaussainHeatmap(map: number[][], sigma: number = 1) {
    const radius = sigma * 3;
    const width = map[0].length;
    const height = map.length;
    const result: number[][] = Array.from({ length: height }, _ =>
        Array.from({ length: width }, _ => 0)
    );
    const s = sigma ** 2;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const left = Math.max(0, x - radius);
            const right = Math.min(width - 1, x + radius);
            const top = Math.max(0, y - radius);
            const bottom = Math.min(height - 1, y + radius);
            let res = 0;
            for (let ky = top; ky < bottom; ky++) {
                for (let kx = left; kx < right; kx++) {
                    const dis = (ky - y) ** 2 + (kx - x) ** 2;
                    const g = Math.E ** (-dis / (2 * s)) / (2 * Math.PI * s);
                    res += map[ky][kx] * g;
                }
            }
            result[y][x] = res;
        }
    }
    return result;
}

/**
 * 归一化热力图
 * @param map 热力图
 */
export function normalizeHeatmap(map: number[][]) {
    const max = Math.max(...map.flat());
    const min = Math.min(...map.flat());
    if (max === min) return map;
    const d = max - min;
    return map.map(line => line.map(v => (v - min) / d));
}
