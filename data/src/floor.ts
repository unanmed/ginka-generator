const numMap: Record<number, number> = {
    0: 0, // 空地
    1: 1, // 墙壁
    21: 2, // 钥匙
    27: 3, // 红宝石
    28: 4, // 蓝宝石
    31: 5, // 血瓶
    81: 6, // 门
    201: 7, // 弱怪
    202: 8, // 中怪
    203: 9, // 强怪
    87: 10, // 楼梯
    88: 10, // 楼梯
    161: 11, // 箭头
    162: 11, // 箭头
    163: 11, // 箭头
    164: 11, // 箭头
    53: 12 // 道具
};

const apeiriaMap: Record<number, number> = {
    0: 0, // 空地
    1: 1, // 墙壁
    224: 1, // 吸血鬼，视为墙壁
    21: 2, // 黄钥匙
    22: 2, // 蓝钥匙
    23: 2, // 红钥匙
    27: 3, // 红宝石
    28: 4, // 蓝宝石
    29: 0, // 绿宝石
    31: 5, // 红血瓶
    32: 5, // 蓝血瓶
    33: 5, // 绿血瓶
    34: 5, // 黄血瓶
    81: 6, // 门
    201: 7, // 弱怪
    202: 8, // 中怪
    203: 9, // 强怪
    87: 10, // 楼梯
    88: 10, // 楼梯
    161: 11, // 箭头
    162: 11, // 箭头
    163: 11, // 箭头
    164: 11, // 箭头
    53: 12, // 幸运金币
    50: 12, // 飞
    49: 12, // 炸
    47: 12 // 破
};

export interface ApeiriaEnemy {
    hp: number;
    atk: number;
    def: number;
}

function convert(
    map: number[][],
    [x, y, w, h]: [number, number, number, number],
    name: string,
    floorId: string,
    numMap: Record<number, number>
) {
    const clipped: number[][] = [];

    for (let nx = x; nx < x + w; nx++) {
        const row: number[] = [];
        for (let ny = y; ny < y + h; ny++) {
            const num = numMap[map[nx][ny]];
            if (num === void 0) {
                console.log(
                    `⚠️  魔塔 ${name} 的楼层 ${floorId} 中出现未知图块类型：${map[nx][ny]}`
                );
            }
            row.push(num ?? 0);
        }
        clipped.push(row);
    }

    return clipped;
}

function convertApeiriaEnemy(
    map: number[][],
    enemyMap: Record<number, ApeiriaEnemy>
) {
    const width = map[0].length;
    const height = map.length;
    const enemy = new Set<ApeiriaEnemy>();
    for (let ny = 0; ny < height; ny++) {
        for (let nx = 0; nx < width; nx++) {
            const tile = map[ny][nx];
            if (tile > 200 && tile <= 280) {
                // 这些是怪物
                if (enemyMap[tile]) enemy.add(enemyMap[tile]);
            }
        }
    }
    const attrs = [...enemy].map(v => (v.atk + v.def) * v.hp);
    const maxAttr = Math.max(...attrs);
    const minAttr = Math.min(...attrs);
    const delta = maxAttr - minAttr;
    for (let ny = 0; ny < height; ny++) {
        for (let nx = 0; nx < width; nx++) {
            const tile = map[ny][nx];
            if (tile > 200 && tile <= 280) {
                // 这些是怪物
                if (enemyMap[tile]) {
                    // 替换为弱怪/中怪/强怪
                    const enemy = enemyMap[tile];
                    const attr = (enemy.atk + enemy.def) * enemy.hp;
                    const ad = attr - minAttr;
                    if (ad < delta / 3) {
                        map[ny][nx] = 201;
                    } else if (ad < (delta * 2) / 3) {
                        map[ny][nx] = 202;
                    } else {
                        map[ny][nx] = 203;
                    }
                }
            }
        }
    }

    return map;
}

export function convertFloor(
    map: number[][],
    clip: [number, number, number, number],
    name: string,
    floorId: string
) {
    return convert(map, clip, name, floorId, numMap);
}

export function convertApeiriaMap(
    map: number[][],
    clip: [number, number, number, number],
    name: string,
    floorId: string,
    enemyMap: Record<number, ApeiriaEnemy>
) {
    return convert(
        convertApeiriaEnemy(map, enemyMap),
        clip,
        name,
        floorId,
        apeiriaMap
    );
}
