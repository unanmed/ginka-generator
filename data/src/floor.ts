import { GinkaConfig } from './types';

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
    91: 11, // 箭头
    92: 11, // 箭头
    93: 11, // 箭头
    94: 11, // 箭头
    53: 12, // 道具
    29: 13 // 绿宝石
};

export interface Enemy {
    hp: number;
    atk: number;
    def: number;
}

function convert(
    map: number[][],
    [x, y, w, h]: [number, number, number, number],
    config: GinkaConfig,
    enemyMap: Record<number, Enemy>
) {
    const clipped: number[][] = [];

    // 1. 裁剪
    for (let nx = x; nx < x + w; nx++) {
        const row: number[] = [];
        for (let ny = y; ny < y + h; ny++) {
            row.push(map[ny][nx]);
        }
        clipped.push(row);
    }

    // 2. 转换怪物
    const enemySet = new Set<Enemy>();
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (tile === 201 || tile === 202 || tile === 203) continue;
            const enemy = enemyMap[tile];
            if (!enemy) continue;
            enemySet.add(enemy);
        }
    }
    const attrs = [...enemySet].map(v => (v.atk + v.def) * v.hp);
    const maxAttr = Math.max(...attrs);
    const minAttr = Math.min(...attrs);
    const delta = maxAttr - minAttr;
    for (let ny = 0; ny < w; ny++) {
        for (let nx = 0; nx < h; nx++) {
            const tile = clipped[ny][nx];
            if (tile === 201 || tile === 202 || tile === 203) continue;
            const enemy = enemyMap[tile];
            if (!enemy) continue;
            // 替换为弱怪/中怪/强怪
            const attr = (enemy.atk + enemy.def) * enemy.hp;
            const ad = attr - minAttr;
            if (ad < delta / 3) {
                clipped[ny][nx] = 7;
            } else if (ad < (delta * 2) / 3) {
                clipped[ny][nx] = 8;
            } else {
                clipped[ny][nx] = 9;
            }
        }
    }

    // 3. 转换一般图块
    const mapping: Record<number, number> = {};
    config.mapping.wall.forEach(v => (mapping[v] = 1));
    config.mapping.key.forEach(v => (mapping[v] = 2));
    config.mapping.redGem.forEach(v => (mapping[v] = 3));
    config.mapping.blueGem.forEach(v => (mapping[v] = 4));
    config.mapping.potion.forEach(v => (mapping[v] = 5));
    config.mapping.door.forEach(v => (mapping[v] = 6));
    config.mapping.item.forEach(v => (mapping[v] = 12));
    config.mapping.greenGem.forEach(v => (mapping[v] = 13));
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (mapping[tile]) clipped[ny][nx] = mapping[tile];
            else if (numMap[tile]) clipped[ny][nx] = numMap[tile];
            else clipped[ny][nx] = 0;
        }
    }

    return clipped;
}

export function convertFloor(
    map: number[][],
    clip: [number, number, number, number],
    config: GinkaConfig,
    enemyMap: Record<number, Enemy>
) {
    return convert(map, clip, config, enemyMap);
}
