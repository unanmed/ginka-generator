import { GinkaConfig } from './types';

const numMap: Record<number, number> = {
    0: 0,
    1: 1,
    2: 2,
    91: 29,
    92: 29,
    93: 29,
    94: 29,
    87: 28,
    88: 28
};

export interface Enemy {
    num: number;
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

    const res: number[][] = Array.from({ length: clipped.length }, () =>
        Array.from({ length: clipped[0].length })
    );

    // 2. 初步映射
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (numMap[tile] !== void 0) {
                res[ny][nx] = numMap[tile];
            }
        }
    }

    // 3. 转换一般图块
    const mapping: Record<number, number> = {};
    const dict = config.mapping;
    dict.wall.forEach(v => (mapping[v] = 1));
    dict.decoration.forEach(v => (mapping[v] = 2));
    dict.floor.forEach(v => (mapping[v] = 28));
    dict.arrow.forEach(v => (mapping[v] = 29));
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (mapping[tile] !== void 0) res[ny][nx] = mapping[tile];
        }
    }

    // 4. 转换含等级图块
    const redGemSet = new Set<number>();
    const blueGemSet = new Set<number>();
    const greenGemSet = new Set<number>();
    const potionSet = new Set<number>();
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (dict.redGem[tile] !== void 0) {
                redGemSet.add(dict.redGem[tile]);
            } else if (dict.blueGem[tile] !== void 0) {
                blueGemSet.add(dict.blueGem[tile]);
            } else if (dict.greenGem[tile] !== void 0) {
                greenGemSet.add(dict.greenGem[tile]);
            } else if (dict.yellowGem[tile] !== void 0) {
                redGemSet.add(dict.yellowGem[tile]);
                blueGemSet.add(dict.yellowGem[tile]);
                greenGemSet.add(dict.yellowGem[tile]);
            } else if (dict.potion[tile] !== void 0) {
                potionSet.add(dict.potion[tile]);
            }
        }
    }
    const minRedGem = Math.min(...redGemSet);
    const maxRedGem = Math.max(...redGemSet);
    const minBlueGem = Math.min(...blueGemSet);
    const maxBlueGem = Math.max(...blueGemSet);
    const minGreenGem = Math.min(...greenGemSet);
    const maxGreenGem = Math.max(...greenGemSet);
    const minPotion = Math.min(...potionSet);
    const maxPotion = Math.max(...potionSet);

    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            if (dict.redGem[tile] !== void 0) {
                const value = dict.redGem[tile];
                if (maxRedGem === minRedGem) {
                    res[ny][nx] = 10;
                } else {
                    const level = Math.floor(
                        ((value - minRedGem) / (maxRedGem - minRedGem)) * 3
                    );
                    res[ny][nx] = 10 + level;
                }
            } else if (dict.blueGem[tile] !== void 0) {
                const value = dict.blueGem[tile];
                if (maxBlueGem === minBlueGem) {
                    res[ny][nx] = 13;
                } else {
                    const level = Math.floor(
                        ((value - minBlueGem) / (maxBlueGem - minBlueGem)) * 3
                    );
                    res[ny][nx] = 13 + level;
                }
            } else if (dict.greenGem[tile] !== void 0) {
                const value = dict.greenGem[tile];
                if (maxGreenGem === minGreenGem) {
                    res[ny][nx] = 16;
                } else {
                    const level = Math.floor(
                        ((value - minGreenGem) / (maxGreenGem - minGreenGem)) *
                            3
                    );
                    res[ny][nx] = 16 + level;
                }
            } else if (dict.yellowGem[tile] !== void 0) {
                const rand = Math.random();
                const value = dict.yellowGem[tile];
                if (rand < 2 / 5) {
                    if (maxRedGem === minRedGem) {
                        res[ny][nx] = 10;
                    } else {
                        const level = Math.floor(
                            ((value - minRedGem) / (maxRedGem - minRedGem)) * 3
                        );
                        res[ny][nx] = 10 + level;
                    }
                } else if (rand < 4 / 5) {
                    if (maxBlueGem === minBlueGem) {
                        res[ny][nx] = 13;
                    } else {
                        const level = Math.floor(
                            ((value - minBlueGem) / (maxBlueGem - minBlueGem)) *
                                3
                        );
                        res[ny][nx] = 13 + level;
                    }
                } else {
                    if (maxGreenGem === minGreenGem) {
                        res[ny][nx] = 16;
                    } else {
                        const level = Math.floor(
                            ((value - minGreenGem) /
                                (maxGreenGem - minGreenGem)) *
                                3
                        );
                        res[ny][nx] = 16 + level;
                    }
                }
            } else if (dict.potion[tile] !== void 0) {
                const value = dict.potion[tile];
                if (maxGreenGem === minGreenGem) {
                    res[ny][nx] = 19;
                } else {
                    const level = Math.floor(
                        ((value - minPotion) / (maxPotion - minPotion)) * 3
                    );
                    res[ny][nx] = 19 + level;
                }
            } else if (dict.door[tile] !== void 0) {
                const level = dict.door[tile];
                res[ny][nx] = 3 + level;
            } else if (dict.key[tile] !== void 0) {
                const level = dict.key[tile];
                res[ny][nx] = 7 + level;
            } else if (dict.item[tile] !== void 0) {
                const level = dict.item[tile];
                res[ny][nx] = 22 + level;
            }
        }
    }

    // 5. 转换怪物
    const enemySet = new Set<Enemy>();
    for (let nx = 0; nx < w; nx++) {
        for (let ny = 0; ny < h; ny++) {
            const tile = clipped[ny][nx];
            const enemy = enemyMap[tile];
            if (!enemy) continue;
            enemySet.add({ ...enemy, num: tile });
        }
    }
    const enemyArr = [...enemySet];
    enemyArr.sort((a, b) => a.num - b.num);

    const attrs = [...enemySet].map(v => (v.atk + v.def) * v.hp);
    const maxAttr = Math.max(...attrs);
    const minAttr = Math.min(...attrs);
    const delta = maxAttr - minAttr;
    for (let ny = 0; ny < w; ny++) {
        for (let nx = 0; nx < h; nx++) {
            const tile = clipped[ny][nx];
            const enemy = enemyMap[tile];
            if (!enemy) continue;
            // 替换为弱怪/中怪/强怪
            const attr = (enemy.atk + enemy.def) * enemy.hp;
            const ad = attr - minAttr;
            if (ad < delta / 3 || delta === 0) {
                res[ny][nx] = 25;
            } else if (ad < (delta * 2) / 3) {
                res[ny][nx] = 26;
            } else {
                res[ny][nx] = 27;
            }
        }
    }

    return res;
}

export function convertFloor(
    map: number[][],
    clip: [number, number, number, number],
    config: GinkaConfig,
    enemyMap: Record<number, Enemy>
) {
    return convert(map, clip, config, enemyMap);
}
