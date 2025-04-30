import { GinkaConfig } from './types';

const numMap: Record<number, number> = {
    0: 0,
    1: 1,
    2: 2,
    91: 30,
    92: 30,
    93: 30,
    94: 30,
    87: 29,
    88: 29
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
    for (let ny = y; ny < y + w; ny++) {
        const row: number[] = [];
        for (let nx = y; nx < x + h; nx++) {
            row.push(map[ny][nx]);
        }
        clipped.push(row);
    }

    const res: number[][] = Array.from({ length: clipped.length }, () =>
        Array.from({ length: clipped[0].length }, () => 0)
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
    dict.floor.forEach(v => (mapping[v] = 29));
    dict.arrow.forEach(v => (mapping[v] = 30));
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
                if (maxRedGem - minRedGem < 1e-8) {
                    res[ny][nx] = 10;
                } else {
                    const level = Math.min(
                        Math.floor(
                            ((value - minRedGem) / (maxRedGem - minRedGem)) * 3
                        ),
                        2
                    );
                    res[ny][nx] = 10 + level;
                }
            } else if (dict.blueGem[tile] !== void 0) {
                const value = dict.blueGem[tile];
                if (maxBlueGem - minBlueGem < 1e-8) {
                    res[ny][nx] = 13;
                } else {
                    const level = Math.min(
                        Math.floor(
                            ((value - minBlueGem) / (maxBlueGem - minBlueGem)) *
                                3
                        ),
                        2
                    );
                    res[ny][nx] = 13 + level;
                }
            } else if (dict.greenGem[tile] !== void 0) {
                const value = dict.greenGem[tile];
                if (maxGreenGem - minGreenGem < 1e-8) {
                    res[ny][nx] = 16;
                } else {
                    const level = Math.min(
                        Math.floor(
                            ((value - minGreenGem) /
                                (maxGreenGem - minGreenGem)) *
                                3
                        ),
                        2
                    );
                    res[ny][nx] = 16 + level;
                }
            } else if (dict.yellowGem[tile] !== void 0) {
                const rand = Math.random();
                const value = dict.yellowGem[tile];
                if (rand < 2 / 5) {
                    if (maxRedGem - minRedGem < 1e-8) {
                        res[ny][nx] = 10;
                    } else {
                        const level = Math.min(
                            Math.floor(
                                ((value - minRedGem) /
                                    (maxRedGem - minRedGem)) *
                                    3
                            ),
                            2
                        );
                        res[ny][nx] = 10 + level;
                    }
                } else if (rand < 4 / 5) {
                    if (maxBlueGem - minBlueGem < 1e-8) {
                        res[ny][nx] = 13;
                    } else {
                        const level = Math.min(
                            Math.floor(
                                ((value - minBlueGem) /
                                    (maxBlueGem - minBlueGem)) *
                                    3
                            ),
                            2
                        );
                        res[ny][nx] = 13 + level;
                    }
                } else {
                    if (maxGreenGem - minGreenGem < 1e-8) {
                        res[ny][nx] = 16;
                    } else {
                        const level = Math.min(
                            Math.floor(
                                ((value - minGreenGem) /
                                    (maxGreenGem - minGreenGem)) *
                                    3
                            ),
                            2
                        );
                        res[ny][nx] = 16 + level;
                    }
                }
            } else if (dict.potion[tile] !== void 0) {
                const value = dict.potion[tile];
                if (maxGreenGem - minGreenGem < 1e-8) {
                    res[ny][nx] = 19;
                } else {
                    const level = Math.min(
                        Math.floor(
                            ((value - minPotion) / (maxPotion - minPotion)) * 4
                        ),
                        3
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

export function getCount(map: number[][], tiles: number[]) {
    let n = 0;
    map.flat().forEach(v => {
        if (tiles.includes(v)) n++;
    });
    return n;
}

export function getRatio(map: number[][], tiles: number[]) {
    const area = map.length * map[0].length;
    return getCount(map, tiles) / area;
}

function range(from: number, to: number) {
    const length = to - from;
    return Array.from({ length }, (_, i) => i + from);
}

export function getGinkaRatio(map: number[][]): number[] {
    const arr: number[] = Array(16).fill(0);
    arr[0] = getRatio(map, [1, ...range(3, 32)]);
    arr[1] = getRatio(map, [1]);
    arr[2] = getRatio(map, [2]);
    arr[3] = getRatio(map, [3, 4, 5, 6]);
    arr[4] = getRatio(map, [26, 27, 28]);
    arr[5] = getRatio(map, range(7, 26));
    arr[6] = getRatio(map, range(10, 19));
    arr[7] = getRatio(map, range(19, 23));
    arr[8] = getRatio(map, [7, 8, 9]);
    arr[9] = getCount(map, [23, 24, 25]);
    arr[10] = getCount(map, [29, 30]);
    return arr;
}
