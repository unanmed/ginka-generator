import {
    IAutoLabelConfig,
    ICodeRunResult,
    IConvertedMap,
    INeededFloorData
} from './types';

/**
 * 运行塔的代码
 * @param project project.min.js 内容
 * @param floors floors.min.js 内容
 */
export function runTowerCode(project: string, floors: string): ICodeRunResult {
    const result: Partial<ICodeRunResult> = {
        issue: [],
        main: {
            floors: {}
        }
    };
    const projectCode =
        project +
        `
        result.data = data_a1e2fb4a_e986_4524_b0da_9b7ba7c0874d;
        result.enemy = enemys_fcae963b_31c9_42b4_b48c_bb48d09f3f80;
        result.map = maps_90f36752_8815_4be8_b32b_d7fad1d0542e;
        result.item = items_296f5d02_12fd_4166_a7c1_b5e830c9ee3a;
    `;
    const main = result.main!;
    try {
        eval(projectCode);
        eval(floors);
        if (Object.keys(main.floors).length === 0) {
            result.issue?.push(`楼层信息为空`);
        }
    } catch {
        result.issue?.push(`代码运行错误`);
    }
    return result as ICodeRunResult;
}

function edge(x: number, y: number, width: number, height: number) {
    return x === 0 || y === 0 || x === width - 1 || y === height - 1;
}

export function convertTowerMap(
    result: ICodeRunResult,
    floor: INeededFloorData,
    config: IAutoLabelConfig
): IConvertedMap {
    const width = floor.map[0].length;
    const height = floor.map.length;

    let hasCannotInOut = false;
    const converted: number[][] = Array.from({ length: height }, () =>
        Array.from<number>({ length: width }).fill(0)
    );

    /** 键表示怪物位置，值表示怪物的生命乘以攻加防 */
    const enemyMap = new Map<number, number>();

    /** 键表示道具位置，值表示道具增加的血、攻、防、盾属性 */
    const itemMap = new Map<number, [number, number, number, number]>();
    // 这些是为了区分宝石，一个地图只有两种宝石的话当然没必要把三种宝石都用上
    const itemHpSet = new Set<number>();
    const itemAtkSet = new Set<number>();
    const itemDefSet = new Set<number>();
    const itemMdefSet = new Set<number>();

    const heroStatus: Record<string, number> = {
        hp: 0,
        atk: 0,
        def: 0,
        mdef: 0
    };

    const thisMap = {
        ratio: 1
    };

    // 给后面的 eval 用的
    const core = {
        values: new Proxy(result.data.values, {
            set() {
                // 防止被修改
                return true;
            }
        }),
        status: {
            hero: new Proxy(heroStatus, {
                set(target, p: string, newValue) {
                    if (typeof newValue !== 'number') return true;
                    if (
                        p !== 'hp' &&
                        p !== 'atk' &&
                        p !== 'def' &&
                        p !== 'mdef'
                    ) {
                        return true;
                    }
                    target[p] = newValue;
                    return true;
                }
            }),
            thisMap: new Proxy(thisMap, {
                set() {
                    // 防止被修改
                    return true;
                }
            })
        }
    };

    core.status.hero.hp = 0;
    core.status.hero.atk = 0;
    core.status.hero.def = 0;
    core.status.hero.mdef = 0;

    const tiles = config.classes;

    for (let nx = 0; nx < width; nx++) {
        for (let ny = 0; ny < height; ny++) {
            const num = floor.map[ny][nx];
            if (num === 0 || num === 17) {
                converted[ny][nx] = tiles.empty;
                continue;
            }
            const loc = `${nx},${ny}`;
            if (floor.changeFloor[loc]) {
                converted[ny][nx] = tiles.entry;
                continue;
            }
            const block = result.map[num];
            if (!block) {
                // 图块不存在说明是额外素材中的内容，默认不可通行，视为墙壁
                converted[ny][nx] = tiles.wall;
                continue;
            }
            // 怪物处理
            if (block.cls === 'enemys' || block.cls === 'enemy48') {
                const enemy = result.enemy[block.id];
                if (!enemy) {
                    converted[ny][nx] = tiles.empty;
                    continue;
                }
                const value = enemy.hp * (enemy.atk + enemy.def);
                enemyMap.set(ny * width + nx, value);
                continue;
            }
            // 道具处理
            if (block.cls === 'items') {
                const item = result.item[block.id];
                if (!item) {
                    converted[ny][nx] = tiles.empty;
                    continue;
                }
                // 先清空内容
                heroStatus.hp = 0;
                heroStatus.atk = 0;
                heroStatus.def = 0;
                heroStatus.mdef = 0;
                if (block.id === 'pickaxe') {
                    const idx = Math.min(tiles.items.length - 1, 0);
                    converted[ny][nx] = tiles.items[idx];
                    continue;
                } else if (block.id === 'bomb') {
                    const idx = Math.min(tiles.items.length - 1, 1);
                    converted[ny][nx] = tiles.items[idx];
                    continue;
                } else if (block.id === 'centerFly') {
                    const idx = Math.min(tiles.items.length - 1, 2);
                    converted[ny][nx] = tiles.items[idx];
                    continue;
                } else if (block.id === 'yellowKey') {
                    const idx = Math.min(tiles.items.length - 1, 0);
                    converted[ny][nx] = tiles.keys[idx];
                    continue;
                } else if (block.id === 'blueKey') {
                    const idx = Math.min(tiles.items.length - 1, 1);
                    converted[ny][nx] = tiles.keys[idx];
                    continue;
                } else if (block.id === 'redKey') {
                    const idx = Math.min(tiles.items.length - 1, 2);
                    converted[ny][nx] = tiles.keys[idx];
                    continue;
                } else if (block.id === 'greenKey') {
                    const idx = Math.min(tiles.items.length - 1, 3);
                    converted[ny][nx] = tiles.keys[idx];
                    continue;
                }
                // 执行道具效果
                if (item.cls === 'items' && item.itemEffect) {
                    try {
                        eval(item.itemEffect);
                    } catch {
                        // 执行失败就清空一下防止被误识别为宝石血瓶
                        heroStatus.hp = 0;
                        heroStatus.atk = 0;
                        heroStatus.def = 0;
                        heroStatus.mdef = 0;
                    }
                }
                const arr: [number, number, number, number] = [
                    heroStatus.hp,
                    heroStatus.atk,
                    heroStatus.def,
                    heroStatus.mdef
                ];
                let isResouce = false;
                // 对每个属性进行判断
                if (heroStatus.hp > 0) {
                    isResouce = true;
                    itemHpSet.add(heroStatus.hp);
                }
                if (heroStatus.atk > 0) {
                    isResouce = true;
                    itemAtkSet.add(heroStatus.atk);
                }
                if (heroStatus.def > 0) {
                    isResouce = true;
                    itemDefSet.add(heroStatus.def);
                }
                if (heroStatus.mdef > 0) {
                    isResouce = true;
                    itemMdefSet.add(heroStatus.mdef);
                }
                if (isResouce) {
                    itemMap.set(ny * width + nx, arr);
                    continue;
                } else {
                    converted[ny][nx] = tiles.empty;
                    continue;
                }
            }
            // 门信息，这种处理方式只能处理 2.7+ 的塔，老塔估计处理不了，不过老塔占比也不大，忽略就好了
            if (block.doorInfo && Object.keys(block.doorInfo.keys).length > 0) {
                if (block.id === 'specialDoor') {
                    converted[ny][nx] = tiles.specialDoors[0];
                    continue;
                } else if ('greenKey' in block.doorInfo.keys) {
                    const idx = Math.min(tiles.commonDoors.length - 1, 3);
                    converted[ny][nx] = tiles.commonDoors[idx];
                } else if ('redKey' in block.doorInfo.keys) {
                    const idx = Math.min(tiles.commonDoors.length - 1, 2);
                    converted[ny][nx] = tiles.commonDoors[idx];
                    continue;
                } else if ('blueKey' in block.doorInfo.keys) {
                    const idx = Math.min(tiles.commonDoors.length - 1, 1);
                    converted[ny][nx] = tiles.commonDoors[idx];
                    continue;
                } else if ('yellowKey' in block.doorInfo.keys) {
                    const idx = Math.min(tiles.commonDoors.length - 1, 0);
                    converted[ny][nx] = tiles.commonDoors[idx];
                    continue;
                } else {
                    // 其余视为绿门
                    const idx = Math.min(tiles.commonDoors.length - 1, 3);
                    converted[ny][nx] = tiles.commonDoors[idx];
                    continue;
                }
            }
            // 不可入和不可出现在还没办法处理，这两个还需要同时考虑背景前景
            const bgNum = floor.bgmap?.[ny]?.[nx] ?? 0;
            const bg2Num = floor.bg2map?.[ny]?.[nx] ?? 0;
            const fgNum = floor.fgmap?.[ny]?.[nx] ?? 0;
            const fg2Num = floor.fg2map?.[ny]?.[nx] ?? 0;
            const bgBlock = result.map[bgNum];
            const bg2Block = result.map[bg2Num];
            const fgBlock = result.map[fgNum];
            const fg2Block = result.map[fg2Num];
            if (
                block.cannotIn ||
                block.cannotOut ||
                bgBlock?.cannotIn ||
                bgBlock?.cannotOut ||
                bg2Block?.cannotIn ||
                bg2Block?.cannotOut ||
                fgBlock?.cannotIn ||
                fgBlock?.cannotOut ||
                fg2Block?.cannotIn ||
                fg2Block?.cannotOut
            ) {
                converted[ny][nx] = tiles.empty;
                hasCannotInOut = true;
                continue;
            }
            // 墙壁处理
            if (block.canPass) {
                converted[ny][nx] = tiles.empty;
                continue;
            } else {
                converted[ny][nx] = tiles.wall;
            }
        }
    }

    // 处理怪物
    const minEnemyValue = Math.min(...enemyMap.values());
    const maxEnemyValue = Math.max(...enemyMap.values());
    const enemyValueDelta = maxEnemyValue - minEnemyValue;

    if (enemyValueDelta <= 0) {
        // 如果怪物战斗力都一样的话...
        enemyMap.forEach((value, pos) => {
            const nx = pos % width;
            const ny = Math.floor(pos / width);
            converted[ny][nx] = tiles.enemies[0];
        });
    } else {
        enemyMap.forEach((value, pos) => {
            const nx = pos % width;
            const ny = Math.floor(pos / width);
            const ratio = (value - minEnemyValue) / enemyValueDelta;
            const idx = Math.min(
                Math.floor(ratio * tiles.enemies.length),
                tiles.enemies.length - 1
            );
            converted[ny][nx] = tiles.enemies[idx];
        });
    }

    // 处理宝石血瓶
    const minHpValue = Math.min(...itemHpSet);
    const maxHpValue = Math.max(...itemHpSet);
    const minAtkValue = Math.min(...itemAtkSet);
    const maxAtkValue = Math.max(...itemAtkSet);
    const minDefValue = Math.min(...itemDefSet);
    const maxDefValue = Math.max(...itemDefSet);
    const minMdefValue = Math.min(...itemMdefSet);
    const maxMdefValue = Math.max(...itemMdefSet);
    const hpValueDelta = maxHpValue - minHpValue;
    const atkValueDelta = maxAtkValue - minAtkValue;
    const defValueDelta = maxDefValue - minDefValue;
    const mdefValueDelta = maxMdefValue - minMdefValue;

    itemMap.forEach(([hp, atk, def, mdef], pos) => {
        const nx = pos % width;
        const ny = Math.floor(pos / width);
        // 资源判定为占比最大的那个
        // 如果只有一种资源且道具包含这种属性，全部使用最低的资源种类
        if (minHpValue === maxHpValue && hp > 0) {
            converted[ny][nx] = tiles.potions[0];
            return;
        }
        if (minAtkValue === maxAtkValue && atk > 0) {
            converted[ny][nx] = tiles.redGems[0];
            return;
        }
        if (minDefValue === maxDefValue && def > 0) {
            converted[ny][nx] = tiles.blueGems[0];
            return;
        }
        if (minMdefValue === maxMdefValue && mdef > 0) {
            converted[ny][nx] = tiles.greenGems[0];
            return;
        }
        const hpRatio = (hp - minHpValue) / hpValueDelta;
        const atkRatio = (atk - minAtkValue) / atkValueDelta;
        const defRatio = (def - minDefValue) / defValueDelta;
        const mdefRatio = (mdef - minMdefValue) / mdefValueDelta;

        // 判断资源种类
        const arr = [hpRatio, atkRatio, defRatio, mdefRatio];
        let maxIndex = 0;
        let maxRatio = 0;
        for (let i = 0; i < arr.length; i++) {
            if (arr[i] > maxRatio) {
                maxRatio = arr[i];
                maxIndex = i;
            }
        }
        // 转换图块，对于宝石来说，一共有三个级别的宝石，如果数值只有两种，那么需要额外判断下使用哪两个级别的宝石
        // 倍数差距大于 3 的就使用一级和三级宝石，小于等于 3 的就使用一级和二级宝石
        // 血瓶不做这个处理
        switch (maxIndex) {
            case 0: {
                // 血瓶
                const idx = Math.min(
                    Math.floor(hpRatio * tiles.potions.length),
                    tiles.potions.length - 1
                );
                converted[ny][nx] = tiles.potions[idx];
                break;
            }
            case 1: {
                // 红宝石，这里不可能只有一种数值了，不需要判断
                const idx = Math.min(
                    Math.floor(atkRatio * tiles.redGems.length),
                    tiles.redGems.length - 1
                );
                converted[ny][nx] = tiles.redGems[idx];
                break;
            }
            case 2: {
                // 蓝宝石，这里不可能只有一种数值了，不需要判断
                const idx = Math.min(
                    Math.floor(defRatio * tiles.blueGems.length),
                    tiles.blueGems.length - 1
                );
                converted[ny][nx] = tiles.blueGems[idx];
                break;
            }
            case 2: {
                // 绿宝石，这里不可能只有一种数值了，不需要判断
                const idx = Math.min(
                    Math.floor(mdefRatio * tiles.greenGems.length),
                    tiles.greenGems.length - 1
                );
                converted[ny][nx] = tiles.greenGems[idx];
                break;
            }
        }
    });

    for (let nx = 0; nx < width; nx++) {
        for (let ny = 0; ny < height; ny++) {
            if (
                typeof converted[ny][nx] !== 'number' ||
                isNaN(converted[ny][nx]) ||
                !isFinite(converted[ny][nx]) ||
                converted[ny][nx] < 0
            ) {
                converted[ny][nx] = tiles.empty;
            }
        }
    }

    return {
        map: converted,
        hasCannotInOut
    };
}
