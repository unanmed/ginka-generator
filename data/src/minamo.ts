import { writeFile } from 'fs-extra';
import { FloorData, readOne, getAllFloors, parseTowerInfo } from './utils';
import { compareMap } from './topology/compare';
import { mirrorMapX, mirrorMapY, rotateMap } from './topology/transform';
import { directions, tileType } from './topology/graph';
import { calculateVisualSimilarity } from './vision/similarity';
import { BaseConfig } from './types';
import { Presets, SingleBar } from 'cli-progress';
import { log } from 'console';

interface MinamoConfig extends BaseConfig {}

interface MinamoTrainData {
    map1: number[][];
    map2: number[][];
    topoSimilarity: number;
    visionSimilarity: number;
    size: [number, number];
}

interface MinamoDataset {
    datasetId: number;
    data: Record<string, MinamoTrainData>;
}

const [output, ...list] = process.argv.slice(2);
// 判断 assigned 模式，此模式下只会对前两个塔处理，会在这两个塔之间对比，而单个塔的地图不会对比
const assigned = list.at(-1) === 'assigned';
if (assigned) list.pop();

function chooseFrom<T>(arr: T[], n: number): T[] {
    const copy = arr.slice();
    for (let i = copy.length - 1; i > 0; i--) {
        let randIndex = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[randIndex]] = [copy[randIndex], copy[i]];
    }
    return copy.slice(0, n);
}

function chooseN(maxCount: number, n: number) {
    return chooseFrom(
        Array(maxCount)
            .fill(0)
            .map((_, i) => i),
        n
    );
}

function choosePair(n: number, max: number = 1000) {
    const totalCount = Math.round((n * (n - 1)) / 2);
    const count = Math.min(totalCount, max);
    const pairs: number[] = [];
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            pairs.push(i * n + j);
        }
    }
    // 直接打乱后取前 count 个
    for (let i = pairs.length - 1; i > 0; i--) {
        let randIndex = Math.floor(Math.random() * (i + 1));
        [pairs[i], pairs[randIndex]] = [pairs[randIndex], pairs[i]];
    }

    return pairs.slice(0, count);
}

function transform(map: number[][], rot: number, flip: number) {
    let res = map;
    for (let i = 0; i < rot; i++) {
        res = rotateMap(res);
    }
    if (flip & 0b01) {
        res = mirrorMapX(res);
    }
    if (flip & 0b10) {
        res = mirrorMapY(res);
    }
    return res;
}

function generateTransformData(
    id1: string,
    id2: string,
    map1: number[][],
    map2: number[][],
    simi: number
) {
    const types: [rot: number, flip: number][] = [];
    for (const rot of [0, 1, 2, 3]) {
        for (const flip of [0b00, 0b01, 0b10, 0b11]) {
            if (rot === 0 && flip === 0) continue;
            types.push([rot, flip]);
        }
    }
    // 随机抽取最多两个
    const trans = chooseFrom(types, Math.floor(Math.random() * 2));
    return trans
        .map(([rot, flip]) => {
            const com1 = `${id1}.${rot}.${flip}:${id1}`;
            const com2 = `${id1}.${rot}.${flip}:${id2}`;
            const com3 = `${id2}.${rot}.${flip}:${id1}`;
            const com4 = `${id2}.${rot}.${flip}:${id2}`;
            const choose = chooseFrom(
                [com1, com2, com3, com4],
                Math.floor(Math.random() * 2)
            );
            const res: [id: string, data: MinamoTrainData][] = [];
            if (choose.includes(com1)) {
                const t = transform(map1, rot, flip);
                res.push([
                    com1,
                    {
                        map1: t,
                        map2: map1,
                        topoSimilarity: 1,
                        visionSimilarity: calculateVisualSimilarity(map1, t),
                        size: [map1[0].length, map1.length]
                    }
                ]);
            }
            if (choose.includes(com2)) {
                const t = transform(map1, rot, flip);
                res.push([
                    com2,
                    {
                        map1: t,
                        map2: map2,
                        topoSimilarity: simi,
                        visionSimilarity: calculateVisualSimilarity(t, map2),
                        size: [map1[0].length, map1.length]
                    }
                ]);
            }
            if (choose.includes(com3)) {
                const t = transform(map2, rot, flip);
                res.push([
                    com3,
                    {
                        map1: t,
                        map2: map1,
                        topoSimilarity: simi,
                        visionSimilarity: calculateVisualSimilarity(t, map1),
                        size: [map1[0].length, map1.length]
                    }
                ]);
            }
            if (choose.includes(com4)) {
                const t = transform(map2, rot, flip);
                res.push([
                    com4,
                    {
                        map1: t,
                        map2: map2,
                        topoSimilarity: 1,
                        visionSimilarity: calculateVisualSimilarity(t, map2),
                        size: [map1[0].length, map1.length]
                    }
                ]);
            }

            return res;
        })
        .flat();
}

function generateSimilarData(id: string, map: number[][]) {
    // 生成最多五个微调地图
    const width = map[0].length;
    const height = map.length;
    const num = Math.floor(Math.random() * 3);
    const res: [id: string, data: MinamoTrainData][] = [];

    for (let i = 0; i < num; i++) {
        const clone = map.map(v => v.slice());
        const prob = Math.random() * 0.3;
        for (let ny = 0; ny < height; ny++) {
            for (let nx = 0; nx < width; nx++) {
                if (Math.random() > prob) {
                    // 有一定的概率进行微调
                    continue;
                }
                if (Math.random() < 0.2) {
                    // 20% 概率与旁边图块互换位置
                    const [dx, dy] =
                        directions[
                            Math.floor(Math.random() * directions.length)
                        ];
                    const px = nx + dx;
                    const py = ny + dy;
                    if (px < 0 || px >= width || py < 0 || py >= height) {
                        continue;
                    }
                    [clone[ny][nx], clone[py][px]] = [
                        clone[py][px],
                        clone[ny][nx]
                    ];
                } else {
                    // 80% 概率替换当前图块
                    clone[ny][nx] = Math.floor(Math.random() * tileType.size);
                }
            }
        }
        const id2 = `${id}.S${i}`;
        const sid = `${id}:${id2}`;
        const simi = compareMap(id, id2, map, clone);
        res.push([
            sid,
            {
                map1: map,
                map2: clone,
                size: [width, height],
                topoSimilarity: simi,
                visionSimilarity: calculateVisualSimilarity(map, clone)
            }
        ]);
    }
    return res;
}

function generatePair(
    data: Record<string, MinamoTrainData>,
    id1: string,
    id2: string,
    map1: number[][],
    map2: number[][],
    size: [number, number]
) {
    const topoSimilarity = compareMap(id1, id2, map1, map2);
    const visionSimilarity = calculateVisualSimilarity(map1, map2);
    const train: MinamoTrainData = {
        map1,
        map2,
        topoSimilarity,
        visionSimilarity,
        size: size
    };
    data[`${id1}:${id2}`] = train;
    // 自身与自身对比的训练集，保证模型对相同地图输出 1
    const self1 = `${id1}:${id1}`;
    const self2 = `${id2}:${id2}`;
    const selfTrain = chooseFrom([self1, self2], Math.floor(Math.random() * 3));
    if (selfTrain.includes(self1) && !data[`${id1}:${id1}`]) {
        const selfTrain1: MinamoTrainData = {
            map1: map1,
            map2: map1,
            topoSimilarity: 1,
            visionSimilarity: 1,
            size: size
        };
        data[`${id1}:${id1}`] = selfTrain1;
    }
    if (selfTrain.includes(self2) && !data[`${id2}:${id2}`]) {
        const selfTrain2: MinamoTrainData = {
            map1: map2,
            map2: map2,
            topoSimilarity: 1,
            visionSimilarity: 1,
            size: size
        };
        data[`${id2}:${id2}`] = selfTrain2;
    }
    // 翻转、旋转训练集
    Object.assign(
        data,
        Object.fromEntries(
            generateTransformData(id1, id2, map1, map2, topoSimilarity)
        )
    );
    // 地图微调训练集
    Object.assign(data, Object.fromEntries(generateSimilarData(id1, map1)));
}

function generateDataset(
    floors: Map<string, FloorData>,
    pairs: number[],
    floorIds: string[]
): Record<string, MinamoTrainData> {
    const data: Record<string, MinamoTrainData> = {};

    const progress = new SingleBar({}, Presets.shades_classic);

    progress.start(pairs.length, 0);

    pairs.forEach((v, i) => {
        const num1 = Math.floor(v / floorIds.length);
        const num2 = v % floorIds.length;
        const id1 = floorIds[num1];
        const id2 = floorIds[num2];
        const map1 = floors.get(id1)?.map;
        const map2 = floors.get(id2)?.map;
        if (!map1 || !map2) return;
        const [w1, h1] = [map1[0].length, map1.length];
        const [w2, h2] = [map2[0].length, map2.length];
        if (w1 !== w2 || h1 !== h2) return;
        generatePair(data, id1, id2, map1, map2, [w1, h1]);
        progress.update(i + 1);
    });

    progress.stop();

    return data;
}

function parseAllData(data: Map<string, FloorData>): MinamoDataset {
    const length = data.size;
    const totalCount = Math.round((length * (length - 1)) / 2);

    const pairs = choosePair(length, 10000);

    console.log(
        `✅ 共发现 ${length} 个楼层，共 ${totalCount} 种组合，选取 ${pairs.length} 个组合`
    );

    const trainData = generateDataset(data, pairs, [...data.keys()]);

    const dataset: MinamoDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: trainData
    };

    return dataset;
}

function generateAssignedData(
    data1: Map<string, FloorData>,
    data2: Map<string, FloorData>
): MinamoDataset {
    const length = data1.size + data2.size;
    const totalCount = data1.size * data2.size;
    const count1 = Math.min(100, data1.size);
    const count2 = Math.min(100, data2.size);
    const keys1 = [...data1.keys()];
    const keys2 = [...data2.keys()];
    const choose1 = chooseFrom(keys1, count1);

    const trainData: Record<string, MinamoTrainData> = {};

    console.log(
        `✅ 共发现 ${length} 个楼层，共 ${totalCount} 种组合，选取 ${
            count1 * count2
        } 个组合`
    );

    const progress = new SingleBar({}, Presets.shades_classic);
    progress.start(count1 * count2, 0);
    let n = 0;

    for (const key1 of choose1) {
        const choose2 = chooseFrom(keys2, count2);
        for (const key2 of choose2) {
            const { map: map1 } = data1.get(key1)!;
            const { map: map2 } = data2.get(key2)!;
            if (!map1 || !map2) continue;
            const [w1, h1] = [map1[0].length, map1.length];
            const [w2, h2] = [map2[0].length, map2.length];
            if (w1 !== w2 || h1 !== h2) continue;
            generatePair(trainData, key1, key2, map1, map2, [w1, h1]);
            n++;
            progress.update(n);
        }
    }

    progress.stop();

    const dataset: MinamoDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: trainData
    };

    return dataset;
}

(async () => {
    if (!assigned) {
        const towers = await Promise.all(
            list.map(v => parseTowerInfo(v, 'minamo-config.json'))
        );
        const floors = await getAllFloors(...towers);
        const results = parseAllData(floors);
        await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
        const size = Object.keys(results.data).length;
        console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个组合`);
    } else {
        const [tower1, tower2] = list;
        if (!tower1 || !tower2) {
            console.log(`⚠️  assigned 模式下必须传入两个塔！`);
            return;
        }
        const data1 = await readOne(tower1);
        const data2 = await readOne(tower2);
        const results = generateAssignedData(data1, data2);
        await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
        const size = Object.keys(results.data).length;
        console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个组合`);
    }
})();
