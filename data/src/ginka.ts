import { exists, writeFile } from 'fs-extra';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { convertFloor } from './floor';
import { mergeDataset } from './utils';

interface GinkaConfig {
    clip: {
        defaults: [number, number, number, number];
        special: Record<string, [number, number, number, number]>;
    };
    data: Record<string, string[]>;
}

interface GinkaTrainData {
    text: string[];
    map: number[][];
    size: [number, number];
}

interface GinkaDataset {
    datasetId: number;
    data: Record<string, GinkaTrainData>;
}

const [output, ...list] = process.argv.slice(2);

async function parseOneFloor(
    path: string,
    name: string,
    floorId: string,
    config: GinkaConfig
): Promise<GinkaTrainData> {
    const floorFile = await readFile(path, 'utf-8');
    const floor: any = JSON.parse(floorFile.split('\n').slice(1).join('\n'));
    const map = floor.map as number[][];
    const clip = config.clip.special[floorId] ?? config.clip.defaults;

    const clipped = convertFloor(map, clip, name, floorId);

    if (!config.data[floorId]) {
        console.log(`⚠️  魔塔 ${name} 的楼层 ${floorId} 不存在描述文本！`);
    }

    const data: GinkaTrainData = {
        text: config.data[floorId],
        map: clipped,
        size: [clipped[0].length, clipped.length]
    };

    return data;
}

async function parseOne(path: string): Promise<GinkaDataset> {
    const dataFile = await readFile(join(path, 'data.js'), 'utf-8');
    const configFile = await readFile(join(path, 'ginka-config.json'), 'utf-8');
    const data: any = JSON.parse(dataFile.split('\n').slice(1).join('\n'));
    const config = JSON.parse(configFile) as GinkaConfig;
    const floorIds = data.main.floorIds as string[];
    const name = data.firstData.name as string;

    const datas = await Promise.all(
        floorIds.map(v =>
            parseOneFloor(join(path, 'floors', `${v}.js`), name, v, config)
        )
    );

    const dataset: GinkaDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: Object.fromEntries(datas.map((v, i) => [floorIds[i], v]))
    };

    return dataset;
}

(async () => {
    const results = await Promise.all(list.map(v => parseOne(v)));
    const dataset = mergeDataset(...results);
    await writeFile(output, JSON.stringify(dataset, void 0), 'utf-8');
    const size = Object.keys(dataset.data).length;
    console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个地图`);
})();
