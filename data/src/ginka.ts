import { writeFile } from 'fs-extra';
import { FloorData, getAllFloors, parseTowerInfo } from './utils';
import { Presets, SingleBar } from 'cli-progress';

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

function parseAllData(data: Map<string, FloorData>) {
    const resolved: Record<string, GinkaTrainData> = {};

    const progress = new SingleBar({}, Presets.shades_classic);
    progress.start(data.size, 0);
    let i = 0;

    data.forEach((floor, key) => {
        const config = floor.config as GinkaConfig;
        const text = config.data[floor.id] ?? [];
        resolved[key] = {
            map: floor.map,
            size: [floor.map[0].length, floor.map.length],
            text: text
        };
        i++;
        progress.update(i);
    });

    const dataset: GinkaDataset = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: resolved
    };

    return dataset;
}

(async () => {
    const towers = await Promise.all(
        list.map(v => parseTowerInfo(v, 'ginka-config.json'))
    );
    const floors = await getAllFloors(...towers);
    const results = parseAllData(floors);
    await writeFile(output, JSON.stringify(results, void 0), 'utf-8');
    const size = Object.keys(results.data).length;
    console.log(`✅ 已处理 ${list.length} 个塔，共 ${size} 个地图`);
})();
