import { SingleBar, Presets } from 'cli-progress';
import { GinkaTrainData, GinkaConfig, GinkaDataset } from 'src/types';
import { FloorData } from 'src/utils';

export function parseGinka(data: Map<string, FloorData>) {
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
