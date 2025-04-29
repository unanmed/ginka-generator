import { SingleBar, Presets } from 'cli-progress';
import { getGinkaRatio } from 'src/floor';
import { GinkaTrainData, GinkaConfig, GinkaDataset } from 'src/types';
import { FloorData } from 'src/utils';

export function parseGinka(data: Map<string, FloorData>) {
    const resolved: Record<string, GinkaTrainData> = {};

    const progress = new SingleBar({}, Presets.shades_classic);
    progress.start(data.size, 0);
    let i = 0;

    data.forEach((floor, key) => {
        const config = floor.config as GinkaConfig;
        const data = config.data[floor.id] ?? {
            tag: Array(64).fill(0)
        };
        resolved[key] = {
            map: floor.map,
            size: [floor.map[0].length, floor.map.length],
            tag: data.tag,
            val: getGinkaRatio(floor.map)
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
