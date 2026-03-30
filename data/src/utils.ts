import { readFile } from 'fs-extra';
import { join } from 'path';
import { BaseConfig, GinkaConfig, TowerInfo } from './types';

export interface DatasetMergable<T> {
    datasetId: number;
    data: Record<string, T>;
}

export interface FloorData {
    map: number[][];
    id: string;
    config: BaseConfig;
}

export function mergeDataset<T>(
    allowDuplicateKeys: boolean,
    ...datasets: DatasetMergable<T>[]
): DatasetMergable<T> {
    if (datasets.length === 1) {
        return datasets[0];
    }
    const usedKeys = new Set<string>();
    const data: Record<string, T> = {};
    datasets.forEach(v => {
        for (const [key, value] of Object.entries(v.data)) {
            if (usedKeys.has(key) && allowDuplicateKeys) {
                const dataKey = `${v.datasetId}/${key}`;
                data[dataKey] = value;
                usedKeys.add(dataKey);
            } else {
                data[key] = value;
                usedKeys.add(key);
            }
        }
    });

    const dataset: DatasetMergable<T> = {
        datasetId: Math.floor(Math.random() * 1e12),
        data: data
    };

    return dataset;
}

export function chooseFrom<T>(arr: T[], n: number): T[] {
    const copy = arr.slice();
    for (let i = copy.length - 1; i > 0; i--) {
        let randIndex = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[randIndex]] = [copy[randIndex], copy[i]];
    }
    return copy.slice(0, n);
}
