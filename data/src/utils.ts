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

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
        throw new Error('Vectors must have same dimension');
    }

    let dot = 0,
        normA = 0,
        normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] ** 2;
        normB += vecB[i] ** 2;
    }

    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function parseTowerInfo(
    path: string,
    configName: string
): Promise<TowerInfo> {
    const dataFile = await readFile(join(path, 'data.js'), 'utf-8');
    const data: any = JSON.parse(dataFile.split('\n').slice(1).join('\n'));
    const configFile = await readFile(join(path, configName), 'utf-8');

    return {
        path: path,
        name: data.firstData.name as string,
        floorIds: data.main.floorIds as string[],
        config: JSON.parse(configFile) as BaseConfig
    };
}

export function mergeFloorIds(...info: TowerInfo[]) {
    const ids: string[] = [];
    info.forEach(v => {
        ids.push(...v.floorIds.map(id => `${v.name}:${id}`));
    });
    return ids;
}

export async function fromJSON(path: string) {
    const file = await readFile(path, 'utf-8');
    const data = JSON.parse(file) as Record<string, number[][]>;
    const clip: Record<string, [number, number, number, number]> = {};
    const config: BaseConfig = {
        clip: {
            defaults: [0, 0, 0, 0],
            special: clip
        }
    };
    const name = (Math.random() * 12).toFixed(0);
    const floorMap = new Map<string, FloorData>();
    for (const [key, value] of Object.entries(data)) {
        const floorData: FloorData = {
            map: value,
            id: key,
            config
        };
        floorMap.set(`${name}:${key}`, floorData);
    }
    return floorMap;
}

export function chooseFrom<T>(arr: T[], n: number): T[] {
    const copy = arr.slice();
    for (let i = copy.length - 1; i > 0; i--) {
        let randIndex = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[randIndex]] = [copy[randIndex], copy[i]];
    }
    return copy.slice(0, n);
}
