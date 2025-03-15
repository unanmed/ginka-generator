import { readFile } from 'fs-extra';
import { join } from 'path';
import { BaseConfig, TowerInfo } from './types';
import { convertFloor } from './floor';

interface DatasetMergable<T> {
    datasetId: number;
    data: Record<string, T>;
}

export interface FloorData {
    map: number[][];
    config: BaseConfig;
}

export function mergeDataset<T>(
    ...datasets: DatasetMergable<T>[]
): DatasetMergable<T> {
    const data: Record<string, T> = {};
    datasets.forEach(v => {
        for (const [key, value] of Object.entries(v.data)) {
            const dataKey = `${v.datasetId}/${key}`;
            data[dataKey] = value;
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

export async function getAllFloors(...info: TowerInfo[]) {
    const floorData = await Promise.all(
        info.map(tower => {
            return Promise.all(
                tower.floorIds.map(async id => {
                    const floorFile = await readFile(
                        join(tower.path, 'floors', `${id}.js`),
                        'utf-8'
                    );
                    const data = JSON.parse(
                        floorFile.split('\n').slice(1).join('\n')
                    );
                    const map = data.map as number[][];
                    // 裁剪地图
                    const { clip } = tower.config;
                    const area = clip.special[id] ?? clip.defaults;
                    return convertFloor(map, area, tower.name, id);
                })
            );
        })
    );
    const maps: Map<string, FloorData> = new Map();
    floorData.forEach((tower, tid) => {
        const name = info[tid].name;
        tower.forEach((map, mid) => {
            const floorId = info[tid].floorIds[mid];
            maps.set(`${name}::${floorId}`, { map, config: info[tid].config });
        });
    });
    return maps;
}

export function mergeFloorIds(...info: TowerInfo[]) {
    const ids: string[] = [];
    info.forEach(v => {
        ids.push(...v.floorIds.map(id => `${v.name}:${id}`));
    });
    return ids;
}
