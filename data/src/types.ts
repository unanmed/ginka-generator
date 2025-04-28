export interface BaseConfig {
    clip: {
        defaults: [number, number, number, number];
        special: Record<string, [number, number, number, number]>;
    };
}

export interface TowerInfo {
    path: string;
    name: string;
    floorIds: string[];
    config: BaseConfig;
}

export interface GinkaConfig extends BaseConfig {
    data: Record<string, string[]>;
    mapping: {
        redGem: number[];
        blueGem: number[];
        greenGem: number[];
        yellowGem: number[];
        item: number[];
        potion: number[];
        key: number[];
        door: number[];
        wall: number[];
    };
}

export interface GinkaTrainData {
    text: string[];
    map: number[][];
    size: [number, number];
}

export interface GinkaDataset {
    datasetId: number;
    data: Record<string, GinkaTrainData>;
}

export interface MinamoConfig extends BaseConfig {}

export interface MinamoTrainData {
    map1: number[][];
    map2: number[][];
    topoSimilarity: number;
    visionSimilarity: number;
    size: [number, number];
}

export interface MinamoDataset {
    datasetId: number;
    data: Record<string, MinamoTrainData>;
}
