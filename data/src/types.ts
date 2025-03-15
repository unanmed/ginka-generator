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
