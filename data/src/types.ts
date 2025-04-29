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

export interface GinkaData {
    tag: number[];
}

export interface GinkaConfig extends BaseConfig {
    data: Record<string, GinkaData>;
    mapping: {
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        redGem: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        blueGem: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        greenGem: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        yellowGem: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        item: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        potion: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        key: Record<number, number>;
        /** 键表示图块，值表示等级或其增加的属性值，0是最低级，以此类推 */
        door: Record<number, number>;
        floor: number[];
        arrow: number[];
        wall: number[];
        decoration: number[];
    };
}

export interface GinkaTrainData {
    tag: number[];
    val: number[];
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
