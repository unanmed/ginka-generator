import { GinkaTopologicalGraphs } from 'src/topology/interface';

export const enum TowerColor {
    White,
    Orange,
    Blue,
    Green,
    Red,
    Purple
}

export interface IConvertedMap {
    /** 地图矩阵 */
    readonly map: number[][];
    /** 是否包含不可入不可出图块 */
    readonly hasCannotInOut: boolean;
}

export interface IConvertedMapInfo {
    /** 转换后的地图 */
    readonly data: IConvertedMap;
    /** 地图信息 */
    readonly info: IFloorInfo;
    /** 地图所属塔信息 */
    readonly tower: ITowerInfo;
    /** 地图 id */
    readonly mapId: string;
}

export interface ITowerInfo {
    /** 作者 id */
    readonly authorId: number;
    /** 塔颜色 */
    readonly color: TowerColor;
    /** 评论数量 */
    readonly comment: number;
    /** 是否是比赛塔 */
    readonly competition: boolean;
    /** 楼层数量 */
    readonly floors: number;
    /** 塔的数字 id */
    readonly id: number;
    /** 塔的英文名 */
    readonly name: string;
    /** 塔的游玩量 */
    readonly people: number;
    /** 塔标签，每一项是对应的标签名 */
    readonly tag: string[];
    /** 塔的名称 */
    readonly title: string;
    /** 测试员列表 */
    readonly topuser: number[];
    /** 通关人数 */
    readonly win: number;
    /** 精美评分，第一项是评分结果，后面五项是选择每个评分的人数 */
    readonly designrate: number[];
    /** 难度评分，第一项是评分结果，后面五项是选择每个评分的人数 */
    readonly hardrate: number[];
}

export interface IFloorInfo {
    /** 楼层所属的塔信息 */
    readonly tower: ITowerInfo;
    /** 楼层拓扑图 */
    readonly topo: GinkaTopologicalGraphs;
    /** 地图矩阵 */
    readonly map: number[][];
    /** 地图整体密度，非空白图块/地图面积 */
    readonly globalDensity: number;
    /** 墙壁密度，墙壁数量/地图面积 */
    readonly wallDensity: number;
    /** 门密度，门数量/地图面积 */
    readonly doorDensity: number;
    /** 怪物密度，怪物数量/地图面积 */
    readonly enemyDensity: number;
    /** 资源密度，资源数量/地图面积，资源包括宝石、血瓶、道具、钥匙 */
    readonly resourceDensity: number;
    /** 宝石密度，宝石数量/地图面积 */
    readonly gemDensity: number;
    /** 血瓶密度，血瓶数量/地图面积 */
    readonly potionDensity: number;
    /** 钥匙密度，钥匙数量/地图面积 */
    readonly keyDensity: number;
    /** 道具密度，道具数量/地图面积，道具指破炸飞这些内容 */
    readonly itemDensity: number;
    /** 入口数量 */
    readonly entryCount: number;
    /** 机关门数量 */
    readonly specialDoorCount: number;
    /** 咸鱼门数量，多层咸鱼门算一个 */
    readonly fishCount: number;
    /** 是否包含只连接了一个节点的分支节点。这种节点相当于门或怪物后面什么都不加，多数是无用的。 */
    readonly hasUselessBranch: boolean;
    /** 墙壁密度标准差 */
    readonly wallDensityStd: number;
}

export interface IAutoLabelConfig {
    /** 地图允许大小 */
    readonly allowedSize: [number, number][];
    /** 是否允许无用节点 */
    readonly allowUselessBranch: boolean;
    /** 最小怪物占比 */
    readonly minEnemyRatio: number;
    /** 最大怪物占比 */
    readonly maxEnemyRatio: number;
    /** 最小墙壁占比 */
    readonly minWallRatio: number;
    /** 最大墙壁占比 */
    readonly maxWallRatio: number;
    /** 血瓶+宝石+道具+钥匙之和最小占比 */
    readonly minResourceRatio: number;
    /** 血瓶+宝石+道具+钥匙之和最小占比 */
    readonly maxResourceRatio: number;
    /** 最小门占比 */
    readonly minDoorRatio: number;
    /** 最大门占比 */
    readonly maxDoorRatio: number;
    /** 最小咸鱼门数量，多层咸鱼门算一个 */
    readonly minFishCount: number;
    /** 最大咸鱼门数量，多层咸鱼门算一个 */
    readonly maxFishCount: number;
    /** 最小入口数量 */
    readonly minEntryCount: number;
    /** 最大入口数量 */
    readonly maxEntryCount: number;

    /** 最大墙壁密度标准差，用于描述一个地图墙壁分布是否均匀的，较大的时候可能是特殊地图，不符合要求 */
    readonly maxWallDensityStd: number;

    /** 是否忽略问题 */
    readonly ignoreIssues: boolean;

    /**
     * 自定义塔过滤器
     * @param info 塔信息
     */
    customTowerFilter?: (info: ITowerInfo) => boolean;

    /**
     * 自定义楼层过滤器
     * @param floor 楼层信息
     */
    customFloorFilter?: (floor: IConvertedMapInfo) => boolean;
}

export interface INeededCoreData {
    readonly main: {
        readonly floorIds: readonly string[];
    };
    readonly firstData: {
        readonly name: string;
    };
    readonly values: {
        readonly redGem: number;
        readonly blueGem: number;
        readonly greenGem: number;
        readonly redPotion: number;
        readonly bluePotion: number;
        readonly greenPotion: number;
        readonly yellowPotion: number;
    };
}

export interface INeededEnemyData {
    readonly hp: number;
    readonly atk: number;
    readonly def: number;
}

export interface INeededMapData {
    readonly cls: string;
    readonly id: string;
    readonly canPass?: boolean;
    readonly trigger?: string;
    readonly script?: string;
    readonly cannotOut?: string[];
    readonly cannotIn?: string[];
    readonly doorInfo?: {
        readonly keys: Record<string, number>;
    };
}

export interface INeededItemData {
    readonly cls: string;
    readonly useItemEvent?: any;
    readonly itemEffect?: string;
    readonly canUseItemEffect?: string;
    readonly equip?: {
        readonly value?: Record<string, number>;
        readonly percentage?: Record<string, number>;
    };
}

export interface INeededFloorData {
    readonly floorId: string;
    readonly map: number[][];
    readonly bgmap?: number[][];
    readonly bg2map?: number[][];
    readonly fgmap?: number[][];
    readonly fg2map?: number[][];
    readonly changeFloor: Record<string, unknown>;
}

export interface ICodeRunResult {
    issue: string[];
    data: INeededCoreData;
    enemy: Record<string, INeededEnemyData>;
    map: Record<number, INeededMapData>;
    item: Record<string, INeededItemData>;
    main: {
        floors: Record<string, INeededFloorData>;
    };
}
