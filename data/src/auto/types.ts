import { GinkaTopologicalGraphs } from '../topology/interface';

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
    readonly topo: IMapTopology;
    /** 最大空地面积 */
    readonly maxEmptyArea: number;
    /** 最大资源区域面积 */
    readonly maxResourceArea: number;
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
    /** 是否包含只连接了一个节点的空白节点。这种节点相当于门或怪物后面什么都不加，多数是无用的。 */
    readonly hasUselessBranch: boolean;
    /** 墙壁密度标准差 */
    readonly wallDensityStd: number;

    /** 墙壁热力图 */
    readonly wallHeatmap: number[][];
    /** 怪物热力图 */
    readonly enemyHeatmap: number[][];
    /** 资源热力图 */
    readonly resourceHeatmap: number[][];
    /** 血瓶热力图 */
    readonly potionHeatmap: number[][];
    /** 宝石热力图 */
    readonly gemHeatmap: number[][];
    /** 钥匙热力图 */
    readonly keyHeatmap: number[][];
    /** 道具热力图 */
    readonly itemHeatmap: number[][];
    /** 入口热力图 */
    readonly entryHeatmap: number[][];
    /** 门热力图 */
    readonly doorHeatmap: number[][];
}

export interface IMapBlockConfig {
    /** 空地的图块数字 */
    readonly empty: number;
    /** 墙的图块数字 */
    readonly wall: number;
    /** 装饰图块 */
    readonly decoration: number;
    /**
     * 普通门，黄、蓝、红、绿等级依次增加，如果数组长度不够，会取最高级的那个，
     * 对于四种基础门之外的门，会进入 `specialDoors` 判断
     */
    readonly commonDoors: number[];
    /** 机关门，普通机关门使用第一项作为结果，对于四种基础门及机关门之外的门，会选用第二项作为结果 */
    readonly specialDoors: [number, number];
    /**
     * 钥匙，黄、蓝、红、绿等级依次增加，如果数组长度不够，会取最高级的那个，
     * 特殊钥匙视为最高级
     */
    readonly keys: number[];
    /** 红宝石图块，平均分为若干档 */
    readonly redGems: number[];
    /** 蓝宝石图块，平均分为若干档 */
    readonly blueGems: number[];
    /** 绿宝石图块，平均分为若干档 */
    readonly greenGems: number[];
    /** 血瓶图块，平均分为若干档 */
    readonly potions: number[];
    /** 道具图块，依次为破炸飞，如果长度不够则取最后一个 */
    readonly items: number[];
    /** 怪物图块，按怪物强度分为若干当，强度为血量乘攻防和 */
    readonly enemies: number[];
    /** 入口图块 */
    readonly entry: number;
}

export interface IAutoLabelConfig {
    /** 地图图块分类 */
    readonly classes: Readonly<IMapBlockConfig>;
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
    /** 最大空地区域面积，超过这个的地图会忽略 */
    readonly maxEmptyArea: number;
    /** 最大资源区域面积，超过这个的地图会忽略 */
    readonly maxResourceArea: number;
    /** 热力图统计算子 */
    readonly heatmapKernel: number;
    /** 热力图高斯模糊的标准差 */
    readonly guassainRadius: number;

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
    readonly events?: Record<string, any[] | { readonly noPass: boolean }>;
    readonly cannotMove?: ('left' | 'right' | 'up' | 'down')[];
    readonly cannotMoveIn?: ('left' | 'right' | 'up' | 'down')[];
}

export interface ICodeRunResult {
    issue: string[];
    data: INeededCoreData;
    enemy: Record<string, INeededEnemyData>;
    /** Tile 数字到其内容的映射 */
    map: Record<number, INeededMapData>;
    item: Record<string, INeededItemData>;
    main: {
        floors: Record<string, INeededFloorData>;
    };
}

export const enum CannotInOut {
    /** 左侧不可入 / 不可出 */
    Left = 0b0001,
    /** 上侧不可入 / 不可出 */
    Top = 0b0010,
    /** 右侧不可入 / 不可出 */
    Right = 0b0100,
    /** 下侧不可入 / 不可出 */
    Bottom = 0b1000
}

export const enum GraphNodeType {
    Empty,
    /** 资源节点，由资源组成 */
    Resource,
    /** 分支节点，由门或怪物组成 */
    Branch,
    /** 入口节点 */
    Entry,
    /** 墙 */
    Wall
}

export const enum ResourceType {
    Hp,
    Atk,
    Def,
    Mdef,
    Item,
    Key
}

export const enum BranchType {
    Door,
    Enemy
}

export interface IMapGraphNodeBase {
    /** 节点类型 */
    readonly type: GraphNodeType;
    /** 当前节点在拓扑图中的索引 */
    readonly index: number;
    /** 此节点包含的所有地图坐标 */
    readonly tiles: Set<number>;
    /** 当前节点的邻居节点 */
    readonly neighbors: Set<MapGraphNode>;
}

export interface IEmptyMapGraphNode extends IMapGraphNodeBase {
    readonly type: GraphNodeType.Empty;
}

export interface IResourceMapGraphNode extends IMapGraphNodeBase {
    readonly type: GraphNodeType.Resource;
    /** 节点包含的资源数量 */
    readonly resources: Map<ResourceType, number>;
}

export interface IBranchMapGraphNode extends IMapGraphNodeBase {
    readonly type: GraphNodeType.Branch;
    /** 分支节点类型 */
    readonly branch: BranchType;
}

export interface IEntryMapGraphNode extends IMapGraphNodeBase {
    readonly type: GraphNodeType.Entry;
}

export type MapGraphNode =
    | IEmptyMapGraphNode
    | IResourceMapGraphNode
    | IBranchMapGraphNode
    | IEntryMapGraphNode;

export interface IMapGraphArea {
    /** 当前区域包含的所有节点 */
    readonly nodes: Set<MapGraphNode>;
}

export interface IMapGraph {
    /** 不可到达的区域 */
    readonly unreachableArea: Set<IMapGraphArea>;
    /** 当前拓扑图包含的区域信息 */
    readonly areas: Set<IMapGraphArea>;
    /** 当前拓扑图的所有入口 */
    readonly entries: Set<IEntryMapGraphNode>;
    /** 坐标至其所在节点的映射，可以根据坐标获取其对应的节点 */
    readonly nodeMap: Map<number, MapGraphNode>;
}

export interface IMapTopology {
    /** 原始地图 */
    readonly originMap: number[][];
    /** 事件层除外的层的地图 */
    readonly otherLayersMap: number[][][];
    /** 经过转换的地图 */
    readonly convertedMap: number[][];
    /** 不可通行标记 */
    readonly noPass: boolean[][];
    /** 不可入标记 */
    readonly cannotIn: number[][];
    /** 不可出标记 */
    readonly cannotOut: number[][];
    /** 地图的拓扑图 */
    readonly graph: IMapGraph;

    /**
     * 判断一个点是否连接至任意一个入口节点，数字表示 y * width + x
     * @param pos 需要判断的点
     * @param ignoredNode 路径中不允许通过的节点
     */
    connectedToAnyEntry(
        pos: number,
        ignoredNode?: (MapGraphNode | number)[]
    ): boolean;

    /**
     * 判断一个点是否连接至指定的入口节点，数字表示 y * width + x
     * @param pos 需要判断的点
     * @param entry 指定入口
     * @param ignoredNode 路径中不允许通过的节点
     */
    connectedToSpecificEntry(
        pos: number,
        entry: number | IEntryMapGraphNode,
        ignoredNode?: (MapGraphNode | number)[]
    ): boolean;
}

export interface IMapTileConverter {
    /**
     * 根据地图原始图块，获取对应的标签图块
     * @param tile 地图原始图块
     */
    getLabeledTile(tile: number): number;

    isEmpty(tile: number): boolean;

    isEntry(tile: number, x: number, y: number, floorId: string): boolean;

    isDoor(tile: number): boolean;

    isEnemy(tile: number): boolean;

    isResource(tile: number): boolean;

    /**
     * 获取指定原始图块在指定位置的通行信息
     * @param tile 地图原始图块
     * @param x 图块所在位置
     * @param y 图块所在位置
     */
    getNoPass(tile: number, x: number, y: number): boolean;

    /**
     * 获取指定原始图块在指定位置的不可入信息
     * @param tile 地图原始图块
     * @param x 图块所在位置
     * @param y 图块所在位置
     */
    getCannotIn(tile: number, x: number, y: number): number;

    /**
     * 获取指定原始图块在指定位置的不可出信息
     * @param tile 地图原始图块
     * @param x 图块所在位置
     * @param y 图块所在位置
     */
    getCannotOut(tile: number, x: number, y: number): number;

    /**
     * 获取指定图块所包含的资源
     */
    getResource(tile: number, x: number, y: number): Map<ResourceType, number>;
}
