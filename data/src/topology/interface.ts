export const enum NodeType {
    Branch,
    Resource
}

export interface ResourceArea {
    /** 节点类型 */
    readonly type: NodeType.Resource;
    /** 每种资源对应的数量 */
    readonly resources: Map<number, number>;
    /** 资源区域包含的所有资源图块坐标索引 */
    readonly members: Set<number>;
    /** 资源区域的邻居节点 */
    readonly neighbor: Set<number>;
}

export interface BranchNode {
    /** 节点类型 */
    readonly type: NodeType.Branch;
    /** 分支节点的邻居节点 */
    readonly neighbor: Set<number>;
    /** 分支节点图块 */
    readonly tile: number;
}

export interface ResourceNode {
    /** 节点类型 */
    readonly type: NodeType.Resource;
    /** 资源类型 */
    readonly resourceType: number;
    /** 邻居节点 */
    readonly neighbor: Set<number>;
    /** 资源节点所属的资源区域 */
    readonly resourceArea: ResourceArea;
}

export type GinkaNode = BranchNode | ResourceNode;

export interface GinkaGraph {
    /** 拓扑图内容，键表示位置，值表示这一点的节点 */
    readonly graph: Map<number, GinkaNode>;
    /** 资源指针，键表示位置，值表示这一点对应的资源节点在 areaMap 的索引 */
    readonly resourceMap: Map<number, number>;
    /** 资源区域列表 */
    readonly areaMap: ResourceArea[];
    /** 这个拓扑图包含的入口位置 */
    readonly visitedEntrance: Set<number>;
    /** 这个拓扑图能够造访的所有位置 */
    readonly visited: Set<number>;
}

export interface GinkaTopologicalGraphs {
    /** 这个地图包含的所有独立的图 */
    readonly graphs: GinkaGraph[];
    /** 每个入口对应哪个图 */
    readonly entranceMap: Map<number, GinkaGraph>;
    /** 这个图从入口开始的不可到达区域 */
    readonly unreachable: Set<number>;
}
