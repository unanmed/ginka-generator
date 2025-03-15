export interface ResourceArea {
    type: 'resource';
    resources: Map<number, number>;
    members: Set<number>;
    neighbor: Set<number>;
}

export interface BranchNode {
    type: 'branch';
    neighbor: Set<number>;
    tile: number;
}

export interface ResourceNode {
    type: 'resource';
    resourceType: number;
    neighbor: Set<number>;
    resourceArea: ResourceArea;
}

export type GinkaNode = BranchNode | ResourceNode;

export interface GinkaGraph {
    /** 拓扑图内容，键表示位置，值表示这一点的节点 */
    graph: Map<number, GinkaNode>;
    /** 资源指针，键表示位置，值表示这一点对应的资源节点在 areaMap 的索引 */
    resourceMap: Map<number, number>;
    /** 资源区域列表 */
    areaMap: ResourceArea[];
    /** 这个拓扑图包含的入口位置 */
    visitedEntrance: Set<number>;
    /** 这个拓扑图能够造访的所有位置 */
    visited: Set<number>;
}

export interface GinkaTopologicalGraphs {
    /** 这个地图包含的所有独立的图 */
    graphs: GinkaGraph[];
    /** 每个入口对应哪个图 */
    entranceMap: Map<number, GinkaGraph>;
    /** 这个图从入口开始的不可到达区域 */
    unreachable: Set<number>;
}
