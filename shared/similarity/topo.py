# Converted Python version of the JS code
import math
from typing import Dict, Set, List, Tuple, Union
from collections import deque, defaultdict

# 拓扑相似度，由 ChatGPT-4o 从 ts 转译而来

class ResourceArea:
    def __init__(self):
        self.type = 'resource'
        self.resources: Dict[int, int] = {}
        self.members: Set[int] = set()
        self.neighbor: Set[int] = set()

class BranchNode:
    def __init__(self, tile: int):
        self.type = 'branch'
        self.tile = tile
        self.neighbor: Set[int] = set()

class ResourceNode:
    def __init__(self, resource_type: int, area: ResourceArea):
        self.type = 'resource'
        self.resourceType = resource_type
        self.neighbor = area.neighbor
        self.resourceArea = area

GinkaNode = Union[BranchNode, ResourceNode]

class GinkaGraph:
    def __init__(self):
        self.graph: Dict[int, GinkaNode] = {}
        self.resourceMap: Dict[int, int] = {}
        self.areaMap: List[ResourceArea] = []
        self.visitedEntrance: Set[int] = set()
        self.visited: Set[int] = set()

class GinkaTopologicalGraphs:
    def __init__(self):
        self.graphs: List[GinkaGraph] = []
        self.entranceMap: Dict[int, GinkaGraph] = {}
        self.unreachable: Set[int] = set()

TILE_TYPE = set(range(13))
BRANCH_TYPE = {6, 7, 8, 9}
ENTRANCE_TYPE = {10, 11}
RESOURCE_TYPE = {0, 2, 3, 4, 5, 10, 11, 12, 13}

directions: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1)
]

def find_resource_nodes(map_: List[List[int]]):
    width, height = len(map_[0]), len(map_)
    visited = set()
    areas = []
    resource_map = {}

    for ny in range(height):
        for nx in range(width):
            tile = map_[ny][nx]
            index = ny * width + nx
            if index in visited or tile not in RESOURCE_TYPE:
                continue
            queue = deque([(nx, ny)])
            area = ResourceArea()
            area.resources[tile] = 1
            area.members.add(index)
            while queue:
                cx, cy = queue.popleft()
                cindex = cy * width + cx
                if cindex in visited:
                    continue
                ctile = map_[cy][cx]
                if ctile not in RESOURCE_TYPE:
                    continue
                visited.add(cindex)
                area.resources[ctile] = area.resources.get(ctile, 0) + 1
                area.members.add(cindex)
                resource_map[cindex] = len(areas)
                for dx, dy in directions:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < width and 0 <= py < height:
                        queue.append((px, py))
            areas.append(area)
    return areas, resource_map

def build_graph_from_entrance(map_: List[List[int]], entrance: int, resource_map: Dict[int, int], area_map: List[ResourceArea]) -> GinkaGraph:
    width, height = len(map_[0]), len(map_)
    graph = GinkaGraph()
    graph.resourceMap = resource_map
    graph.areaMap = area_map

    visited = graph.visited
    visited_entrance = graph.visitedEntrance
    visited_entrance.add(entrance)

    branch_nodes = set()
    queue = deque([(entrance % width, entrance // width)])

    while queue:
        x, y = queue.popleft()
        index = y * width + x
        if index in visited:
            continue
        tile = map_[y][x]
        if tile in ENTRANCE_TYPE:
            visited_entrance.add(index)
        if tile in BRANCH_TYPE:
            branch_nodes.add(index)
        visited.add(index)
        for dx, dy in directions:
            px, py = x + dx, y + dy
            if 0 <= px < width and 0 <= py < height and map_[py][px] != 1:
                queue.append((px, py))

    for v in branch_nodes:
        x, y = v % width, v // width
        if v not in graph.graph:
            graph.graph[v] = BranchNode(map_[y][x])
        node = graph.graph[v]
        for dx, dy in directions:
            px, py = x + dx, y + dy
            if 0 <= px < width and 0 <= py < height:
                index = py * width + px
                if index in branch_nodes:
                    node.neighbor.add(index)
                elif index in resource_map:
                    area = area_map[resource_map[index]]
                    area.neighbor.add(v)
                    for m in area.members:
                        node.neighbor.add(m)

    for area in area_map:
        for index in area.members:
            x, y = index % width, index // width
            tile = map_[y][x]
            if tile == 0:
                continue
            node = ResourceNode(tile, area)
            graph.graph[index] = node

    return graph

def build_topological_graph(map_: List[List[int]]) -> GinkaTopologicalGraphs:
    width, height = len(map_[0]), len(map_)
    entrances = set()
    entrances = {y * width + x for y in range(height) for x in range(width) if map_[y][x] in ENTRANCE_TYPE}
    area_map, resource_map = find_resource_nodes(map_)

    top_graph = GinkaTopologicalGraphs()
    used_entrance = set()
    total_visited = set()

    for entrance in entrances:
        if entrance in used_entrance:
            continue
        graph = build_graph_from_entrance(map_, entrance, resource_map, area_map)
        top_graph.graphs.append(graph)
        for ent in graph.visitedEntrance:
            used_entrance.add(ent)
            top_graph.entranceMap[ent] = graph
        total_visited.update(graph.visited)

    for y in range(height):
        for x in range(width):
            index = y * width + x
            if index not in total_visited and map_[y][x] != 1:
                top_graph.unreachable.add(index)

    return top_graph

class WLNode:
    def __init__(self, pos: int, label: str):
        self.originalPos = pos
        self.originalLabel = label
        self.currentLabel = label
        self.neighbors: List['WLNode'] = []

def encode_node_labels(graph: GinkaGraph) -> List[WLNode]:
    node_map = {}
    nodes = []
    for pos, node in graph.graph.items():
        label = f"B:{node.tile}" if node.type == 'branch' else f"R:{node.resourceType}"
        wl_node = WLNode(pos, label)
        node_map[pos] = wl_node
        nodes.append(wl_node)

    for node in nodes:
        g_node = graph.graph[node.originalPos]
        for neighbor in g_node.neighbor:
            if neighbor in node_map:
                node.neighbors.append(node_map[neighbor])

    return nodes

def weisfeiler_lehman_iteration(nodes: List[WLNode], iterations: int, decay: float = 0.6) -> Dict[str, float]:
    label_history = []
    for _ in range(iterations):
        new_labels = []
        for node in nodes:
            neighbor_labels = sorted(n.currentLabel for n in node.neighbors)
            composite = f"{node.currentLabel}|{','.join(neighbor_labels)}"[:8192]
            new_labels.append(composite)
        for node, new_label in zip(nodes, new_labels):
            node.currentLabel = new_label
        label_history.append(new_labels[:])

    weight = 1.0
    label_counts = defaultdict(float)
    for layer in label_history:
        for label in layer:
            label_counts[label] += weight
        weight *= decay
    for node in nodes:
        label_counts[node.originalLabel] += weight
    return dict(label_counts)

def vectorize_features(features: Dict[str, float], vocab: List[str]) -> List[float]:
    vec = [0.0] * len(vocab)
    for label, count in features.items():
        if label in vocab:
            idx = vocab.index(label)
            vec[idx] += count
    return vec

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def wl_kernel(graph_a: GinkaGraph, graph_b: GinkaGraph, iterations: int = 3) -> float:
    nodes_a = encode_node_labels(graph_a)
    nodes_b = encode_node_labels(graph_b)
    features_a = weisfeiler_lehman_iteration(nodes_a, iterations)
    features_b = weisfeiler_lehman_iteration(nodes_b, iterations)
    vocab = list(set(features_a.keys()) | set(features_b.keys()))
    vec_a = vectorize_features(features_a, vocab)
    vec_b = vectorize_features(features_b, vocab)
    return cosine_similarity(vec_a, vec_b)

def overall_similarity(a: GinkaTopologicalGraphs, b: GinkaTopologicalGraphs) -> float:
    graphs_a = a.graphs
    graphs_b = b.graphs

    total_similarity = 0.0
    compared_graphs: Set[GinkaGraph] = set()

    for ga in graphs_a:
        max_similarity = 0.0
        max_graph = None
        for gb in graphs_b:
            if gb in compared_graphs:
                continue
            min_nodes = min(len(ga.graph), len(gb.graph))
            iterations = max(1, math.ceil(math.log(min_nodes)))
            similarity = wl_kernel(ga, gb, iterations)
            if similarity > max_similarity and not math.isnan(similarity):
                max_similarity = similarity
                max_graph = gb
            if similarity == 1:
                break
        total_similarity += max_similarity
        if max_graph:
            compared_graphs.add(max_graph)

    reduction = 1 / (1 + abs(len(a.unreachable) - len(b.unreachable)))
    if not graphs_a:
        return 0.0
    return math.sqrt(total_similarity / len(graphs_a)) * reduction
