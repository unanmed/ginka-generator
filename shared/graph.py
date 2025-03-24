import torch
from torch_geometric.data import Data, Batch

def differentiable_convert_to_data(map_probs: torch.Tensor) -> Data:
    """
    可导的图结构转换（返回PyG Data对象）
    map_probs: [C, H, W]
    返回：
    Data(x=[N,C], edge_index=[2,E], edge_attr=[E,C])
    """
    C, H, W = map_probs.shape
    device = map_probs.device
    N = H * W

    # 1. 节点特征（保留所有节点）
    node_features = map_probs.view(C, -1).T  # [N, C]

    # 2. 构建所有可能的边连接（预计算）
    # 生成坐标网格
    rows, cols = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    node_indices = rows * W + cols

    # 水平连接（右邻居）
    right_src = node_indices[:, :-1].flatten()
    right_dst = node_indices[:, 1:].flatten()
    
    # 垂直连接（下邻居）
    down_src = node_indices[:-1, :].flatten()
    down_dst = node_indices[1:, :].flatten()

    # 合并边列表
    edge_src = torch.cat([right_src, down_src]).to(device)
    edge_dst = torch.cat([right_dst, down_dst]).to(device)
    edge_index = torch.stack([edge_src, edge_dst])  # [2, E]

    # 3. 计算可导的边权重（排除墙类型）
    wall_class_idx = 1  # 假设类型1是墙
    src_probs = 1.0 - map_probs[wall_class_idx].flatten()[edge_src]  # [E]
    dst_probs = 1.0 - map_probs[wall_class_idx].flatten()[edge_dst]  # [E]
    edge_mask = (src_probs * dst_probs).unsqueeze(1)  # [E, 1]

    # 4. 边特征计算（保持可导）
    src_feat = map_probs[:, edge_src//W, edge_src%W].T  # [E, C]
    dst_feat = map_probs[:, edge_dst//W, edge_dst%W].T  # [E, C]
    edge_attr = (src_feat + dst_feat) / 2 * edge_mask  # [E, C]

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=N
    )

def convert_soft_map_to_graph(map_probs: torch.Tensor):
    """
    直接使用 Softmax 概率构建 soft 图结构
    """
    C, H, W = map_probs.shape  # [32, H, W]
    N = H * W
    device = map_probs.device

    # 计算 soft 节点特征
    node_features = map_probs.view(C, N).T  # [N, C]

    # 计算 soft 邻接边（基于 soft 权重）
    edge_list = []
    for r in range(H):
        for c in range(W):
            node = r * W + c
            if c + 1 < W:
                right = node + 1
                edge_list.append([node, right])
            if r + 1 < H:
                down = node + W
                edge_list.append([node, down])

    edge_index = torch.tensor(edge_list).t().to(device)

    # 计算 soft 边权重（基于 Softmax 概率）
    soft_edge_weight = (map_probs[:, edge_index[0] // W, edge_index[0] % W] + 
                        map_probs[:, edge_index[1] // W, edge_index[1] % W]) / 2

    return Data(x=node_features, edge_index=edge_index, edge_attr=soft_edge_weight)

def batch_convert_soft_map_to_graph(batch_map_probs):
    """
    处理 batch 维度，将 [B, C, H, W] 转换为批量图结构 Batch
    """
    B, C, H, W = batch_map_probs.shape  # 获取 batch 维度
    batch_graphs = []

    for i in range(B):
        graph = differentiable_convert_to_data(batch_map_probs[i])  # 处理单个样本
        batch_graphs.append(graph)

    # 合并所有图为批量 Batch
    return Batch.from_data_list(batch_graphs)
