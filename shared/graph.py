import torch
from torch_geometric.data import Data, Batch

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
        graph = convert_soft_map_to_graph(batch_map_probs[i])  # 处理单个样本
        batch_graphs.append(graph)

    # 合并所有图为批量 Batch
    return Batch.from_data_list(batch_graphs)
