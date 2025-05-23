import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, grid

def differentiable_convert_to_data(map_probs: torch.Tensor) -> Data:
    """
    可导的图结构转换（返回 PyG Data 对象）
    map_probs: [C, H, W]
    返回：
    Data(x=[N, C], edge_index=[2, E], edge_attr=[E, C])
    """
    C, H, W = map_probs.shape
    device = map_probs.device
    N = H * W

    # 1. 节点特征
    node_features = map_probs.view(C, H * W).T  # [N, C]
    edge_index, _ = grid(H, W)
    edge_index = edge_index.to(device)

    # 2. 构建所有可能的边连接
    # node_indices = torch.arange(N, device=device).view(H, W)

    # # 水平连接（右邻居）
    # right_src = node_indices[:, :-1].flatten()
    # right_dst = node_indices[:, 1:].flatten()
    
    # # 垂直连接（下邻居）
    # down_src = node_indices[:-1, :].flatten()
    # down_dst = node_indices[1:, :].flatten()

    # # 合并边列表（双向）
    # edge_src = torch.cat([right_src, down_src])
    # edge_dst = torch.cat([right_dst, down_dst])
    # edge_index = torch.cat([
    #     torch.stack([edge_src, edge_dst], dim=0),
    #     torch.stack([edge_dst, edge_src], dim=0)  # 反向连接
    # ], dim=1).to(device, dtype=torch.long)

    # # 3. 计算边特征
    # src_feat = map_probs[:, edge_src // W, edge_src % W].T  # [E, C]
    # dst_feat = map_probs[:, edge_dst // W, edge_dst % W].T  # [E, C]
    # edge_attr = (src_feat + dst_feat) / 2  # [E, C]
    
    # edge_index, edge_attr = add_self_loops(edge_index, edge_attr)

    return Data(
        x=node_features,
        edge_index=edge_index,
        # edge_attr=edge_attr,
        num_nodes=N
    )

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
