import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

def convert_map_to_graph(map):
    rows = len(map)
    cols = len(map[0])
    node_indices = {}
    valid_nodes = []
    node_counter = 0

    for r in range(rows):
        for c in range(cols):
            if map[r][c] != 1:  # 排除墙体
                node_indices[(r, c)] = node_counter
                valid_nodes.append((r, c, map[r][c]))  # (行, 列, 地形类型)
                node_counter += 1

    edge_list = []
    for (r, c, _) in valid_nodes:
        node = node_indices[(r, c)]
        if c + 1 < cols and (r, c + 1) in node_indices:
            edge_list.append((node, node_indices[(r, c + 1)]))
        if r + 1 < rows and (r + 1, c) in node_indices:
            edge_list.append((node, node_indices[(r + 1, c)]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    node_features = torch.tensor([node_type for (_, _, node_type) in valid_nodes], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index)

def soft_convert_map_to_graph(map_tensor, tau=0.5, threshold=0.1):
    """
    将地图批量转换为 GNN 可用的 Graph Data，使用 soft 策略
    """
    B, H, W, C = map_tensor.shape
    N = H * W  

    # 使用 Gumbel-Softmax 确保是 one-hot 形式
    y = F.gumbel_softmax(map_tensor.view(B, N, C), tau=tau, hard=True)  # [B, N, C]

    # 计算整数索引（用于 embedding）
    node_features = y.argmax(dim=-1).long().unsqueeze(-1)  # [B, N, 1]

    # 取出墙体的 soft mask
    wall_mask = y[:, :, 1]  # [B, N]

    adjacency_matrix = torch.zeros(B, N, N, device=map_tensor.device)

    def get_index(r, c):
        return r * W + c

    for r in range(H):
        for c in range(W):
            idx = get_index(r, c)
            if c + 1 < W:
                right_idx = get_index(r, c + 1)
                adjacency_matrix[:, idx, right_idx] = (1 - wall_mask[:, idx]) * (1 - wall_mask[:, right_idx])
            if r + 1 < H:
                down_idx = get_index(r + 1, c)
                adjacency_matrix[:, idx, down_idx] = (1 - wall_mask[:, idx]) * (1 - wall_mask[:, down_idx])

    edge_index_list, edge_weight_list = [], []
    for b in range(B):
        adj_bin = (adjacency_matrix[b] > threshold).float()
        edge_index = torch.nonzero(adj_bin, as_tuple=False).T  # [2, E]
        edge_weight = adjacency_matrix[b][edge_index[0], edge_index[1]]  

        edge_index_list.append(edge_index)
        edge_weight_list.append(edge_weight)

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_weight = torch.cat(edge_weight_list, dim=0)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

class DynamicGraphConverter(nn.Module):
    def __init__(self, map_size=13):
        super().__init__()
        self.map_size = map_size
        self.n_nodes = map_size * map_size
        
        # 预计算所有可能的边索引组合（包括对角线）
        self.base_edge_index = self._precompute_base_edges()
        
    def _precompute_base_edges(self):
        """预生成全连接边索引（包含所有可能邻接）"""
        edge_list = []
        directions = [
            (0, 1),   # 右
            (1, 0),   # 下
        ]
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                node = r * self.map_size + c
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.map_size and 0 <= nc < self.map_size:
                        neighbor = nr * self.map_size + nc
                        edge_list.append([node, neighbor])
                        
        return torch.tensor(edge_list).t().contiguous().unique(dim=1)

    def forward(self, map_probs, tau=0.5):
        B, C, H, W = map_probs.shape
        device = map_probs.device
        
        self.base_edge_index = self.base_edge_index.to(device)
        
        # 1. 节点特征离散化（保持可导）
        node_logits = map_probs.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        hard_nodes = F.gumbel_softmax(node_logits, tau=tau, hard=True)
        node_ids = hard_nodes.argmax(dim=-1)  # [B, N]

        # 2. 动态边权重计算
        wall_mask = (node_ids == 1).float()  # 假设类别1是墙体
        edge_weights = self._compute_dynamic_weights(wall_mask)

        # 3. 构建动态图
        batch_data = []
        for b in range(B):
            # 动态过滤无效边（与墙体相连的边）
            valid_mask = (edge_weights[b] > 0.1).squeeze(-1)
            dynamic_edge_index = self.base_edge_index[:, valid_mask]
            dynamic_edge_attr = edge_weights[b][valid_mask]
            
            data = Data(
                x=node_ids[b], 
                edge_index=dynamic_edge_index,
                edge_attr=dynamic_edge_attr
            )
            batch_data.append(data)
            
        return Batch.from_data_list(batch_data)

    def _compute_dynamic_weights(self, wall_mask):
        """基于墙体存在性计算动态边权重"""
        # wall_mask: [B, N]
        src_nodes = self.base_edge_index[0]  # [E]
        dst_nodes = self.base_edge_index[1]  # [E]
        
        # 边权重 = 1 - (源是墙 OR 目标墙)
        weights = 1 - torch.logical_or(
            wall_mask[:, src_nodes],
            wall_mask[:, dst_nodes]
        ).float()  # [B, E]
        
        return weights.unsqueeze(-1)  # [B, E, 1]
