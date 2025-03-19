import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

def convert_soft_map_to_graph(map_probs):
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

class DynamicGraphConverter(nn.Module):
    def __init__(self, map_size=13):
        super().__init__()
        self.map_size = map_size
        self.n_nodes = map_size * map_size
        self.base_edge_index = self._precompute_base_edges()

    def _precompute_base_edges(self):
        edge_list = []
        directions = [(0, 1), (1, 0)]
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

        # 1. 计算可微的节点 ID
        node_logits = map_probs.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        hard_nodes = F.gumbel_softmax(node_logits, tau=tau, hard=True)
        node_ids = (hard_nodes * torch.arange(C, device=device).view(1, 1, -1)).sum(dim=-1).long()

        # 2. 计算 soft 壁障 mask
        wall_mask = torch.sigmoid((node_ids - 1) * 10)  # 类别 1 代表墙体，soft 处理
        edge_weights = self._compute_dynamic_weights(wall_mask)

        # 3. 构建动态图
        batch_data = []
        for b in range(B):
            soft_mask = torch.sigmoid((edge_weights[b] - 0.1) * 10)  # 软门控
            dynamic_edge_attr = edge_weights[b] * soft_mask  # 仍然保留梯度
            
            data = Data(
                x=node_ids[b], 
                edge_index=self.base_edge_index,
                edge_attr=dynamic_edge_attr
            )
            batch_data.append(data)

        return Batch.from_data_list(batch_data)

    def _compute_dynamic_weights(self, wall_mask):
        src_nodes = self.base_edge_index[0]
        dst_nodes = self.base_edge_index[1]
        
        # 让梯度能正确回传
        weights = 1 - (wall_mask[:, src_nodes] + wall_mask[:, dst_nodes]) / 2  
        return weights.unsqueeze(-1)
