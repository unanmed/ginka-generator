import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from minamo.model.model import MinamoModel
from shared.graph import batch_convert_soft_map_to_graph

CLASS_NUM = 32
ILLEGAL_MAX_NUM = 12

def get_not_allowed(classes: list[int], include_illegal=False):
    res = list()
    for num in range(0, CLASS_NUM):
        if not num in classes:
            if num > ILLEGAL_MAX_NUM:
                if include_illegal:
                    res.append(num)
            else:
                res.append(num)
            
    return res

def outer_border_constraint_loss(pred: torch.Tensor, allowed_classes=[1, 11]):
    """
    强制地图最外圈像素必须为指定类别（墙或箭头）
    
    参数:
        pred: 模型输出的概率分布，形状 [B, C, H, W]
        allowed_classes: 允许出现在外圈的类别列表（默认[1,11]）
        penalty_scale: 惩罚强度系数
    
    返回:
        loss: 标量损失值
    """
    B, C, H, W = pred.shape
    
    # 创建外圈mask [H, W]
    border_mask = torch.zeros((H, W), dtype=torch.bool, device=pred.device)
    border_mask[0, :] = True      # 第一行
    border_mask[-1, :] = True     # 最后一行
    border_mask[:, 0] = True      # 第一列
    border_mask[:, -1] = True     # 最后一列
    
    # 提取所有允许和不允许类别的概率和 [B, H, W]
    unallowed_probs = pred[:, get_not_allowed(allowed_classes, include_illegal=True), :, :].sum(dim=1)
    
    # 获取外圈区域允许类别的概率 [B, N_pixels]
    border_unallowed = unallowed_probs[:, border_mask]
    
    target = torch.zeros_like(border_unallowed)
    loss_unallowed = F.mse_loss(border_unallowed, target)
    
    return loss_unallowed

def inner_constraint_loss(pred: torch.Tensor, allowed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]):
    """限定内部允许出现的图块种类

    Args:
        pred (torch.Tensor): 模型输出的概率分布 [B, C, H, W]
        unallowed (list, optional): 在地图中部（处最外圈）允许出现的图块种类. Defaults to [11].
    """
    B, C, H, W = pred.shape
    
    # 创建内部 mask [H, W]
    mask = torch.ones((H, W), dtype=torch.bool, device=pred.device)
    mask[0, :] = False      # 第一行
    mask[-1, :] = False     # 最后一行
    mask[:, 0] = False      # 第一列
    mask[:, -1] = False     # 最后一列
    
    # 提取所有允许和不允许类别的概率和 [B, H, W]
    unallowed_probs = pred[:, get_not_allowed(allowed, include_illegal=True), :, :].sum(dim=1)
    
    # 获取外圈区域允许类别的概率 [B, N_pixels]
    inner_unallowed = unallowed_probs[:, mask]
    
    target = torch.zeros_like(inner_unallowed)
    loss_unallowed = F.mse_loss(inner_unallowed, target)
    
    return loss_unallowed

def _create_distance_kernel(size):
    """生成一个环状衰减核"""
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = size // 2
    dist = torch.sqrt((x - center)**2 + (y - center)**2)
    kernel = 1 / (dist + 1)
    kernel /= kernel.sum()  # 归一化
    return kernel.unsqueeze(0).unsqueeze(0), 1 / kernel.sum()  # [1,1,H,W]

def entrance_constraint_loss(
    pred: torch.Tensor,
    entrance_classes=[10, 11],  # 假设10是楼梯，11是箭头
    min_distance=9,
    presence_threshold=0.8,
    lambda_presence=1.0,
    lambda_spacing=0.5
):
    """
    入口约束损失函数
    
    参数:
        pred: 模型输出的概率分布 [B, C, H, W]
        entrance_classes: 入口类别列表
        min_distance: 最小间隔距离（对应卷积核尺寸）
        presence_threshold: 存在性概率阈值
        lambda_presence: 存在性损失权重
        lambda_spacing: 间距约束权重
    
    返回:
        total_loss: 综合损失值
    """
    B, C, H, W = pred.shape
    entrance_probs = pred[:, entrance_classes, :, :].sum(dim=1) # [B, H, W]

    # 计算存在性损失：鼓励至少有一个高置信度入口
    max_per_sample = entrance_probs.view(B, -1).max(dim=1)[0] # [B, H*W] -> [B, 1]
    presence_loss = F.relu(presence_threshold - max_per_sample).mean()

    # 生成空间权重掩码（中心衰减）
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_weight = 1 - torch.sqrt(((x-W//2)/W*2)**2 + ((y-H//2)/H*2)**2)
    center_weight = center_weight.clamp(0,1).to(pred.device)  # [H,W]

    # 概率密度感知的间距计算
    kernel, cw = _create_distance_kernel(min_distance)  # 自定义函数生成权重核
    kernel = kernel.to(pred.device)
    density_map = F.conv2d(entrance_probs.unsqueeze(1), kernel, padding=min_distance-1)
    
    spacing_loss = density_map.mean()

    ###########################
    # 区域加权综合损失
    ###########################
    total_loss = (
        lambda_presence * presence_loss +
        lambda_spacing * (spacing_loss * center_weight).mean()
    )
    return total_loss

def adaptive_count_loss(
    pred_probs: torch.Tensor, 
    target_map: torch.Tensor,
    class_list: list = list(range(32)),
    margin_ratio: float = 0.1,  # 降低margin比例以更严格
    zero_margin_scale: float = 0.1,  # 减少零类别的margin
    lambda_entropy: float = 0.2,  # 增大熵约束权重
    lambda_local: float = 0.2,
    lambda_max: float = 0,  # 新增最大概率约束
    grid_size: int = 4,  # 减小局部网格尺寸
    eps: float = 1e-3
) -> torch.Tensor:
    """
    改进版自适应图块数量约束损失，增强局部匹配和概率确定性
    """
    B, C, H, W = pred_probs.shape
    device = pred_probs.device
    total_loss = 0.0
    valid_classes = 0
    
    # 预计算地图面积
    map_area = math.sqrt(H * W)
    
    # 动态调整零类别的margin：基于预测中最小的非零概率
    min_nonzero_prob = pred_probs[:, class_list].max(dim=1).values.mean()
    dynamic_zero_margin = zero_margin_scale * min_nonzero_prob * map_area

    # 计算每个类别的数量损失
    for cls in class_list:
        pred_count = pred_probs[:, cls].sum(dim=(1,2))  # 预测数量
        true_count = target_map[:, cls].sum(dim=(1,2))  # 真实数量
        
        zero_mask = (true_count == 0)
        dynamic_margin = torch.where(
            zero_mask,
            dynamic_zero_margin,
            margin_ratio * true_count
        )
        
        safe_true = true_count + eps * zero_mask
        abs_error = torch.abs(pred_count - true_count)
        rel_error = abs_error / safe_true
        
        # 调整损失函数形状，远离目标时惩罚更大
        loss_per_class = torch.where(
            abs_error <= dynamic_margin,
            rel_error ** 2,  # 近目标时二次损失
            (rel_error - 0.5 * margin_ratio) ** 2  # 远目标时二次增长
        )
        
        # 零类别使用更严格的绝对误差惩罚
        loss_per_class = torch.where(
            zero_mask,
            F.relu(abs_error - dynamic_zero_margin) ** 2 / map_area,
            loss_per_class
        )
        
        total_loss += loss_per_class.mean()
        valid_classes += 1
    
    total_loss /= valid_classes  # 平均类别损失
    
    # 改进的熵约束：每个像素的熵
    def entropy_loss(pred_probs):
        entropy_per_pixel = -torch.sum(pred_probs * torch.log(pred_probs + 1e-6), dim=1)
        return entropy_per_pixel.mean()  # 所有像素的平均熵
    
    total_loss += lambda_entropy * entropy_loss(pred_probs)

    # 新增最大概率约束：鼓励每个位置概率尖锐化
    max_probs = pred_probs.max(dim=1)[0]  # 每个位置的最大概率
    max_loss = (1 - max_probs).mean()  # 鼓励接近1
    total_loss += lambda_max * max_loss

    # 改进局部损失：约束局部区域内的数量
    def local_count_loss(pred_probs, target_probs, grid_size):
        grid_area = grid_size ** 2
        # 计算每个grid内的预测数量
        pred_counts = F.avg_pool2d(pred_probs, grid_size, stride=grid_size) * grid_area
        target_counts = F.avg_pool2d(target_probs, grid_size, stride=grid_size) * grid_area
        # 使用L1损失更鲁棒
        return F.l1_loss(pred_counts, target_counts)
    
    total_loss += lambda_local * local_count_loss(pred_probs, target_map, grid_size)

    return total_loss

def illegal_tile_loss(
    pred_probs: torch.Tensor, 
    legal_classes: int = 13, 
    temperature: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    非法图块惩罚损失函数
    
    参数:
        pred_probs: 模型输出的概率分布 [B, C, H, W]
        legal_classes: 合法图块数量（0-based，默认0-12为合法）
        temperature: 概率锐化温度系数（0.1-1.0）
        eps: 数值稳定性保护
    
    返回:
        loss: 标量损失值
    """
    B, C, H, W = pred_probs.shape
    
    # 提取非法图块概率（类别13及之后）
    illegal_probs = pred_probs[:, legal_classes:, :, :]  # [B, C_illegal, H, W]
    
    # 概率锐化（增强高概率区域的惩罚）
    sharpened_probs = torch.exp(torch.log(illegal_probs + eps) / temperature)
    sharpened_probs = sharpened_probs / (sharpened_probs.sum(dim=1, keepdim=True) + eps)
    
    # 空间敏感权重（关注高置信度非法区域）
    with torch.no_grad():
        # 计算每个像素的非法概率置信度
        confidence = illegal_probs.max(dim=1)[0]  # [B, H, W]
        # 生成注意力权重（高置信度区域权重加倍）
        spatial_weights = 1 + torch.sigmoid(10*(confidence - 0.5))
    
    # 逐像素计算非法概率损失
    per_pixel_loss = torch.log(1 + illegal_probs.sum(dim=1))  # [B, H, W]
    
    # 加权空间损失
    weighted_loss = (per_pixel_loss * spatial_weights).mean()
    
    # 类别平衡因子（抑制高频非法类别）
    class_balance = 1 + torch.var(illegal_probs.mean(dim=(0,2,3)))  # [C_illegal]
    
    return weighted_loss * class_balance.mean()

def entrance_spatial_constraint(
    pred_probs: torch.Tensor, 
    arrow_class: int = 11, 
    stair_class: int = 10,
    border_width: int = 1,
    lambda_arrow: float = 1.0,
    lambda_stair: float = 1.0
) -> torch.Tensor:
    """
    入口空间约束损失函数
    
    参数:
        pred_probs: 模型输出的概率分布 [B, C, H, W]
        arrow_class: 箭头入口类别索引
        stair_class: 楼梯入口类别索引
        border_width: 边缘区域宽度（默认1表示最外圈）
        lambda_arrow: 箭头约束权重
        lambda_stair: 楼梯约束权重
    
    返回:
        loss: 标量损失值
    """
    B, C, H, W = pred_probs.shape
    
    ##########################################
    # 1. 区域掩码生成
    ##########################################
    # 生成边缘区域掩码 [H, W]
    edge_mask = torch.zeros((H, W), dtype=torch.bool, device=pred_probs.device)
    # 上下边缘
    edge_mask[:border_width, :] = True
    edge_mask[-border_width:, :] = True
    # 左右边缘（排除已标记的角落）
    edge_mask[:, :border_width] = True
    edge_mask[:, -border_width:] = True
    
    # 生成中间区域掩码 [H, W]
    center_mask = ~edge_mask
    
    ##########################################
    # 2. 边缘区域约束（只能出现箭头）
    ##########################################
    
    # 抑制边缘出现楼梯的概率 [B, N_edge_pixels]
    edge_stair_probs = pred_probs[:, stair_class][:, edge_mask]
    edge_stair_penalty = F.relu(edge_stair_probs - 0.1).mean()  # 允许10%以下
    
    ##########################################
    # 3. 中间区域约束（只能出现楼梯）
    ##########################################
    
    # 抑制中间出现箭头的概率 [B, N_center_pixels]
    center_arrow_probs = pred_probs[:, arrow_class][:, center_mask]
    center_arrow_penalty = F.relu(center_arrow_probs - 0.1).mean()  # 允许10%以下
    
    ##########################################
    # 4. 综合损失
    ##########################################
    total_loss = (
        lambda_arrow * edge_stair_penalty +
        lambda_stair * center_arrow_penalty
    )
    
    return total_loss

class GinkaLoss(nn.Module):
    def __init__(self, minamo: MinamoModel, weight=[0.5, 0.2, 0.1, 0.2]):
        """Ginka Model 损失函数部分

        Args:
            weight (list, optional): 每一个损失函数的权重，从第 0 项开始，依次是：
                1. Minamo 相似度损失
                2. 图块种类损失，要求内部不出现箭头，外圈只出现箭头和墙壁，不允许出现非法图块
                3. 入口间距及存在性损失
                4. 怪物、道具、门数量损失
        """
        super().__init__()
        self.weight = weight
        self.minamo = minamo
        
    def forward(self, pred, target, target_vision_feat, target_topo_feat):
        # 地图结构损失
        class_loss = outer_border_constraint_loss(pred) + inner_constraint_loss(pred)
        entrance_loss = entrance_constraint_loss(pred) + entrance_spatial_constraint(pred)
        count_loss = adaptive_count_loss(pred, target)
        
        # 使用 Minamo Model 计算相似度
        graph = batch_convert_soft_map_to_graph(pred)
        pred_vision_feat, pred_topo_feat = self.minamo(pred, graph)
        
        vision_sim = F.cosine_similarity(pred_vision_feat, target_vision_feat, dim=-1)
        topo_sim = F.cosine_similarity(pred_topo_feat, target_topo_feat, dim=-1)
        minamo_sim = 0.2 * vision_sim + 0.8 * topo_sim
        minamo_loss = (1.0 - minamo_sim).mean()
        
        print(
            minamo_loss.item(),
            class_loss.item(),
            entrance_loss.item(),
            count_loss.item()
        )
        
        losses = [
            minamo_loss * self.weight[0],
            class_loss * self.weight[1],
            entrance_loss * self.weight[2],
            count_loss * self.weight[3]
        ]
    
        # 梯度归一化
        scaled_losses = [loss / (loss.detach() + 1e-6) for loss in losses]
        total_loss = sum(scaled_losses)
        return total_loss, sum(losses)
