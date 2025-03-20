import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from minamo.model.model import MinamoModel
from shared.graph import batch_convert_soft_map_to_graph

def outer_border_constraint_loss(pred: torch.Tensor, allowed_classes=[1, 11], penalty_scale=1.0):
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
    
    # 提取所有允许类别的概率和 [B, H, W]
    allowed_probs = pred[:, allowed_classes, :, :].sum(dim=1)
    
    # 获取外圈区域允许类别的概率 [B, N_pixels]
    border_allowed = allowed_probs[:, border_mask]
    
    # 计算不符合要求的概率（反向损失）
    # 1 - 允许类别的概率 = 禁止类别的概率和
    border_violation = 1 - border_allowed
    
    # 使用平滑的Huber损失替代直接均值
    loss = F.huber_loss(
        border_violation, 
        torch.zeros_like(border_violation), 
        delta=0.1, 
        reduction='mean'
    )
    
    return penalty_scale * loss

def _create_distance_kernel(size):
    """生成带距离权重的卷积核"""
    kernel = torch.zeros(2*size-1, 2*size-1)
    center = size-1
    for i in range(2*size-1):
        for j in range(2*size-1):
            dist = math.sqrt((i-center)**2 + (j-center)**2)
            kernel[i,j] = 1 / (1 + dist)  # 距离越近权重越高
    return kernel.view(1,1,2*size-1,2*size-1)

def entrance_constraint_loss(
    pred: torch.Tensor,
    entrance_classes=[10, 11],  # 假设10是楼梯，11是箭头
    min_distance=9,
    presence_threshold=0.9,
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
    entrance_probs = pred[:, entrance_classes].sum(dim=1)

    ###########################
    # 改进的存在性约束
    ###########################
    # 计算存在性损失：鼓励至少有一个高置信度入口
    max_per_sample = entrance_probs.view(B, -1).max(dim=1)[0]
    presence_loss = F.relu(presence_threshold - max_per_sample).mean()

    ###########################
    # 改进的间距约束
    ###########################
    # 生成空间权重掩码（中心衰减）
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_weight = 1 - torch.sqrt(((x-W//2)/W*2)**2 + ((y-H//2)/H*2)**2)
    center_weight = center_weight.clamp(0,1).to(pred.device)  # [H,W]

    # 概率密度感知的间距计算
    kernel = _create_distance_kernel(min_distance).to(pred.device)  # 自定义函数生成权重核
    density_map = F.conv2d(entrance_probs.unsqueeze(1), kernel, padding=min_distance-1)
    
    # 平滑惩罚函数：S形曲线
    spacing_loss = torch.sigmoid(10*(density_map - 0.5)).mean()  # 密度>0.5时快速上升
    
    # print(entrance_probs)
    print(presence_loss.item(), (density_map).mean().item(), center_weight.mean().item())

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
    margin_ratio: float = 0.2,
    zero_margin_scale: float = 0.3,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    自适应图块数量约束损失函数
    
    参数:
        pred_probs: 预测概率分布 [B, C, H, W]
        target_map: 真实地图 [B, C, H, W]
        class_list: 需要约束的类别列表
        margin_ratio: 允许的相对误差范围（如0.2表示±20%）
        zero_margin_scale: 参考数量为0时的允许余量系数（余量=scale*sqrt(H*W))
        eps: 数值稳定性常数
    
    返回:
        loss: 标量损失值
    """
    B, C, H, W = pred_probs.shape
    device = pred_probs.device
    total_loss = 0.0
    valid_classes = 0
    
    # 预计算地图面积用于余量计算
    map_area = math.sqrt(H * W)
    
    for cls in class_list:
        # 预测数量（概率和）
        pred_count = pred_probs[:, cls].sum(dim=(1,2))  # [B]
        # 真实数量
        true_count = target_map[:, cls].sum(dim=(1,2))  # [B]
        
        # 动态容差计算
        with torch.no_grad():
            # 当真实数量为0时的允许上限
            zero_mask = (true_count == 0)
            dynamic_margin = torch.where(
                zero_mask,
                zero_margin_scale * map_area,  # 允许存在少量
                margin_ratio * true_count      # 相对误差范围
            )
        
        # 误差计算（考虑数值稳定性）
        safe_true = true_count + eps * zero_mask  # 零真实值时添加微小量
        abs_error = torch.abs(pred_count - true_count)
        rel_error = abs_error / safe_true
        
        # 双阶段损失函数
        # 阶段一：误差在容差范围内时使用二次函数（强梯度）
        # 阶段二：超出容差时转为线性（稳定训练）
        loss_per_class = torch.where(
            abs_error <= dynamic_margin,
            (rel_error ** 2) * 0.5,        # 区间内强梯度
            rel_error - (0.5 * margin_ratio)  # 区间外稳定梯度
        )
        
        # 零真实值特殊处理：仅惩罚超出余量部分
        loss_per_class = torch.where(
            zero_mask,
            F.relu(abs_error - dynamic_margin) / map_area,  # 归一化处理
            loss_per_class
        )
        
        total_loss += loss_per_class.mean()
        valid_classes += 1
    
    return total_loss / valid_classes  # 类别平均

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
    # 提取边缘区域的箭头概率 [B, N_edge_pixels]
    edge_arrow_probs = pred_probs[:, arrow_class][:, edge_mask]
    
    # 边缘应最大化箭头概率（最小化1 - arrow_prob）
    edge_arrow_loss = (1 - edge_arrow_probs).mean()
    
    # 抑制边缘出现楼梯的概率 [B, N_edge_pixels]
    edge_stair_probs = pred_probs[:, stair_class][:, edge_mask]
    edge_stair_penalty = F.relu(edge_stair_probs - 0.1).mean()  # 允许10%以下
    
    ##########################################
    # 3. 中间区域约束（只能出现楼梯）
    ##########################################
    # 提取中间区域的楼梯概率 [B, N_center_pixels]
    center_stair_probs = pred_probs[:, stair_class][:, center_mask]
    
    # 中间应最大化楼梯概率（最小化1 - stair_prob）
    center_stair_loss = (1 - center_stair_probs).mean()
    
    # 抑制中间出现箭头的概率 [B, N_center_pixels]
    center_arrow_probs = pred_probs[:, arrow_class][:, center_mask]
    center_arrow_penalty = F.relu(center_arrow_probs - 0.1).mean()  # 允许10%以下
    
    ##########################################
    # 4. 综合损失
    ##########################################
    total_loss = (
        lambda_arrow * (edge_arrow_loss + edge_stair_penalty) +
        lambda_stair * (center_stair_loss + center_arrow_penalty)
    )
    
    return total_loss

class GinkaLoss(nn.Module):
    def __init__(self, minamo: MinamoModel, weight=[0.5, 0.15, 0.15, 0.1, 0.1]):
        """Ginka Model 损失函数部分

        Args:
            weight (list, optional): 每一个损失函数的权重，从第 0 项开始，依次是：
                1. Minamo 相似度损失
                2. 外圈墙壁损失
                3. 入口间距及存在性损失
                4. 怪物、道具、门数量损失
                5. 非法图块损失
        """
        super().__init__()
        self.weight = weight
        self.minamo = minamo
        
    def forward(self, pred, target, target_vision_feat, target_topo_feat):
        # 地图结构损失
        border_loss = outer_border_constraint_loss(pred)
        entrance_loss = entrance_constraint_loss(pred) * 0.5 + entrance_spatial_constraint(pred) * 0.5
        count_loss = adaptive_count_loss(pred, target)
        illegal_loss = illegal_tile_loss(pred)
        
        # 使用 Minamo Model 计算相似度
        graph = batch_convert_soft_map_to_graph(pred)
        pred_vision_feat, pred_topo_feat = self.minamo(pred, graph)
        
        vision_sim = F.cosine_similarity(pred_vision_feat, target_vision_feat, dim=-1)
        topo_sim = F.cosine_similarity(pred_topo_feat, target_topo_feat, dim=-1)
        minamo_sim = 0.3 * vision_sim + 0.7 * topo_sim
        minamo_loss = (1.0 - minamo_sim).mean()
        
        print(
            minamo_loss.item(),
            border_loss.item(),
            entrance_loss.item(),
            count_loss.item(),
            illegal_loss.item()
        )
        
        losses = [
            minamo_loss * self.weight[0],
            border_loss * self.weight[1] * 0.1,
            entrance_loss * self.weight[2],
            count_loss * self.weight[3],
            illegal_loss * self.weight[4]
        ]
    
        # 梯度归一化
        scaled_losses = [loss / (loss.detach() + 1e-6) for loss in losses]
        total_loss = sum(scaled_losses)
        return total_loss