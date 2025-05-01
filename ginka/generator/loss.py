import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from shared.graph import batch_convert_soft_map_to_graph
from shared.constant import VISION_WEIGHT, TOPO_WEIGHT
from ..critic.model import MinamoModel

CLASS_NUM = 32
ILLEGAL_MAX_NUM = 30

STAGE_ALLOWED = [
    [],
    [0, 1, 2, 29, 30],
    [3, 4, 5, 6, 26, 27, 28],
    list(range(7, 26))
]

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

def outer_border_constraint_loss(pred: torch.Tensor, allowed_classes=[*list(range(0, 29)), 30]):
    """
    强制地图最外圈像素必须为指定类别（墙或箭头）
    
    参数:
        pred: 模型输出的概率分布，形状 [B, C, H, W]
        allowed_classes: 允许出现在外圈的类别列表
    
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

def inner_constraint_loss(pred: torch.Tensor, allowed=list(range(0, 30))):
    """限定内部允许出现的图块种类

    Args:
        pred (torch.Tensor): 模型输出的概率分布 [B, C, H, W]
        allowed (list, optional): 在地图中部（除最外圈）允许出现的图块种类
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
    return kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

def entrance_constraint_loss(
    pred: torch.Tensor,
    entrance_classes=[29, 30],
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
    kernel = _create_distance_kernel(min_distance)  # 自定义函数生成权重核
    kernel = kernel.to(pred.device)
    density_map = F.conv2d(entrance_probs.unsqueeze(1), kernel, padding=min_distance-1)
    
    spacing_loss = density_map.mean()

    # 区域加权综合损失
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

def input_head_illegal_loss(input_map, allowed_classes=(0, 1)):
    C = input_map.shape[1]
    mask = torch.ones(C, device=input_map.device)
    mask[list(allowed_classes)] = 0  # 屏蔽允许的类别，其余为 1
    illegal_class_penalty = (input_map * mask.view(1, -1, 1, 1)).sum() / input_map.numel()
    
    return illegal_class_penalty

def input_head_wall_loss(input_map, max_wall_ratio=0.2, wall_class=1):
    wall_prob = input_map[:, wall_class]  # [B, H, W]
    wall_ratio = wall_prob.mean()         # 计算平均墙体占比
    wall_penalty = torch.clamp(wall_ratio - max_wall_ratio, min=0.0)  # 超过则惩罚
    
    return wall_penalty

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
        entrance_loss = entrance_constraint_loss(pred)
        count_loss = adaptive_count_loss(pred, target)
        
        # 使用 Minamo Model 计算相似度
        graph = batch_convert_soft_map_to_graph(pred)
        pred_vision_feat, pred_topo_feat = self.minamo(pred, graph)
        
        vision_sim = F.cosine_similarity(pred_vision_feat, target_vision_feat, dim=1)
        topo_sim = F.cosine_similarity(pred_topo_feat, target_topo_feat, dim=1)
        minamo_sim = 0 * vision_sim + 1 * topo_sim
        # tqdm.write(f"{vision_sim.mean().item():.12f}, {topo_sim.mean().item():.12f}")
        minamo_loss = (1.0 - minamo_sim).mean()
        
        tqdm.write(
            f"{minamo_loss.item():.12f}, {class_loss.item():.12f}, {entrance_loss.item():.12f}, {count_loss.item():.12f}"
        )
        
        losses = [
            minamo_loss * self.weight[0] * 4,
            class_loss * self.weight[1],
            entrance_loss * self.weight[2],
            count_loss * self.weight[3]
        ]
    
        return sum(losses)
    
# 对图像数据进行插值
def interpolate_data(real_data, fake_data, epsilon):
    return epsilon * real_data + (1 - epsilon) * fake_data

# 对节点特征进行插值，但保持边连接关系不变
def interpolate_graph_features(real_graph, fake_graph, epsilon=0.5):
    # 插值节点特征
    x_real, x_fake = real_graph.x, fake_graph.x
    x_interp = epsilon * x_real + (1 - epsilon) * x_fake
    
    # 保持边连接关系和边特征不变
    edge_index_interp = real_graph.edge_index  # 保持边连接关系
    edge_attr_interp = real_graph.edge_attr  # 如果有边特征，保持不变
    
    return Data(x=x_interp, edge_index=edge_index_interp, edge_attr=edge_attr_interp)
    
def js_divergence(p, q, eps=1e-6, softmax=False):
    if softmax:
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
    # softmax 后变成概率分布    
    m = 0.5 * (p + q)
    
    # log_softmax 以供 kl_div 使用
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    log_m = torch.log(m + eps)

    kl_pm = F.kl_div(log_p, log_m, reduction='batchmean', log_target=True)  # KL(p || m)
    kl_qm = F.kl_div(log_q, log_m, reduction='batchmean', log_target=True)  # KL(q || m)

    return torch.log1p(0.5 * (kl_pm + kl_qm))

def immutable_penalty_loss(
    pred: torch.Tensor, input: torch.Tensor, modifiable_classes: list[int]
) -> torch.Tensor:
    """
    惩罚模型修改不可更改区域的损失。
    
    Args:
        input: 模型输出 [B, C, H, W]，概率分布 (softmax 前)
        target: 原始输入图 [B, C, H, W]，概率分布 (softmax 前)
        modifiable_classes: 允许被修改的类别列表
    """
    not_allowed = get_not_allowed(modifiable_classes, include_illegal=True)
    input_mask = pred[:, not_allowed, :, :]
    with torch.no_grad():
        target_mask = torch.argmax(input[:, not_allowed, :, :], dim=1)
        target_mask = F.one_hot(target_mask, num_classes=len(not_allowed)).permute(0, 3, 1, 2).float()

    # 差异区域（模型试图改变的地方）
    penalty = F.cross_entropy(input_mask, target_mask)

    return penalty

class WGANGinkaLoss:
    def __init__(self, lambda_gp=100, weight=[1, 0.5, 50, 0.2, 0.2, 0.2]):
        # weight: 判别器损失，CE 损失，不可修改类型损失，图块类型损失，入口存在性损失，多样性损失
        self.lambda_gp = lambda_gp  # 梯度惩罚系数
        self.weight = weight
        
    def compute_gradient_penalty(self, critic, stage, real_data, fake_data, tag_cond, val_cond):
        # 进行插值
        batch_size = real_data.size(0)
        epsilon_data = torch.randn(batch_size, 1, 1, 1, device=real_data.device)
        interp_data = interpolate_data(real_data, fake_data, epsilon_data).to(real_data.device)
        interp_graph = batch_convert_soft_map_to_graph(interp_data).to(real_data.device)
        
        # 对图像进行反向传播并计算梯度
        interp_data.requires_grad_()
        interp_graph.x.requires_grad_()
        
        _, d_vis_score, d_topo_score = critic(interp_data, interp_graph, stage, tag_cond, val_cond)
        
        # 计算梯度
        grad_vis = torch.autograd.grad(
            outputs=d_vis_score, inputs=interp_data,
            grad_outputs=torch.ones_like(d_vis_score),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_topo = torch.autograd.grad(
            outputs=d_topo_score, inputs=interp_graph.x,
            grad_outputs=torch.ones_like(d_topo_score),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # 计算梯度的 L2 范数
        grad_norm_vis = grad_vis.view(batch_size, -1).norm(2, dim=1)
        grad_norm_topo = grad_topo.view(batch_size, -1).norm(2, dim=1)
        # 计算梯度惩罚项
        gp_loss_vis = ((grad_norm_vis - 1.0) ** 2).mean()
        gp_loss_topo = ((grad_norm_topo - 1.0) ** 2).mean()
        gp_loss = gp_loss_vis * VISION_WEIGHT + gp_loss_topo * TOPO_WEIGHT
        # print(grad_norm_topo.mean().item(), grad_norm_vis.mean().item())

        return gp_loss
        
    def discriminator_loss(
        self, critic, stage: int, real_data: torch.Tensor, fake_data: torch.Tensor,
        tag_cond: torch.Tensor, val_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ 判别器损失函数 """
        fake_data = F.softmax(fake_data, dim=1)
        real_graph = batch_convert_soft_map_to_graph(real_data)
        fake_graph = batch_convert_soft_map_to_graph(fake_data)
        real_scores, _, _ = critic(real_data, real_graph, stage, tag_cond, val_cond)
        fake_scores, _, _ = critic(fake_data, fake_graph, stage, tag_cond, val_cond)
        
        # Wasserstein 距离
        d_loss = fake_scores.mean() - real_scores.mean()
        grad_loss = self.compute_gradient_penalty(critic, stage, real_data, fake_data, tag_cond, val_cond)
        
        total_loss = d_loss + self.lambda_gp * grad_loss
        
        return total_loss, d_loss

    def generator_loss(self, critic, stage, mask_ratio, real, fake: torch.Tensor, input, tag_cond, val_cond) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 生成器损失函数 """
        probs_fake = F.softmax(fake, dim=1)
        fake_graph = batch_convert_soft_map_to_graph(probs_fake)
        
        fake_scores, _, _ = critic(probs_fake, fake_graph, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        ce_loss = F.cross_entropy(fake, real) * (1 - mask_ratio) # 蒙版越大，交叉熵损失权重越小
        immutable_loss = immutable_penalty_loss(fake, input, STAGE_ALLOWED[stage])
        constraint_loss = inner_constraint_loss(probs_fake)
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            ce_loss * self.weight[1],
            immutable_loss * self.weight[2],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
        
        # print(-js_divergence(fake_a, fake_b).item())
        
        return sum(losses), minamo_loss, ce_loss, immutable_loss
    
    def generator_loss_total(self, critic, stage, fake, tag_cond, val_cond) -> torch.Tensor:
        probs_fake = F.softmax(fake, dim=1)
        fake_graph = batch_convert_soft_map_to_graph(probs_fake)
        
        fake_scores, _, _ = critic(probs_fake, fake_graph, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        constraint_loss = inner_constraint_loss(probs_fake)
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
            
        return sum(losses)
    
    def generator_loss_total_with_input(self, critic, stage, fake, input, tag_cond, val_cond) -> torch.Tensor:
        probs_fake = F.softmax(fake, dim=1)
        fake_graph = batch_convert_soft_map_to_graph(probs_fake)
        
        fake_scores, _, _ = critic(probs_fake, fake_graph, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        immutable_loss = immutable_penalty_loss(fake, input, STAGE_ALLOWED[stage])
        constraint_loss = inner_constraint_loss(probs_fake)
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            immutable_loss * self.weight[2],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
            
        return sum(losses)

    def generator_input_head_loss(self, probs: torch.Tensor) -> torch.Tensor:
        probs_a, probs_b = probs.chunk(2, dim=0)
        
        losses = [
            input_head_illegal_loss(probs),
            input_head_wall_loss(probs),
            -js_divergence(probs_a, probs_b, softmax=False) * 0.3
        ]
        
        return sum(losses)
