import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

CLASS_NUM = 32
ILLEGAL_MAX_NUM = 30

STAGE_CHANGEABLE = [
    [],
    [0, 1, 2, 29, 30],
    [3, 4, 5, 6, 26, 27, 28],
    list(range(7, 26))
]

STAGE_ALLOWED = [
    [],
    STAGE_CHANGEABLE[1],
    [*STAGE_CHANGEABLE[1], *STAGE_CHANGEABLE[2]],
    [*STAGE_CHANGEABLE[1], *STAGE_CHANGEABLE[2], *STAGE_CHANGEABLE[3]]
]

DENSITY_MAP = [
    [1, *list(range(3, 30))],
    [1],
    [2],
    [3, 4, 5, 6],
    [26, 27, 28],
    list(range(7, 26)),
    list(range(10, 19)),
    [19, 20, 21, 22],
    [7, 8, 9],
    [23, 24, 25],
    [29, 30]
]

DENSITY_WEIGHTS = [
    1,
    1.5,
    0.5,
    5,
    4,
    3,
    3,
    3,
    5,
    10,
    20
]

DENSITY_STAGE = [
    [],
    [1, 2],
    [1, 2, 3, 4],
    list(range(0, 10))
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

def input_head_illegal_loss(input_map, allowed_classes=[0, 1, 2]):
    C = input_map.shape[1]
    unallowed = get_not_allowed(allowed_classes, include_illegal=True)
    illegal = input_map[:, unallowed, :, :]
    penalty = F.l1_loss(illegal, torch.zeros_like(illegal, device=illegal.device))

    return penalty

def input_head_wall_loss(input_map, max_wall_ratio=0.2, wall_class=[1, 2]):
    wall_prob = input_map[:, wall_class]  # [B, H, W]
    wall_ratio = wall_prob.mean()         # 计算平均墙体占比
    wall_penalty = torch.clamp(wall_ratio - max_wall_ratio, min=0.0)  # 超过则惩罚
    
    return wall_penalty

def compute_multi_density_loss(probs, target_densities, tile_list):
    """
    pred: [B, C, H, W]
    target_densities: [B, N] - N 个目标类别密度
    class_indices: [N] - 对应类别通道索引
    """
    losses = []
    for i, classes in enumerate(DENSITY_MAP):
        class_map = probs[:, classes, :, :]
        pred_density = torch.mean(class_map, dim=(1, 2, 3))
        if i in tile_list:
            loss = F.mse_loss(pred_density, target_densities[:, i])
            losses.append(loss * DENSITY_WEIGHTS[i])
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
    penalty = torch.clamp(F.cross_entropy(input_mask, target_mask) - 0.2, min=0)

    return penalty

def modifiable_penalty_loss(
    probs: torch.Tensor, input: torch.Tensor, modifiable_classes: list[int]
) -> torch.Tensor:
    target_modifiable = input[:, modifiable_classes, :, :]
    pred_modifiable = probs[:, modifiable_classes, :, :]
    existed = torch.clamp(target_modifiable - pred_modifiable, min=0.0, max=1.0)
    penalty = F.mse_loss(existed, torch.zeros_like(existed, device=existed.device))
    
    return penalty

def illegal_penalty_loss(pred: torch.Tensor, legal_classes: list[int]):
    not_allowed = get_not_allowed(legal_classes, include_illegal=True)
    input_mask = pred[:, not_allowed, :, :]
    target = torch.zeros_like(input_mask)
    penalty = F.cross_entropy(input_mask, target)
    return penalty

class WGANGinkaLoss:
    def __init__(self, lambda_gp=100, weight=[1, 0.4, 20, 0.2, 0.2, 0.05, 0.4]):
        # weight: 
        # 1. 判别器损失及图块维持损失（可修改部分的已有内容不可修改）
        # 2. CE 损失
        # 3. 不可修改类型损失和非法图块损失
        # 4. 图块类型损失
        # 5. 入口存在性损失
        # 6. 多样性损失
        # 7. 密度损失
        self.lambda_gp = lambda_gp  # 梯度惩罚系数
        self.weight = weight
        
    def compute_gradient_penalty(self, critic, stage, real_data, fake_data, tag_cond, val_cond):
        # 进行插值
        batch_size = real_data.size(0)
        epsilon_data = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interp_data = interpolate_data(real_data, fake_data, epsilon_data).to(real_data.device)
        
        # 对图像进行反向传播并计算梯度
        interp_data.requires_grad_()
        
        d_score = critic(interp_data, stage, tag_cond, val_cond)
        
        # 计算梯度
        grad = torch.autograd.grad(
            outputs=d_score, inputs=interp_data,
            grad_outputs=torch.ones_like(d_score),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # 计算梯度的 L2 范数
        grad_norm = grad.reshape(batch_size, -1).norm(2, dim=1)
        # 计算梯度惩罚项
        gp_loss = ((grad_norm - 1.0) ** 2).mean()
        # print(grad_norm_topo.mean().item(), grad_norm_vis.mean().item())

        return gp_loss
        
    def discriminator_loss(
        self, critic, stage: int, real_data: torch.Tensor, fake_data: torch.Tensor,
        tag_cond: torch.Tensor, val_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ 判别器损失函数 """
        fake_data = F.softmax(fake_data, dim=1)
        real_scores = critic(real_data, stage, tag_cond, val_cond)
        fake_scores = critic(fake_data, stage, tag_cond, val_cond)
        
        # Wasserstein 距离
        d_loss = fake_scores.mean() - real_scores.mean()
        grad_loss = self.compute_gradient_penalty(critic, stage, real_data, fake_data, tag_cond, val_cond)
        
        total_loss = d_loss + self.lambda_gp * grad_loss
        
        return total_loss, d_loss

    def generator_loss(self, critic, stage, mask_ratio, real, fake: torch.Tensor, input, tag_cond, val_cond) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 生成器损失函数 """
        probs_fake = F.softmax(fake, dim=1)
        
        fake_scores = critic(probs_fake, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        ce_loss = F.cross_entropy(fake, real) * (1 - mask_ratio) # 蒙版越大，交叉熵损失权重越小
        immutable_loss = immutable_penalty_loss(fake, input, STAGE_CHANGEABLE[stage])
        constraint_loss = inner_constraint_loss(probs_fake)
        density_loss = compute_multi_density_loss(probs_fake, val_cond, DENSITY_STAGE[stage])
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            ce_loss * self.weight[1],
            immutable_loss * self.weight[2],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
            density_loss * self.weight[6],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
                    
        return sum(losses), ce_loss
    
    def generator_loss_total(self, critic, stage, fake, tag_cond, val_cond) -> torch.Tensor:
        probs_fake = F.softmax(fake, dim=1)
        
        fake_scores = critic(probs_fake, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        illegal_loss = illegal_penalty_loss(probs_fake, STAGE_ALLOWED[stage])
        constraint_loss = inner_constraint_loss(probs_fake)
        density_loss = compute_multi_density_loss(probs_fake, val_cond, DENSITY_STAGE[stage])
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            illegal_loss * self.weight[2],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
            density_loss * self.weight[6],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
            
        return sum(losses)
    
    def generator_loss_total_with_input(self, critic, stage, fake, input, tag_cond, val_cond) -> torch.Tensor:
        probs_fake = F.softmax(fake, dim=1)
        
        fake_scores = critic(probs_fake, stage, tag_cond, val_cond)
        minamo_loss = -torch.mean(fake_scores)
        immutable_loss = immutable_penalty_loss(fake, input, STAGE_CHANGEABLE[stage])
        constraint_loss = inner_constraint_loss(probs_fake)
        density_loss = compute_multi_density_loss(probs_fake, val_cond, DENSITY_STAGE[stage])
        
        fake_a, fake_b = fake.chunk(2, dim=0)
        
        losses = [
            minamo_loss * self.weight[0],
            immutable_loss * self.weight[2],
            constraint_loss * self.weight[3],
            -js_divergence(fake_a, fake_b, softmax=True) * self.weight[5],
            density_loss * self.weight[6],
        ]
        
        if stage == 1:
            # 第一个阶段检查入口存在性
            entrance_loss = entrance_constraint_loss(probs_fake)
            losses.append(entrance_loss * self.weight[4])
            
        return sum(losses)

    def generator_input_head_loss(self, critic, probs: torch.Tensor, tag_cond, val_cond) -> torch.Tensor:
        head_scores = -torch.mean(critic(probs, 0, tag_cond, val_cond))
        probs_a, probs_b = probs.chunk(2, dim=0)
        
        losses = [
            head_scores,
            input_head_illegal_loss(probs) * 50,
            -js_divergence(probs_a, probs_b, softmax=False) * 0.5
        ]
        
        return sum(losses)

class RNNGinkaLoss:
    def __init__(self):
        pass
    
    def rnn_loss(self, fake, target):
        target = F.one_hot(target, num_classes=32).float()
        return F.cross_entropy(fake, target)
