import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from minamo.model.model import MinamoModel
from shared.graph import convert_soft_map_to_graph

def wall_border_loss(pred: torch.Tensor, allow_border=[1, 11]):
    """地图最外层是否为墙"""
    # 计算 softmax 概率
    B, C, H, W = pred.shape

    # 构造一个 [H, W] 的布尔 mask，选取最外圈的像素
    border_mask = torch.zeros((H, W), dtype=torch.bool, device=pred.device)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    # 对允许的类别求概率和（即该像素为允许类别的总概率）
    allowed_prob = pred[:, allow_border, :, :].sum(dim=1)  # [B, H, W]

    # 只计算边界区域的损失：对于边界上的每个像素，要求 allowed_prob 越高越好
    border_allowed_prob = allowed_prob[:, border_mask]  # [B, N_border_pixels]

    # 损失为 -log(allowed_prob)
    loss = 1 - border_allowed_prob.mean()

    return loss

def internal_wall_loss(pred, wall_class=1, threshold=2.5):
    """
    针对内部区域（排除最外圈）设计的损失函数：
    当内部任意 2×2 区域的 wall 类别概率之和超过阈值时，施加惩罚。

    参数:
        pred: 模型输出，形状 [B, C, H, W]
        wall_class: 对应墙壁的类别索引（这里假设墙壁数字为1）
        threshold: 2×2 区域概率之和的阈值，超过此值时施加惩罚。可根据实际情况调节。
    
    返回:
        loss: 内部墙壁连续区域的平均惩罚损失
    """    
    # 取出对应墙壁类别的概率图 [B, H, W]
    wall_probs = pred[:, wall_class, :, :]
    
    # 排除最外圈，取内部区域 (H, W 均减去2)
    interior = wall_probs[:, 1:-1, 1:-1]  # [B, H-2, W-2]
    
    # 构造一个 2×2 的卷积核，全为 1，用于检测局部连续墙壁的概率之和
    kernel = torch.ones((1, 1, 2, 2), device=pred.device)
    
    # 对内部区域进行卷积操作，计算每个 2×2 区域内的概率和
    # 需要将 interior 扩展一个通道维度
    conv_result = F.conv2d(interior.unsqueeze(1), kernel, stride=1, padding=0)
    # conv_result 的形状为 [B, 1, H-3, W-3]
    
    # 对于每个 2×2 区域，如果概率和超过 threshold，则产生惩罚
    # 这里采用 ReLU 计算超出部分，确保损失为非负
    penalty = F.relu(conv_result - threshold)
    
    # 取平均作为损失值
    loss = penalty.mean()
    return loss

def entrance_loss(pred, stairs_class=10, arrow_class=11):
    """
    针对地图生成的额外约束损失：
    - 保证最外圈不出现楼梯类型入口（数字10）
    - 保证内部区域不出现箭头类型入口（数字11）
    
    参数:
        pred: 模型输出，形状 [B, C, H, W]
        stairs_class: 楼梯入口对应的类别（数字10）
        arrow_class: 箭头入口对应的类别（数字11）
    
    返回:
        loss: 针对入口出现的惩罚损失
    """
    # 先将 logits 转为概率分布
    B, C, H, W = pred.shape

    # 构造最外圈 mask：外圈为 True，其余为 False
    outer_mask = torch.zeros((H, W), dtype=torch.bool, device=pred.device)
    outer_mask[0, :] = True
    outer_mask[-1, :] = True
    outer_mask[:, 0] = True
    outer_mask[:, -1] = True

    # 内部区域 mask
    interior_mask = ~outer_mask  # 取反

    # 提取对应类别的概率图
    stairs_probs = pred[:, stairs_class, :, :]  # 楼梯概率 [B, H, W]
    arrow_probs = pred[:, arrow_class, :, :]    # 箭头概率 [B, H, W]

    # 从最外圈提取楼梯概率；用 mask 索引时：张量[:, mask] 会将每个样本的外圈像素展平
    outer_stairs = stairs_probs[:, outer_mask]  # [B, num_outer_pixels]
    # 从内部区域提取箭头概率
    interior_arrow = arrow_probs[:, interior_mask]  # [B, num_interior_pixels]

    # 损失设计：使得这些概率尽量接近 0，直接使用均值惩罚
    outer_loss = outer_stairs.mean()
    interior_loss = interior_arrow.mean()

    total_loss = outer_loss + interior_loss
    return total_loss

def entrance_distance_and_presence_loss(
    pred,
    arrow_class=11, stairs_class=10, 
    arrow_min_threshold=0.5, stairs_min_threshold=0.5,
    lambda_arrow_presence=1.0, lambda_stairs_presence=1.0
):
    """
    入口损失同时考虑：
        1. 局部距离约束：防止同一类型入口过于靠近
        2. 存在性约束：鼓励至少放置一个入口
    
    箭头入口要求局部 (9x9) 内最多只有一个入口；
    楼梯入口要求在一个窗口（地图尺寸一半）内只出现一个楼梯入口。
    
    参数:
        pred: 模型输出, shape [B, C, H, W]
        arrow_class: 箭头入口类别（默认 11）
        stairs_class: 楼梯入口类别（默认 10）
        arrow_min_threshold: 箭头入口全局最小平均概率要求（可根据任务调节）
        stairs_min_threshold: 楼梯入口全局最小平均概率要求
        lambda_arrow_presence: 箭头入口存在性损失权重
        lambda_stairs_presence: 楼梯入口存在性损失权重
    返回:
        total_loss: 综合入口距离与存在性损失
    """
    # 将 logits 转换为概率分布
    B, C, H, W = pred.shape

    # 提取箭头和楼梯的概率图
    arrow_probs = pred[:, arrow_class, :, :]   # [B, H, W]
    stairs_probs = pred[:, stairs_class, :, :]   # [B, H, W]

    #### 局部距离约束 ####
    # 箭头：构造 9x9 卷积核，半径 4
    kernel_arrow = torch.ones((1, 1, 9, 9), device=pred.device)
    local_arrow_sum = F.conv2d(arrow_probs.unsqueeze(1), kernel_arrow, padding=4)
    # 减去自身概率，计算多余的局部累积
    arrow_excess = local_arrow_sum - arrow_probs.unsqueeze(1)
    arrow_distance_loss = F.relu(arrow_excess).mean()

    # 楼梯：使用窗口大小为 (W//2, H//2)
    kernel_size_stairs = (9, 9)
    kernel_stairs = torch.ones((1, 1, kernel_size_stairs[0], kernel_size_stairs[1]), device=pred.device)
    pad_stairs = ((kernel_size_stairs[0] - 1) // 2, (kernel_size_stairs[1] - 1) // 2)
    local_stairs_sum = F.conv2d(stairs_probs.unsqueeze(1), kernel_stairs, padding=pad_stairs)
    stairs_excess = local_stairs_sum - stairs_probs.unsqueeze(1)
    stairs_distance_loss = F.relu(stairs_excess).mean()

    #### 存在性约束 ####
    # 计算每个样本中箭头的最大概率
    global_arrow_max = arrow_probs.view(B, -1).max(dim=1)[0]  # [B]
    global_stairs_max = stairs_probs.view(B, -1).max(dim=1)[0]  # [B]
    
    # 取 batch 平均（或者你可以对每个样本分别计算损失再求平均）
    global_arrow_max = global_arrow_max.mean()
    global_stairs_max = global_stairs_max.mean()

    # 如果全局均值低于预期阈值，则施加额外惩罚
    arrow_presence_loss = F.relu(arrow_min_threshold - global_arrow_max)
    stairs_presence_loss = F.relu(stairs_min_threshold - global_stairs_max)
    
    ap_weighted = lambda_arrow_presence * arrow_presence_loss
    sp_weighted = lambda_stairs_presence * stairs_presence_loss

    # 总入口损失：局部距离约束 + 存在性约束（加权）
    total_loss = arrow_distance_loss + stairs_distance_loss \
                 + min(ap_weighted, sp_weighted)
    return total_loss

def monster_consecutive_loss(pred, monster_classes=[7,8,9], threshold=2.9):
    """
    检查横向和纵向是否存在连续超过三个的怪物（类别 7,8,9）。
    
    参数:
      pred: 模型输出，形状 [B, C, H, W]
      monster_classes: 待检测的怪物类别列表
      threshold: 滑动窗口内概率和的阈值，若超过则施加惩罚
                 （对于连续三个像素，如果每个像素概率接近 1，则窗口和接近 3）
    
    返回:
      loss: 惩罚损失（数值越高表示连续怪物区域越严重）
    """
    # 将 logits 转换为概率分布
    B, C, H, W = pred.shape
    
    # 得到怪物整体概率图：将类别 7,8,9 的概率相加
    monster_probs = pred[:, monster_classes, :].sum(dim=1)  # [B, H, W]
    
    # 注意：monster_probs 越高说明该像素更有可能是怪物
    
    # --- 横向检测 ---
    # 构造一个 (1,3) 的卷积核，全 1
    kernel_horiz = torch.ones((1, 1, 1, 3), device=pred.device)
    # 对 monster_probs 加一个 channel 维度，使形状为 [B, 1, H, W]
    conv_horiz = F.conv2d(monster_probs.unsqueeze(1), kernel_horiz, padding=(0,1))
    # conv_horiz 的每个值表示相邻三个像素的怪物概率和
    
    # --- 纵向检测 ---
    # 构造一个 (3,1) 的卷积核，全 1
    kernel_vert = torch.ones((1, 1, 3, 1), device=pred.device)
    conv_vert = F.conv2d(monster_probs.unsqueeze(1), kernel_vert, padding=(1,0))
    # conv_vert 的每个值表示垂直连续三个像素的怪物概率和
    
    # 对两个方向的窗口，如果概率和超过阈值，则计算超出部分的惩罚
    penalty_horiz = F.relu(conv_horiz - threshold)
    penalty_vert  = F.relu(conv_vert - threshold)
    
    # 将两个方向的惩罚损失取平均（或者直接相加）
    loss = penalty_horiz.mean() + penalty_vert.mean()
    return loss

def illegal_block_loss(pred, used_classes=12, mode='mean'):
    """
    对未使用类别（例如 12 ~ 31）的预测概率施加惩罚，
    鼓励模型输出仅集中在 0 ~ 11 上。
    
    参数:
      pred: 模型输出，形状 [B, num_classes, H, W]
      used_classes: 已经使用的类别数（例如 12 表示只使用 0-11）
      mode: 'mean' 使用平均概率，或 'mse' 使用均方误差
    
    返回:
      penalty: 标量惩罚损失
    """
    B, C, H, W = pred.shape
    # 选取非法类别的概率（注意：这一步会得到非法图块在每个像素上的概率）
    illegal_probs = pred[:, range(used_classes, 32), :, :]  # [B, len(illegal_classes), H, W]
    
    # 我们可以将非法图块的概率在类别维度上求和，得到每个像素的非法激活值
    illegal_activation = illegal_probs.sum(dim=1)  # [B, H, W]
    
    # 接下来我们计算整个图上非法激活的“数量”
    # 例如，可以直接对整个 batch 内非法激活求和
    total_illegal = illegal_activation.sum() / B  # 标量
    
    # 计算损失值：使用负指数函数。注意如果非法激活很小，总损失接近 exp(0)=1
    loss = torch.sqrt(total_illegal).mean()
    return loss

def integrated_count_loss(probs, target, class_list=[0,1,2,3,4,5,6,7,8,9], tolerance=0.5):
    """
    对每个类别分别计算数量匹配损失，再取平均。
    
    参数:
      probs: 模型输出的概率，形状 [B, num_classes, H, W]
      target: 真实标签，形状 [B, H, W]，类别取值在 0 ~ 使用范围-1 内
      class_list: 需要计算的类别列表
      tolerance: 每个类别允许的相对误差（例如 0.15 表示 15%）
    
    返回:
      loss: 对每个类别数量匹配损失取平均后的标量
    """
    total_loss = 0.0
    count = 0
    B, C, H, W = probs.shape
    
    for cls in class_list:
        # 预测数量：对于当前类别，所有像素的预测概率和
        pred_count = probs[:, cls, :, :].sum()
        # 真实数量：统计 target 中属于当前类别的像素数量
        true_count = (target == cls).float().sum()
        
        if true_count == 0:
            # 参考地图中不包含该类别，允许最多出现 (sqrt(地图尺寸) / 2) 个单位的概率输出
            cls_loss = F.relu(pred_count - math.sqrt(H * W) / 2)
        else:
            # 计算相对误差
            rel_error = torch.abs(pred_count - true_count) / (true_count)
            cls_loss = F.relu(rel_error - tolerance)
        
        total_loss += cls_loss
        count += 1
        
    # 求平均每个类别的损失
    avg_loss = total_loss / count
    return avg_loss

class GinkaLoss(nn.Module):
    def __init__(self, minamo: MinamoModel, weight=[0.35, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1]):
        """Ginka Model 损失函数部分

        Args:
            weight (list, optional): 每一个损失函数的权重，从第 0 项开始，依次是：
                1. 拓扑图损失
                2. 外圈墙壁损失
                3. 内层 2*2 墙壁损失
                4. 要求外层只能有箭头，内层只能有楼梯的损失
                5. 入口间距及存在性损失
                6. 连续怪物损失
                7. 非法图块损失
                8. 怪物、道具、门数量损失
        """
        super().__init__()
        self.weight = weight
        self.minamo = minamo
        
    def forward(self, pred, target, target_vision_feat, target_topo_feat):
        # 地图结构损失
        border_loss = wall_border_loss(pred)
        wall_loss = internal_wall_loss(pred)
        entry_loss = entrance_loss(pred)
        entry_dis_loss = entrance_distance_and_presence_loss(pred, )
        enemy_loss = monster_consecutive_loss(pred)
        valid_block_loss = illegal_block_loss(pred, used_classes=12, mode="mean")
        count_loss = integrated_count_loss(pred, target)
        
        # 使用 Minamo Model 计算相似度
        graph = convert_soft_map_to_graph(pred)
        pred_vision_feat, pred_topo_feat = self.minamo(pred, graph)
        
        vision_sim = F.cosine_similarity(pred_vision_feat, target_vision_feat, dim=-1)
        topo_sim = F.cosine_similarity(pred_topo_feat, target_topo_feat, dim=-1)
        minamo_sim = 0.3 * vision_sim + 0.7 * topo_sim
        minamo_loss = torch.exp(-1 * (minamo_sim - 0.8)).mean()
        
        print(
            minamo_loss.item(),
            border_loss.item(),
            wall_loss.item(),
            entry_loss.item(),
            entry_dis_loss.item(),
            enemy_loss.item(),
            valid_block_loss.item(),
            count_loss.item()
        )
        
        return (
            minamo_loss * self.weight[0] +
            border_loss * self.weight[1] +
            wall_loss * self.weight[2] +
            entry_loss * self.weight[3] +
            entry_dis_loss * self.weight[4] +
            enemy_loss * self.weight[5] +
            valid_block_loss * self.weight[6] +
            count_loss * self.weight[7]
        )