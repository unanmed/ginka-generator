import argparse
import math
import os
import sys
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from .vqvae.quantize import VectorQuantizer
from .vqvae.model import GinkaVQVAE
from .maskGIT.model import GinkaMaskGIT
from .dataset import GinkaSeperatedDataset
from shared.image import matrix_to_image_cv

# 三阶段级联地图生成训练脚本
#
# 整体架构：
#   VQ-VAE（三组独立编码器 vq1/vq2/vq3）将三阶段地图上下文分别编码为离散潜变量，
#   再由共用 VectorQuantizer 统一量化为 z_q；
#   三个独立 MaskGIT（mg1/mg2/mg3）分别以 z_q 和 struct_inject 为条件，
#   逐阶段迭代解码地图图块序列。
#
# 三阶段生成目标：
#   stage1 → floor / wall（地图骨架）
#   stage2 → door / monster / entrance（功能性实体）
#   stage3 → resource（资源点）

# 图块 ID 定义：
# 0. 空地   1. 墙壁   2. 门   3. 资源   4. 怪物   5. 入口   6. 掩码（MASK_TOKEN）

# 共用 VQ-VAE 超参
# 三组编码器（vq1/vq2/vq3）共享相同超参，分别对三阶段地图上下文独立编码
VQ_L = 2 # 码字序列长度（每个编码器输出 L 个码字，量化后合并为 L*3）
VQ_K = 8 # codebook 大小（离散码本条目数）
VQ_D_Z = 64 # 码字维度
VQ_BETA = 0.5 # commit loss 权重（防止编码器输出漂离 codebook）
VQ_GAMMA = 0.0 # entropy loss 权重（当前未启用）
VQ_LAYERS = 3 # VQ-VAE Transformer 层数
VQ_DIM_FF = 512 # VQ-VAE 前馈网络隐层维度
VQ_D_MODEL = 128 # VQ-VAE Transformer 模型维度
VQ_NHEAD = 4 # VQ-VAE 多头注意力头数

# 第一阶段 MaskGIT 超参
STAGE1_MG_DMODEL = 256
STAGE1_MG_NHEAD = 4
STAGE1_MG_NUM_LAYERS = 6
STAGE1_MG_DIM_FF = 1024

# 第二阶段 MaskGIT 超参
STAGE2_MG_DMODEL = 256
STAGE2_MG_NHEAD = 4
STAGE2_MG_NUM_LAYERS = 6
STAGE2_MG_DIM_FF = 1024

# 第三阶段 MaskGIT 超参
STAGE3_MG_DMODEL = 256
STAGE3_MG_NHEAD = 4
STAGE3_MG_NUM_LAYERS = 6
STAGE3_MG_DIM_FF = 1024

# 三阶段 Focal Loss 损失权重（可调节各阶段对总损失的贡献比例）
STAGE1_FOCAL_WEIGHT = 1.0
STAGE2_FOCAL_WEIGHT = 1.0
STAGE3_FOCAL_WEIGHT = 1.0

# 各阶段 VQ commit loss 权重（当前未单独使用，统一由 VQ_BETA 控制）
STAGE1_VQ_WEIGHT = 0.5
STAGE2_VQ_WEIGHT = 0.5
STAGE3_VQ_WEIGHT = 0.5

# 全局参数
NUM_CLASSES = 7 # 图块类型数
MASK_TOKEN = 6 # 掩码图块
MAP_W = 13 # 地图宽度
MAP_H = 13 # 地图高度
MAP_SIZE = MAP_W * MAP_H # 地图大小
GENERATE_STEP = 18 # MaskGIT 采样步数
SUBSET_WEIGHTS = (0.5, 0.3, 0.2) # 每个子集的概率

MG_Z_DROPOUT = 0.1 # z 隐变量 Dropout 概率
MG_STRUCT_DROPOUT = 0.1 # 结构参量 Dropout 概率

# 损失参数
FOCAL_GAMMA = 2.0 # Focal Loss 参数
VQ_BETA = 0.5 # 承诺损失权重

# 训练超参
BATCH_SIZE = 64 # 每批样本数
LR = 1e-4 # AdamW 初始学习率
MIN_LR = 1e-6 # 余弦退火最低学习率
WEIGHT_DECAY = 1e-4 # L2 正则化系数
EPOCHS = 400 # 总训练轮数
CHECKPOINT = 20 # 每隔多少 epoch 保存检查点并执行验证

device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

disable_tqdm = not sys.stdout.isatty()

def _str2bool(v: str):
    if isinstance(v, bool): return v
    if v.lower() in ('true', '1', 'yes'): return True
    if v.lower() in ('false', '0', 'no'): return False
    raise argparse.ArgumentTypeError(f"布尔值应为 True/False，收到: {v!r}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="三阶段级联训练")
    parser.add_argument("--resume", type=_str2bool, default=False)
    parser.add_argument("--state", type=str, default="", help="续训时检查点路径")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--load_optim", type=_str2bool, default=True)
    return parser.parse_args()

def build_model(device: torch.device):
    # 三组 VQ-VAE 编码器：各自独立编码一个阶段的地图上下文（encoder_stage1/2/3）
    # 输出形状均为 [B, L, d_z]，拼接后送入共用 quantizer
    vq_kwargs = dict(
        num_classes=NUM_CLASSES, L=VQ_L, K=VQ_K, d_model=VQ_D_MODEL,
        nhead=VQ_NHEAD, num_layers=VQ_LAYERS, dim_ff=VQ_DIM_FF, map_h=MAP_H, map_w=MAP_W
    )
    vq1 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage1 上下文（floor/wall）
    vq2 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage2 上下文（door/monster/entrance）
    vq3 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage3 上下文（resource）

    # 三个独立 MaskGIT 解码器，均接收完整的三阶段 z_q 作为条件
    mg1 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE1_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE1_MG_DIM_FF,
        nhead=STAGE1_MG_NHEAD, num_layers=STAGE1_MG_NUM_LAYERS, map_h=MAP_H, map_w=MAP_W
    ).to(device)
    mg2 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE2_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE2_MG_DIM_FF,
        nhead=STAGE2_MG_NHEAD, num_layers=STAGE2_MG_NUM_LAYERS, map_h=MAP_H, map_w=MAP_W
    ).to(device)
    mg3 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE3_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE3_MG_DIM_FF,
        nhead=STAGE3_MG_NHEAD, num_layers=STAGE3_MG_NUM_LAYERS, map_h=MAP_H, map_w=MAP_W
    ).to(device)

    # 六个模型参数合并到同一优化器，端到端联合训练
    all_params = (
        list(vq1.parameters()) + list(vq2.parameters()) + list(vq3.parameters()) +
        list(mg1.parameters()) + list(mg2.parameters()) + list(mg3.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=LR, weight_decay=1e-4)
    # 余弦退火：从 LR 线性衰减至 MIN_LR，周期为全部训练轮数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

    # 共用 VectorQuantizer：不参与梯度更新，仅在前向时做码本查表
    quantizer = VectorQuantizer(K=VQ_K, d_z=VQ_D_Z).to(device)

    return vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler

def focal_loss(logits, target):
    # logits: [B, L, C]，需转为 [B, C, L] 以匹配 cross_entropy 期望格式
    ce = F.cross_entropy(logits.permute(0, 2, 1), target, reduction='none')
    pt = torch.exp(-ce)  # pt = 模型对正确类的预测概率
    # Focal Loss：对高置信度样本降低权重，让模型更专注于难样本
    focal = ((1 - pt) ** FOCAL_GAMMA) * ce
    return focal.mean()

def random_struct(device: torch.device) -> torch.Tensor:
    # 随机采样一组结构参量，用于无条件自由生成
    # struct_inject 格式：[cond_sym(0-7), cond_room(0-2), cond_branch(0-2), cond_outer(0-1)]
    cond_sym = random.randint(0, 7) # 地图对称类型
    cond_room = random.randint(0, 2) # 房间数量档位
    cond_branch = random.randint(0, 2) # 分支复杂度档位
    cond_outer = random.randint(0, 1) # 是否有外围走廊
    return torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer]).unsqueeze(0).to(device)

def random_density(device: torch.device) -> torch.Tensor:
    # 随机采样一组密度参量，用于自由生成
    # density_inject 格式：[door_level(0-2), monster_level(0-2), resource_level(0-2)]
    door_lv = random.randint(0, 2)
    monster_lv = random.randint(0, 2)
    resource_lv = random.randint(0, 2)
    return torch.LongTensor([door_lv, monster_lv, resource_lv]).unsqueeze(0).to(device)

def maskgit_sample(
    model: torch.nn.Module, inp: torch.Tensor, z: torch.Tensor,
    struct: torch.Tensor, density: torch.Tensor, steps: int,
    target_tiles: list[int] | None = None, keep_fixed: bool = True
) -> np.ndarray:
    # target_tiles: 本阶段负责生成的图块 ID 列表；None 表示接受所有类别（stage1）
    # keep_fixed=True：锁定输入中已有的非掩码/非空地位，使上一阶段结构保持不变
    # keep_fixed=False：结构位保留，但每步结束后空地重新标为 MASK（探索模式）
    #
    # 有 target_tiles 时的核心逻辑：
    #   每步只从预测中选出置信度最高的若干 target_tile 候选揭开，
    #   其余已有结构（墙/门等非空地非掩码）原样保留，
    #   空地与掩码保持为 MASK，等待后续步骤继续填充。
    current = inp.clone()
    has_target = target_tiles is not None
    if has_target:
        target_tensor = torch.tensor(target_tiles, dtype=torch.long, device=inp.device)

    # 迭代去掩码：每步根据置信度分数重新决定掩码位置
    for step in range(steps):
        logits = model(current, z, struct, density)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()

        confidences = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

        # 余弦退火调度：随步数推进，保留掩码的位置数量递减至 0
        ratio = math.cos(((step + 1) / steps) * math.pi / 2)
        num_to_mask = math.floor(ratio * MAP_SIZE)

        if not has_target:
            # stage1：无 target 约束，仅锁定 fixed 位（若 keep_fixed）
            if keep_fixed:
                fixed_mask = (current[0] != MASK_TOKEN)
                sampled[0, fixed_mask] = current[0, fixed_mask]
                confidences[0, fixed_mask] = 1.0
            if num_to_mask > 0:
                _, mask_indices = torch.topk(confidences[0], k=num_to_mask, largest=False)
                sampled[0].scatter_(0, mask_indices, MASK_TOKEN)
            current = sampled
        else:
            # 有 target_tiles：基于当前 current 构建下一状态
            # 结构位：current 中非空地、非掩码的位置（来自上一阶段，始终保留）
            struct_mask = (current[0] != MASK_TOKEN) & (current[0] != 0)
            # 候选位：sampled 为目标图块且不覆盖结构位
            candidate_mask = torch.isin(sampled[0], target_tensor) & ~struct_mask
            # 对候选位按置信度排序，选出置信度最高的若干位揭开
            cand_count = candidate_mask.sum().item()
            reveal_count = max(0, int(cand_count) - num_to_mask)
            next_state = current[0].clone()
            if reveal_count > 0 and cand_count > 0:
                cand_indices = candidate_mask.nonzero(as_tuple=True)[0]
                cand_conf = confidences[0][cand_indices]
                top_k = min(reveal_count, cand_conf.size(0))
                _, top_idx = torch.topk(cand_conf, k=top_k, largest=True)
                reveal_indices = cand_indices[top_idx]
                next_state[reveal_indices] = sampled[0][reveal_indices]
                # 后处理：进度未超 75% 时，随机将新揭开位的 20%-40% 再次掩码，
                # 抑制目标图块过密生成；后期不再压制，确保最终能全部揭开
                if step / steps <= 0.75:
                    suppress_ratio = random.uniform(0.2, 0.4)
                    suppress_k = max(1, int(reveal_indices.size(0) * suppress_ratio))
                    suppress_perm = torch.randperm(
                        reveal_indices.size(0), device=inp.device
                    )[:suppress_k]
                    next_state[reveal_indices[suppress_perm]] = MASK_TOKEN
            # 结构位原样保留，其余未揭开的置为 MASK
            non_struct_non_revealed = (next_state == current[0]) & ~struct_mask
            next_state[non_struct_non_revealed & (next_state != MASK_TOKEN)] = MASK_TOKEN
            # free 模式下，空地也重新标为 MASK（允许下一步继续填充）
            if not keep_fixed:
                next_state[next_state == 0] = MASK_TOKEN
            current = next_state.unsqueeze(0)

        if (current[0] == MASK_TOKEN).sum() == 0:
            break

    # 兜底：若仍有残余掩码位，按模式填充
    still_masked = (current[0] == MASK_TOKEN)
    if still_masked.any():
        if has_target:
            # 目标模式下，未被填充的位置视为空地（不属于本阶段负责的图块）
            current[0, still_masked] = 0
        else:
            logits = model(current, z, struct, density)
            current[0, still_masked] = torch.argmax(logits[0, still_masked], dim=-1)

    return current[0].cpu().numpy().reshape(MAP_H, MAP_W)

def full_generate_random_z(
    input: torch.Tensor,
    struct: torch.Tensor,
    density: torch.Tensor,
    models: list[torch.nn.Module],
    device: torch.device,
    keep_fixed: tuple[bool, bool, bool] = (True, True, True)
) -> tuple:
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    with torch.no_grad():
        z = quantizer.sample(1, VQ_L, device)

        # stage1：生成 floor/wall 骨架
        pred1_np = maskgit_sample(mg1, input.clone(), z, struct, density, GENERATE_STEP, keep_fixed=keep_fixed[0])
        inp2 = torch.tensor(pred1_np.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp2[inp2 == 0] = MASK_TOKEN  # 空地位交由 stage2 填充

        # stage2：在骨架上生成 door(2)/monster(4)/entrance(5)，非零结果覆盖合并
        pred2_np = maskgit_sample(
            mg2, inp2, z, struct, density, GENERATE_STEP, target_tiles=[2, 4, 5], keep_fixed=keep_fixed[1]
        )
        merged12 = pred1_np.copy()
        merged12[pred2_np != 0] = pred2_np[pred2_np != 0]
        inp3 = torch.tensor(merged12.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp3[inp3 == 0] = MASK_TOKEN

        # stage3：填充 resource(3)
        pred3_np = maskgit_sample(
            mg3, inp3, z, struct, density, GENERATE_STEP, target_tiles=[3], keep_fixed=keep_fixed[2]
        )
        merged123 = merged12.copy()
        merged123[pred3_np != 0] = pred3_np[pred3_np != 0]

    return pred1_np, merged12, merged123

def full_generate_specific_z(
    input: torch.Tensor,
    z: torch.Tensor,
    struct: torch.Tensor,
    density: torch.Tensor,
    models: list[torch.nn.Module],
    device: torch.device,
    keep_fixed: tuple[bool, bool, bool] = (True, True, True)
) -> tuple:
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    with torch.no_grad():
        # 与 full_generate_random_z 相同的三阶段级联，但使用给定的 z
        pred1_np = maskgit_sample(mg1, input.clone(), z, struct, density, GENERATE_STEP, keep_fixed=keep_fixed[0])
        inp2 = torch.tensor(pred1_np.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp2[inp2 == 0] = MASK_TOKEN

        pred2_np = maskgit_sample(
            mg2, inp2, z, struct, density, GENERATE_STEP, target_tiles=[2, 4, 5], keep_fixed=keep_fixed[1]
        )
        merged12 = pred1_np.copy()
        merged12[pred2_np != 0] = pred2_np[pred2_np != 0]
        inp3 = torch.tensor(merged12.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp3[inp3 == 0] = MASK_TOKEN

        pred3_np = maskgit_sample(
            mg3, inp3, z, struct, density, GENERATE_STEP, target_tiles=[3], keep_fixed=keep_fixed[2]
        )
        merged123 = merged12.copy()
        merged123[pred3_np != 0] = pred3_np[pred3_np != 0]

    return pred1_np, merged12, merged123

def annotate(img: np.ndarray, text: str) -> np.ndarray:
    # 在图片左上角叠加文字标注（黑色描边 + 白色填充，确保任意背景下可读）
    img = img.copy()
    cv2.putText(img, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(img, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

def annotate_labels(
    img: np.ndarray,
    struct: torch.Tensor,
    density: torch.Tensor
) -> np.ndarray:
    # 两行标注：第一行结构标签，第二行密度标签
    lv = ['Low', 'Medium', 'High']
    s = struct.tolist()
    d = density.tolist()
    line1 = f"sym:{s[0]} room:{lv[s[1]]} branch:{lv[s[2]]} outer:{s[3]}"
    line2 = f"door:{lv[d[0]]} enemy:{lv[d[1]]} res:{lv[d[2]]}"
    img = img.copy()
    for text, y in [(line1, 12), (line2, 24)]:
        cv2.putText(img, text, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(img, text, (2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return img

def rand_keep() -> tuple[bool, bool, bool]:
    b = random.choice([True, False])
    return (b, b, b)

def keep_label(kf: tuple[bool, bool, bool]) -> str:
    return 'fix' if kf[0] else 'free'

# 验证可视化 part1：3×3 网格；行1=编码器输入，行2=掩码输入，行3=三阶段预测（合并）
def visualize_part1(batch, logits1, logits2, logits3, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    pred1 = torch.argmax(logits1[0], dim=-1).cpu().numpy().reshape(MAP_H, MAP_W)
    pred2 = torch.argmax(logits2[0], dim=-1).cpu().numpy().reshape(MAP_H, MAP_W)
    pred3 = torch.argmax(logits3[0], dim=-1).cpu().numpy().reshape(MAP_H, MAP_W)

    enc1_np = batch["encoder_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    enc2_np = batch["encoder_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    enc3_np = batch["encoder_stage3"][0].numpy().reshape(MAP_H, MAP_W)
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    inp2_np = batch["input_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    inp3_np = batch["input_stage3"][0].numpy().reshape(MAP_H, MAP_W)

    # 将各阶段掩码输入中的 MASK 位用模型预测值填充，保留非掩码位原值
    result1 = inp1_np.copy()
    result1[inp1_np == MASK_TOKEN] = pred1[inp1_np == MASK_TOKEN]
    result2 = inp2_np.copy()
    result2[inp2_np == MASK_TOKEN] = pred2[inp2_np == MASK_TOKEN]
    result3 = inp3_np.copy()
    result3[inp3_np == MASK_TOKEN] = pred3[inp3_np == MASK_TOKEN]

    rows = [
        [to_img(enc1_np), to_img(enc2_np), to_img(enc3_np)],
        [to_img(inp1_np), to_img(inp2_np), to_img(inp3_np)],
        [to_img(result1), to_img(result2), to_img(result3)],
    ]
    grid = np.ones((3 * img_h + 4 * SEP, 3 * img_w + 4 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

# 验证可视化 part2：行1=真实地图三阶段，行2=stage1 输入与使用真实 z 自回归生成的各阶段结果
def visualize_part2(batch, z_q, models, device, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    inp1_t = batch["input_stage1"][0:1].to(device).reshape(1, MAP_SIZE)
    struct_t = batch["struct_inject"][0:1].to(device)
    density_t = batch["density_inject"][0:1].to(device)
    kf = rand_keep()
    auto_pred1_np, auto_merged12, auto_merged123 = full_generate_specific_z(
        inp1_t, z_q[0:1], struct_t, density_t, models, device, keep_fixed=kf
    )
    kf_label = 'fix' if kf[0] else 'free'

    enc1_np = batch["encoder_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    enc2_np = batch["encoder_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    enc3_np = batch["encoder_stage3"][0].numpy().reshape(MAP_H, MAP_W)
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)

    struct_cpu = batch["struct_inject"][0]
    density_cpu = batch["density_inject"][0]

    rows = [
        [to_img(enc1_np), to_img(enc2_np), to_img(enc3_np)],
        [
            annotate(to_img(inp1_np), kf_label),
            annotate_labels(to_img(auto_pred1_np), struct_cpu, density_cpu),
            annotate_labels(to_img(auto_merged12), struct_cpu, density_cpu),
            annotate_labels(to_img(auto_merged123), struct_cpu, density_cpu)
        ],
    ]
    grid = np.ones((2 * img_h + 3 * SEP, 4 * img_w + 5 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

# 验证可视化 part3：2×3 网格；行1=参考输入+相同 struct 随机 z 生成，行2=随机 struct 生成
def visualize_part3(batch, models, device, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    inp1_t = batch["input_stage1"][0:1].to(device).reshape(1, MAP_SIZE)
    struct_ref = batch["struct_inject"][0:1].to(device)
    density_ref = batch["density_inject"][0:1].to(device)
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    struct_cpu = batch["struct_inject"][0]
    density_cpu = batch["density_inject"][0]

    row1 = [to_img(inp1_np)]
    for _ in range(2):
        kf = rand_keep()
        _, _, merged123 = full_generate_random_z(inp1_t, struct_ref, density_ref, models, device, keep_fixed=kf)
        row1.append(annotate_labels(to_img(merged123), struct_cpu, density_cpu))

    row2 = []
    for _ in range(3):
        kf = rand_keep()
        rnd_struct = random_struct(device)
        rnd_density = random_density(device)
        _, _, merged123 = full_generate_random_z(inp1_t, rnd_struct, rnd_density, models, device, keep_fixed=kf)
        row2.append(annotate_labels(to_img(merged123), rnd_struct[0].cpu(), rnd_density[0].cpu()))

    rows = [row1, row2]
    grid = np.ones((2 * img_h + 3 * SEP, 3 * img_w + 4 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

# 验证可视化 part4：2×3 网格；以少量随机墙壁作为种子，纯随机 struct+z 自由生成
def visualize_part4(models, device, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    n_walls = random.randint(math.floor(MAP_SIZE * 0.02), math.floor(MAP_SIZE * 0.06))
    seed = torch.full((1, MAP_SIZE), MASK_TOKEN, dtype=torch.long, device=device)
    wall_pos = torch.randperm(MAP_SIZE, device=device)[:n_walls]
    seed[0, wall_pos] = 1
    seed_np = seed[0].cpu().numpy().reshape(MAP_H, MAP_W)

    results = []
    for _ in range(5):
        kf = rand_keep()
        rnd_struct = random_struct(device)
        rnd_density = random_density(device)
        _, _, merged123 = full_generate_random_z(seed, rnd_struct, rnd_density, models, device, keep_fixed=kf)
        results.append(annotate_labels(to_img(merged123), rnd_struct[0].cpu(), rnd_density[0].cpu()))

    row1 = [to_img(seed_np)] + results[:2]
    row2 = results[2:]
    rows = [row1, row2]
    grid = np.ones((2 * img_h + 3 * SEP, 3 * img_w + 4 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

def visualize_validate(
    batch, logits1, logits2, logits3, z_q,
    models: list[torch.nn.Module], device: torch.device, tile_dict, epoch: int, batch_idx: int
):
    save_dir = f"result/seperated/e{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f"{save_dir}/val{batch_idx}.png", visualize_part1(batch, logits1, logits2, logits3, tile_dict))
    cv2.imwrite(f"{save_dir}/full{batch_idx}.png", visualize_part2(batch, z_q, models, device, tile_dict))
    cv2.imwrite(f"{save_dir}/rand{batch_idx}.png", visualize_part3(batch, models, device, tile_dict))
    cv2.imwrite(f"{save_dir}/dvar{batch_idx}.png", visualize_density_var(batch, z_q, models, device, tile_dict))

# 密度对照图：随机种子+随机结构，5 张随机密度生成，2×3 网格（左上角为种子图）
def visualize_density_cmp(models, device, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    n_walls = random.randint(math.floor(MAP_SIZE * 0.02), math.floor(MAP_SIZE * 0.06))
    seed = torch.full((1, MAP_SIZE), MASK_TOKEN, dtype=torch.long, device=device)
    wall_pos = torch.randperm(MAP_SIZE, device=device)[:n_walls]
    seed[0, wall_pos] = 1
    seed_np = seed[0].cpu().numpy().reshape(MAP_H, MAP_W)
    rnd_struct = random_struct(device)
    struct_cpu = rnd_struct[0].cpu()
    gen_imgs = []
    for _ in range(5):
        rnd_density = random_density(device)
        density_cpu = rnd_density[0].cpu()
        _, _, merged123 = full_generate_random_z(seed, rnd_struct, rnd_density, models, device)
        gen_imgs.append(annotate_labels(to_img(merged123), struct_cpu, density_cpu))
    row1 = [to_img(seed_np)] + gen_imgs[:2]
    row2 = gen_imgs[2:]
    rows = [row1, row2]
    grid = np.ones((2 * img_h + 3 * SEP, 3 * img_w + 4 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

# 固定 z 和结构条件，使用 5 个随机密度各生成一次，2×3 网格（左上角为参考地图）
def visualize_density_var(batch, z_q, models, device, tile_dict):
    SEP = 3
    TILE_SIZE = 32
    img_h = MAP_H * TILE_SIZE
    img_w = MAP_W * TILE_SIZE

    def to_img(mat):
        return matrix_to_image_cv(mat, tile_dict, TILE_SIZE)

    inp1_t = batch["input_stage1"][0:1].to(device).reshape(1, MAP_SIZE)
    struct_t = batch["struct_inject"][0:1].to(device)
    struct_cpu = batch["struct_inject"][0]
    ref_np = batch["encoder_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    gen_imgs = []
    for _ in range(5):
        rnd_density = random_density(device)
        density_cpu = rnd_density[0].cpu()
        _, _, merged123 = full_generate_specific_z(
            inp1_t, z_q[0:1], struct_t, rnd_density, models, device
        )
        gen_imgs.append(annotate_labels(to_img(merged123), struct_cpu, density_cpu))
    row1 = [to_img(ref_np)] + gen_imgs[:2]
    row2 = gen_imgs[2:]
    rows = [row1, row2]
    grid = np.ones((2 * img_h + 3 * SEP, 3 * img_w + 4 * SEP, 3), dtype=np.uint8) * 255
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = SEP + r * (img_h + SEP)
            x = SEP + c * (img_w + SEP)
            grid[y:y + img_h, x:x + img_w] = img
    return grid

def validate(dataloader: DataLoader, models: list[torch.nn.Module], device: torch.device, tile_dict, epoch: int):
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    # 切换为推理模式（关闭 Dropout / BatchNorm 统计更新）
    for m in [vq1, vq2, vq3, mg1, mg2, mg3]:
        m.eval()

    # 累计各阶段损失（跨所有 batch 求和，最终除以 batch 数得到均值）
    loss1_total = torch.Tensor([0]).to(device)
    loss2_total = torch.Tensor([0]).to(device)
    loss3_total = torch.Tensor([0]).to(device)
    commit_total = torch.Tensor([0]).to(device)

    # 按档位（0/1/2）累计实体计数差（L1），用于诊断密度条件可控性
    # 结构：{tile_id: {level: [累计误差, 样本数]}}
    density_l1 = {
        2: {0: [0.0, 0], 1: [0.0, 0], 2: [0.0, 0]}, # door
        4: {0: [0.0, 0], 1: [0.0, 0], 2: [0.0, 0]}, # monster
        3: {0: [0.0, 0], 1: [0.0, 0], 2: [0.0, 0]}, # resource
    }
    # 三类实体对应的 density_inject 索引
    tile_density_idx = {2: 0, 4: 1, 3: 2}

    idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="Validate Progress", disable=disable_tqdm):

            # 三阶段各自的掩码输入、预测目标和 VQ 编码器输入
            inp1 = batch["input_stage1"].to(device).reshape(-1, MAP_SIZE)
            target1 = batch["target_stage1"].to(device).reshape(-1, MAP_SIZE)
            enc1 = batch["encoder_stage1"].to(device).reshape(-1, MAP_SIZE)

            inp2 = batch["input_stage2"].to(device).reshape(-1, MAP_SIZE)
            target2 = batch["target_stage2"].to(device).reshape(-1, MAP_SIZE)
            enc2 = batch["encoder_stage2"].to(device).reshape(-1, MAP_SIZE)

            inp3 = batch["input_stage3"].to(device).reshape(-1, MAP_SIZE)
            target3 = batch["target_stage3"].to(device).reshape(-1, MAP_SIZE)
            enc3 = batch["encoder_stage3"].to(device).reshape(-1, MAP_SIZE)

            struct = batch["struct_inject"].to(device)
            density = batch["density_inject"].to(device)

            # VQ 编码：各阶段独立编码后拼接、量化
            z_e1 = vq1(enc1) # [B, L, d_z]
            z_e2 = vq2(enc2)
            z_e3 = vq3(enc3)

            z_e_all = torch.cat([z_e1, z_e2, z_e3], dim=1) # [B, L*3, d_z]
            z_q, _, commit_loss = quantizer(z_e_all) # [B, L*3, d_z]

            # 三阶段 MaskGIT 推理（均以完整 z_q、struct 和 density 为条件）
            logits1 = mg1(inp1, z_q, struct, density)
            logits2 = mg2(inp2, z_q, struct, density)
            logits3 = mg3(inp3, z_q, struct, density)

            loss1_total += focal_loss(logits1, target1)
            loss2_total += focal_loss(logits2, target2)
            loss3_total += focal_loss(logits3, target3)
            commit_total += commit_loss

            # 计算 argmax 预测并统计各档位密度 L1（预测计数与真实计数之差的绝对值）
            pred2_map = torch.argmax(logits2, dim=-1).cpu() # [B, MAP_SIZE]
            pred3_map = torch.argmax(logits3, dim=-1).cpu()
            true2_map = target2.cpu() # [B, MAP_SIZE]
            true3_map = target3.cpu()
            density_cpu = batch["density_inject"] # [B, 3]
            for b in range(pred2_map.size(0)):
                for tile_id, d_idx in tile_density_idx.items():
                    if tile_id == 3:
                        pred_map = pred3_map[b]
                        true_map = true3_map[b]
                    else:
                        pred_map = pred2_map[b]
                        true_map = true2_map[b]
                    pred_count = float((pred_map == tile_id).sum().item())
                    true_count = float((true_map == tile_id).sum().item())
                    lv = int(density_cpu[b, d_idx].item())
                    density_l1[tile_id][lv][0] += abs(pred_count - true_count)
                    density_l1[tile_id][lv][1] += 1

            # 每个 batch 生成三种可视化图（val/full/rand）
            visualize_validate(batch, logits1, logits2, logits3, z_q, models, device, tile_dict, epoch, idx)
            idx += 1

    # 输出密度 L1 统计（各档位的平均实体计数，供诊断密度条件效果）
    lv_names = ['Low', 'Medium', 'High']
    tile_names = {2: 'door', 4: 'enemy', 3: 'resource'}
    for tile_id in [2, 4, 3]:
        parts = []
        for lv in range(3):
            acc, cnt = density_l1[tile_id][lv]
            avg = acc / cnt if cnt > 0 else 0.0
            parts.append(f"{lv_names[lv]}={avg:.2f}")
        tqdm.write(f"  density {tile_names[tile_id]}: {' '.join(parts)}")

    save_dir = f"result/seperated/e{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    # 每个 epoch 额外生成：无条件自由生成图 + 全局密度对照图
    cv2.imwrite(f"{save_dir}/free.png", visualize_part4(models, device, tile_dict))
    cv2.imwrite(f"{save_dir}/density_cmp.png", visualize_density_cmp(models, device, tile_dict))

    # 恢复训练模式
    for m in [vq1, vq2, vq3, mg1, mg2, mg3]:
        m.train()

    return loss1_total, loss2_total, loss3_total, commit_total

def train(device: torch.device):
    args = parse_arguments()

    models = build_model(device)
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    tqdm.write(f"Device: {device}")
    model_list = [
        ("vq1", vq1), ("vq2", vq2), ("vq3", vq3),
        ("mg1", mg1), ("mg2", mg2), ("mg3", mg3),
        ("quantizer", quantizer)
    ]
    total_params = 0
    for name, m in model_list:
        n = sum(p.numel() for p in m.parameters())
        total_params += n
        tqdm.write(f"{name}: {n:,} params")
    tqdm.write(f"Total: {total_params:,} params")

    start_epoch = 0

    if args.resume:
        # 从指定检查点恢复：加载所有模型权重及训练状态
        ckpt = torch.load(args.state, map_location=device)
        vq1.load_state_dict(ckpt["vq1"])
        vq2.load_state_dict(ckpt["vq2"])
        vq3.load_state_dict(ckpt["vq3"])
        mg1.load_state_dict(ckpt["mg1"])
        mg2.load_state_dict(ckpt["mg2"])
        mg3.load_state_dict(ckpt["mg3"])
        quantizer.load_state_dict(ckpt["quantizer"])
        # load_optim=False 时可跳过优化器/调度器恢复（适合调整学习率后继续训练）
        if args.load_optim and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if args.load_optim and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)  # 从上次保存的 epoch 继续
        tqdm.write(f"Resumed from epoch {start_epoch}: {args.state}")

    os.makedirs("result/seperated", exist_ok=True)

    dataset = GinkaSeperatedDataset(
        args.train, subset_weights=SUBSET_WEIGHTS
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    dataset_val = GinkaSeperatedDataset(
        args.validate, subset_weights=SUBSET_WEIGHTS
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=min(BATCH_SIZE, len(dataset_val) // 8), shuffle=True
    )

    # 预加载图块图像，键为文件名（不含扩展名），用于可视化时将 ID 映射为像素图
    tile_dict = {}
    for f in os.listdir("tiles"):
        name = os.path.splitext(f)[0]
        img = cv2.imread(f"tiles/{f}", cv2.IMREAD_UNCHANGED)
        if img is not None:
            tile_dict[name] = img

    for epoch in tqdm(range(start_epoch, EPOCHS), desc="Seperated Training", disable=disable_tqdm):
        loss_total = torch.Tensor([0]).to(device)
        loss1_total = torch.Tensor([0]).to(device)
        loss2_total = torch.Tensor([0]).to(device)
        loss3_total = torch.Tensor([0]).to(device)
        commit_total = torch.Tensor([0]).to(device)

        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            # 三阶段各自的掩码输入序列、预测目标和编码器上下文
            inp1 = batch["input_stage1"].to(device).reshape(-1, MAP_SIZE)
            target1 = batch["target_stage1"].to(device).reshape(-1, MAP_SIZE)
            enc1 = batch["encoder_stage1"].to(device).reshape(-1, MAP_SIZE)

            inp2 = batch["input_stage2"].to(device).reshape(-1, MAP_SIZE)
            target2 = batch["target_stage2"].to(device).reshape(-1, MAP_SIZE)
            enc2 = batch["encoder_stage2"].to(device).reshape(-1, MAP_SIZE)

            inp3 = batch["input_stage3"].to(device).reshape(-1, MAP_SIZE)
            target3 = batch["target_stage3"].to(device).reshape(-1, MAP_SIZE)
            enc3 = batch["encoder_stage3"].to(device).reshape(-1, MAP_SIZE)

            # 结构条件向量：[cond_sym, cond_room, cond_branch, cond_outer]
            struct = batch["struct_inject"].to(device)
            density = batch["density_inject"].to(device)

            optimizer.zero_grad()

            # VQ 编码：各阶段编码器分别处理各自上下文切片
            z_e1 = vq1(enc1) # [B, L, d_z]
            z_e2 = vq2(enc2)
            z_e3 = vq3(enc3)

            # 合并三阶段编码后量化
            z_e_all = torch.cat([z_e1, z_e2, z_e3], dim=1) # [B, L*3, d_z]
            z_q, _, commit_loss = quantizer(z_e_all) # [B, L*3, d_z]

            # 三阶段 MaskGIT 前向（均接收完整三阶段 z_q、struct 和 density 条件）
            logits1 = mg1(inp1, z_q, struct, density)
            logits2 = mg2(inp2, z_q, struct, density)
            logits3 = mg3(inp3, z_q, struct, density)

            # 三阶段 Focal Loss + VQ commit loss 加权求和
            loss1 = focal_loss(logits1, target1)
            loss2 = focal_loss(logits2, target2)
            loss3 = focal_loss(logits3, target3)
            loss1_weighted = STAGE1_FOCAL_WEIGHT * loss1
            loss2_weighted = STAGE2_FOCAL_WEIGHT * loss2
            loss3_weighted = STAGE3_FOCAL_WEIGHT * loss3
            commit_weighted = VQ_BETA * commit_loss
            loss = loss1_weighted + loss2_weighted + loss3_weighted + commit_weighted

            loss.backward()
            optimizer.step()

            # detach 后累加，避免保留计算图占用显存
            loss_total += loss.detach()
            loss1_total += loss1.detach()
            loss2_total += loss2.detach()
            loss3_total += loss3.detach()
            commit_total += commit_loss.detach()

        # 每个 epoch 结束后更新学习率
        scheduler.step()

        data_length = len(dataloader)
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"E: {epoch + 1} | Loss: {loss_total.item() / data_length:.6f} | "
            f"L1: {loss1_total.item() / data_length:.6f} | "
            f"L2: {loss2_total.item() / data_length:.6f} | "
            f"L3: {loss3_total.item() / data_length:.6f} | "
            f"VQ: {commit_total.item() / data_length:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # 每 CHECKPOINT 个 epoch 执行一次验证、可视化和检查点保存
        if (epoch + 1) % CHECKPOINT == 0:
            losses = validate(dataloader_val, models, device, tile_dict, epoch + 1)
            loss1_total, loss2_total, loss3_total, commit_total = losses
            loss1_weighted = STAGE1_FOCAL_WEIGHT * loss1_total
            loss2_weighted = STAGE2_FOCAL_WEIGHT * loss2_total
            loss3_weighted = STAGE3_FOCAL_WEIGHT * loss3_total
            commit_weighted = VQ_BETA * commit_total
            loss_total = loss1_weighted + loss2_weighted + loss3_weighted + commit_weighted

            data_length = len(dataloader_val)
            tqdm.write(
                f"[Validate {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"E: {epoch + 1} | Loss: {loss_total.item() / data_length:.6f} | "
                f"L1: {loss1_total.item() / data_length:.6f} | "
                f"L2: {loss2_total.item() / data_length:.6f} | "
                f"L3: {loss3_total.item() / data_length:.6f} | "
                f"VQ: {commit_total.item() / data_length:.6f} | "
            )

            ckpt_path = f"result/seperated/sep-{epoch + 1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "vq1": vq1.state_dict(),
                "vq2": vq2.state_dict(),
                "vq3": vq3.state_dict(),
                "mg1": mg1.state_dict(),
                "mg2": mg2.state_dict(),
                "mg3": mg3.state_dict(),
                "quantizer": quantizer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    # 训练结束后保存最终完整权重（含优化器状态，可用于后续续训或推理）
    final_path = "result/seperated.pth"
    torch.save({
        "epoch": EPOCHS,
        "vq1": vq1.state_dict(),
        "vq2": vq2.state_dict(),
        "vq3": vq3.state_dict(),
        "mg1": mg1.state_dict(),
        "mg2": mg2.state_dict(),
        "mg3": mg3.state_dict(),
        "quantizer": quantizer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, final_path)
    tqdm.write(f"Training complete. Final model saved: {final_path}")

if __name__ == "__main__":
    train(device)
