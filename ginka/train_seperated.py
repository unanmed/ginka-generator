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
VQ_D_MODEL = 64 # VQ-VAE Transformer 模型维度
VQ_NHEAD = 8 # VQ-VAE 多头注意力头数

# 第一阶段 MaskGIT 超参
STAGE1_MG_DMODEL = 192
STAGE1_MG_NHEAD = 8
STAGE1_MG_NUM_LAYERS = 6
STAGE1_MG_DIM_FF = 1024

# 第二阶段 MaskGIT 超参
STAGE2_MG_DMODEL = 192
STAGE2_MG_NHEAD = 8
STAGE2_MG_NUM_LAYERS = 6
STAGE2_MG_DIM_FF = 1024

# 第三阶段 MaskGIT 超参
STAGE3_MG_DMODEL = 192
STAGE3_MG_NHEAD = 8
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
SUBSET2_WALL_PROB = 0.7 # 子集2 进行墙壁掩码的概率
SUBSET_WEIGHTS = (0.5, 0.3, 0.2) # 每个子集的概率

MG_Z_DROPOUT = 0.1 # z 隐变量 Dropout 概率
MG_STRUCT_DROPOUT = 0.1 # 结构参量 Dropout 概率

# 损失参数
FOCAL_GAMMA = 2.0 # Focal Loss 参数
VQ_BETA = 0.5 # 承诺损失权重

# 训练超参
BATCH_SIZE = 64       # 每批样本数
LR = 1e-4             # AdamW 初始学习率
MIN_LR = 1e-6         # 余弦退火最低学习率
WEIGHT_DECAY = 1e-4   # L2 正则化系数
EPOCHS = 400          # 总训练轮数
CHECKPOINT = 20       # 每隔多少 epoch 保存检查点并执行验证

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
        nhead=VQ_NHEAD, num_layers=VQ_LAYERS, dim_ff=VQ_DIM_FF, map_size=MAP_SIZE
    )
    vq1 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage1 上下文（floor/wall）
    vq2 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage2 上下文（door/monster/entrance）
    vq3 = GinkaVQVAE(**vq_kwargs).to(device) # 编码 stage3 上下文（resource）

    # 三个独立 MaskGIT 解码器，均接收完整的三阶段 z_q 作为条件
    mg1 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE1_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE1_MG_DIM_FF,
        nhead=STAGE1_MG_NHEAD, num_layers=STAGE1_MG_NUM_LAYERS, map_size=MAP_SIZE
    ).to(device)
    mg2 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE2_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE2_MG_DIM_FF,
        nhead=STAGE2_MG_NHEAD, num_layers=STAGE2_MG_NUM_LAYERS, map_size=MAP_SIZE
    ).to(device)
    mg3 = GinkaMaskGIT(
        num_classes=NUM_CLASSES, d_model=STAGE3_MG_DMODEL, d_z=VQ_D_Z, dim_ff=STAGE3_MG_DIM_FF,
        nhead=STAGE3_MG_NHEAD, num_layers=STAGE3_MG_NUM_LAYERS, map_size=MAP_SIZE
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
    quantizer = VectorQuantizer(K=VQ_K, d_z=VQ_D_Z)

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
    cond_sym    = random.randint(0, 7)  # 地图对称类型
    cond_room   = random.randint(0, 2)  # 房间数量档位
    cond_branch = random.randint(0, 2)  # 分支复杂度档位
    cond_outer  = random.randint(0, 1)  # 是否有外围走廊
    return torch.LongTensor([cond_sym, cond_room, cond_branch, cond_outer]).unsqueeze(0).to(device)

def maskgit_sample(
    model: torch.nn.Module, inp: torch.Tensor, z: torch.Tensor,
    struct: torch.Tensor, steps: int
) -> np.ndarray:
    current = inp.clone()

    # 迭代去掩码：每步根据置信度分数重新决定掩码位置
    for step in range(steps):
        logits = model(current, z, struct)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()

        confidences = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

        # 余弦退火调度：随步数推进，保留掩码的位置数量递减至 0
        ratio = math.cos(((step + 1) / steps) * math.pi / 2)
        num_to_mask = math.floor(ratio * MAP_SIZE)

        # 输入中已有的非掩码位（来自上一阶段）保持不变
        fixed_mask = (current[0] != MASK_TOKEN)
        sampled[0, fixed_mask] = current[0, fixed_mask]
        confidences[0, fixed_mask] = 1.0

        if num_to_mask > 0:
            # 将置信度最低的位重新掩码，留待下一步重新预测
            _, mask_indices = torch.topk(confidences[0], k=num_to_mask, largest=False)
            sampled[0].scatter_(0, mask_indices, MASK_TOKEN)

        current = sampled

        if (current[0] == MASK_TOKEN).sum() == 0:
            break

    # 兜底：若仍有残余掩码位（理论上不应发生），用 argmax 确定性填充
    still_masked = (current[0] == MASK_TOKEN)
    if still_masked.any():
        logits = model(current, z, struct)
        current[0, still_masked] = torch.argmax(logits[0, still_masked], dim=-1)

    return current[0].cpu().numpy().reshape(MAP_H, MAP_W)

def full_generate_random_z(
    input: torch.Tensor,
    struct: torch.Tensor,
    models: list[torch.nn.Module],
    device: torch.device
) -> tuple:
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    with torch.no_grad():
        z = quantizer.sample(1, VQ_L, device)

        # stage1：生成 floor/wall 骨架
        pred1_np = maskgit_sample(mg1, input.clone(), z, struct, GENERATE_STEP)
        inp2 = torch.tensor(pred1_np.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp2[inp2 == 0] = MASK_TOKEN  # 空地位交由 stage2 填充

        # stage2：在骨架上生成 door/monster/entrance，非零结果覆盖合并
        pred2_np = maskgit_sample(mg2, inp2, z, struct, GENERATE_STEP)
        merged12 = pred1_np.copy()
        merged12[pred2_np != 0] = pred2_np[pred2_np != 0]
        inp3 = torch.tensor(merged12.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp3[inp3 == 0] = MASK_TOKEN

        # stage3：填充 resource
        pred3_np = maskgit_sample(mg3, inp3, z, struct, GENERATE_STEP)
        merged123 = merged12.copy()
        merged123[pred3_np != 0] = pred3_np[pred3_np != 0]

    return pred1_np, merged12, merged123

def full_generate_specific_z(
    input: torch.Tensor,
    z: torch.Tensor,
    struct: torch.Tensor,
    models: list[torch.nn.Module],
    device: torch.device
) -> tuple:
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

    with torch.no_grad():
        # 与 full_generate_random_z 相同的三阶段级联，但使用给定的 z
        pred1_np = maskgit_sample(mg1, input.clone(), z, struct, GENERATE_STEP)
        inp2 = torch.tensor(pred1_np.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp2[inp2 == 0] = MASK_TOKEN

        pred2_np = maskgit_sample(mg2, inp2, z, struct, GENERATE_STEP)
        merged12 = pred1_np.copy()
        merged12[pred2_np != 0] = pred2_np[pred2_np != 0]
        inp3 = torch.tensor(merged12.flatten(), dtype=torch.long, device=device).reshape(1, MAP_SIZE)
        inp3[inp3 == 0] = MASK_TOKEN

        pred3_np = maskgit_sample(mg3, inp3, z, struct, GENERATE_STEP)
        merged123 = merged12.copy()
        merged123[pred3_np != 0] = pred3_np[pred3_np != 0]

    return pred1_np, merged12, merged123

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

    pred3_merged = pred1.copy()
    pred3_merged[pred2 != 0] = pred2[pred2 != 0]
    pred3_merged[pred3 != 0] = pred3[pred3 != 0]

    enc1_np = batch["encoder_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    enc2_np = batch["encoder_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    enc3_np = batch["encoder_stage3"][0].numpy().reshape(MAP_H, MAP_W)
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    inp2_np = batch["input_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    inp3_np = batch["input_stage3"][0].numpy().reshape(MAP_H, MAP_W)

    rows = [
        [to_img(enc1_np), to_img(enc2_np), to_img(enc3_np)],
        [to_img(inp1_np), to_img(inp2_np), to_img(inp3_np)],
        [to_img(pred1), to_img(pred2), to_img(pred3_merged)],
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
    auto_pred1_np, auto_merged12, auto_merged123 = full_generate_specific_z(
        inp1_t, z_q[0:1], struct_t, models, device
    )

    enc1_np = batch["encoder_stage1"][0].numpy().reshape(MAP_H, MAP_W)
    enc2_np = batch["encoder_stage2"][0].numpy().reshape(MAP_H, MAP_W)
    enc3_np = batch["encoder_stage3"][0].numpy().reshape(MAP_H, MAP_W)
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)

    rows = [
        [to_img(enc1_np), to_img(enc2_np), to_img(enc3_np)],
        [to_img(inp1_np), to_img(auto_pred1_np), to_img(auto_merged12), to_img(auto_merged123)],
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
    inp1_np = batch["input_stage1"][0].numpy().reshape(MAP_H, MAP_W)

    row1 = [to_img(inp1_np)]
    for _ in range(2):
        _, _, merged123 = full_generate_random_z(inp1_t, struct_ref, models, device)
        row1.append(to_img(merged123))

    row2 = []
    for _ in range(3):
        _, _, merged123 = full_generate_random_z(inp1_t, random_struct(device), models, device)
        row2.append(to_img(merged123))

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
        _, _, merged123 = full_generate_random_z(seed, random_struct(device), models, device)
        results.append(to_img(merged123))

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

            # VQ 编码：各阶段独立编码后拼接、量化
            z_e1 = vq1(enc1) # [B, L, d_z]
            z_e2 = vq2(enc2)
            z_e3 = vq3(enc3)

            z_e_all = torch.cat([z_e1, z_e2, z_e3], dim=1) # [B, L*3, d_z]
            z_q, _, commit_loss = quantizer(z_e_all) # [B, L*3, d_z]

            # 三阶段 MaskGIT 推理（均以完整 z_q 和 struct 为条件）
            logits1 = mg1(inp1, z_q, struct)
            logits2 = mg2(inp2, z_q, struct)
            logits3 = mg3(inp3, z_q, struct)

            loss1_total += focal_loss(logits1, target1)
            loss2_total += focal_loss(logits2, target2)
            loss3_total += focal_loss(logits3, target3)
            commit_total += commit_loss

            # 每个 batch 生成三种可视化图（val/full/rand）
            visualize_validate(batch, logits1, logits2, logits3, z_q, models, device, tile_dict, epoch, idx)
            idx += 1

    # 每个 epoch 额外生成一张无条件自由生成图（不依赖任何 batch 样本）
    save_dir = f"result/seperated/e{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f"{save_dir}/free.png", visualize_part4(models, device, tile_dict))

    # 恢复训练模式
    for m in [vq1, vq2, vq3, mg1, mg2, mg3]:
        m.train()

    return loss1_total, loss2_total, loss3_total, commit_total

def train(device: torch.device):
    args = parse_arguments()

    models = build_model(device)
    vq1, vq2, vq3, mg1, mg2, mg3, quantizer, optimizer, scheduler = models

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
        args.train, subset_weights=SUBSET_WEIGHTS, subset2_wall_prob=SUBSET2_WALL_PROB
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    dataset_val = GinkaSeperatedDataset(
        args.validate, subset_weights=SUBSET_WEIGHTS, subset2_wall_prob=SUBSET2_WALL_PROB
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

            optimizer.zero_grad()

            # VQ 编码：各阶段编码器分别处理各自上下文切片
            z_e1 = vq1(enc1) # [B, L, d_z]
            z_e2 = vq2(enc2)
            z_e3 = vq3(enc3)

            # 合并三阶段编码后量化
            z_e_all = torch.cat([z_e1, z_e2, z_e3], dim=1) # [B, L*3, d_z]
            z_q, _, commit_loss = quantizer(z_e_all) # [B, L*3, d_z]

            # 三阶段 MaskGIT 前向（均接收完整三阶段 z_q）
            logits1 = mg1(inp1, z_q, struct)
            logits2 = mg2(inp2, z_q, struct)
            logits3 = mg3(inp3, z_q, struct)

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
