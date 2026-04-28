"""
联合训练脚本：VQ-VAE + MaskGIT

总损失 = L_CE（MaskGIT 重建损失）+ beta * L_commit + gamma * L_entropy
       + lambda * L_consist（z 一致性约束，方案 A）

验证阶段对四种子集（A/B/C/D）分别输出图片，
每条样本额外采样 N_Z_SAMPLES 个随机 z，
便于直观对比同条件不同 z 下的生成差异。

用法示例：
    python -m ginka.train_vq
    python -m ginka.train_vq --resume True --state result/joint/joint-10.pth
"""

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

from .vqvae.model import GinkaVQVAE
from .maskGIT.model import GinkaMaskGIT
from .dataset import GinkaVQDataset
from shared.image import matrix_to_image_cv

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
BATCH_SIZE       = 64
NUM_CLASSES      = 16
MASK_TOKEN       = 15
GENERATE_STEP    = 18    # 推理时 MaskGIT 迭代步数
MAP_SIZE         = 13 * 13
MAP_H = MAP_W    = 13
LABEL_SMOOTHING  = 0.0
WALL_MASK_RATIO  = 0.8

# VQ-VAE 超参
VQ_L      = 32   # summary token 数量（即 z 的序列长度）
VQ_K      = 1    # codebook 大小
VQ_D_Z    = 128   # codebook 嵌入维度
VQ_D_MODEL= 192
VQ_NHEAD  = 8
VQ_LAYERS = 4
VQ_DIM_FF = 512
VQ_BETA   = 0.5  # commit loss 权重
VQ_GAMMA  = 0.0   # entropy loss 权重

# MaskGIT 超参
MG_D_MODEL  = 256
MG_NHEAD    = 8
MG_LAYERS   = 6
MG_DIM_FF   = 2048
MG_Z_DROPOUT     = 0.1  # 训练时以此概率把 z 替换为随机噪声
MG_STRUCT_DROPOUT= 0.1  # 训练时以此概率将结构标签替换为 null（无条件占位）

# 一致性约束超参（方案 A）
CONSIST_LAMBDA = 0.1   # z 一致性损失权重
CONSIST_TEMP   = 2.0   # 计算软嵌入时对 logits 施加的温度（>1 平滑分布，降低 gap）

# 验证时对每条样本额外采样的 z 数量（0 = 只用真实 z）
N_Z_SAMPLES = 3

# 四个子集 A/B/C/D 的采样权重（训练集与验证集共用）
SUBSET_WEIGHTS = (0.5, 0.2, 0.2, 0.1)

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs("result/joint", exist_ok=True)
os.makedirs("result/joint_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="VQ-VAE + MaskGIT 联合训练")
    parser.add_argument("--resume",      type=bool, default=False)
    parser.add_argument("--state",       type=str,  default="result/joint/joint-10.pth",
                        help="续训时加载的检查点路径")
    parser.add_argument("--train",       type=str,  default="ginka-dataset.json")
    parser.add_argument("--validate",    type=str,  default="ginka-eval.json")
    parser.add_argument("--epochs",      type=int,  default=100)
    parser.add_argument("--checkpoint",  type=int,  default=5,
                        help="每隔多少 epoch 保存检查点并验证")
    parser.add_argument("--load_optim",  type=bool, default=True)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# MaskGIT 推理（cosine schedule 迭代解码）
# ---------------------------------------------------------------------------
@torch.no_grad()
def maskgit_generate(
    model_mg: GinkaMaskGIT,
    z: torch.Tensor,
    steps: int = GENERATE_STEP,
    init_map: torch.Tensor = None,
    struct_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    迭代生成地图（cosine schedule unmasking）。

    Args:
        model_mg: GinkaMaskGIT（eval 模式）
        z:        [B, L, d_z]  条件 z
        steps:    解码步数
        init_map: [B, MAP_SIZE] 可选初始地图；非 MASK 位置在生成过程中保持固定。
                  为 None 时从全 MASK 开始（自由生成）。

    Returns:
        map_out: [B, MAP_SIZE]
    """
    B = z.shape[0]
    if init_map is None:
        map_seq = torch.full((B, MAP_SIZE), MASK_TOKEN, device=device)
    else:
        map_seq = init_map.clone().to(device)

    # 记录初始 MASK 位置，这些位置才需要生成
    generatable = (map_seq == MASK_TOKEN)   # [B, S] bool

    for step in range(steps):
        if not generatable.any():
            break

        logits  = model_mg(map_seq, z, struct_cond=struct_cond) # [B, S, C]
        probs   = F.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        sampled = dist.sample()                                 # [B, S]

        # 计算置信度；固定位置设为 +inf（确保不会被选为“继续保持 MASK”）
        confidences = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        confidences = confidences.masked_fill(~generatable, float('inf'))

        ratio   = math.cos(((step + 1) / steps) * math.pi / 2)
        new_map = map_seq.clone()

        for b in range(B):
            n_gen  = int(generatable[b].sum().item())
            n_keep = int(ratio * n_gen)   # 本步仍保持 MASK 的位置数

            if n_keep > 0:
                _, keep_idx = torch.topk(confidences[b], k=n_keep, largest=False)
                pred_b = sampled[b].clone()
                pred_b[keep_idx] = MASK_TOKEN
                new_map[b] = torch.where(generatable[b], pred_b, map_seq[b])
            else:
                new_map[b] = torch.where(generatable[b], sampled[b], map_seq[b])

        map_seq = new_map

    return map_seq

# ---------------------------------------------------------------------------
# 可视化工具
# ---------------------------------------------------------------------------
def make_map_image(map_flat: torch.Tensor, tile_dict: dict) -> np.ndarray:
    """将 [MAP_SIZE] 的 tensor 转成 RGB 图片（numpy）。"""
    arr = map_flat.cpu().numpy().reshape(MAP_H, MAP_W)
    return matrix_to_image_cv(arr, tile_dict)


def hstack_images(imgs: list, gap: int = 4, color=(255, 255, 255)) -> np.ndarray:
    """将多张图片横向拼接，之间插入竖线；高度不一致时底部补齐背景色。"""
    max_h = max(img.shape[0] for img in imgs)

    def _pad_h(img):
        dh = max_h - img.shape[0]
        if dh == 0:
            return img
        pad = np.full((dh, img.shape[1], 3), color, dtype=np.uint8)
        return np.concatenate([img, pad], axis=0)

    vline = np.full((max_h, gap, 3), color, dtype=np.uint8)
    result = _pad_h(imgs[0])
    for img in imgs[1:]:
        result = np.concatenate([result, vline, _pad_h(img)], axis=1)
    return result


def grid_images(imgs: list, gap: int = 4, bg_color=(255, 255, 255)) -> np.ndarray:
    """将图片列表排成两行网格（上行 ceil(N/2)，下行 floor(N/2)），方便查看。"""
    n = len(imgs)
    if n == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if n == 1:
        return imgs[0]

    mid = math.ceil(n / 2)
    top_row = hstack_images(imgs[:mid], gap, bg_color)
    bot_imgs = imgs[mid:]

    if not bot_imgs:
        return top_row

    bot_row = hstack_images(bot_imgs, gap, bg_color)

    # 补齐宽度（右侧填充背景色）
    tw, bw = top_row.shape[1], bot_row.shape[1]
    if tw > bw:
        pad = np.full((bot_row.shape[0], tw - bw, 3), bg_color, dtype=np.uint8)
        bot_row = np.concatenate([bot_row, pad], axis=1)
    elif bw > tw:
        pad = np.full((top_row.shape[0], bw - tw, 3), bg_color, dtype=np.uint8)
        top_row = np.concatenate([top_row, pad], axis=1)

    hline = np.full((gap, top_row.shape[1], 3), bg_color, dtype=np.uint8)
    return np.concatenate([top_row, hline, bot_row], axis=0)


def label_image(img: np.ndarray, text: str, font_scale: float = 0.45) -> np.ndarray:
    """在图片顶部加一行文字标签（就地修改并返回）。"""
    bar_h = 16
    bar = np.full((bar_h, img.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(
        bar, text, (2, bar_h - 3),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (200, 200, 200), 1, cv2.LINE_AA
    )
    return np.concatenate([bar, img], axis=0)


def struct_cond_to_text(sc: torch.Tensor) -> str:
    """
    将 struct_cond [4] LongTensor 解码为可读字符串。

    sc 顺序：[cond_sym, cond_room, cond_branch, cond_outer]
      cond_sym  : sym_h*4 + sym_v*2 + sym_c，取值 0-6，7=null
      cond_room : roomCountLevel 0-2，3=null
      cond_branch: branchLevel 0-2，3=null
      cond_outer : outerWall 0-1，2=null
    """
    sym_val, room_val, branch_val, outer_val = (int(x) for x in sc.tolist())

    # 对称性
    if sym_val == 7:
        sym_str = "sym:-"
    else:
        flags = []
        if sym_val & 4: flags.append("H")
        if sym_val & 2: flags.append("V")
        if sym_val & 1: flags.append("C")
        sym_str = "sym:" + ("".join(flags) if flags else "none")

    # 房间数量等级
    room_map = {0: "room:lo", 1: "room:mid", 2: "room:hi", 3: "room:-"}
    room_str = room_map.get(room_val, f"room:{room_val}")

    # 分支等级
    branch_map = {0: "br:lo", 1: "br:mid", 2: "br:hi", 3: "br:-"}
    branch_str = branch_map.get(branch_val, f"br:{branch_val}")

    # 外墙
    outer_map = {0: "wall:N", 1: "wall:Y", 2: "wall:-"}
    outer_str = outer_map.get(outer_val, f"wall:{outer_val}")

    return f"{sym_str} {room_str} {branch_str} {outer_str}"


def annotate_struct(img: np.ndarray, sc: torch.Tensor) -> np.ndarray:
    """在图片底部追加一行结构标签注释（深蓝底白字）。"""
    text  = struct_cond_to_text(sc)
    bar_h = 14
    bar   = np.full((bar_h, img.shape[1], 3), (60, 30, 10), dtype=np.uint8)
    cv2.putText(
        bar, text, (2, bar_h - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (180, 220, 255), 1, cv2.LINE_AA
    )
    return np.concatenate([img, bar], axis=0)


def make_random_wall_seed(ratio_min: float = 0.02, ratio_max: float = 0.08) -> torch.Tensor:
    """
    在全 MASK 地图上随机放置少量墙壁作为推理种子，用于完全随机生成场景。

    Returns:
        [1, MAP_SIZE]  MASK=15 背景 + 随机置少量墙壁（tile=1）
    """
    ratio  = random.uniform(ratio_min, ratio_max)
    n_wall = max(2, int(MAP_SIZE * ratio))
    seed   = torch.full((1, MAP_SIZE), MASK_TOKEN, dtype=torch.long, device=device)
    idx    = torch.randperm(MAP_SIZE)[:n_wall]
    seed[0, idx] = 1   # wall
    return seed


def make_random_struct_cond() -> torch.Tensor:
    """
    生成一个随机结构条件，所有标签均取合法非-null 值。

    Returns:
        [1, 4] LongTensor，顺序 [cond_sym, cond_room, cond_branch, cond_outer]
    """
    from .maskGIT.model import SYM_VOCAB, ROOM_VOCAB, BRANCH_VOCAB, OUTER_VOCAB
    sym    = random.randint(0, SYM_VOCAB    - 2)   # 0-6
    room   = random.randint(0, ROOM_VOCAB   - 2)   # 0-2
    branch = random.randint(0, BRANCH_VOCAB - 2)   # 0-2
    outer  = random.randint(0, OUTER_VOCAB  - 2)   # 0-1
    return torch.tensor([[sym, room, branch, outer]], dtype=torch.long, device=device)

@torch.no_grad()
def validate(
    model_vq: GinkaVQVAE,
    model_mg: GinkaMaskGIT,
    dataloader_val: DataLoader,
    tile_dict: dict,
    epoch: int,
):
    """
    验证函数：计算 val loss 并输出 5 类推理场景的对比图。

    场景说明（按 epoch 建立子文件夹，避免图片堆积）：
      场景1 (scene1_completion) : 子集 A，标准随机掩码补全
          列: ground truth | masked input | z_real pred | z_real gen | z_rand×N
      场景2 (scene2_wall)       : 子集 B，仅墙壁+空地 → 生成完整地图
          列: ground truth | wall-only input | z_real gen | z_rand×N
      场景3 (scene3_sparse)     : 子集 C，稀疏墙壁条件 → 生成完整地图
          列: ground truth | sparse wall input | z_real gen | z_rand×N
      场景4 (scene4_entrance)   : 子集 D，墙壁+入口 → 生成完整地图
          列: ground truth | wall+entrance input | z_real gen | z_rand×N
      场景5 (scene5_random)     : 无数据集参照，随机稀疏墙壁种子 → 完全随机生成
          列: random seed | z_rand×(N+1)
    """
    model_vq.eval()
    model_mg.eval()

    # 按 epoch 建立独立子文件夹，保留每次验证结果方便回溯
    epoch_dir = f"result/joint_img/e{epoch:04d}"
    os.makedirs(epoch_dir, exist_ok=True)

    val_loss_total = 0.0
    val_steps      = 0
    captured = {s: None for s in ('A', 'B', 'C', 'D')}

    # ── 计算 val loss + 捕获各子集样本 ──────────────────────────────────────
    for batch in tqdm(dataloader_val, desc="Validating", leave=False, disable=disable_tqdm):
        raw_map    = batch["raw_map"].to(device)      # [B, 169]
        masked_map = batch["masked_map"].to(device)   # [B, 169]
        target_map = batch["target_map"].to(device)   # [B, 169]
        subsets    = batch["subset"]                  # list of str
        B          = raw_map.shape[0]

        z_q, _, _, vq_loss, _, _ = model_vq(raw_map)
        struct_cond_b = batch["struct_cond"].to(device)  # [B, 4]
        logits = model_mg(masked_map, z_q, struct_cond=struct_cond_b)
        mask   = (masked_map == MASK_TOKEN)

        ce_loss   = F.cross_entropy(
            logits.permute(0, 2, 1), target_map,
            reduction='none', label_smoothing=LABEL_SMOOTHING
        )
        masked_ce = (ce_loss * mask).sum() / (mask.sum() + 1e-6)
        val_loss_total += (masked_ce + vq_loss).item()
        val_steps      += 1

        for i in range(B):
            s = subsets[i]
            if captured[s] is None:
                captured[s] = {
                    "raw":         raw_map[i:i+1].clone(),
                    "masked":      masked_map[i:i+1].clone(),
                    "z_q":         z_q[i:i+1].clone(),
                    "struct_cond": struct_cond_b[i:i+1].clone(),
                }

        if all(v is not None for v in captured.values()):
            break

    # ── 公共辅助：对给定条件地图随机采样 n 次 z 并迭代生成（无条件）──────────────
    def _rand_gens(cond_map, n):
        imgs = []
        for i in range(n):
            z_r = model_vq.sample(1, device)
            gen = maskgit_generate(model_mg, z_r, init_map=cond_map)  # struct_cond=None 无条件
            imgs.append(label_image(make_map_image(gen[0], tile_dict), f"z_rand_{i + 1}"))
        return imgs

    # ── 公共辅助：对给定条件地图随机采样 n 次 z 并迭代生成（随机结构标签）────────
    def _rand_gens_with_struct(cond_map, n):
        imgs = []
        for i in range(n):
            z_r = model_vq.sample(1, device)
            sc_r = make_random_struct_cond()           # [1, 4] 随机合法标签
            gen  = maskgit_generate(model_mg, z_r, init_map=cond_map, struct_cond=sc_r)
            img  = label_image(make_map_image(gen[0], tile_dict), f"z_rand_{i + 1}")
            img  = annotate_struct(img, sc_r[0])
            imgs.append(img)
        return imgs

    # ── 场景1：标准掩码补全（子集 A）─────────────────────────────────────────
    if captured['A'] is not None:
        cap = captured['A']
        raw, cond, z_q, sc = cap['raw'], cap['masked'], cap['z_q'], cap['struct_cond']

        real_img  = label_image(make_map_image(raw[0],  tile_dict), "ground truth")
        cond_img  = label_image(make_map_image(cond[0], tile_dict), "masked input")

        # 单步 argmax 预测（观察模型对掩码位置的瞬时判断）
        pred      = model_mg(cond, z_q, struct_cond=sc).argmax(dim=-1)[0]
        pred_img  = label_image(make_map_image(pred, tile_dict), "z_real pred")

        # 迭代生成（从掩码输入出发，真实 z）
        gen_real  = maskgit_generate(model_mg, z_q, init_map=cond, struct_cond=sc)
        gen_r_img = label_image(make_map_image(gen_real[0], tile_dict), "z_real gen")

        # 对使用了真实 struct_cond 的图片追加标签注释
        sc0 = sc[0]
        real_img  = annotate_struct(real_img,  sc0)
        cond_img  = annotate_struct(cond_img,  sc0)
        pred_img  = annotate_struct(pred_img,  sc0)
        gen_r_img = annotate_struct(gen_r_img, sc0)

        row = [real_img, cond_img, pred_img, gen_r_img] + _rand_gens(cond, N_Z_SAMPLES)
        cv2.imwrite(f"{epoch_dir}/scene1_completion.png", grid_images(row))

    # ── 场景2：墙壁辅助生成（子集 B）─────────────────────────────────────────
    if captured['B'] is not None:
        cap = captured['B']
        raw, cond, z_q, sc = cap['raw'], cap['masked'], cap['z_q'], cap['struct_cond']

        real_img  = label_image(make_map_image(raw[0],  tile_dict), "ground truth")
        cond_img  = label_image(make_map_image(cond[0], tile_dict), "wall-only input")
        gen_real  = maskgit_generate(model_mg, z_q, init_map=cond, struct_cond=sc)
        gen_r_img = label_image(make_map_image(gen_real[0], tile_dict), "z_real gen")

        sc0 = sc[0]
        real_img  = annotate_struct(real_img,  sc0)
        cond_img  = annotate_struct(cond_img,  sc0)
        gen_r_img = annotate_struct(gen_r_img, sc0)

        row = [real_img, cond_img, gen_r_img] + _rand_gens(cond, N_Z_SAMPLES)
        cv2.imwrite(f"{epoch_dir}/scene2_wall.png", grid_images(row))

    # ── 场景3：稀疏墙壁条件生成（子集 C）────────────────────────────────────
    if captured['C'] is not None:
        cap = captured['C']
        raw, cond, z_q, sc = cap['raw'], cap['masked'], cap['z_q'], cap['struct_cond']

        real_img  = label_image(make_map_image(raw[0],  tile_dict), "ground truth")
        cond_img  = label_image(make_map_image(cond[0], tile_dict), "sparse wall input")
        gen_real  = maskgit_generate(model_mg, z_q, init_map=cond, struct_cond=sc)
        gen_r_img = label_image(make_map_image(gen_real[0], tile_dict), "z_real gen")

        sc0 = sc[0]
        real_img  = annotate_struct(real_img,  sc0)
        cond_img  = annotate_struct(cond_img,  sc0)
        gen_r_img = annotate_struct(gen_r_img, sc0)

        row = [real_img, cond_img, gen_r_img] + _rand_gens(cond, N_Z_SAMPLES)
        cv2.imwrite(f"{epoch_dir}/scene3_sparse.png", grid_images(row))

    # ── 场景4：墙壁+入口条件生成（子集 D）───────────────────────────────────
    if captured['D'] is not None:
        cap = captured['D']
        raw, cond, z_q, sc = cap['raw'], cap['masked'], cap['z_q'], cap['struct_cond']

        real_img  = label_image(make_map_image(raw[0],  tile_dict), "ground truth")
        cond_img  = label_image(make_map_image(cond[0], tile_dict), "wall+entrance input")
        gen_real  = maskgit_generate(model_mg, z_q, init_map=cond, struct_cond=sc)
        gen_r_img = label_image(make_map_image(gen_real[0], tile_dict), "z_real gen")

        sc0 = sc[0]
        real_img  = annotate_struct(real_img,  sc0)
        cond_img  = annotate_struct(cond_img,  sc0)
        gen_r_img = annotate_struct(gen_r_img, sc0)

        row = [real_img, cond_img, gen_r_img] + _rand_gens(cond, N_Z_SAMPLES)
        cv2.imwrite(f"{epoch_dir}/scene4_entrance.png", grid_images(row))

    # ── 场景5：完全随机生成（无数据集参照）──────────────────────────────────
    # 5a：随机结构标签 — 验证结构导向能力
    rand_seed_a = make_random_wall_seed()
    seed_img_a  = label_image(make_map_image(rand_seed_a[0], tile_dict), "random seed")
    row_a = [seed_img_a] + _rand_gens_with_struct(rand_seed_a, N_Z_SAMPLES + 1)
    cv2.imwrite(f"{epoch_dir}/scene5a_random_cond.png", grid_images(row_a))

    # 5b：无条件（struct_cond=None）— 验证基线生成质量
    rand_seed_b = make_random_wall_seed()
    seed_img_b  = label_image(make_map_image(rand_seed_b[0], tile_dict), "random seed")
    row_b = [seed_img_b] + _rand_gens(rand_seed_b, N_Z_SAMPLES + 1)
    cv2.imwrite(f"{epoch_dir}/scene5b_random_uncond.png", grid_images(row_b))

    avg_val_loss = val_loss_total / max(val_steps, 1)
    return avg_val_loss

# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------
def train():
    print(f"Using device: {device}")
    args = parse_arguments()

    # ---- 模型 ----
    model_vq = GinkaVQVAE(
        num_classes=NUM_CLASSES,
        L=VQ_L,   K=VQ_K,   d_z=VQ_D_Z,
        d_model=VQ_D_MODEL, nhead=VQ_NHEAD,
        num_layers=VQ_LAYERS, dim_ff=VQ_DIM_FF,
        map_size=MAP_SIZE,
        beta=VQ_BETA, gamma=VQ_GAMMA,
    ).to(device)

    model_mg = GinkaMaskGIT(
        num_classes=NUM_CLASSES,
        d_model=MG_D_MODEL,  d_z=VQ_D_Z,
        dim_ff=MG_DIM_FF,    nhead=MG_NHEAD,
        num_layers=MG_LAYERS,
        map_size=MAP_SIZE,
        z_dropout=MG_Z_DROPOUT,
        struct_dropout=MG_STRUCT_DROPOUT,
    ).to(device)

    vq_params  = sum(p.numel() for p in model_vq.parameters())
    mg_params  = sum(p.numel() for p in model_mg.parameters())
    print(f"VQ-VAE  参数量: {vq_params:,}  ({vq_params/1e6:.3f}M)")
    print(f"MaskGIT 参数量: {mg_params:,}  ({mg_params/1e6:.3f}M)")
    print(f"Total   参数量: {vq_params+mg_params:,}  ({(vq_params+mg_params)/1e6:.3f}M)")

    # ---- 数据集 ----
    dataset_train = GinkaVQDataset(
        args.train,
        subset_weights=SUBSET_WEIGHTS,
        wall_mask_ratio=WALL_MASK_RATIO,
    )
    dataset_val = GinkaVQDataset(
        args.validate,
        subset_weights=SUBSET_WEIGHTS,
        room_thresholds=dataset_train.room_th,
        branch_thresholds=dataset_train.branch_th,
        wall_mask_ratio=WALL_MASK_RATIO,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=8, shuffle=True,
        num_workers=0,
    )

    # ---- 优化器（联合训练，两个模型共用一个 optimizer）----
    all_params = list(model_vq.parameters()) + list(model_mg.parameters())
    optimizer  = optim.AdamW(all_params, lr=2e-4, weight_decay=1e-2)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- 续训 ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.state, map_location=device)
        model_vq.load_state_dict(ckpt["vq_state"],  strict=False)
        model_mg.load_state_dict(ckpt["mg_state"],  strict=False)
        if args.load_optim and ckpt.get("optim_state") is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 epoch {start_epoch} 接续训练。")

    # ---- tile 贴图（用于验证可视化）----
    tile_dict = {}
    for file in os.listdir("tiles"):
        name = os.path.splitext(file)[0]
        img  = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
        if img is not None:
            tile_dict[name] = img

    # ---- 训练循环 ----
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs),
                      desc="Joint Training", disable=disable_tqdm):
        model_vq.train()
        model_mg.train()

        loss_total      = 0.0
        ce_total        = 0.0
        vq_loss_total   = 0.0
        commit_total    = 0.0
        entropy_total   = 0.0
        consist_total   = 0.0
        subset_stats    = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        for batch in tqdm(dataloader_train, leave=False,
                          desc="Epoch Progress", disable=disable_tqdm):
            raw_map    = batch["raw_map"].to(device)      # [B, 169]
            masked_map = batch["masked_map"].to(device)   # [B, 169]
            target_map = batch["target_map"].to(device)   # [B, 169]

            for s in batch["subset"]:
                subset_stats[s] = subset_stats.get(s, 0) + 1

            # ---- 前向传播 ----
            # 1. VQ-VAE 编码真实地图 → z_q, z_e
            z_q, z_e, _, vq_loss, commit_loss, entropy_loss = model_vq(raw_map)  # z_q/z_e: [B, L, d_z]

            # 2. MaskGIT 以掩码地图 + z + 结构标签预测原始 tile
            struct_cond = batch["struct_cond"].to(device)  # [B, 4]
            logits = model_mg(masked_map, z_q, struct_cond=struct_cond)  # [B, 169, C]

            # 3. 只对被 mask 的位置计算 CE loss
            mask   = (masked_map == MASK_TOKEN)           # [B, 169] bool
            ce_loss = F.cross_entropy(
                logits.permute(0, 2, 1), target_map,
                reduction='none', label_smoothing=LABEL_SMOOTHING
            )
            masked_ce = (ce_loss * mask).sum() / (mask.sum() + 1e-6)

            # 4. z 一致性约束（方案 A）：将 MaskGIT 的 logits 经温度平滑后
            #    与 VQ 编码器的 tile embedding 做加权求和，得到软嵌入序列，
            #    再送入编码器得到 z_pred_e，约束其与真实 z_e 对齐。
            #    梯度从 z_pred_e 回传到 MaskGIT 的 logits（以及 VQ encoder 的权重）；
            #    z_e 作为 detach 后的监督目标，不产生梯度。
            soft_probs = F.softmax(logits / CONSIST_TEMP, dim=-1)            # [B, H*W, V]
            tile_emb   = model_vq.tile_embedding.weight                      # [V, d_model]
            soft_emb   = soft_probs @ tile_emb                               # [B, H*W, d_model]
            z_pred_e   = model_vq.encode_soft(soft_emb)                      # [B, L, d_z]
            consist_loss = F.mse_loss(z_pred_e, z_e.detach())

            # 5. 联合损失
            loss = masked_ce + vq_loss + CONSIST_LAMBDA * consist_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            loss_total    += loss.detach().item()
            ce_total      += masked_ce.detach().item()
            vq_loss_total += vq_loss.detach().item()
            commit_total  += commit_loss.detach().item()
            entropy_total += entropy_loss.detach().item()
            consist_total += consist_loss.detach().item()

        scheduler.step()

        n = len(dataloader_train)
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch + 1:4d} | "
            f"Loss {loss_total/n:.5f}  "
            f"CE {ce_total/n:.5f}  "
            f"VQ {vq_loss_total/n:.5f}  "
            f"Commit {commit_total/n:.5f}  "
            f"Entropy {entropy_total/n:.5f}  "
            f"Consist {consist_total/n:.5f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | "
            f"Subsets {subset_stats}"
        )

        # ---- 检查点 + 验证 ----
        if (epoch + 1) % args.checkpoint == 0:
            ckpt_path = f"result/joint/joint-{epoch + 1}.pth"
            torch.save({
                "epoch":      epoch + 1,
                "vq_state":   model_vq.state_dict(),
                "mg_state":   model_mg.state_dict(),
                "optim_state":optimizer.state_dict(),
            }, ckpt_path)
            tqdm.write(f"  检查点已保存: {ckpt_path}")

            val_loss = validate(
                model_vq, model_mg, dataloader_val, tile_dict, epoch + 1
            )
            tqdm.write(
                f"[Validate] Epoch {epoch + 1:4d} | Val Loss {val_loss:.5f}"
            )
            # 恢复训练模式
            model_vq.train()
            model_mg.train()

    print("训练结束。")
    torch.save({
        "epoch":    start_epoch + args.epochs,
        "vq_state": model_vq.state_dict(),
        "mg_state": model_mg.state_dict(),
    }, "result/joint/joint_final.pth")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
