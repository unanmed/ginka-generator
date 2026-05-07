"""
三阶段级联训练脚本：各阶段独立训练，使用 GinkaStageDataset。

总损失 = L_CE（只对本阶段负责的 tile 位置计算）+ beta * L_commit + gamma * L_entropy

各阶段分工：
  stage=1  结构骨架：floor(0) + wall(1)
  stage=2  功能元素：door(2) + monster(4) + entrance(5)
  stage=3  资源放置：resource(3)

用法示例：
    python -m ginka.train_stage --stage 1
    python -m ginka.train_stage --stage 2
    python -m ginka.train_stage --stage 3
    python -m ginka.train_stage --stage 1 --resume True --state result/stage1/stage1-10.pth
    python -m ginka.train_stage --stage 2 --pretrain_vq result/joint/joint-50.pth
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
from .dataset import GinkaStageDataset
from shared.image import matrix_to_image_cv

# ---------------------------------------------------------------------------
# 各阶段配置
# ---------------------------------------------------------------------------

# 共用 VQ-VAE 超参
VQ_L = 2        # 码字序列长度
VQ_K = 8        # codebook 大小
VQ_D_Z = 64     # 码字维度
VQ_BETA = 0.5   # commit loss 权重
VQ_GAMMA = 0.0  # entropy loss 权重
VQ_LAYERS = 3
VQ_DIM_FF = 512
VQ_D_MODEL = 64
VQ_NHEAD = 8

# 各阶段 MaskGIT 超参（按任务复杂度差异化配置）
STAGE_MG_CONFIGS = {
    1: dict(d_model=256, nhead=8, num_layers=6, dim_ff=2048),  # 结构骨架，最重要
    2: dict(d_model=192, nhead=8, num_layers=4, dim_ff=1024),  # 功能元素
    3: dict(d_model=128, nhead=8, num_layers=3, dim_ff=512),   # 资源放置，最简单
}

# 各阶段监控的 tile 集合（用于分类别召回率统计）
STAGE_TILE_SETS = {
    1: {0: "floor", 1: "wall"},
    2: {2: "door", 4: "monster", 5: "entrance"},
    3: {3: "resource"},
}

# 各阶段损失权重（可单独调节 CE 与 VQ 损失的平衡）
# stage3 的 resource 极稀疏，大幅上调 ce_weight 以补偿类别不均衡
STAGE_LOSS_CONFIG = {
    1: dict(ce_weight=1.0, vq_weight=1.0),  # 结构骨架，标准权重
    2: dict(ce_weight=1.5, vq_weight=0.5),  # 功能元素较稀疏，上调 CE
    3: dict(ce_weight=3.0, vq_weight=0.5),  # resource 极稀疏，显著上调 CE
}

NUM_CLASSES = 7
MASK_TOKEN = 6
MAP_SIZE = 13 * 13
MAP_H = MAP_W = 13
FOCAL_GAMMA = 2.0
GENERATE_STEP = 18
BATCH_SIZE = 64
WALL_MASK_RATIO = 0.8

MG_Z_DROPOUT = 0.1
MG_STRUCT_DROPOUT = 0.1

SUBSET_WEIGHTS = (0.5, 0.2, 0.2, 0.1)

device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

disable_tqdm = not sys.stdout.isatty()

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('true', '1', 'yes'): return True
    if v.lower() in ('false', '0', 'no'): return False
    raise argparse.ArgumentTypeError(f"布尔值应为 True/False，收到: {v!r}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="三阶段级联训练")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--resume", type=_str2bool, default=False)
    parser.add_argument(
        "--state", type=str, default="",
        help="续训时加载的检查点路径（自动推断 stage{N}/stage{N}-*.pth）",
    )
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=_str2bool, default=True)
    parser.add_argument(
        "--freeze_vq", type=_str2bool, default=False,
        help="冻结 VQ 编码器，只训练 MaskGIT（适合加载预训练编码器后热身）",
    )
    parser.add_argument(
        "--pretrain_vq", type=str, default="",
        help="从 train_vq.py 的联合训练检查点中导入对应通道的 VQ 编码器权重",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Focal Loss（与 train_vq.py 一致）
# ---------------------------------------------------------------------------

def focal_loss(logits, targets, gamma=FOCAL_GAMMA, reduction='none'):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    fl = (1.0 - pt) ** gamma * ce
    if reduction == 'mean': return fl.mean()
    if reduction == 'sum': return fl.sum()
    return fl


def masked_focal_loss(logits, targets, loss_mask, gamma=FOCAL_GAMMA):
    """
    只对 loss_mask 为 True 的位置计算 focal loss 均值。

    Args:
        logits:    [B, C, H*W]
        targets:   [B, H*W]
        loss_mask: [B, H*W] bool
    """
    per_token = focal_loss(logits, targets, gamma, reduction='none')  # [B, H*W]
    selected = per_token[loss_mask]
    if selected.numel() == 0:
        return per_token.mean()
    return selected.mean()

# ---------------------------------------------------------------------------
# MaskGIT 推理（cosine schedule）
# ---------------------------------------------------------------------------

@torch.no_grad()
def maskgit_generate(
    model_mg: GinkaMaskGIT,
    z: torch.Tensor,
    steps: int = GENERATE_STEP,
    init_map: torch.Tensor = None,
    struct_cond: torch.Tensor = None,
) -> torch.Tensor:
    """
    迭代生成地图（cosine schedule unmasking）。

    Args:
        init_map: 可选初始地图；非 MASK 位置在生成中保持不变。

    Returns:
        [B, MAP_SIZE] LongTensor
    """
    B = z.shape[0]
    map_seq = (
        torch.full((B, MAP_SIZE), MASK_TOKEN, device=device)
        if init_map is None else init_map.clone().to(device)
    )

    generatable = (map_seq == MASK_TOKEN)

    for step in range(steps):
        if not generatable.any():
            break

        logits = model_mg(map_seq, z, struct_cond=struct_cond)  # [B, S, C]
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()
        confidences = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        confidences = confidences.masked_fill(~generatable, float('inf'))

        ratio = math.cos(((step + 1) / steps) * math.pi / 2)
        new_map = map_seq.clone()

        for b in range(B):
            n_gen = int(generatable[b].sum().item())
            n_keep = int(ratio * n_gen)
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
# 可视化工具（与 train_vq.py 保持一致）
# ---------------------------------------------------------------------------

def make_map_image(map_flat, tile_dict):
    arr = map_flat.cpu().numpy().reshape(MAP_H, MAP_W)
    return matrix_to_image_cv(arr, tile_dict)


def hstack_images(imgs, gap=4, color=(255, 255, 255)):
    max_h = max(img.shape[0] for img in imgs)

    def _pad(img):
        dh = max_h - img.shape[0]
        return img if dh == 0 else np.concatenate(
            [img, np.full((dh, img.shape[1], 3), color, dtype=np.uint8)], axis=0)

    vline = np.full((max_h, gap, 3), color, dtype=np.uint8)
    result = _pad(imgs[0])
    for img in imgs[1:]:
        result = np.concatenate([result, vline, _pad(img)], axis=1)
    return result


def grid_images(imgs, gap=4, bg=(255, 255, 255)):
    n = len(imgs)
    if n == 0: return np.zeros((1, 1, 3), dtype=np.uint8)
    if n == 1: return imgs[0]
    mid = math.ceil(n / 2)
    top = hstack_images(imgs[:mid], gap, bg)
    bot_imgs = imgs[mid:]
    if not bot_imgs: return top
    bot = hstack_images(bot_imgs, gap, bg)
    tw, bw = top.shape[1], bot.shape[1]
    if tw > bw:
        bot = np.concatenate(
            [bot, np.full((bot.shape[0], tw - bw, 3), bg, dtype=np.uint8)], axis=1)
    elif bw > tw:
        top = np.concatenate(
            [top, np.full((top.shape[0], bw - tw, 3), bg, dtype=np.uint8)], axis=1)
    hline = np.full((gap, top.shape[1], 3), bg, dtype=np.uint8)
    return np.concatenate([top, hline, bot], axis=0)


def label_image(img, text, font_scale=0.45):
    bar = np.full((16, img.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    cv2.putText(
        bar, text, (2, 13), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return np.concatenate([bar, img], axis=0)


def make_random_struct_cond():
    from .maskGIT.model import SYM_VOCAB, ROOM_VOCAB, BRANCH_VOCAB, OUTER_VOCAB
    return torch.tensor([[
        random.randint(0, SYM_VOCAB - 2),
        random.randint(0, ROOM_VOCAB - 2),
        random.randint(0, BRANCH_VOCAB - 2),
        random.randint(0, OUTER_VOCAB - 2),
    ]], dtype=torch.long, device=device)

# ---------------------------------------------------------------------------
# 按阶段构造推理初始地图
# ---------------------------------------------------------------------------

def make_stage_init(stage: int, context_map: torch.Tensor) -> torch.Tensor:
    """
    根据阶段构造 MaskGIT 的推理初始地图。

    Stage 1: 全 MASK（或保留稀疏 wall 种子）
    Stage 2: 保留 floor/wall 上下文，其余 → MASK
    Stage 3: 保留完整上下文（floor/wall/door/monster/entrance），resource → MASK
    """
    init = context_map.clone()

    if stage == 1:
        # 全 MASK（不依赖上下文地图）
        init = torch.full_like(init, MASK_TOKEN)

    elif stage == 2:
        # 保留 floor/wall，其余 → MASK
        mask = ~torch.isin(init, torch.tensor([0, 1], device=init.device))
        init[mask] = MASK_TOKEN

    else:  # stage == 3
        # 保留非 resource，resource → MASK
        init[init == 3] = MASK_TOKEN

    return init


def make_random_wall_seed(ratio_min=0.02, ratio_max=0.08):
    ratio = random.uniform(ratio_min, ratio_max)
    n_wall = max(2, int(MAP_SIZE * ratio))
    seed = torch.full((1, MAP_SIZE), MASK_TOKEN, dtype=torch.long, device=device)
    idx = torch.randperm(MAP_SIZE)[:n_wall]
    seed[0, idx] = 1
    return seed

# ---------------------------------------------------------------------------
# 验证函数
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    stage: int,
    enc: GinkaVQVAE,
    model_mg: GinkaMaskGIT,
    dataloader_val: DataLoader,
    tile_dict: dict,
    epoch: int,
    n_rand: int = 3,
):
    enc.eval()
    model_mg.eval()

    epoch_dir = f"result/stage{stage}_img/e{epoch:04d}"
    os.makedirs(epoch_dir, exist_ok=True)

    val_loss_total = 0.0
    val_steps = 0
    captured = {s: None for s in ('A', 'B', 'C', 'D')}

    for batch in tqdm(dataloader_val, desc="Validating", leave=False, disable=disable_tqdm):
        raw_map = batch["raw_map"].to(device)
        vq_slice = batch["vq_slice"].to(device)
        stage_input = batch["stage_input"].to(device)
        target_map = batch["target_map"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        struct_cond = batch["struct_cond"].to(device)
        subsets = batch["subset"]

        z_q, _, _, vq_loss, _, _ = enc(vq_slice)
        logits = model_mg(stage_input, z_q, struct_cond=struct_cond)

        ce = masked_focal_loss(logits.permute(0, 2, 1), target_map, loss_mask)
        val_loss_total += (ce + vq_loss).item()
        val_steps      += 1

        for i in range(raw_map.shape[0]):
            s = subsets[i]
            if captured[s] is None:
                captured[s] = {
                    "raw": raw_map[i:i+1].clone(),
                    "stage_input": stage_input[i:i+1].clone(),
                    "z_q": z_q[i:i+1].clone(),
                    "struct_cond": struct_cond[i:i+1].clone(),
                }

        if all(v is not None for v in captured.values()):
            break

    # ---- 可视化：每个子集一张图 ----------------------------------------
    for sub, cap in captured.items():
        if cap is None:
            continue

        raw_img = label_image(make_map_image(cap["raw"][0], tile_dict), "GT")
        inp_img = label_image(make_map_image(cap["stage_input"][0], tile_dict), f"stage{stage} input")

        # 真实 z 的迭代生成
        init = make_stage_init(stage, cap["stage_input"][0].unsqueeze(0))
        gen = maskgit_generate(
            model_mg, cap["z_q"],
            init_map=init, struct_cond=cap["struct_cond"],
        )
        gen_img = label_image(make_map_image(gen[0], tile_dict), "z_real gen")

        # 随机 z 的生成
        rand_imgs = []
        for i in range(n_rand):
            z_r = enc.sample(1, device)
            sc_r = make_random_struct_cond()
            init2 = make_stage_init(stage, cap["raw"][0].unsqueeze(0))
            gen_r = maskgit_generate(model_mg, z_r, init_map=init2, struct_cond=sc_r)
            rand_imgs.append(label_image(make_map_image(gen_r[0], tile_dict), f"z_rand_{i+1}"))

        row = [raw_img, inp_img, gen_img] + rand_imgs
        cv2.imwrite(f"{epoch_dir}/subset_{sub}.png", grid_images(row))

    # ---- 场景：完全自主生成 -----------------------------------------------
    # stage1：从随机稀疏墙壁种子出发（完全不依赖 GT）
    # stage2：以验证集中采样的 floor/wall 结构为上下文，随机 z₂（模拟级联推理）
    # stage3：以验证集中采样的完整功能地图为上下文，随机 z₃（模拟级联推理）
    context_pool = [cap["raw"][0] for cap in captured.values() if cap is not None]

    rand_free = []
    for i in range(n_rand + 1):
        z_r  = enc.sample(1, device)
        sc_r = make_random_struct_cond()

        if stage == 1:
            # 稀疏 wall 种子作为提示，模型自主补全 floor/wall
            init = make_random_wall_seed()
        else:
            # 从验证集上下文池中轮流取一张图作为前序阶段的输出
            ctx = context_pool[i % len(context_pool)].unsqueeze(0)
            # make_stage_init 会自动将本阶段负责的 tile 位置替换为 MASK
            init = make_stage_init(stage, ctx)

        gen = maskgit_generate(model_mg, z_r, init_map=init, struct_cond=sc_r)
        rand_free.append(label_image(make_map_image(gen[0], tile_dict), f"free_{i+1}"))
    cv2.imwrite(f"{epoch_dir}/scene_free_random.png", grid_images(rand_free))

    return val_loss_total / max(val_steps, 1)

# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train():
    print(f"Using device: {device}")
    args = parse_arguments()
    stage = args.stage

    result_dir = f"result/stage{stage}"
    result_img_dir = f"result/stage{stage}_img"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_img_dir, exist_ok=True)

    # ---- VQ 编码器（单路）----
    mg_cfg = STAGE_MG_CONFIGS[stage]
    enc = GinkaVQVAE(
        num_classes=NUM_CLASSES,
        L=VQ_L,
        K=VQ_K,
        d_z=VQ_D_Z,
        d_model=VQ_D_MODEL,
        nhead=VQ_NHEAD,
        num_layers=VQ_LAYERS,
        dim_ff=VQ_DIM_FF,
        map_size=MAP_SIZE,
        beta=VQ_BETA,
        gamma=VQ_GAMMA,
    ).to(device)

    model_mg = GinkaMaskGIT(
        num_classes=NUM_CLASSES,
        d_model=mg_cfg["d_model"],
        d_z=VQ_D_Z,
        dim_ff=mg_cfg["dim_ff"],
        nhead=mg_cfg["nhead"],
        num_layers=mg_cfg["num_layers"],
        map_size=MAP_SIZE,
        z_dropout=MG_Z_DROPOUT,
        struct_dropout=MG_STRUCT_DROPOUT,
    ).to(device)

    enc_params = sum(p.numel() for p in enc.parameters())
    mg_params = sum(p.numel() for p in model_mg.parameters())
    print(f"[Stage {stage}] VQ Encoder 参数量: {enc_params:,}  ({enc_params/1e6:.3f}M)")
    print(f"[Stage {stage}] MaskGIT   参数量: {mg_params:,}  ({mg_params/1e6:.3f}M)")

    # ---- 数据集 ----
    dataset_train = GinkaStageDataset(
        args.train,
        stage=stage,
        subset_weights=SUBSET_WEIGHTS,
        wall_mask_ratio=WALL_MASK_RATIO,
    )
    dataset_val = GinkaStageDataset(
        args.validate,
        stage=stage,
        subset_weights=SUBSET_WEIGHTS,
        room_thresholds=dataset_train.room_th,
        branch_thresholds=dataset_train.branch_th,
        wall_mask_ratio=WALL_MASK_RATIO,
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # ---- 优化器 ----
    all_params = list(enc.parameters()) + list(model_mg.parameters())
    optimizer = optim.AdamW(all_params, lr=2e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ---- 权重加载 ----
    start_epoch = 0

    if args.pretrain_vq:
        # 从 train_vq.py 的联合训练检查点加载对应通道的 VQ 编码器
        ckpt = torch.load(args.pretrain_vq, map_location=device)
        enc_key = f"enc{stage}"
        if enc_key in ckpt:
            enc.load_state_dict(ckpt[enc_key], strict=False)
            print(f"已从 {args.pretrain_vq} 加载 {enc_key} 权重。")
        else:
            print(f"警告：检查点中未找到 {enc_key}，跳过权重加载。")

    if args.resume:
        state_path = args.state or f"{result_dir}/stage{stage}-latest.pth"
        ckpt = torch.load(state_path, map_location=device)
        enc.load_state_dict(ckpt["enc"], strict=False)
        model_mg.load_state_dict(ckpt["mg_state"], strict=False)
        if args.load_optim and ckpt.get("optim_state") is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 epoch {start_epoch} 接续训练。")

    # ---- tile 贴图 ----
    tile_dict = {}
    for f in os.listdir("tiles"):
        name = os.path.splitext(f)[0]
        img = cv2.imread(f"tiles/{f}", cv2.IMREAD_UNCHANGED)
        if img is not None:
            tile_dict[name] = img

    # ---- 冻结 VQ 编码器（可选）----
    if args.freeze_vq:
        for p in enc.parameters():
            p.requires_grad_(False)
        print(f"[Stage {stage}] VQ 编码器已冻结。")

    # ---- 训练循环 ----
    for epoch in tqdm(
        range(start_epoch, start_epoch + args.epochs),
        desc=f"Stage{stage} Training",
        disable=disable_tqdm,
    ):
        enc.train()
        model_mg.train()

        loss_total = 0.0
        ce_total = 0.0
        vq_loss_total = 0.0
        subset_stats = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # 按 tile 统计召回率（用于监控各类 tile 的预测准确性）
        tile_correct = {tid: 0 for tid in STAGE_TILE_SETS[stage]}
        tile_total = {tid: 0 for tid in STAGE_TILE_SETS[stage]}

        for batch in tqdm(
            dataloader_train,
            leave=False,
            desc="Epoch Progress",
            disable=disable_tqdm,
        ):
            raw_map = batch["raw_map"].to(device)
            vq_slice = batch["vq_slice"].to(device)
            stage_input = batch["stage_input"].to(device)
            target_map = batch["target_map"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            struct_cond = batch["struct_cond"].to(device)

            for s in batch["subset"]:
                subset_stats[s] = subset_stats.get(s, 0) + 1

            # ---- 前向传播 ----
            z_q, _, _, vq_loss, commit_loss, entropy_loss = enc(vq_slice)
            logits = model_mg(stage_input, z_q, struct_cond=struct_cond)  # [B, S, C]

            # ---- 仅对本阶段 tile 位置计算 focal loss ----
            ce_loss  = masked_focal_loss(logits.permute(0, 2, 1), target_map, loss_mask)
            loss_cfg = STAGE_LOSS_CONFIG[stage]
            loss     = loss_cfg["ce_weight"] * ce_loss + loss_cfg["vq_weight"] * vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            loss_total += loss.detach().item()
            ce_total += ce_loss.detach().item()
            vq_loss_total += vq_loss.detach().item()

            # ---- 分 tile 召回率统计 ----
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # [B, S]
                for tid in STAGE_TILE_SETS[stage]:
                    gt_mask = (target_map == tid) & loss_mask
                    tile_total[tid] += gt_mask.sum().item()
                    tile_correct[tid] += (preds[gt_mask] == tid).sum().item()

        scheduler.step()

        n = len(dataloader_train)
        recall_str = "  ".join(
            f"{STAGE_TILE_SETS[stage][tid]}={tile_correct[tid]/(tile_total[tid]+1e-6):.2%}"
            for tid in STAGE_TILE_SETS[stage]
        )
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch + 1:4d} | "
            f"Loss {loss_total/n:.5f}  "
            f"Focal {ce_total/n:.5f}  "
            f"VQ {vq_loss_total/n:.5f} | "
            f"Recall: {recall_str} | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | "
            f"Subsets {subset_stats}"
        )

        # ---- 检查点 + 验证 ----
        if (epoch + 1) % args.checkpoint == 0:
            ckpt_path = f"{result_dir}/stage{stage}-{epoch + 1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "stage": stage,
                "enc": enc.state_dict(),
                "mg_state": model_mg.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, ckpt_path)
            tqdm.write(f"  检查点已保存: {ckpt_path}")

            val_loss = validate(stage, enc, model_mg, dataloader_val, tile_dict, epoch + 1)
            tqdm.write(f"[Validate] Epoch {epoch + 1:4d} | Val Loss {val_loss:.5f}")

            enc.train()
            model_mg.train()

    # ---- 最终存档 ----
    torch.save({
        "epoch": start_epoch + args.epochs,
        "stage": stage,
        "enc": enc.state_dict(),
        "mg_state": model_mg.state_dict(),
    }, f"{result_dir}/stage{stage}_final.pth")
    print(f"[Stage {stage}] 训练结束。")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
