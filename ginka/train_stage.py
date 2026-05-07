"""
三阶段级联训练脚本：各阶段独立训练，使用 GinkaStageDataset。

总损失 = L_CE（只对本阶段负责的 tile 位置计算）+ beta * L_commit + gamma * L_entropy

各阶段分工：
  stage=1  结构骨架：floor(0) + wall(1)
  stage=2  功能元素：door(2) + monster(4) + entrance(5)
  stage=3  资源放置：resource(3)

用法示例：
    python -m ginka.train_stage --stage 0               # 三阶段联合训练（推荐）
    python -m ginka.train_stage --stage 1               # 只训练 stage1
    python -m ginka.train_stage --stage 0 --resume True --state result/joint-50.pth
    python -m ginka.train_stage --stage 0 --pretrain_vq result/joint/joint-50.pth
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
    parser.add_argument(
        "--stage", type=int, required=True, choices=[0, 1, 2, 3],
        help="训练阶段：1/2/3 单独训练，0 = 依次训练全部三个阶段",
    )
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

    # ---- 场景：完全自主生成（仅单阶段时执行，多阶段由级联验证统一覆盖）------
    if True:  # 占位，避免缩进塌陷；单阶段验证不做级联，跳过
        pass

    return val_loss_total / max(val_steps, 1)

# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def _build_stage(stage: int, args):
    """初始化单个阶段的模型、数据集，返回状态字典（不含优化器）。"""
    result_dir = f"result/stage{stage}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"result/stage{stage}_img", exist_ok=True)

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
    print(f"[Stage {stage}] VQ={enc_params/1e6:.2f}M  MaskGIT={mg_params/1e6:.2f}M")

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

    if args.pretrain_vq:
        ckpt = torch.load(args.pretrain_vq, map_location=device)
        enc_key = f"enc{stage}"
        if enc_key in ckpt:
            enc.load_state_dict(ckpt[enc_key], strict=False)
            print(f"[Stage {stage}] 已加载预训练 VQ 权重。")
        else:
            print(f"[Stage {stage}] 警告：检查点中未找到 {enc_key}。")

    if args.freeze_vq:
        for p in enc.parameters():
            p.requires_grad_(False)
        print(f"[Stage {stage}] VQ 编码器已冻结。")

    return {
        "stage": stage,
        "enc": enc,
        "model_mg": model_mg,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "result_dir": result_dir,
    }


# ---------------------------------------------------------------------------
def train():
    print(f"Using device: {device}")
    args = parse_arguments()
    stages = [1, 2, 3] if args.stage == 0 else [args.stage]

    # ---- tile 贴图（一次性加载，所有阶段共用）----
    tile_dict = {}
    for f in os.listdir("tiles"):
        name = os.path.splitext(f)[0]
        img = cv2.imread(f"tiles/{f}", cv2.IMREAD_UNCHANGED)
        if img is not None:
            tile_dict[name] = img

    # ---- 初始化各阶段 ----
    states = {stage: _build_stage(stage, args) for stage in stages}

    # ---- 合并优化器（所有阶段参数统一管理）----
    all_params = []
    for st in states.values():
        all_params += list(st["enc"].parameters()) + list(st["model_mg"].parameters())
    optimizer = optim.AdamW(all_params, lr=2e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ---- 续训 ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.state, map_location=device)
        for stage in stages:
            st = states[stage]
            st["enc"].load_state_dict(ckpt[f"enc{stage}"], strict=False)
            st["model_mg"].load_state_dict(ckpt[f"mg{stage}"], strict=False)
        if args.load_optim and ckpt.get("optim_state") is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 epoch {start_epoch} 接续训练。")

    # ---- 数据集对齐：以最短的 dataloader 为准，zip 迭代 ----
    # 单阶段时直接用该阶段的 dataloader；多阶段时 zip 保证每个 batch 各阶段同步推进
    def _epoch_iters():
        loaders = [states[s]["dataloader_train"] for s in stages]
        return zip(*loaders)

    # ---- 训练循环 ----
    for epoch in tqdm(
        range(start_epoch, start_epoch + args.epochs),
        desc="Training",
        disable=disable_tqdm,
    ):
        for st in states.values():
            st["enc"].train()
            st["model_mg"].train()

        loss_totals = {s: 0.0 for s in stages}
        ce_totals  = {s: 0.0 for s in stages}
        vq_totals  = {s: 0.0 for s in stages}
        n_batches = 0

        for batches in tqdm(
            _epoch_iters(),
            leave=False,
            desc="Batch",
            disable=disable_tqdm,
        ):
            optimizer.zero_grad()
            total_loss = 0.0

            for stage, batch in zip(stages, batches):
                st = states[stage]
                vq_slice = batch["vq_slice"].to(device)
                stage_input = batch["stage_input"].to(device)
                target_map = batch["target_map"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                struct_cond = batch["struct_cond"].to(device)

                z_q, _, _, vq_loss, _, _ = st["enc"](vq_slice)
                logits = st["model_mg"](stage_input, z_q, struct_cond=struct_cond)

                ce_loss = masked_focal_loss(logits.permute(0, 2, 1), target_map, loss_mask)
                cfg = STAGE_LOSS_CONFIG[stage]
                loss = cfg["ce_weight"] * ce_loss + cfg["vq_weight"] * vq_loss
                total_loss = total_loss + loss
                loss_totals[stage] += loss.detach().item()
                ce_totals[stage] += ce_loss.detach().item()
                vq_totals[stage] += vq_loss.detach().item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            n_batches += 1

        scheduler.step()

        n = max(n_batches, 1)
        total_avg = sum(loss_totals.values()) / n
        stage_loss_str = "  ".join(
            f"S{s}[focal={ce_totals[s]/n:.4f} vq={vq_totals[s]/n:.4f}]" for s in stages
        )
        lr_now = scheduler.get_last_lr()[0]
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch + 1:4d} | "
            f"Total {total_avg:.4f} | {stage_loss_str} | "
            f"LR {lr_now:.2e}"
        )

        # ---- 检查点 + 验证 ----
        if (epoch + 1) % args.checkpoint == 0:
            # 保存联合检查点
            ckpt_data = {"epoch": epoch + 1, "optim_state": optimizer.state_dict()}
            for stage in stages:
                st = states[stage]
                ckpt_data[f"enc{stage}"] = st["enc"].state_dict()
                ckpt_data[f"mg{stage}"] = st["model_mg"].state_dict()
            ckpt_path = f"result/stage{stages[-1]}/joint-{epoch + 1}.pth"
            torch.save(ckpt_data, ckpt_path)
            tqdm.write(f"  检查点已保存: {ckpt_path}")

            # 各阶段验证
            val_loss_total = 0.0
            for stage in stages:
                st = states[stage]
                vl = validate(
                    stage, st["enc"], st["model_mg"],
                    st["dataloader_val"], tile_dict, epoch + 1,
                )
                val_loss_total += vl
                tqdm.write(f"  [Stage {stage}] Val Loss {vl:.5f}")

            # 级联自由生成（stage1→stage2→stage3）
            if len(stages) == 3:
                _cascade_free_validate(states, tile_dict, epoch + 1)

            for st in states.values():
                st["enc"].train()
                st["model_mg"].train()

    # ---- 最终存档 ----
    ckpt_data = {"epoch": start_epoch + args.epochs}
    for stage in stages:
        st = states[stage]
        ckpt_data[f"enc{stage}"] = st["enc"].state_dict()
        ckpt_data[f"mg{stage}"] = st["model_mg"].state_dict()
    torch.save(ckpt_data, "result/joint_final.pth")
    print("训练结束。")


@torch.no_grad()
def _cascade_free_validate(states: dict, tile_dict: dict, epoch: int, n: int = 4):
    """
    三阶段级联自由生成：stage1 生成结果 → stage2 上下文 → stage3 上下文，
    最终只展示 stage3 的完整地图（已含所有 tile）。
    """
    epoch_dir = f"result/cascade_img/e{epoch:04d}"
    os.makedirs(epoch_dir, exist_ok=True)

    imgs = []
    for i in range(n):
        sc = make_random_struct_cond()

        # Stage 1：全 MASK → 生成 floor/wall
        z1 = states[1]["enc"].sample(1, device)
        init1 = make_random_wall_seed()
        map1 = maskgit_generate(states[1]["model_mg"], z1, init_map=init1, struct_cond=sc)

        # Stage 2：以 stage1 结果为上下文，生成 door/monster/entrance
        z2 = states[2]["enc"].sample(1, device)
        init2 = make_stage_init(2, map1)
        map2 = maskgit_generate(states[2]["model_mg"], z2, init_map=init2, struct_cond=sc)

        # Stage 3：以 stage2 结果为上下文，生成 resource
        z3 = states[3]["enc"].sample(1, device)
        init3 = make_stage_init(3, map2)
        map3 = maskgit_generate(states[3]["model_mg"], z3, init_map=init3, struct_cond=sc)

        imgs.append(label_image(make_map_image(map3[0], tile_dict), f"cascade_{i+1}"))

    cv2.imwrite(f"{epoch_dir}/cascade_free.png", grid_images(imgs))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
