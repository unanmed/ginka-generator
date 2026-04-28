"""
VQ 编码器预训练脚本（方案 D）

目标：在联合训练开始前，先单独预训练 VQ 编码器，使其学到地图的大致语义分类。
解码头（VQDecodeHead）仅在预训练阶段使用，结束后丢弃，权重不迁移到联合训练。

训练流程（对应设计文档方案 D 三阶段）：
    阶段 0（本脚本）：编码器 + 临时解码头，全图重建目标
    阶段 1（在 train_vq.py 中）：编码器冻结 + MaskGIT 热身，启用 --freeze_vq
    阶段 2（在 train_vq.py 中）：完整联合训练，编码器用较小 LR

用法示例：
    python -m ginka.train_pretrain
    python -m ginka.train_pretrain --resume True --state result/pretrain/pretrain-20.pth
    # 预训练完成后，传入权重路径启动联合训练阶段 1：
    python -m ginka.train_vq --resume True --state result/pretrain/pretrain_final.pth
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .vqvae.model import GinkaVQVAE, VQDecodeHead
from .dataset import load_data

# ---------------------------------------------------------------------------
# 超参数（须与 train_vq.py 中 VQ-VAE 配置保持一致）
# ---------------------------------------------------------------------------
BATCH_SIZE   = 64
NUM_CLASSES  = 16
MAP_SIZE     = 13 * 13
MAP_H = MAP_W = 13

# VQ-VAE 超参（保持与 train_vq.py 一致）
VQ_L      = 2
VQ_K      = 4
VQ_D_Z    = 128
VQ_D_MODEL= 192
VQ_NHEAD  = 8
VQ_LAYERS = 4
VQ_DIM_FF = 512
VQ_BETA   = 0.5
VQ_GAMMA  = 0.0

# 解码头超参（与编码器对称：同等层数和 FFN 宽度）
DH_NHEAD   = 8    # Cross-Attention 头数（VQ_D_Z=128 可被 8 整除）
DH_DIM_FF  = 512  # FFN 隐层维度（与编码器 VQ_DIM_FF 一致）
DH_LAYERS  = 4    # 解码层数（与编码器 VQ_LAYERS 一致）

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs("result/pretrain", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

# ---------------------------------------------------------------------------
# 简单数据集：仅返回 raw_map，无子集划分，无掩码
# ---------------------------------------------------------------------------
class GinkaPretrainDataset(Dataset):
    """
    预训练专用数据集，仅提供完整原始地图（raw_map）和随机数据增强。

    不做子集划分与掩码处理；重建目标为全图所有 169 个位置。
    """

    def __init__(self, data_path: str):
        self.data = load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        arr = np.array(item['map'], dtype=np.int64)     # [H, W]

        # 随机旋转 / 翻转数据增强
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            arr = np.rot90(arr, k).copy()
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() > 0.5:
            arr = np.flipud(arr).copy()

        raw_map = torch.tensor(arr.reshape(-1), dtype=torch.long)   # [H*W]
        return raw_map

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="VQ 编码器预训练（方案 D）")
    parser.add_argument("--resume",     type=bool, default=False)
    parser.add_argument("--state",      type=str,  default="result/pretrain/pretrain-20.pth",
                        help="续训时加载的检查点路径")
    parser.add_argument("--train",      type=str,  default="ginka-dataset.json")
    parser.add_argument("--validate",   type=str,  default="ginka-eval.json")
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--checkpoint", type=int,  default=5,
                        help="每隔多少 epoch 保存检查点并输出验证指标")
    parser.add_argument("--load_optim", type=bool, default=True)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 验证：计算全图 top-1 准确率及关键类别（墙壁）召回率
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model_vq: GinkaVQVAE,
    decode_head: VQDecodeHead,
    dataloader_val: DataLoader,
) -> dict:
    model_vq.eval()
    decode_head.eval()

    total, correct = 0, 0
    wall_tp, wall_gt = 0, 0     # wall tile=1 的召回
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    class_total   = torch.zeros(NUM_CLASSES, dtype=torch.long)

    for raw_map in tqdm(dataloader_val, desc="Validating", leave=False, disable=disable_tqdm):
        raw_map = raw_map.to(device)                      # [B, H*W]

        z_q, _, _, _, _, _ = model_vq(raw_map)
        logits = decode_head(z_q)                         # [B, H*W, C]
        pred   = logits.argmax(dim=-1)                    # [B, H*W]

        correct += (pred == raw_map).sum().item()
        total   += raw_map.numel()

        # 墙壁召回
        wall_mask = (raw_map == 1)
        wall_tp  += (pred[wall_mask] == 1).sum().item()
        wall_gt  += wall_mask.sum().item()

        # 逐类别统计
        for c in range(NUM_CLASSES):
            mask_c = (raw_map == c)
            class_correct[c] += (pred[mask_c] == c).sum().item()
            class_total[c]   += mask_c.sum().item()

    acc      = correct / max(total, 1)
    wall_rec = wall_tp / max(wall_gt, 1)

    # 有样本的类别逐一统计
    per_class = {}
    for c in range(NUM_CLASSES):
        if class_total[c] > 0:
            per_class[c] = class_correct[c].item() / class_total[c].item()

    return {"acc": acc, "wall_recall": wall_rec, "per_class": per_class}

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

    decode_head = VQDecodeHead(
        num_classes=NUM_CLASSES,
        d_z=VQ_D_Z,
        map_size=MAP_SIZE,
        nhead=DH_NHEAD,
        dim_ff=DH_DIM_FF,
        num_layers=DH_LAYERS,
    ).to(device)

    vq_params = sum(p.numel() for p in model_vq.parameters())
    dh_params = sum(p.numel() for p in decode_head.parameters())
    print(f"VQ-VAE     参数量: {vq_params:,}  ({vq_params/1e6:.3f}M)")
    print(f"DecodeHead 参数量: {dh_params:,}  ({dh_params/1e6:.3f}M)")

    # ---- 数据集 ----
    dataset_train = GinkaPretrainDataset(args.train)
    dataset_val   = GinkaPretrainDataset(args.validate)
    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0,
    )
    print(f"训练集: {len(dataset_train)} 条  验证集: {len(dataset_val)} 条")

    # ---- 优化器 ----
    all_params = list(model_vq.parameters()) + list(decode_head.parameters())
    optimizer  = optim.AdamW(all_params, lr=2e-4, weight_decay=1e-2)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- 续训 ----
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.state, map_location=device)
        model_vq.load_state_dict(ckpt["vq_state"],  strict=False)
        if "dh_state" in ckpt:
            decode_head.load_state_dict(ckpt["dh_state"], strict=False)
        if args.load_optim and ckpt.get("optim_state") is not None:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"从 epoch {start_epoch} 接续训练。")

    # ---- 训练循环 ----
    for epoch in tqdm(range(start_epoch, start_epoch + args.epochs),
                      desc="VQ Pretrain", disable=disable_tqdm):
        model_vq.train()
        decode_head.train()

        loss_total    = 0.0
        ce_total      = 0.0
        commit_total  = 0.0
        entropy_total = 0.0

        for raw_map in tqdm(dataloader_train, leave=False,
                            desc="Epoch Progress", disable=disable_tqdm):
            raw_map = raw_map.to(device)                  # [B, H*W]

            # 1. 编码
            z_q, _, _, vq_loss, commit_loss, entropy_loss = model_vq(raw_map)

            # 2. 解码→全图重建
            logits = decode_head(z_q)                     # [B, H*W, C]
            ce_loss = F.cross_entropy(
                logits.permute(0, 2, 1), raw_map          # [B, C, H*W] vs [B, H*W]
            )

            # 3. 总损失（重建 + VQ 正则）
            loss = ce_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            loss_total    += loss.detach().item()
            ce_total      += ce_loss.detach().item()
            commit_total  += commit_loss.detach().item()
            entropy_total += entropy_loss.detach().item()

        scheduler.step()

        n = len(dataloader_train)
        tqdm.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch + 1:4d} | "
            f"Loss {loss_total/n:.5f}  "
            f"CE {ce_total/n:.5f}  "
            f"Commit {commit_total/n:.5f}  "
            f"Entropy {entropy_total/n:.5f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f}"
        )

        # ---- 检查点 + 验证 ----
        if (epoch + 1) % args.checkpoint == 0:
            ckpt_path = f"result/pretrain/pretrain-{epoch + 1}.pth"
            torch.save({
                "epoch":       epoch + 1,
                "vq_state":    model_vq.state_dict(),
                "dh_state":    decode_head.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, ckpt_path)
            tqdm.write(f"  检查点已保存: {ckpt_path}")

            metrics = validate(model_vq, decode_head, dataloader_val)
            acc_str = f"  [Validate] Acc {metrics['acc']:.4f}  Wall Recall {metrics['wall_recall']:.4f}"

            # 输出有样本的类别准确率
            pc = metrics["per_class"]
            detail = "  ".join(
                f"c{c}={v:.3f}" for c, v in sorted(pc.items()) if v < 1.0
            )
            if detail:
                acc_str += f"\n           Per-class: {detail}"
            tqdm.write(acc_str)

            model_vq.train()
            decode_head.train()

    # ---- 保存最终 VQ 编码器权重 ----
    final_path = "result/pretrain/pretrain_final.pth"
    torch.save({
        "epoch":    start_epoch + args.epochs,
        "vq_state": model_vq.state_dict(),
        # 不保存解码头：联合训练阶段不需要
    }, final_path)
    print(f"\n预训练完成。编码器权重已保存至: {final_path}")
    print(f"联合训练阶段 1 启动命令（编码器冻结热身）：")
    print(f"  python -m ginka.train_vq --resume True --state {final_path} --freeze_vq True")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
