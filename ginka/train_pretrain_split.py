"""
三通道分拆预训练脚本（方案 B）

三路编码器各自负责一个语义通道：
  通道 1：空间骨架（floor+wall），损失仅计算 wall(1) 位置
  通道 2：关卡门控（floor+wall+door+mob+entrance），损失仅计算 {2,9,10} 位置
  通道 3：收集资源（完整地图），损失仅计算 {3,4,5,6,7,8} 位置

预训练完成后保存各通道编码器权重（不含解码头），
供联合训练脚本 train_vq.py 加载并拼接 z。

用法示例：
    python -m ginka.train_pretrain_split
    python -m ginka.train_pretrain_split --resume True --state result/pretrain_split/split-10.pth
    # 预训练完成后指定权重路径启动联合训练：
    python -m ginka.train_vq --pretrain_split result/pretrain_split/split_final.pth
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .vqvae.model import GinkaVQVAE, VQDecodeHead
from .dataset import GinkaSplitDataset
from .utils import masked_focal

# ---------------------------------------------------------------------------
# 超参数
# ---------------------------------------------------------------------------
BATCH_SIZE   = 64
NUM_CLASSES  = 16
MAP_SIZE     = 13 * 13
FOCAL_GAMMA  = 2.0

# 通道 1：空间骨架（floor+wall）
CH1_KEEP     = {0, 1}          # 编码器输入保留的 tile
CH1_LOSS     = {1}             # 损失计算范围（仅 wall）
CH1_D_MODEL  = 128
CH1_NHEAD    = 4

# 通道 2：关卡门控
CH2_KEEP     = {0, 1, 2, 9, 10}
CH2_LOSS     = {2, 9, 10}
CH2_D_MODEL  = 64
CH2_NHEAD    = 4

# 通道 3：收集资源
CH3_KEEP     = None            # 完整地图，无需切片
CH3_LOSS     = {3, 4, 5, 6, 7, 8}
CH3_D_MODEL  = 64
CH3_NHEAD    = 4

# 三路共用的 VQ 超参
VQ_L      = 2
VQ_K      = 16
VQ_D_Z    = 64
VQ_LAYERS = 2
VQ_DIM_FF = 256
VQ_BETA   = 0.25   # commit loss 权重
VQ_GAMMA  = 0.1    # entropy loss 权重

# 解码头超参（三路共用相同规格）
DH_NHEAD   = 4
DH_DIM_FF  = 256
DH_LAYERS  = 2

# ---------------------------------------------------------------------------
# 设备
# ---------------------------------------------------------------------------
device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs("result/pretrain_split", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="三通道分拆 VQ 编码器预训练（方案 B）")
    parser.add_argument("--resume",     type=bool, default=False)
    parser.add_argument("--state",      type=str,  default="result/pretrain_split/split-10.pth",
                        help="续训时加载的检查点路径")
    parser.add_argument("--train",      type=str,  default="ginka-dataset.json")
    parser.add_argument("--validate",   type=str,  default="ginka-eval.json")
    parser.add_argument("--epochs",     type=int,  default=60)
    parser.add_argument("--checkpoint", type=int,  default=5,
                        help="每隔多少 epoch 保存检查点并输出验证指标")
    parser.add_argument("--load_optim", type=bool, default=True)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 验证：各通道专属 tile 召回率 + codebook 使用熵
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    enc1, enc2, enc3,
    head1, head2, head3,
    dataloader_val: DataLoader,
) -> dict:
    for m in [enc1, enc2, enc3, head1, head2, head3]:
        m.eval()

    # 每类 tile 的 tp / gt 计数
    ch1_tp, ch1_gt = 0, 0                      # wall(1)
    ch2_tp = {t: 0 for t in CH2_LOSS}          # {2,9,10}
    ch2_gt = {t: 0 for t in CH2_LOSS}
    ch3_tp = {t: 0 for t in CH3_LOSS}          # {3,4,5,6,7,8}
    ch3_gt = {t: 0 for t in CH3_LOSS}

    # codebook 使用频次（用于熵估算）
    codebook_counts = [
        torch.zeros(VQ_K, dtype=torch.long),   # 通道 1
        torch.zeros(VQ_K, dtype=torch.long),   # 通道 2
        torch.zeros(VQ_K, dtype=torch.long),   # 通道 3
    ]

    for batch in tqdm(dataloader_val, desc="Validating", leave=False, disable=disable_tqdm):
        raw_map = batch["raw_map"].to(device)
        s1      = batch["slice1"].to(device)
        s2      = batch["slice2"].to(device)
        s3      = batch["slice3"].to(device)

        # 通道 1
        z_q1, _, idx1, _, _, _ = enc1(s1)
        logits1 = head1(z_q1)
        pred1   = logits1.argmax(dim=-1)        # [B, H*W]
        wall_m  = (raw_map == 1)
        ch1_tp += (pred1[wall_m] == 1).sum().item()
        ch1_gt += wall_m.sum().item()
        for code in idx1.view(-1).cpu():
            codebook_counts[0][code] += 1

        # 通道 2
        z_q2, _, idx2, _, _, _ = enc2(s2)
        logits2 = head2(z_q2)
        pred2   = logits2.argmax(dim=-1)
        for t in CH2_LOSS:
            m = (raw_map == t)
            ch2_tp[t] += (pred2[m] == t).sum().item()
            ch2_gt[t] += m.sum().item()
        for code in idx2.view(-1).cpu():
            codebook_counts[1][code] += 1

        # 通道 3
        z_q3, _, idx3, _, _, _ = enc3(s3)
        logits3 = head3(z_q3)
        pred3   = logits3.argmax(dim=-1)
        for t in CH3_LOSS:
            m = (raw_map == t)
            ch3_tp[t] += (pred3[m] == t).sum().item()
            ch3_gt[t] += m.sum().item()
        for code in idx3.view(-1).cpu():
            codebook_counts[2][code] += 1

    def _entropy(counts):
        """估算 codebook 使用熵（bits）。"""
        counts = counts.float() + 1e-8
        p = counts / counts.sum()
        return float(-(p * torch.log2(p)).sum())

    metrics = {
        "ch1_wall_recall": ch1_tp / max(ch1_gt, 1),
        "ch2_recall": {t: ch2_tp[t] / max(ch2_gt[t], 1) for t in CH2_LOSS},
        "ch3_recall": {t: ch3_tp[t] / max(ch3_gt[t], 1) for t in CH3_LOSS},
        "codebook_entropy": [_entropy(c) for c in codebook_counts],
    }
    return metrics

# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------
def train():
    print(f"Using device: {device}")
    args = parse_arguments()

    # ---- 三路编码器 ----
    enc1 = GinkaVQVAE(
        num_classes=NUM_CLASSES, L=VQ_L, K=VQ_K, d_z=VQ_D_Z,
        d_model=CH1_D_MODEL, nhead=CH1_NHEAD, num_layers=VQ_LAYERS,
        dim_ff=VQ_DIM_FF, beta=VQ_BETA, gamma=VQ_GAMMA,
    ).to(device)

    enc2 = GinkaVQVAE(
        num_classes=NUM_CLASSES, L=VQ_L, K=VQ_K, d_z=VQ_D_Z,
        d_model=CH2_D_MODEL, nhead=CH2_NHEAD, num_layers=VQ_LAYERS,
        dim_ff=VQ_DIM_FF, beta=VQ_BETA, gamma=VQ_GAMMA,
    ).to(device)

    enc3 = GinkaVQVAE(
        num_classes=NUM_CLASSES, L=VQ_L, K=VQ_K, d_z=VQ_D_Z,
        d_model=CH3_D_MODEL, nhead=CH3_NHEAD, num_layers=VQ_LAYERS,
        dim_ff=VQ_DIM_FF, beta=VQ_BETA, gamma=VQ_GAMMA,
    ).to(device)

    # ---- 三路解码头（预训练专用，训练后丢弃）----
    head1 = VQDecodeHead(
        num_classes=NUM_CLASSES, d_z=VQ_D_Z, map_size=MAP_SIZE,
        nhead=DH_NHEAD, dim_ff=DH_DIM_FF, num_layers=DH_LAYERS,
    ).to(device)

    head2 = VQDecodeHead(
        num_classes=NUM_CLASSES, d_z=VQ_D_Z, map_size=MAP_SIZE,
        nhead=DH_NHEAD, dim_ff=DH_DIM_FF, num_layers=DH_LAYERS,
    ).to(device)

    head3 = VQDecodeHead(
        num_classes=NUM_CLASSES, d_z=VQ_D_Z, map_size=MAP_SIZE,
        nhead=DH_NHEAD, dim_ff=DH_DIM_FF, num_layers=DH_LAYERS,
    ).to(device)

    # ---- 优化器（三路同步训练） ----
    optimizer = optim.AdamW(
        list(enc1.parameters()) + list(enc2.parameters()) + list(enc3.parameters()) +
        list(head1.parameters()) + list(head2.parameters()) + list(head3.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    start_epoch = 0

    # ---- 续训 ----
    if args.resume:
        ckpt = torch.load(args.state, map_location=device)
        enc1.load_state_dict(ckpt["enc1"])
        enc2.load_state_dict(ckpt["enc2"])
        enc3.load_state_dict(ckpt["enc3"])
        head1.load_state_dict(ckpt["head1"])
        head2.load_state_dict(ckpt["head2"])
        head3.load_state_dict(ckpt["head3"])
        if args.load_optim and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}: {args.state}")

    # ---- 数据集 ----
    ds_train = GinkaSplitDataset(args.train)
    ds_val   = GinkaSplitDataset(args.validate)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练集大小: {len(ds_train)}，验证集大小: {len(ds_val)}")

    total_params = (
        sum(p.numel() for p in enc1.parameters()) +
        sum(p.numel() for p in enc2.parameters()) +
        sum(p.numel() for p in enc3.parameters())
    )
    print(f"编码器总参数量（三路）: {total_params:,}  ({total_params / 1e6:.3f}M)")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, args.epochs):
        for m in [enc1, enc2, enc3, head1, head2, head3]:
            m.train()

        total_loss = 0.0
        ch_losses  = [0.0, 0.0, 0.0]

        for batch in tqdm(dl_train, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=disable_tqdm):
            raw_map = batch["raw_map"].to(device)
            s1      = batch["slice1"].to(device)
            s2      = batch["slice2"].to(device)
            s3      = batch["slice3"].to(device)

            optimizer.zero_grad()

            # ─── 通道 1 ───
            z_q1, _, _, vq_loss1, commit_loss1, entropy_loss1 = enc1(s1)
            logits1 = head1(z_q1)                             # [B, H*W, C]
            fl1     = masked_focal(logits1, raw_map, CH1_LOSS, gamma=FOCAL_GAMMA)
            loss1   = fl1 + VQ_BETA * commit_loss1 + VQ_GAMMA * entropy_loss1

            # ─── 通道 2 ───
            z_q2, _, _, vq_loss2, commit_loss2, entropy_loss2 = enc2(s2)
            logits2 = head2(z_q2)
            fl2     = masked_focal(logits2, raw_map, CH2_LOSS, gamma=FOCAL_GAMMA)
            loss2   = fl2 + VQ_BETA * commit_loss2 + VQ_GAMMA * entropy_loss2

            # ─── 通道 3 ───
            z_q3, _, _, vq_loss3, commit_loss3, entropy_loss3 = enc3(s3)
            logits3 = head3(z_q3)
            fl3     = masked_focal(logits3, raw_map, CH3_LOSS, gamma=FOCAL_GAMMA)
            loss3   = fl3 + VQ_BETA * commit_loss3 + VQ_GAMMA * entropy_loss3

            loss = loss1 + loss2 + loss3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc1.parameters()) + list(enc2.parameters()) + list(enc3.parameters()) +
                list(head1.parameters()) + list(head2.parameters()) + list(head3.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss   += loss.item()
            ch_losses[0] += loss1.item()
            ch_losses[1] += loss2.item()
            ch_losses[2] += loss3.item()

        n_batches = len(dl_train)
        print(
            f"[{epoch + 1:03d}] total={total_loss / n_batches:.4f}  "
            f"ch1={ch_losses[0] / n_batches:.4f}  "
            f"ch2={ch_losses[1] / n_batches:.4f}  "
            f"ch3={ch_losses[2] / n_batches:.4f}"
        )

        # ---- 检查点 & 验证 ----
        if (epoch + 1) % args.checkpoint == 0 or epoch + 1 == args.epochs:
            metrics = validate(enc1, enc2, enc3, head1, head2, head3, dl_val)
            print(
                f"  验证  ch1_wall_recall={metrics['ch1_wall_recall']:.3f}  "
                f"ch2_recall={metrics['ch2_recall']}  "
                f"ch3_recall={metrics['ch3_recall']}"
            )
            print(
                f"  codebook_entropy  ch1={metrics['codebook_entropy'][0]:.3f}  "
                f"ch2={metrics['codebook_entropy'][1]:.3f}  "
                f"ch3={metrics['codebook_entropy'][2]:.3f}"
            )

            ts = datetime.now().strftime("%m%d-%H%M")
            ckpt_path = f"result/pretrain_split/split-{epoch + 1}.pth"
            torch.save({
                "epoch":     epoch + 1,
                "enc1":      enc1.state_dict(),
                "enc2":      enc2.state_dict(),
                "enc3":      enc3.state_dict(),
                "head1":     head1.state_dict(),
                "head2":     head2.state_dict(),
                "head3":     head3.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics":   metrics,
                "ts":        ts,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ---- 保存最终编码器权重（供联合训练加载） ----
    final_path = "result/pretrain_split/split_final.pth"
    torch.save({
        "epoch": args.epochs,
        "enc1":  enc1.state_dict(),
        "enc2":  enc2.state_dict(),
        "enc3":  enc3.state_dict(),
        # 解码头不迁移，不保存
    }, final_path)
    print(f"\n预训练完成，编码器权重已保存至: {final_path}")
    print("接下来运行联合训练（阶段 1 冻结热身）：")
    print(f"  python -m ginka.train_vq --pretrain_split {final_path} --freeze_vq True")


if __name__ == "__main__":
    train()
