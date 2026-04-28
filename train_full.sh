#!/usr/bin/env bash
# ==============================================================================
# 三阶段完整训练流水线（方案 B：三通道分拆 VQ 编码器）
#
# 阶段 0  三通道分拆预训练       train_pretrain_split.py
#           enc1(floor+wall) / enc2(+door+mob+entrance) / enc3(全图)
#           各自仅对本通道 tile 计算 masked Focal Loss
# 阶段 1  MaskGIT 热身          train_vq.py  --pretrain_split  --freeze_vq True
#           三路编码器权重冻结，仅训练 MaskGIT
# 阶段 2  完整联合训练           train_vq.py
#           三路编码器 + MaskGIT 全量优化
#
# 用法：
#   bash train_full.sh                  # 从头开始三阶段训练
#   bash train_full.sh --skip 1         # 跳过阶段 0，从阶段 1 开始
#   bash train_full.sh --skip 2         # 跳过阶段 0-1，直接阶段 2
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 超参配置（按需修改）
# ------------------------------------------------------------------------------
TRAIN_DATA="ginka-dataset.json"
EVAL_DATA="ginka-eval.json"

# 阶段 0：三通道分拆预训练
P0_EPOCHS=100
P0_CHECKPOINT=10
P0_FINAL="result/pretrain_split/split_final.pth"

# 阶段 1：冻结编码器热身
P1_EPOCHS=50
P1_CHECKPOINT=10
P1_FINAL="result/joint/warmup_final.pth"

# 阶段 2：完整联合训练
P2_EPOCHS=400
P2_CHECKPOINT=20

# 从哪个阶段开始（0 = 从头）；命令行 --skip N 可覆盖此值
START_PHASE=0

# ------------------------------------------------------------------------------
# 解析命令行参数
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)
            START_PHASE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"; exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
log() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

# ------------------------------------------------------------------------------
# 阶段 0：三通道分拆 VQ 编码器预训练
# ------------------------------------------------------------------------------
if [[ $START_PHASE -le 0 ]]; then
    log "阶段 0 / 3  三通道分拆 VQ 预训练  (epochs=${P0_EPOCHS})"
    mkdir -p result/pretrain_split
    python3 -u -m ginka.train_pretrain_split \
        --train       "$TRAIN_DATA"   \
        --validate    "$EVAL_DATA"    \
        --epochs      "$P0_EPOCHS"    \
        --checkpoint  "$P0_CHECKPOINT"

    [[ -f "$P0_FINAL" ]] || die "阶段 0 未生成预期检查点：$P0_FINAL"
    log "阶段 0 完成 → $P0_FINAL"
else
    [[ -f "$P0_FINAL" ]] || die "跳过阶段 0 但找不到检查点：$P0_FINAL"
    log "阶段 0 已跳过（使用现有检查点 $P0_FINAL）"
fi

# ------------------------------------------------------------------------------
# 阶段 1：MaskGIT 热身（三路 VQ 编码器冻结）
# ------------------------------------------------------------------------------
if [[ $START_PHASE -le 1 ]]; then
    log "阶段 1 / 3  MaskGIT 热身（三路 VQ 冻结）  (epochs=${P1_EPOCHS})"
    mkdir -p result/joint
    python3 -u -m ginka.train_vq \
        --train          "$TRAIN_DATA"   \
        --validate       "$EVAL_DATA"    \
        --pretrain_split "$P0_FINAL"     \
        --load_optim     False           \
        --freeze_vq      True            \
        --epochs         "$P1_EPOCHS"    \
        --checkpoint     "$P1_CHECKPOINT"

    # 取阶段 1 最后一个检查点，固定为阶段 2 入口
    _P1_LAST=$(ls -t result/joint/joint-*.pth 2>/dev/null | head -1)
    [[ -n "$_P1_LAST" ]] || die "阶段 1 未生成任何检查点（result/joint/joint-*.pth）"
    cp "$_P1_LAST" "$P1_FINAL"
    log "阶段 1 完成 → $P1_FINAL（来自 $_P1_LAST）"
else
    [[ -f "$P1_FINAL" ]] || die "跳过阶段 1 但找不到检查点：$P1_FINAL"
    log "阶段 1 已跳过（使用现有检查点 $P1_FINAL）"
fi

# ------------------------------------------------------------------------------
# 阶段 2：完整联合训练（三路编码器 + MaskGIT 全量）
# ------------------------------------------------------------------------------
if [[ $START_PHASE -le 2 ]]; then
    log "阶段 2 / 3  完整联合训练  (epochs=${P2_EPOCHS})"
    python3 -u -m ginka.train_vq \
        --train       "$TRAIN_DATA"   \
        --validate    "$EVAL_DATA"    \
        --resume      True            \
        --state       "$P1_FINAL"     \
        --load_optim  False           \
        --freeze_vq   False           \
        --epochs      "$P2_EPOCHS"    \
        --checkpoint  "$P2_CHECKPOINT"

    log "阶段 2 完成"
fi

# ------------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  三阶段训练全部完成                      ║"
echo "╚══════════════════════════════════════════╝"
