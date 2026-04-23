import argparse
import math
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from perlin_numpy import generate_fractal_noise_2d
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import GinkaJointDataset
from .heatmap.diffusion import Diffusion
from .heatmap.model import GinkaHeatmapModel
from .maskGIT.model import GinkaMaskGIT
from .utils import nms_sampling
from shared.image import matrix_to_image_cv


# 地图与 token 基础配置
NUM_CLASSES = 16
MASK_TOKEN = 15
MAP_W = 13
MAP_H = 13
HEATMAP_CHANNEL = 9
GENERATE_STEP = 8

# 训练批次与损失配置
BATCH_SIZE = 64
VAL_BATCH_DIVIDER = 64
LABEL_SMOOTHING = 0
CE_WEIGHT = 0.5  # 联合训练里 MaskGIT 监督项的权重
DROP_RATE = 0.2  # CFG 训练时随机丢弃条件热力图的概率

# MaskGIT 模型结构
NUM_LAYERS = 4
D_MODEL = 192

# Diffusion 模型结构与噪声过程
NUM_LAYERS_DIFFUSION = 4
D_MODEL_DIFFUSION = 128
T_DIFFUSION = 100
MIN_MASK = 0
MAX_MASK = 1

# 验证预览配置
PREVIEW_CFG_WEIGHT = 5  # 预览生成时使用的 CFG 强度
RANDOM_PREVIEW_COUNT = 5  # 每次验证额外生成的随机预览数量

device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)
os.makedirs("result", exist_ok=True)
os.makedirs("result/joint", exist_ok=True)
os.makedirs("result/joint_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()


def parse_arguments():
    # 解析联合训练脚本的命令行参数。
    parser = argparse.ArgumentParser(description="joint training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_heatmap", type=str, default="result/ginka_heatmap.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    parser.add_argument("--maskgit_path", type=str, default="result/ginka_transformer.pth")
    args = parser.parse_args()
    return args


def load_heatmap_checkpoint(model, optimizer, args):
    # 加载预训练 Diffusion 权重，并在需要时恢复优化器状态。
    if not args.state_heatmap:
        return

    if not os.path.exists(args.state_heatmap):
        raise FileNotFoundError(f"Heatmap checkpoint not found: {args.state_heatmap}")

    checkpoint = torch.load(args.state_heatmap, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    if args.resume and args.load_optim and checkpoint.get("optim_state") is not None:
        optimizer.load_state_dict(checkpoint["optim_state"])

    print("Loaded Diffusion model state.")


def freeze_module(module: torch.nn.Module):
    # 冻结模块参数，使其在联合训练中只作为固定监督器使用。
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def predict_x0(diffusion: Diffusion, x_t: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor):
    # 根据当前时刻的噪声预测还原 x0 热力图估计。
    sqrt_ab = diffusion.sqrt_ab[t][:, None, None, None]
    sqrt_one_minus_ab = diffusion.sqrt_one_minus_ab[t][:, None, None, None]
    x0 = (x_t - sqrt_one_minus_ab * pred_noise) / sqrt_ab
    return x0


def maskgit_joint_loss(maskgit, generated_heatmap: torch.Tensor, target_map: torch.Tensor):
    # 用冻结的 MaskGIT 对 Diffusion 生成的热力图施加地图级监督。
    batch_size, height, width = target_map.shape
    target_tokens = target_map.view(batch_size, height * width)
    canvas = torch.full_like(target_tokens, MASK_TOKEN)
    losses = []

    for step in range(GENERATE_STEP):
        current_mask = canvas == MASK_TOKEN
        if current_mask.sum().item() == 0:
            break

        # 保证前向传播可导
        logits = maskgit(canvas, generated_heatmap)
        ce = F.cross_entropy(
            logits.permute(0, 2, 1),
            target_tokens,
            label_smoothing=LABEL_SMOOTHING
        )
        losses.append(ce)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            sampled_tiles = torch.argmax(probs, dim=-1)
            confidences = torch.gather(probs, -1, sampled_tiles.unsqueeze(-1)).squeeze(-1)

            ratio = math.cos(((step + 1) / GENERATE_STEP) * math.pi / 2)
            num_to_mask = math.floor(ratio * target_tokens.shape[1])

            if num_to_mask > 0:
                _, mask_indices = torch.topk(confidences, k=num_to_mask, largest=False)
                sampled_tiles = sampled_tiles.scatter(1, mask_indices, MASK_TOKEN)

            canvas = sampled_tiles

    if not losses:
        return torch.zeros((), device=generated_heatmap.device)

    return torch.stack(losses).mean()


def load_tile_dict():
    # 加载用于可视化地图的图块贴图。
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
    return tile_dict


def get_nms_sampling_count():
    # 为随机点图预览采样每个通道的点数量。
    return [
        np.random.randint(20, 40),
        np.random.randint(10, 20),
        np.random.randint(10, 30),
        np.random.randint(4, 12),
        np.random.randint(4, 12),
        np.random.randint(2, 6),
        np.random.randint(0, 2),
        np.random.randint(1, 3),
        np.random.randint(2, 10)
    ]


def maskgit_generate(maskgit, batch_size: int, heatmap: torch.Tensor):
    # 使用冻结的 MaskGIT 把热力图解码为完整地图。
    generated_map = torch.full((batch_size, MAP_H * MAP_W), MASK_TOKEN, device=device)
    for step in range(GENERATE_STEP):
        logits = maskgit(generated_map, heatmap)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        sampled_tiles = dist.sample()
        confidences = torch.gather(probs, -1, sampled_tiles.unsqueeze(-1)).squeeze(-1)

        ratio = math.cos(((step + 1) / GENERATE_STEP) * math.pi / 2)
        num_to_mask = math.floor(ratio * MAP_H * MAP_W)

        if num_to_mask > 0:
            _, mask_indices = torch.topk(confidences, k=num_to_mask, largest=False)
            sampled_tiles = sampled_tiles.scatter(1, mask_indices, MASK_TOKEN)

        generated_map = sampled_tiles
        if (generated_map == MASK_TOKEN).sum() == 0:
            break

    return generated_map


def full_generate(heatmap_model, maskgit, cond_heatmap: torch.Tensor, diffusion: Diffusion):
    # 执行完整预览生成流程：点图 -> 热力图 -> 地图。
    fake_heatmap_cond = diffusion.sample(heatmap_model, cond_heatmap)
    fake_heatmap_uncond = diffusion.sample(heatmap_model, torch.zeros_like(cond_heatmap))
    fake_heatmap = fake_heatmap_uncond + PREVIEW_CFG_WEIGHT * (fake_heatmap_uncond - fake_heatmap_cond)
    return maskgit_generate(maskgit, cond_heatmap.shape[0], fake_heatmap)


def save_random_previews(model, maskgit, diffusion, tile_dict):
    # 额外生成随机点图预览，便于观察模型的开放式生成效果。
    for preview_idx in range(RANDOM_PREVIEW_COUNT):
        cond_array = np.ndarray([1, HEATMAP_CHANNEL, MAP_H, MAP_W])
        sampling_count = get_nms_sampling_count()
        for channel in range(HEATMAP_CHANNEL):
            noise = generate_fractal_noise_2d((16, 16), (4, 4), 1)[0:MAP_H, 0:MAP_W]
            cond_array[0, channel] = nms_sampling(noise, sampling_count[channel])

        generated_map = full_generate(model, maskgit, torch.FloatTensor(cond_array).to(device), diffusion)
        generated_img = matrix_to_image_cv(generated_map.view(1, MAP_H, MAP_W)[0].cpu().numpy(), tile_dict)
        cv2.imwrite(f"result/joint_img/g-{preview_idx}.png", generated_img)


def validate(model, maskgit, diffusion, dataloader, ce_weight: float, tile_dict):
    # 执行数值验证，并保存生成地图预览图。
    model.eval()
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_maskgit_loss = 0.0

    with torch.no_grad():
        preview_idx = 0
        for batch in tqdm(dataloader, desc="Validating", leave=False, disable=disable_tqdm):
            cond_heatmap = batch["cond_heatmap"].to(device)
            target_heatmap = batch["target_heatmap"].to(device)
            target_map = batch["target_map"].to(device)
            batch_size, _, map_height, map_width = target_heatmap.shape

            t = torch.randint(1, T_DIFFUSION, [batch_size], device=device)
            noise = torch.randn_like(target_heatmap)
            x_t = diffusion.q_sample(target_heatmap, t, noise)

            pred_noise = model(x_t, cond_heatmap, t)
            diffusion_loss = F.mse_loss(pred_noise, noise)

            generated_heatmap = predict_x0(diffusion, x_t, pred_noise, t)
            maskgit_loss = maskgit_joint_loss(maskgit, generated_heatmap, target_map)

            loss = diffusion_loss + ce_weight * maskgit_loss
            total_loss += loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_maskgit_loss += maskgit_loss.item()

            # 预览生成结果
            generated_map = full_generate(model, maskgit, cond_heatmap, diffusion)
            generated_img = matrix_to_image_cv(
                generated_map.view(batch_size, map_height, map_width)[0].cpu().numpy(),
                tile_dict,
            )
            cv2.imwrite(f"result/joint_img/{preview_idx}.png", generated_img)
            preview_idx += 1

        save_random_previews(model, maskgit, diffusion, tile_dict)

    size = max(len(dataloader), 1)
    return {
        "loss": total_loss / size,
        "diffusion_loss": total_diffusion_loss / size,
        "maskgit_loss": total_maskgit_loss / size,
    }


def train():
    # 联合训练 Diffusion，使其同时受到噪声重建和冻结 MaskGIT 的监督。
    print(f"Using {device.type} to train model.")

    args = parse_arguments()
    tile_dict = load_tile_dict()

    maskgit = GinkaMaskGIT(
        num_classes=NUM_CLASSES,
        heatmap_channel=HEATMAP_CHANNEL,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
    ).to(device)
    if not os.path.exists(args.maskgit_path):
        raise FileNotFoundError(f"MaskGIT checkpoint not found: {args.maskgit_path}")
    maskgit_state = torch.load(args.maskgit_path, map_location=device)
    maskgit.load_state_dict(maskgit_state["model_state"])
    freeze_module(maskgit)
    print("Loaded and froze MaskGIT model state.")

    model = GinkaHeatmapModel(
        T=T_DIFFUSION,
        heatmap_dim=HEATMAP_CHANNEL,
        d_model=D_MODEL_DIFFUSION,
        num_layers=NUM_LAYERS_DIFFUSION,
    ).to(device)
    diffusion = Diffusion(device, T=T_DIFFUSION)

    dataset = GinkaJointDataset(args.train, min_mask=MIN_MASK, max_mask=MAX_MASK)
    dataset_val = GinkaJointDataset(args.validate, min_mask=MIN_MASK, max_mask=MAX_MASK)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=max(1, BATCH_SIZE // VAL_BATCH_DIVIDER),
        shuffle=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    load_heatmap_checkpoint(model, optimizer, args)

    for epoch in tqdm(range(args.epochs), desc="Joint Training", disable=disable_tqdm):
        model.train()
        epoch_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_maskgit_loss = 0.0

        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            cond_heatmap = batch["cond_heatmap"].to(device)
            target_heatmap = batch["target_heatmap"].to(device)
            target_map = batch["target_map"].to(device)
            batch_size = target_heatmap.shape[0]

            optimizer.zero_grad()

            t = torch.randint(1, T_DIFFUSION, [batch_size], device=device)
            noise = torch.randn_like(target_heatmap)
            x_t = diffusion.q_sample(target_heatmap, t, noise)

            cond_for_diffusion = cond_heatmap
            use_unconditional_branch = False
            if np.random.rand() < DROP_RATE:
                cond_for_diffusion = torch.zeros_like(cond_heatmap)
                use_unconditional_branch = True

            pred_noise = model(x_t, cond_for_diffusion, t)
            diffusion_loss = F.mse_loss(pred_noise, noise)

            pred_noise_for_joint = pred_noise
            if use_unconditional_branch:
                pred_noise_for_joint = model(x_t, cond_heatmap, t)

            generated_heatmap = predict_x0(diffusion, x_t, pred_noise_for_joint, t)
            maskgit_loss = maskgit_joint_loss(maskgit, generated_heatmap, target_map)

            loss = diffusion_loss + CE_WEIGHT * maskgit_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_maskgit_loss += maskgit_loss.item()

        scheduler.step()

        train_size = max(len(dataloader), 1)
        tqdm.write(
            f"[Epoch {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"E: {epoch + 1} | "
            f"Loss: {epoch_loss / train_size:.6f} | "
            f"Diffusion: {epoch_diffusion_loss / train_size:.6f} | "
            f"MaskGIT: {epoch_maskgit_loss / train_size:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if (epoch + 1) % args.checkpoint == 0:
            checkpoint_path = f"result/joint/ginka-joint-{epoch + 1}.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )

            metrics = validate(model, maskgit, diffusion, dataloader_val, CE_WEIGHT, tile_dict)
            tqdm.write(
                f"[Validate {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"E: {epoch + 1} | "
                f"Loss: {metrics['loss']:.6f} | "
                f"Diffusion: {metrics['diffusion_loss']:.6f} | "
                f"MaskGIT: {metrics['maskgit_loss']:.6f}"
            )

    print("Train ended.")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        "result/ginka_joint_heatmap.pth",
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    train()