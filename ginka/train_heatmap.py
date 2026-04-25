import argparse
import os
import sys
import math
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from perlin_numpy import generate_fractal_noise_2d
from tqdm import tqdm
from torch.utils.data import DataLoader
from .maskGIT.model import GinkaMaskGIT
from .dataset import GinkaHeatmapDataset
from shared.image import matrix_to_image_cv
from .heatmap.model import GinkaHeatmapModel
from .heatmap.diffusion import Diffusion
from .utils import nms_sampling

# 图块定义：
# 0. 空地, 1. 墙壁, 2. 门, 3. 钥匙, 4. 红宝石, 5. 蓝宝石, 6. 绿宝石, 7. 血瓶
# 8. 道具, 9. 怪物, 10. 入口, 15. 掩码 token

# 热力图定义
# 0. 墙壁热力图, 1. 怪物热力图, 2. 资源热力图, 3. 血瓶热力图, 4. 宝石热力图, 5. 钥匙热力图
# 6. 道具热力图, 7. 入口热力图, 8. 门热力图

BATCH_SIZE = 128
VAL_BATCH_DIVIDER = 64
NUM_CLASSES = 16
MASK_TOKEN = 15
GENERATE_STEP = 8
MAP_W = 13
MAP_H = 13
HEATMAP_CHANNEL = 9
LABEL_SMOOTHING = 0
BLUR_MIN_SIZE = 3
BLUR_MAX_SIZE = 9
RAND_RATIO = 0.15
# MaskGIT 生成设置
USE_MASK_GIT_PREVIEW = True
NUM_LAYERS = 4
D_MODEL = 192
# Diffusion 生成设置
NUM_LAYERS_DIFFUSION = 4
D_MODEL_DIFFUSION = 128
T_DIFFUSION = 100
MIN_MASK = 0
MAX_MASK = 1
W = 5 # CFG 参数

device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)
os.makedirs("result", exist_ok=True)
os.makedirs("result/heatmap", exist_ok=True)
os.makedirs("result/final_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/heatmap/ginka-100.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    parser.add_argument("--use_maskgit", type=bool, default=True)
    parser.add_argument("--maskgit_path", type=str, default="result/ginka_transformer.pth")
    args = parser.parse_args()
    return args

def train():
    print(f"Using {device.type} to train model.")
    
    args = parse_arguments()
    
    if args.use_maskgit:
        maskGIT = GinkaMaskGIT(
            num_classes=NUM_CLASSES, heatmap_channel=HEATMAP_CHANNEL,
            num_layers=NUM_LAYERS, d_model=D_MODEL
        ).to(device)
        maskGIT.eval()
    model = GinkaHeatmapModel(
        T=T_DIFFUSION, heatmap_dim=HEATMAP_CHANNEL, d_model=D_MODEL_DIFFUSION,
        num_layers=NUM_LAYERS_DIFFUSION
    ).to(device)
    
    diffusion = Diffusion(device)
    
    dataset = GinkaHeatmapDataset(args.train, min_mask=MIN_MASK, max_mask=MAX_MASK)
    dataset_val = GinkaHeatmapDataset(args.validate, min_mask=MIN_MASK, max_mask=MAX_MASK)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE // VAL_BATCH_DIVIDER, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles/{file}", cv2.IMREAD_UNCHANGED)
    
    # 接续训练
    if args.resume:
        data_ginka = torch.load(args.state_ginka, map_location=device)

        model.load_state_dict(data_ginka["model_state"], strict=False)
        
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer.load_state_dict(data_ginka["optim_state"])
    
        print("Train from loaded state.")
        
    if args.use_maskgit:
        data_maskGIT = torch.load(args.maskgit_path, map_location=device)
        maskGIT.load_state_dict(data_maskGIT["model_state"])
        print("Loaded MaskGIT model state.")
    
    for epoch in tqdm(range(args.epochs), desc="Diffusion Training", disable=disable_tqdm):
        loss_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            cond_heatmap = batch["cond_heatmap"].to(device)
            target_heatmap = batch["target_heatmap"].to(device) * 2 - 1
            B, C, H, W = target_heatmap.shape

            optimizer.zero_grad()

            t = torch.randint(1, T_DIFFUSION, [B], device=device)
            noise = torch.randn_like(target_heatmap)
            
            x_t = diffusion.q_sample(target_heatmap, t, noise)
            
            # CFG 随机概率没有输入条件
            if np.random.rand() < 0.2:
                cond_heatmap = torch.zeros_like(cond_heatmap)
            
            pred_noise = model(x_t, cond_heatmap, t)
            
            loss = F.mse_loss(pred_noise, noise)
            
            loss.backward()
            optimizer.step()
            loss_total += loss.detach()
            
        scheduler.step()
                
        avg_loss = loss_total.item() / len(dataloader)
        tqdm.write(
            f"[Epoch {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"E: {epoch + 1} | Loss: {avg_loss:.6f} | " +
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # 每若干轮输出一次图片，并保存检查点
        if (epoch + 1) % args.checkpoint == 0:
            # 保存检查点
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, f"result/heatmap/ginka-{epoch + 1}.pth")
        
            val_loss_total = torch.Tensor([0]).to(device)
            model.eval()
            with torch.no_grad():
                idx = 0
                for batch in tqdm(dataloader_val, desc="Validating", leave=False, disable=disable_tqdm):
                    # 1. 验证集验证
                    cond_heatmap = batch["cond_heatmap"].to(device)
                    target_heatmap = batch["target_heatmap"].to(device) * 2 - 1
                    B, C, H, W = target_heatmap.shape

                    t = torch.randint(1, T_DIFFUSION, [B], device=device)
                    noise = torch.randn_like(target_heatmap)
                    
                    x_t = diffusion.q_sample(target_heatmap, t, noise)
                    
                    pred_noise = model(x_t, cond_heatmap, t)
                    
                    loss = F.mse_loss(pred_noise, noise)
                    
                    val_loss_total += loss.detach()
                
                    # 2. 从头完整生成，并使用训练好的 MaskGIT 生成地图
                    if args.use_maskgit:
                        map = full_generate(model, maskGIT, cond_heatmap, diffusion)
                            
                        generated_img = matrix_to_image_cv(map.view(B, H, W)[0].cpu().numpy(), tile_dict)
                        cv2.imwrite(f"result/final_img/{idx}.png", generated_img)
                    
                    idx += 1
                    
                # 3. 完全随机生成五张图
                if args.use_maskgit:
                    for i in range(0, 5):
                        ar = np.ndarray([1, HEATMAP_CHANNEL, MAP_H, MAP_W])
                        k = get_nms_sampling_count()
                        for c in range(0, HEATMAP_CHANNEL):
                            noise = generate_fractal_noise_2d((16, 16), (4, 4), 1)[0:MAP_H,0:MAP_W]
                            ar[0,c] = nms_sampling(noise, k[c])
                        
                        map = full_generate(model, maskGIT, torch.FloatTensor(ar).to(device), diffusion)
                        generated_img = matrix_to_image_cv(map.view(1, H, W)[0].cpu().numpy(), tile_dict)
                        cv2.imwrite(f"result/final_img/g-{i}.png", generated_img)
                    
            avg_loss_val = val_loss_total.item() / len(dataloader_val)
            tqdm.write(
                f"[Validate {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] E: {epoch + 1} | " +
                f"Loss: {avg_loss_val:.6f}"
            )
            
    print("Train ended.")
    torch.save({
        "model_state": model.state_dict(),
    }, f"result/ginka_heatmap.pth")
    
def get_nms_sampling_count():
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

def full_generate(heatmap, maskGIT, cond_heatmap: torch.Tensor, diffusion: Diffusion):
    fake_heatmap_cond = (diffusion.sample(heatmap, cond_heatmap) + 1) / 2
    fake_heatmap_uncond = (diffusion.sample(heatmap, torch.zeros_like(cond_heatmap)) + 1) / 2
    fake_heatmap = fake_heatmap_uncond + W * (fake_heatmap_uncond - fake_heatmap_cond) # [B, C, H, W]
    return maskGIT_generate(maskGIT, cond_heatmap.shape[0], fake_heatmap)

def maskGIT_generate(maskGIT, B: int, heatmap: torch.Tensor):
    # heatmap: [B, C, H, W]
    map = torch.full((B, MAP_H * MAP_W), MASK_TOKEN).to(device)
    for i in range(GENERATE_STEP):
        # 1. 预测
        logits = maskGIT(map, heatmap) # [1, H * W, num_classes]
        probs = F.softmax(logits, dim=-1)
        
        # 2. 采样（为了多样性，这里可以使用概率采样而不是取最大值）
        dist = torch.distributions.Categorical(probs)
        sampled_tiles = dist.sample() # [1, H * W]
        
        # 3. 计算置信度 (模型对采样结果的信心程度)
        confidences = torch.gather(probs, -1, sampled_tiles.unsqueeze(-1)).squeeze(-1)
        
        # 4. 决定本轮要固定多少个格子 (上凸函数逻辑)
        ratio = math.cos(((i + 1) / GENERATE_STEP) * math.pi / 2)
        num_to_mask = math.floor(ratio * MAP_H * MAP_W)
        
        # 5. 更新画布：保留置信度最高的部分，其余位置设回 MASK
        # 注意：这里逻辑上通常是保留当前步预测中置信度最高的，并结合已有的非 mask 部分
        if num_to_mask > 0:
            _, mask_indices = torch.topk(confidences, k=num_to_mask, largest=False)
            sampled_tiles = sampled_tiles.scatter(1, mask_indices, MASK_TOKEN)
        
        map = sampled_tiles
        if (map == MASK_TOKEN).sum() == 0:
            break
        
    return map
    

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
