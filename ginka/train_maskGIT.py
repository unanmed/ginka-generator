import argparse
import os
import sys
import random
import math
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from .maskGIT.model import GinkaMaskGIT
from .dataset import GinkaMaskGITDataset
from shared.image import matrix_to_image_cv
from .maskGIT.mask import MapMask

# 手工标注标签定义（暂时不用）：
# 0. 蓝海, 1. 红海, 2: 室内, 3. 野外, 4. 左右对称, 5. 上下对称, 6. 伪对称, 7. 咸鱼层,
# 8. 剧情层, 9. 水层, 10. 爽塔, 11. Boss层, 12. 纯Boss层, 13. 多房间, 14. 多走廊, 15. 道具风
# 16. 区域入口, 17. 区域连接, 18. 有机关门, 19. 道具层, 20. 斜向对称, 21. 左右通道, 22. 上下通道, 23. 多机关门
# 24. 中心对称, 25. 部分对称, 26. 鱼骨

# 自动标注标签定义（暂时不用）：
# 0. 左右对称, 1. 上下对称, 2. 中心对称, 3. 斜向对称, 4. 伪对称, 5. 多房间, 6. 多走廊
# 32. 平面塔, 33. 转换塔, 34. 道具塔

# 标量值定义：
# 0. 整体密度，非空白图块/地图面积，空白图块还包括装饰图块
# 1. 墙体密度，墙壁/地图面积
# 2. 门密度，门数量/地图面积
# 3. 怪物密度，怪物数量/地图面积
# 4. 资源密度，资源数量/地图面积
# 5. 宝石密度，宝石数量/地图面积
# 6. 血瓶密度，血瓶数量/地图面积
# 7. 钥匙密度，钥匙数量/地图面积
# 8. 道具密度，道具数量/地图面积
# 9. 入口数量

# 图块定义：
# 0. 空地, 1. 墙壁, 2. 门, 3. 钥匙, 4. 红宝石, 5. 蓝宝石, 6. 绿宝石, 7. 血瓶
# 8. 道具, 9. 怪物, 10. 入口, 15. 掩码 token

BATCH_SIZE = 128
VAL_BATCH_DIVIDER = 128
NUM_CLASSES = 16
MASK_TOKEN = 15
GENERATE_STEP = 8
MAP_SIZE = 13 * 13
MASK_PROBS = [0.5, 0.5] # 纯随机，分块随机

device = torch.device(
    "cuda:1" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available()
    else "cpu"
)
os.makedirs("result", exist_ok=True)
os.makedirs("result/transformer", exist_ok=True)
os.makedirs("result/transformer_img", exist_ok=True)

disable_tqdm = not sys.stdout.isatty()

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--state_ginka", type=str, default="result/vae/ginka-100.pth")
    parser.add_argument("--train", type=str, default="ginka-dataset.json")
    parser.add_argument("--validate", type=str, default="ginka-eval.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--load_optim", type=bool, default=True)
    args = parser.parse_args()
    return args

def train():
    print(f"Using {device.type} to train model.")
    
    args = parse_arguments()
    
    model = GinkaMaskGIT(num_classes=NUM_CLASSES).to(device)
    masker = MapMask([0.5, 0.5])
    
    dataset = GinkaMaskGITDataset(args.train, device)
    dataset_val = GinkaMaskGITDataset(args.validate, device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE // VAL_BATCH_DIVIDER, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # 自定义调度器允许在 self_prob 提高时重置调度器信息并提高学习率以适应学习
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 用于生成图片
    tile_dict = dict()
    for file in os.listdir('tiles2'):
        name = os.path.splitext(file)[0]
        tile_dict[name] = cv2.imread(f"tiles2/{file}", cv2.IMREAD_UNCHANGED)
    
    # 接续训练
    if args.resume:
        data_ginka = torch.load(args.state_ginka, map_location=device)

        model.load_state_dict(data_ginka["model_state"], strict=False)
        
        if args.load_optim:
            if data_ginka.get("optim_state") is not None:
                optimizer.load_state_dict(data_ginka["optim_state"])
    
        print("Train from loaded state.")
        
    for epoch in tqdm(range(args.epochs), desc="VAE Training", disable=disable_tqdm):
        loss_total = torch.Tensor([0]).to(device)
        
        for batch in tqdm(dataloader, leave=False, desc="Epoch Progress", disable=disable_tqdm):
            target_map = batch["target_map"].to(device)
            cond = batch["cond"].to(device)
            heatmap = batch["heatmap"].to(device)
            B, H, W = target_map.shape

            target_map = target_map.view(B, H * W)
            rand = torch.randn_like(heatmap).to(device) * 0.05
            if random.random() > 0.5:
                heatmap = heatmap + rand
            
            mask = np.zeros((B, H * W))
            for i in range(B):
                mask[i] = masker.mask(H, W)
            
            mask = torch.from_numpy(mask).to(torch.bool).to(device)
            
            # 掩码
            masked_input = target_map.clone()
            masked_input[mask] = MASK_TOKEN # 填充为 [MASK] 标记
            
            logits = model(masked_input, cond, heatmap)
            
            loss = F.cross_entropy(logits.permute(0, 2, 1), target_map, reduction='none', label_smoothing=0.1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
            
            optimizer.zero_grad()
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
            }, f"result/transformer/ginka-{epoch + 1}.pth")
        
            val_loss_total = torch.Tensor([0]).to(device)
            model.eval()
            with torch.no_grad():
                idx = 0
                gap = 5
                color = (255, 255, 255)  # 白色
                vline = np.full((416, gap, 3), color, dtype=np.uint8)  # 垂直分割线
                for batch in tqdm(dataloader_val, desc="Validating generator.", leave=False, disable=disable_tqdm):
                    # 1. 常规生成
                    target_map = batch["target_map"].to(device)
                    cond = batch["cond"].to(device)
                    heatmap = batch["heatmap"].to(device)
                    B, H, W = target_map.shape
                    target_map = target_map.view(B, H * W)
                    
                    mask = np.zeros((B, H * W))
                    for i in range(B):
                        mask[i] = masker.mask(H, W)
                    
                    mask = torch.from_numpy(mask).to(torch.bool).to(device)
                    
                    # 2. 生成掩码矩阵
                    masked_input = target_map.clone()
                    masked_input[mask] = MASK_TOKEN # 填充为 [MASK] 标记
                    
                    logits = model(masked_input, cond, heatmap)
                    
                    loss = F.cross_entropy(logits.permute(0, 2, 1), target_map, reduction='none', label_smoothing=0.1)
                    loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-6)
                    
                    val_loss_total += loss.detach()
                    
                    fake_map = torch.argmax(logits, dim=2).view(B, H, W).cpu().numpy()
                    fake_img = matrix_to_image_cv(fake_map[0], tile_dict)
                    real_map = target_map.view(B, H, W).cpu().numpy()
                    real_img = matrix_to_image_cv(real_map[0], tile_dict)
                    img = np.block([[real_img], [vline], [fake_img]])
                    cv2.imwrite(f"result/transformer_img/{idx}.png", img)
                    
                    idx += 1
                    
                    # 2. 从头完整生成
                    map = torch.full((1, MAP_SIZE), MASK_TOKEN).to(device)
                    for i in range(GENERATE_STEP):
                        # 1. 预测
                        logits = model(map, cond, heatmap) # [1, H * W, num_classes]
                        probs = F.softmax(logits, dim=-1)
                        
                        # 2. 采样（为了多样性，这里可以使用概率采样而不是取最大值）
                        dist = torch.distributions.Categorical(probs)
                        sampled_tiles = dist.sample() # [1, H * W]
                        
                        # 3. 计算置信度 (模型对采样结果的信心程度)
                        confidences = torch.gather(probs, -1, sampled_tiles.unsqueeze(-1)).squeeze(-1)
                        
                        # 4. 决定本轮要固定多少个格子 (上凸函数逻辑)
                        ratio = math.cos(((i + 1) / GENERATE_STEP) * math.pi / 2)
                        num_to_mask = math.floor(ratio * MAP_SIZE)
                        
                        # 5. 更新画布：保留置信度最高的部分，其余位置设回 MASK
                        # 注意：这里逻辑上通常是保留当前步预测中置信度最高的，并结合已有的非 mask 部分
                        if num_to_mask > 0:
                            _, mask_indices = torch.topk(confidences, k=num_to_mask, largest=False)
                            sampled_tiles = sampled_tiles.scatter(1, mask_indices, MASK_TOKEN)
                        
                        map = sampled_tiles
                        if (map == MASK_TOKEN).sum() == 0:
                            break
                        
                    generated_img = matrix_to_image_cv(map.view(1, H, W)[0].cpu().numpy(), tile_dict)
                    cv2.imwrite(f"result/transformer_img/g-{idx}.png", generated_img)
                    
            avg_loss_val = val_loss_total.item() / len(dataloader_val)
            tqdm.write(
                f"[Validate {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] E: {epoch + 1} | " +
                f"Loss: {avg_loss_val:.6f}"
            )
            
    print("Train ended.")
    torch.save({
        "model_state": model.state_dict(),
    }, f"result/ginka_transformer.pth")
    

if __name__ == "__main__":
    torch.set_num_threads(4)
    train()
